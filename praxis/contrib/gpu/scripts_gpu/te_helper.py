import os

from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes

try:
    import transformer_engine.jax as te
    import transformer_engine.jax.flax as te_flax
    import transformer_engine.jax.praxis as te_praxis
    _IS_TRANSFORMER_ENGINE_INSTALLED = True
    import praxis.layers.repeats as praxis_repeat
    # This is to make Repeat module correctly generate collections we need.
    praxis_repeat.SCAN_VARIABLE_AXES.update({base_layer.NON_PAX_VAR_COLLECTION[1]: 0, # 1-idx = params_axes
                                            te.fp8.FP8Helper.FP8_COLLECTION_NAME:0})
    TE_PIPELINE_EXTRA_VMAP_VAR_AXES = {
        base_layer.NON_PAX_VAR_COLLECTION[1]: 0, # 1-idx = params_axes
        te.fp8.FP8Helper.FP8_COLLECTION_NAME:0
    }

    TE_PIPELINE_EXTRA_SCAN_VAR_BROADCAST = [te.fp8.FP8Helper.FP8_COLLECTION_NAME]

except ModuleNotFoundError as e:
    _IS_TRANSFORMER_ENGINE_INSTALLED = False
    TE_PIPELINE_EXTRA_VMAP_VAR_AXES = {}
    TE_PIPELINE_EXTRA_SCAN_VAR_BROADCAST = []

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor


class TransformerEngineHelperBase:

    @staticmethod
    def get_fprop_caller_of_stack_transformer(fprop, deterministic):
        raise NotImplementedError

    @staticmethod
    def set_layer_params_to_stack_transformer(stacked_transformer_obj, layer_p, layer_id):
        raise NotImplementedError

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(xformer_layer_p):
        raise NotImplementedError



class TENotInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    def get_fprop_caller_of_stack_transformer(fprop, deterministic):
        return fprop

    @staticmethod
    def set_layer_params_to_stack_transformer(stacked_transformer_obj, layer_p, layer_id):
        layer_p.name = f'layer_{layer_id}'
        layer_p.use_cross_attention = stacked_transformer_obj.use_cross_attention
        layer_p.num_heads = stacked_transformer_obj.num_heads
        layer_p.dim_per_head = stacked_transformer_obj.dim_per_head
        layer_p.input_dims = stacked_transformer_obj.model_dims
        layer_p.packed_input = stacked_transformer_obj.packed_input
        layer_p.atten_dropout_prob = stacked_transformer_obj.atten_dropout_prob or stacked_transformer_obj.dropout_prob
        layer_p.residual_dropout_prob = (
            stacked_transformer_obj.residual_dropout_prob or stacked_transformer_obj.dropout_prob
        )
        layer_p.relu_dropout_prob = stacked_transformer_obj.relu_dropout_prob or stacked_transformer_obj.dropout_prob
        layer_p.hidden_dims = stacked_transformer_obj.hidden_dims
        if stacked_transformer_obj.local_window_size is not None:
            if isinstance(stacked_transformer_obj.local_window_size[0], tuple):
                p_i.tr_atten_tpl.local_window_size = stacked_transformer_obj.local_window_size[i]
            else:
                p_i.tr_atten_tpl.local_window_size = stacked_transformer_obj.local_window_size

        if stacked_transformer_obj.residual_droppath_prob > 0.0:
            layer_p.residual_droppath_prob = (
                stacked_transformer_obj.residual_droppath_prob * layer_id / max(1, stacked_transformer_obj.num_layers)
            )
        return layer_p

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(xformer_layer_p):
        return xformer_layer_p.tr_atten_tpl.activation_split_dims_mapping.bld


class TEInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    def get_fprop_caller_of_stack_transformer(_, deterministic):
        def _fprop(
            transformer,
            x_in,
            paddings,
            attention_mask,
            cross_inputs,
            cross_attention_mask,
            segment_pos
        ):
            x_out = transformer(
                inputs=x_in,
                attention_mask=attention_mask,
                encoded=cross_inputs,
                encoder_decoder_mask=cross_attention_mask,
                deterministic=deterministic)
            return x_out
        return _fprop


    @staticmethod
    def set_layer_params_to_stack_transformer(stacked_transformer_obj, _, layer_id):
        te_transformer_tpl = pax_fiddle.Config(te_praxis.TransformerLayer,
            name=f'layer_{layer_id}',
            layernorm_type='layernorm',
            zero_centered_gamma = True,
            mlp_activations=('gelu',),
            use_bias=True,
            self_attn_mask_type='causal',
            enable_relative_embedding=False,
            scaled_query_init=False,
            scale_attn_logits=True,
            transpose_batch_sequence=False
        )

        te_transformer_tpl.logical_axes_rules = te_flax.extend_logical_axis_rules(tuple())
        te_transformer_tpl.params_init = stacked_transformer_obj.params_init
        te_transformer_tpl.dtype = stacked_transformer_obj.fprop_dtype
        te_transformer_tpl.layer_type = te_praxis.TransformerLayerType.DECODER if stacked_transformer_obj.use_cross_attention \
                        else te_praxis.TransformerLayerType.ENCODER
        te_transformer_tpl.num_attention_heads = stacked_transformer_obj.num_heads
        te_transformer_tpl.hidden_size = stacked_transformer_obj.model_dims
        te_transformer_tpl.mlp_hidden_size = stacked_transformer_obj.hidden_dims
        te_transformer_tpl.layernorm_epsilon = stacked_transformer_obj.transformer_layer_params_tpl.ln_tpl.epsilon
        te_transformer_tpl.dropout_rng_name = base_layer.RANDOM
        te_transformer_tpl.attention_dropout = stacked_transformer_obj.atten_dropout_prob or stacked_transformer_obj.dropout_prob
        te_transformer_tpl.hidden_dropout = stacked_transformer_obj.residual_dropout_prob or stacked_transformer_obj.dropout_prob
        te_transformer_tpl.intermediate_dropout = stacked_transformer_obj.relu_dropout_prob or stacked_transformer_obj.dropout_prob
        if stacked_transformer_obj.residual_droppath_prob > 0.0:
            te_transformer_tpl.drop_path = (
                stacked_transformer_obj.residual_droppath_prob * layer_id / max(1, stacked_transformer_obj.num_layers)
            )

        assert stacked_transformer_obj.dim_per_head == stacked_transformer_obj.model_dims // stacked_transformer_obj.num_heads
        assert stacked_transformer_obj.packed_input == False
        assert len(stacked_transformer_obj.moe_layers) == 0
        assert stacked_transformer_obj.ngrammer_tpls is None
        assert stacked_transformer_obj.local_window_size is None

        return te_transformer_tpl

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(_):
        rules = te_flax.extend_logical_axis_rules(tuple())
        # rules [(batch_axis_name, ('replicat', 'data'))', ...)]
        batch_mapping = rules[0][1]
        hidden_tp_mapping = rules[4][1]
        # [Batch, Seqlen, Hidden]
        bld_mapping = [batch_mapping, None, hidden_tp_mapping]
        return bld_mapping




class TransformerEngineHelper(TransformerEngineHelperBase):

    @staticmethod
    def is_enabled_te():
        enable_te = bool(int((os.environ.get("ENABLE_TE", False))))
        return (_IS_TRANSFORMER_ENGINE_INSTALLED and enable_te)

    @staticmethod
    def get_helper():
        if TransformerEngineHelper.is_enabled_te():
            return TEInstalledHelper
        return TENotInstalledHelper

    @staticmethod
    def get_fprop_caller_of_stack_transformer(fprop, deterministic):
        return TransformerEngineHelper.get_helper().get_fprop_caller_of_stack_transformer(
                    fprop, deterministic)

    @staticmethod
    def set_layer_params_to_stack_transformer(stacked_transformer_obj, layer_p, layer_id):
        return TransformerEngineHelper.get_helper().set_layer_params_to_stack_transformer(
                    stacked_transformer_obj, layer_p, layer_id)

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(xformer_layer_p):
        return TransformerEngineHelper.get_helper().get_bld_mapping_for_pipelined_transformer(
                    xformer_layer_p)
