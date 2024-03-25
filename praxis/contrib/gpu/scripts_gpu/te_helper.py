import os

from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis import layers
from praxis.layers.checkpoint_policy import AutodiffCheckpointType
from praxis.layers import activations
from praxis.layers import attentions, grouped_query_attention, multi_query_attention
from praxis.layers import embedding_softmax
from praxis.layers import normalizations

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

    ENABLE_TE_SP = bool(int(os.environ.get('ENABLE_TE_SP', 0)))

except ModuleNotFoundError as e:
    _IS_TRANSFORMER_ENGINE_INSTALLED = False
    TE_PIPELINE_EXTRA_VMAP_VAR_AXES = {}
    TE_PIPELINE_EXTRA_SCAN_VAR_BROADCAST = []
    ENABLE_TE_SP = False

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
    def get_input_bld(original_bld, batch_axes, mdl_axis):
        # This is used to specify the sharding pattern of inputs to TransformerLayers.
        raise NotImplementedError

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(xformer_layer_p):
        raise NotImplementedError

    @staticmethod
    def check_checkpoint_policy(tpl):
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

        if stacked_transformer_obj.residual_droppath_prob > 0.0:
            layer_p.residual_droppath_prob = (
                stacked_transformer_obj.residual_droppath_prob * layer_id / max(1, stacked_transformer_obj.num_layers)
            )
        return layer_p

    @staticmethod
    def get_input_bld(original_bld, *_):
        return original_bld

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(xformer_layer_p):
        return xformer_layer_p.tr_atten_tpl.activation_split_dims_mapping.bld

    @staticmethod
    def check_checkpoint_policy(_):
        """Every checkpoint policy is valid without TE"""
        pass


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
            enable_relative_embedding=False,
            enable_sequence_parallel=ENABLE_TE_SP,
            transpose_batch_sequence=False
        )

        def update_ln_te_tpl(te_tpl, transformer_layer_tpl):
            # TE requires all normalization are the same
            assert transformer_layer_tpl.ln_tpl == transformer_layer_tpl.tr_fflayer_tpl.ln_tpl
            ln_tpl = transformer_layer_tpl.ln_tpl
            if issubclass(ln_tpl.cls, normalizations.LayerNorm):
                te_tpl.layernorm_type = 'layernorm'
                assert ln_tpl.use_scale
                assert ln_tpl.use_bias
            elif issubclass(ln_tpl.cls, normalizations.RmsNorm):
                te_tpl.layernorm_type = 'rmsnorm'
            else:
                raise ValueError(f'Unsupported {ln_tpl.cls=}, LayerNorm, RmsNorm are supported.')
            te_tpl.zero_centered_gamma = not ln_tpl.direct_scale
            te_tpl.layernorm_epsilon = ln_tpl.epsilon
            return te_tpl

        def update_ff_te_tpl(te_tpl, ff_layer_tpl):
            mlp_activations = ()
            if ff_layer_tpl.use_gated_activation:
               mlp_activations += ('linear',)

            if issubclass(ff_layer_tpl.activation_tpl.cls, activations.Identity):
                mlp_activations += ('linear',)
            else:
                mlp_activations += (ff_layer_tpl.activation_tpl.cls.__name__.lower(),)

            te_tpl.mlp_activations = mlp_activations
            return te_tpl

        def update_attn_te_tpl(te_tpl, attn_tpl):
            if issubclass(attn_tpl.cls, attentions.DotProductAttention):
                # Check the DotProductAttention parameters are aligned to TE's attention
                assert attn_tpl.internal_enable_query_scale or attn_tpl.scale_logits_by_head_dims
                assert not (attn_tpl.internal_enable_query_scale and attn_tpl.scale_logits_by_head_dims)
                assert not attn_tpl.internal_enable_per_dim_scale
                assert not attn_tpl.scale_query_by_dim_per_head
                assert not attn_tpl.dconv_qkv
                assert not attn_tpl.internal_gshard_gaussian_init
                assert attn_tpl.relative_bias_tpl is None
                assert attn_tpl.attention_extra_logit is None
                assert attn_tpl.ngrammer_tpl is None
                te_tpl.enable_rotary_pos_emb = attn_tpl.use_rotary_position_emb
                if issubclass(attn_tpl.rotary_position_emb_tpl.cls, embedding_softmax.RotaryPositionalEmbedding):
                    te_tpl.rotary_pos_emb_group_method = 'alternate'
            elif issubclass(attn_tpl.cls, grouped_query_attention.GroupedQueryAttention):
                te_tpl.num_gqa_groups = attn_tpl.num_kv_heads
                if attn_tpl.rope_min_max_timescales is not None:
                    te_tpl.enable_rotary_pos_emb = True
                    te_tpl.rotary_pos_emb_windows = attn_tpl.rope_min_max_timescales
                assert attn_tpl.atten_temp == 1.
            elif issubclass(attn_tpl.cls, multi_query_attention.MultiQueryDotProductAttention):
                te_tpl.num_gqa_groups = attn_tpl.num_kv_heads
                te_tpl.enable_rotary_pos_emb = attn_tpl.use_rotary_position_emb
                if issubclass(attn_tpl.rotary_position_emb_tpl.cls, embedding_softmax.RotaryPositionalEmbedding):
                    te_tpl.rotary_pos_emb_group_method = 'alternate'
            else:
                raise ValueError(f'Unsupported {attn_tpl.cls=}')
            assert attn_tpl.atten_logit_cap <= 0., 'atten_logit_cap > 0. is not supported in TE'
            te_tpl.scaled_query_init = False
            te_tpl.scale_attn_logits = True
            return te_tpl

        transformer_layer_tpl = stacked_transformer_obj.transformer_layer_params_tpl
        # Update TE normalization layer configs
        te_transformer_tpl = update_ln_te_tpl(te_transformer_tpl, transformer_layer_tpl)
        # Update TE feed forward layer configs
        te_transformer_tpl = update_ff_te_tpl(te_transformer_tpl, transformer_layer_tpl.tr_fflayer_tpl)
        # Update TE attention layer configs
        te_transformer_tpl = update_attn_te_tpl(te_transformer_tpl, transformer_layer_tpl.tr_atten_tpl)
        # TE currently only allow the bias config to be same between feed forward, qkv proj, out proj
        assert (transformer_layer_tpl.tr_fflayer_tpl.has_bias ==
            transformer_layer_tpl.tr_atten_tpl.use_bias), "TE only allows same bias settings."
        te_transformer_tpl.use_bias = transformer_layer_tpl.tr_fflayer_tpl.has_bias
        te_transformer_tpl.self_attn_mask_type = 'causal' \
            if stacked_transformer_obj.mask_self_attention else 'padding'

        te_transformer_tpl.logical_axes_rules = te_flax.extend_logical_axis_rules(tuple())
        te_transformer_tpl.params_init = stacked_transformer_obj.params_init
        te_transformer_tpl.dtype = stacked_transformer_obj.fprop_dtype
        te_transformer_tpl.layer_type = te_praxis.TransformerLayerType.DECODER if stacked_transformer_obj.use_cross_attention \
                        else te_praxis.TransformerLayerType.ENCODER
        te_transformer_tpl.num_attention_heads = stacked_transformer_obj.num_heads
        te_transformer_tpl.hidden_size = stacked_transformer_obj.model_dims
        te_transformer_tpl.mlp_hidden_size = stacked_transformer_obj.hidden_dims
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

        return te_transformer_tpl

    @staticmethod
    def get_input_bld(_, batch_axes, mdl_axis):
        if ENABLE_TE_SP:
            return [batch_axes, mdl_axis, None]
        return [batch_axes, None, None]

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(_):
        rules = te_flax.extend_logical_axis_rules(tuple())
        # rules [(batch_axis_name, ('replicat', 'data'))', ...)]
        batch_mapping = rules[0][1]
        hidden_tp_mapping = rules[4][1]
        # [Batch, Seqlen, Hidden]
        bld_mapping = [batch_mapping, None, hidden_tp_mapping]
        return bld_mapping

    @staticmethod
    def check_checkpoint_policy(tpl):
        """Some checkpoint policies are not compatible with TE fused attention"""
        if issubclass(tpl.cls, layers.transformers.StackedTransformer):
            remat = tpl.remat
            attention_dropout = tpl.atten_dropout_prob or tpl.dropout_prob
        elif issubclass(tpl.cls, layers.transformers.StackedTransformerRepeated):
            if not issubclass(tpl.block.cls, layers.transformers.StackedTransformer):
                return
            remat = True  # Current StackedTransformerRepeat always enables remat
            attention_dropout = tpl.block.atten_dropout_prob or tpl.block.dropout_prob
        else:
            raise ValueError(f'Unsupported class={tpl.cls}')

        supported_checkpoint_policies = [
            AutodiffCheckpointType.SAVE_CONTEXT,
            AutodiffCheckpointType.SAVE_CONTEXT_AND_OUT_PROJ,
            AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B,
            AutodiffCheckpointType.SAVE_QUANTIZED,
            AutodiffCheckpointType.SAVE_DOT_EXCEPT_LOGITS_FFN1,
            AutodiffCheckpointType.SAVE_DOT_EXCEPT_LOGITS]
        fused_attn_enabled = int(os.getenv("NVTE_FUSED_ATTN", "0"))
        if remat and fused_attn_enabled and attention_dropout > 0.:
            assert tpl.checkpoint_policy in supported_checkpoint_policies, \
            "Fused attn in TE only permits policies that save 'context' tensors when dropout is " \
            "enabled. This restriction is due to the maintenance of the dropout offset within TE, " \
            "which is incompatible with the JAX remat. Consequently, it's necessary to bypass " \
            "recomputation in the attention layer when fused attention is activated. The supported " \
            f"checkpoint_policies are {supported_checkpoint_policies} but the provided " \
            f"checkpoint_policy is '{tpl.checkpoint_policy}'."


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
    def get_input_bld(original_bld, batch_axes, mdl_axis):
        return TransformerEngineHelper.get_helper().get_input_bld(
                    original_bld, batch_axes, mdl_axis)

    @staticmethod
    def get_bld_mapping_for_pipelined_transformer(xformer_layer_p):
        return TransformerEngineHelper.get_helper().get_bld_mapping_for_pipelined_transformer(
                    xformer_layer_p)

    @staticmethod
    def check_checkpoint_policy(tpl):
        return TransformerEngineHelper.get_helper().check_checkpoint_policy(tpl)
