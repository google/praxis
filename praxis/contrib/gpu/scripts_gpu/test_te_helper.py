from praxis import base_hyperparams
from praxis import layers
from praxis import pax_fiddle
from praxis.contrib.gpu.scripts_gpu.te_helper import TransformerEngineHelper
from paxml.contrib.gpu.scripts_gpu.llama_utils import BaseLLaMA
from paxml.contrib.gpu.scripts_gpu.configs import Synthetic5B
from paxml.tasks.lm.params.lm_cloud import SyntheticDataset

import transformer_engine.jax.praxis as te_praxis


class SyntheticLLaMA7B(BaseLLaMA, SyntheticDataset):
    pass


class TestGPT5B():

    def test_te_tpl_convert(self):
        task = Synthetic5B().task()
        st_tpl = task.model.lm_tpl.stacked_transformer_tpl.block
        te_tpl = TransformerEngineHelper().set_layer_params_to_stack_transformer(st_tpl, None, 0)
        te_cls = base_hyperparams.instantiate(te_tpl)
        assert te_cls.hidden_size == st_tpl.model_dims
        assert te_cls.mlp_hidden_size == st_tpl.hidden_dims
        assert te_cls.num_attention_heads == st_tpl.num_heads
        assert te_cls.num_gqa_groups == te_cls.num_attention_heads
        assert te_cls.layernorm_type == 'layernorm'
        assert te_cls.layernorm_epsilon == 1e-5
        assert te_cls.zero_centered_gamma == True
        assert te_cls.hidden_dropout == 0.
        assert te_cls.hidden_dropout_dims == ()
        assert te_cls.attention_dropout == 0.
        assert te_cls.intermediate_dropout == 0.
        assert te_cls.intermediate_dropout_dims == ()
        assert te_cls.mlp_activations == ('gelu',)
        assert te_cls.use_bias == True
        assert te_cls.apply_residual_connection_post_layernorm == False
        assert te_cls.output_layernorm == False
        assert te_cls.float32_attention_logits == False
        assert te_cls.layer_type == te_praxis.TransformerLayerType.ENCODER
        assert te_cls.self_attn_mask_type == 'padding_causal'
        assert te_cls.self_attn_bias_type == None
        assert te_cls.enable_rotary_pos_emb == False
        assert te_cls.rotary_pos_emb_windows == (1, 10000)
        assert te_cls.enable_relative_embedding == False
        assert te_cls.drop_path == 0.
        assert te_cls.transpose_batch_sequence == False
        assert te_cls.scale_attn_logits == True
        assert te_cls.scaled_query_init == False


class TestLLaMA7B():

    def test_te_tpl_convert(self):
        task = SyntheticLLaMA7B().task()
        st_tpl = task.model.lm_tpl.stacked_transformer_tpl
        te_tpl = TransformerEngineHelper().set_layer_params_to_stack_transformer(st_tpl, None, 0)
        te_cls = base_hyperparams.instantiate(te_tpl)
        assert te_cls.hidden_size == 4096
        assert te_cls.mlp_hidden_size == 16384
        assert te_cls.num_attention_heads == 32
        assert te_cls.num_gqa_groups == 32
        assert te_cls.layernorm_type == 'rmsnorm'
        assert te_cls.layernorm_epsilon == 1e-5
        assert te_cls.zero_centered_gamma == False
        assert te_cls.hidden_dropout == 0.
        assert te_cls.hidden_dropout_dims == ()
        assert te_cls.attention_dropout == 0.
        assert te_cls.intermediate_dropout == 0.
        assert te_cls.intermediate_dropout_dims == ()
        assert te_cls.mlp_activations == ('linear', 'silu')
        assert te_cls.use_bias == False
        assert te_cls.apply_residual_connection_post_layernorm == False
        assert te_cls.output_layernorm == False
        assert te_cls.float32_attention_logits == False
        assert te_cls.layer_type == te_praxis.TransformerLayerType.ENCODER
        assert te_cls.self_attn_mask_type == 'padding_causal'
        assert te_cls.self_attn_bias_type == None
        assert te_cls.enable_rotary_pos_emb == True
        assert te_cls.rotary_pos_emb_windows == (1, 10000)
        assert te_cls.rotary_pos_emb_group_method == 'consecutive'
        assert te_cls.enable_relative_embedding == False
        assert te_cls.drop_path == 0.
        assert te_cls.transpose_batch_sequence == False
        assert te_cls.scale_attn_logits == True
        assert te_cls.scaled_query_init == False
