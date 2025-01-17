# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch


from ..base import NeuronModelBase
from ..config import NeuronConfig
from ..decoder import DecoderLmHeadForSamplingNoEmbedding
from ..utils import interleave_mlp
from .config import LlamaConfig
from .hlo import LlamaForSamplingNoEmbeddingHlo
from .modules import LlamaForCausalLM


class LlamaForSampling(NeuronModelBase):
    def __init__(
        self,
        config,
        *,
        n_positions=2048,
        batch_size=1,
        amp="f32",
        tp_degree=2,
        neuron_config=None,
        **kwargs,
    ):
        config = LlamaConfig(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(LlamaForCausalLM, config)
        self.context_pre_hook = None
        self.context_hook = None
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        self.n_positions = n_positions
        self.batch_size = batch_size
        hlo_builder = LlamaForSamplingNoEmbeddingHlo(
            config, neuron_config=self.neuron_config
        )
        self.decoder_param_set = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree,
            n_positions=self.n_positions,
            n_active_tokens=1,
            batch_size=self.batch_size,
            attention_head_size=config.attention_head_size,
            amp=amp,
            num_layers=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_kv_head=config.num_key_value_heads,
            neuron_config=self.neuron_config,
            allow_pad=True,
            builder=hlo_builder,
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(model_obj=self)
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(
            model_obj=self
        )

    def load_weights(self):
        self.materialize_embeddings()

        for layer in self.chkpt_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            is_unit_scale = False
            new_layer = self.decoder_lm_head.new_layer(is_unit_scale=is_unit_scale)
            if self.neuron_config.has_pre_attention_norm:
                new_layer.add_pre_attention_layer_norm(
                    layer.input_layernorm.weight.detach(), None
                )
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, None)
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, None)
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, None)
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(
                    attn.o_proj.weight.T.detach(), None, sharding=0, transposed=True
                )
            else:
                new_layer.add_attention_output(
                    attn.o_proj.weight.detach(), None, sharding=1, transposed=False
                )

            if self.neuron_config.fused_rmsnorm_mlp:
                dummy_post_attention_ln_weight = torch.ones_like(
                    layer.post_attention_layernorm.weight.detach()
                )
                new_layer.add_pre_mlp_layer_norm(dummy_post_attention_ln_weight, None)
            else:
                new_layer.add_pre_mlp_layer_norm(
                    layer.post_attention_layernorm.weight.detach(), None
                )

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            if self.neuron_config.fused_rmsnorm_mlp:
                fused_pre_mlp_ln_gate_weight = (
                    mlp.gate_proj.weight
                    * layer.post_attention_layernorm.weight.detach().to(
                        dtype=mlp.gate_proj.weight.dtype
                    )
                )
                new_layer.add_parameter(
                    fused_pre_mlp_ln_gate_weight.T, sharding=1, allow_pad=True
                )
                fused_pre_mlp_ln_up_weight = (
                    mlp.up_proj.weight
                    * layer.post_attention_layernorm.weight.detach().to(
                        dtype=mlp.up_proj.weight.dtype
                    )
                )
                new_layer.add_parameter(
                    fused_pre_mlp_ln_up_weight.T, sharding=1, allow_pad=True
                )
                new_layer.add_parameter(
                    mlp.down_proj.weight.T, sharding=0, allow_pad=True
                )
            elif self.neuron_config.fuse_mlp:
                assert all(
                    getattr(mlp, attr, None) for attr in ["gate_proj", "up_proj"]
                ), "fuse_mlp need to have gate and up proj weights"
                assert all(
                    getattr(mlp, attr, None).weight.shape[0] % self.config.tp_degree
                    == 0
                    for attr in ["gate_proj", "up_proj"]
                ), f" mlp weights are not  divisible tp_degree {self.config.tp_degree}"
                mlp_in_weight = interleave_mlp(
                    mlp.gate_proj.weight,
                    mlp.up_proj.weight,
                    tp_degree=self.config.tp_degree,
                    dim=0,
                )
                new_layer.add_mlp_input(mlp_in_weight.T.detach(), None)
                new_layer.add_mlp_output(
                    sharding=1,
                    transposed=False,
                )
            else:
                new_layer.add_parameter(
                    mlp.gate_proj.weight.T,
                    sharding=1,
                    allow_pad=True,
                    allow_transform=True,
                )
                new_layer.add_parameter(
                    mlp.up_proj.weight.T,
                    sharding=1,
                    allow_pad=True,
                    allow_transform=True,
                )
                new_layer.add_parameter(
                    mlp.down_proj.weight, sharding=1, allow_pad=True
                )
            new_layer.to_neuron()
            layer.nullify()

        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)
        ln_f.nullify()

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()

        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(
                self.chkpt_model.model.embed_tokens.weight,
                sharding=1,
                allow_pad=True,
            )
        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()
        self.maybe_nullify_embeddings()

    def materialize_embeddings(self):
        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

    def maybe_nullify_embeddings(self):
        if self.neuron_config.on_device_embedding:
            self.chkpt_model.model.embed_tokens.nullify()

    def init_rest_of_model(self):
        self.decoder_lm_head.use_executor = True

        model = self.decoder_lm_head.build_weight_shared(
            share_caches=True, new=self.decoder_lm_head_for_context
        )
        model.use_executor = True
        self.decoder_lm_head_for_context = model
