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
import os
import itertools
import warnings

import torch
from transformers import PretrainedConfig
from transformers_neuronx import compiler
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.llama.hlo import LlamaForSamplingNoEmbeddingHlo


from .base import NeuronModelBase, NeuronBaseSerializer
from .utils import (
    maybe_pad_tensor,
    round_up_to_divisor,
    interleave_qkv,
    get_qkv_padding,
    pad_interleaved,
    get_pad_size,
)


class DecoderLmHeadForSamplingNoEmbedding(NeuronBaseSerializer):
    def __init__(
        self,
        tp_degree,
        n_positions,
        n_active_tokens,
        batch_size,
        amp,
        config: PretrainedConfig,
        neuron_config=None,
        allow_pad=True,
        is_prefill=True,
        builder=None,
        tag=None,
    ):
        super().__init__()
        self.tp_degree = tp_degree
        if isinstance(n_positions, (list, tuple)):
            # Legacy Tnx encoding for continuous batching
            assert len(n_positions) == 1 and neuron_config.continuous_batching
            n_positions = n_positions[0]
        self.n_positions = n_positions
        self.n_active_tokens = n_active_tokens
        self.batch_size = batch_size
        self.config = config
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.is_prefill = is_prefill
        self.amp = amp
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        self.layers = []
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        self.lm_head_bias = None
        self.logits_indices = None
        self.inputs_sdim = None
        self.inputs_builder = None
        self.embedding_builder = None
        self.layer_builder = None
        self.ln_lm_head_params = []
        self.ln_lm_head_builder = None
        self.program = None
        self.pre_layer_parameters = []
        self.pre_layer_builder = None
        self.allow_pad = allow_pad
        self.use_executor = False
        self.return_ranks = -1
        self.builder = builder
        self.check_gqa_fallback()
        self.tag = tag
        self._cpu_compile = False

    def check_gqa_fallback(self):
        """
        Check if a fallback mechanism is needed for a user-provided GQA config.

        The initial fallback mechanism will be that no special GQA configuration
        is used. This will attempt to evenly distribute Q and KV heads to all
        NeuronCores.

        The second (safest) fallback mechanism will replicate the KV heads to be
        equal to the number of Q heads. This makes the GQA model look identical
        to an MHA model.
        """
        gqa = self.neuron_config.group_query_attention

        if gqa is None:
            # MHA Early exit - This avoids emitting irrelevant GQA warnings
            if self.n_head == self.n_kv_head:
                return
            self.neuron_config.group_query_attention = constants.GQA.SHARD_OVER_HEADS

        if gqa == constants.GQA.REPLICATED_HEADS:
            return

        if gqa == constants.GQA.SHARD_OVER_BATCH:
            success = True
            for batch_size in self.batch_size:
                if batch_size % self.tp_degree != 0:
                    warnings.warn(
                        f'Cannot enable "{gqa}" when a batch size '
                        f"({batch_size} in {self.batch_size}) is not evenly "
                        f"divisible by the tensor parallel degree "
                        f"({self.tp_degree})"
                    )
                    success = False
                    self.neuron_config.group_query_attention = (
                        constants.GQA.SHARD_OVER_HEADS
                    )
            if success:
                return

        if gqa == constants.GQA.ALL_GATHER_HEADS:
            if (self.n_kv_head * self.attention_head_size) % self.tp_degree != 0:
                warnings.warn(
                    f'Cannot enable "{gqa}" when the hidden size of KV '
                    f"({self.n_kv_head} x {self.attention_head_size}) is not evenly divisible "
                    f"by the tensor parallel degree ({self.tp_degree})"
                )
                self.neuron_config.group_query_attention = (
                    constants.GQA.SHARD_OVER_HEADS
                )

            if self.n_head % self.tp_degree != 0:
                # try pad on n_head, if pad_size could be evenly disible by n_kv_head,
                # then we can evenly distribute same number of padding q_head to each k/v head
                pad_size = get_pad_size(self.n_head, self.tp_degree)

                if pad_size % self.n_kv_head == 0:
                    return
                else:
                    warnings.warn(
                        f'Cannot enable "{gqa}" when the number of padding {pad_size} need for query '
                        f"attention heads ({self.n_head}) with the tensor parallel degree ({self.tp_degree}) "
                        f"is not divisible by KV heads ({self.n_kv_head})"
                    )
                    self.neuron_config.group_query_attention = (
                        constants.GQA.SHARD_OVER_HEADS
                    )
            else:
                return

        if self.n_kv_head % self.tp_degree != 0:
            warnings.warn(
                f"KV head replication will be enabled since the number of KV "
                f"heads ({self.n_kv_head}) is not evenly divisible by the "
                f"tensor parallel degree ({self.tp_degree})"
            )
            self.neuron_config.group_query_attention = constants.GQA.REPLICATED_HEADS

    def init_context_decoder(
        self,
        model_obj: NeuronModelBase,
    ):
        cls = type(self)
        decoder_lm_head = {}
        assert self.batch_size == 1 or self.neuron_config.continuous_batching
        decoder_lm_head = cls(
            tp_degree=self.tp_degree,
            n_positions=self.n_positions,
            n_active_tokens=self.n_positions,
            batch_size=1,
            amp=self.amp,
            config=self.config,
            neuron_config=self.neuron_config,
            allow_pad=self.allow_pad,
            is_prefill=True,
            builder=self.builder,
            tag="context",
        )
        model_obj.register_for_serialization(decoder_lm_head)
        return decoder_lm_head

    def init_token_decoder(self, model_obj: NeuronModelBase):
        cls = type(self)
        decoder_lm_head = cls(
            tp_degree=self.tp_degree,
            n_positions=self.n_positions,
            n_active_tokens=1,
            batch_size=self.batch_size,
            amp=self.amp,
            config=self.config,
            neuron_config=self.neuron_config,
            allow_pad=True,
            is_prefill=False,
            builder=self.builder,
            tag="token",
        )
        model_obj.register_for_serialization(decoder_lm_head)
        decoder_lm_head.add_inputs_builder(self.builder.inputs)
        if hasattr(self.builder, "pre_layer"):
            decoder_lm_head.add_pre_layer_builder(self.builder.pre_layer)
        decoder_lm_head.add_layer_builder(self.builder.layer)
        decoder_lm_head.add_ln_lm_head_builder(self.builder.ln_lm_head)
        if hasattr(self.builder, "embedding"):
            decoder_lm_head.add_embedding_builder(self.builder.embedding)
        return decoder_lm_head

    def enable_executor(self, return_ranks=-1):
        self.return_ranks = return_ranks
        self.program.enable_executor()

    def add_inputs_builder(self, inputs_builder):
        self.inputs_builder = inputs_builder

    def add_embedding_builder(self, embedding_builder):
        self.embedding_builder = embedding_builder

    def add_pre_layer_parameter(self, param, sharding=None, allow_pad=False):
        self.pre_layer_parameters.append((param, sharding, allow_pad))

    def add_pre_layer_builder(self, builder):
        self.pre_layer_builder = builder

    def add_layer_builder(self, layer_builder):
        self.layer_builder = layer_builder

    def add_ln_lm_head_builder(self, ln_lm_head_builder):
        self.ln_lm_head_builder = ln_lm_head_builder

    def new_layer(self, is_unit_scale=False):
        layer = DecoderLayer(
            self.tp_degree,
            self.n_positions,
            self.batch_size,
            self.attention_head_size,
            n_head=self.n_head,
            amp=self.amp,
            n_kv_head=self.n_kv_head,
            neuron_config=self.neuron_config,
            allow_pad=self.allow_pad,
            n_active_tokens=self.n_active_tokens,
            layer_num=len(self.layers),
            is_unit_scale=is_unit_scale,
        )
        layer._cpu_compile = self._cpu_compile
        self.layers.append(layer)
        return layer

    def add_final_layer_norm(self, weight, bias):
        self.ln_f_weight = weight
        self.ln_f_bias = bias

    def add_lm_head(self, weight, bias=None):
        self.lm_head_weight = weight
        self.lm_head_bias = bias

    def to_neuron(self):
        manipulator = MaybeParallelTensorManipulator(
            self.tp_degree, on_cpu=self._cpu_compile
        )
        self.pre_layer_parameters = self._prepare_pre_layer_params(
            manipulator, self.pre_layer_parameters
        )

        self.ln_f_weight = manipulator.duplicate(self.ln_f_weight)
        self.ln_f_bias = manipulator.duplicate(self.ln_f_bias)
        _, vocab_size = self.lm_head_weight.shape
        # Pad vocab size such that it can be divided by the following factor
        divisor = int(os.environ.get("NEURON_VOCAB_PAD_DIVISOR", str(self.tp_degree)))
        vocab_pad = get_pad_size(vocab_size, divisor)
        lm_head_weight = torch.nn.functional.pad(
            self.lm_head_weight, (0, vocab_pad, 0, 0)
        )
        self.lm_head_weight = manipulator.shard_along(lm_head_weight, dim=1)
        ln_lm_head_params = [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if self.lm_head_bias is not None:
            self.lm_head_bias = manipulator.shard_along(self.lm_head_bias, dim=0)
            ln_lm_head_params.append(self.lm_head_bias)
        self.ln_lm_head_params = ln_lm_head_params
        self.program = self._build_program()

    def build_weight_shared(
        self,
        share_caches=False,
        new=None,
    ):
        if new is None:
            cls = type(self)
            new = cls(
                self.tp_degree,
                self.n_positions,
                self.n_active_tokens,
                self.batch_size,
                amp=self.amp,
                config=self.config,
                neuron_config=self.neuron_config,
                allow_pad=self.allow_pad,
                is_prefill=self.is_prefill,
            )
        new.add_inputs_builder(self.inputs_builder)
        new.add_embedding_builder(self.embedding_builder)
        new.add_pre_layer_builder(self.pre_layer_builder)
        new.add_layer_builder(self.layer_builder)
        new.add_ln_lm_head_builder(self.ln_lm_head_builder)
        new._cpu_compile = self._cpu_compile
        for layer in self.layers:
            new_layer = new.new_layer()
            new_layer.assign_parameters(layer)
            if share_caches:
                new_layer.assign_caches(layer)
            else:
                new_layer.init_caches()
            new_layer.extra_parameters = layer.extra_parameters
        new.pre_layer_parameters = self.pre_layer_parameters
        new.add_final_layer_norm(self.ln_f_weight, self.ln_f_bias)
        new.add_lm_head(self.lm_head_weight, self.lm_head_bias)
        ln_lm_head_params = [new.ln_f_weight, new.ln_f_bias, new.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if new.lm_head_bias is not None:
            ln_lm_head_params.append(new.lm_head_bias)
        new.ln_lm_head_params = ln_lm_head_params
        new.program = new._build_program()
        return new

    def setup(self):
        self.program.setup(
            self.layers, self.pre_layer_parameters, self.ln_lm_head_params
        )
        if self.use_executor:
            self.enable_executor()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, *inputs):
        """
        This path makes the assumption that inputs are correctly sized for a
        sequence length of 1. This allows us to avoid checking buckets, slicing,
        etc.
        """
        if self.use_executor:
            outputs = self.program.execute(
                *inputs,
                return_ranks=self.return_ranks,
            )
        else:
            self.program.inputs_host_to_device(inputs)
            self.program.run()
            outputs = self.program.maybe_logits_device_to_host(
                return_ranks=self.return_ranks
            )

        return outputs

    def _prepare_pre_layer_params(self, manipulator, pre_layer_parameters):
        extras = []
        for param, dim, allow_pad in pre_layer_parameters:
            if allow_pad and dim is not None:
                if param.shape[dim] % self.tp_degree != 0:
                    size = round_up_to_divisor(param.shape[dim], self.tp_degree)
                    param = maybe_pad_tensor(param, dim, size)
            extras.append(manipulator.duplicate_or_shard_along(param, dim))
        return extras

    def _build_program(self):
        hlo_module = self._hlo_fully_unrolled(self.n_positions, self.batch_size)
        num_inputs = len(self.inputs_sdim)
        return DecoderProgramFullyUnrolled(
            self.neuron_config,
            self.layers,
            hlo_module,
            num_inputs,
            self.tp_degree,
            self.n_positions,
            self.batch_size,
            tag=self.tag,
            on_cpu=self._cpu_compile,
        )

    def _hlo_unroll(
        self,
        hidden,
        tensors,
        layers_caches,
        layers_weights,
        pre_layer_params,
        lm_head_params,
    ):
        last_token_id = tensors[2]
        hidden = self._hlo_embedding(hidden, tensors, pre_layer_params)
        hidden, tensors = self._hlo_pre_layer(hidden, tensors, pre_layer_params)
        hidden, out_caches = self._hlo_layers(
            hidden,
            tensors,
            self.layers,
            layers_caches,
            layers_weights,
            alias_caches=False,
        )
        logits = self.ln_lm_head_builder(
            hidden,
            last_token_id,
            *lm_head_params,
            is_prefill=self.is_prefill,
        )
        return logits, out_caches

    def _hlo_fully_unrolled(self, n_positions, batch_size):
        self.builder.n_positions = n_positions

        def fully_unrolled(scribe):
            dtype = getattr(scribe, self.amp)

            # Create user parameters
            (hidden, *tensors), self.inputs_sdim = self.inputs_builder(
                scribe, dtype, self.n_active_tokens, batch_size
            )
            param_builder = DecoderParameterBuilder(scribe, len(self.inputs_sdim))

            # Create inputs for all weights & caches
            in_caches, layers_weights, pre_layer_params, lm_head_params = (
                self._hlo_parameters(n_positions, batch_size, param_builder)
            )

            # Unroll the graph
            logits, out_caches = self._hlo_unroll(
                hidden,
                tensors,
                in_caches,
                layers_weights,
                pre_layer_params,
                lm_head_params,
            )
            self._hlo_cache_aliases(in_caches, out_caches)

            # Set the output
            out_caches = itertools.chain(*out_caches)
            if self.neuron_config.log_softmax_scores:
                logits, scores = self._hlo_post_layer(logits)
                outputs = [logits, scores, *out_caches]
            else:
                outputs = [logits, *out_caches]

            # Filter out the None's in outputs
            outputs = [o for o in outputs if o is not None]
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)

        return compiler.compile_py_func(fully_unrolled)

    def _hlo_parameters(self, n_positions, batch_size, param_builder):
        layers_caches, layers_weights = self._hlo_layers_params(
            param_builder, self.layers, n_positions
        )
        pre_layer_params = self._hlo_pre_layer_params(param_builder)
        lm_head_params = self._hlo_lm_head_params(param_builder)
        return layers_caches, layers_weights, pre_layer_params, lm_head_params

    def all_parameters(self, n_positions, batch_size):
        """
        Get all the parameters for the current model.

        NOTE: It is extremely important that these tensors are returned in the
              same order as the parameters returned in the _hlo_parameters
              function. If this is not done correctly, the HLO parameter and
              the corresponding weight tensor cannot be assocated.
        """
        parameters = list()

        # Layer caches
        for layer in self.layers:
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                parameters.append(cache)

        # Layer weights
        for layer in self.layers:
            parameters.extend(layer.all_parameters())

        # Prelayer parameters
        parameters.extend(self.pre_layer_parameters)

        # LM head parameters
        parameters.append(self.ln_f_weight)
        parameters.append(self.ln_f_bias)
        parameters.append(self.lm_head_weight)
        parameters.append(self.lm_head_bias)

        return parameters

    def valid_parameters(self, n_positions, batch_size):
        parameters = self.all_parameters(n_positions, batch_size)
        return [par for par in parameters if par is not None]

    def _hlo_pre_layer_params(self, param_builder):
        params = []
        for param in self.pre_layer_parameters:
            param = param_builder.from_tensor(param)
            param = hlo.transfer_with_static_ring(param)
            params.append(param)
        return params

    def _hlo_pre_layer(self, hidden, tensors, params, position_ids=None):
        if self.pre_layer_builder is not None:
            (hidden, *tensors) = self.pre_layer_builder(hidden, *tensors, *params)
        return hidden, tensors

    def _hlo_embedding(self, hidden, tensors, params):
        # Only insert embedding operation when on-device embedding is being used
        if not self.neuron_config.on_device_embedding:
            return hidden

        assert self.embedding_builder is not None, (
            "On-device embedding may only be used on models which provide this functionality"
        )
        hidden = self.embedding_builder(hidden, *tensors, *params)
        return hidden

    def _hlo_layers_params(self, param_builder, layers, n_positions):
        layers_caches = []
        dim_size = {0: n_positions}
        for layer in layers:
            layer_caches = []
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                par = param_builder.from_tensor(cache, dim_size=dim_size)
                layer_caches.append(par)
            layers_caches.append(layer_caches)
        layers_weights = []
        for layer in layers:
            layer_weights = [
                param_builder.from_tensor(weight) for weight in layer.all_parameters()
            ]
            layers_weights.append(layer_weights)
        return layers_caches, layers_weights

    def _hlo_layers(
        self, hidden, tensors, layers, layers_caches, layers_weights, alias_caches=True
    ):
        output_caches = []
        for idx, (layer, caches, weights) in enumerate(
            zip(layers, layers_caches, layers_weights)
        ):
            in_caches = [maybe_transfer_with_static_ring(cache) for cache in caches]
            weights = [maybe_transfer_with_static_ring(weight) for weight in weights]
            is_first_last_layer = True if idx == 0 or idx == len(layers) - 1 else False
            if isinstance(self.layer_builder.__self__, LlamaForSamplingNoEmbeddingHlo):
                # Positional information is needed for fused residual adds in kernels in Llama 3
                hidden, *out_caches = self.layer_builder(
                    hidden,
                    *tensors,
                    *in_caches,
                    *weights,
                    is_first_last_layer=is_first_last_layer,
                )
            else:
                hidden, *out_caches = self.layer_builder(
                    hidden, *tensors, *in_caches, *weights
                )
            output_caches.append(out_caches)

        if alias_caches:
            self._hlo_cache_aliases(layers_caches, output_caches)

        return hidden, output_caches

    def _hlo_cache_aliases(self, in_caches, out_caches):
        assert len(in_caches) == len(out_caches)
        for src, dst in zip(itertools.chain(*in_caches), itertools.chain(*out_caches)):
            if dst is not None:
                assert src is not None, "out_cache must alias with a valid cache!"
                dst.set_alias_to(src, must=True)

    def _hlo_lm_head_params(self, param_builder):
        ln_f_weight = param_builder.from_tensor(self.ln_f_weight)
        ln_f_bias = param_builder.from_tensor(self.ln_f_bias)
        head_weight = param_builder.from_tensor(self.lm_head_weight)
        head_bias = param_builder.from_tensor(self.lm_head_bias)
        ln_f_weight = maybe_transfer_with_static_ring(ln_f_weight)
        ln_f_bias = maybe_transfer_with_static_ring(ln_f_bias)
        head_weight = maybe_transfer_with_static_ring(head_weight)
        head_bias = maybe_transfer_with_static_ring(head_bias)
        return ln_f_weight, ln_f_bias, head_weight, head_bias

    def _hlo_post_layer(self, logits):
        return self.post_layer_builder(logits)

    # Mainly used for serialization purposes.
    # Defines how to access all the kernels.
    def get_all_kernels(self):
        return [self.program.kernel]


def read_n_active_tokens(hlo_module):
    return hlo_module.host_program_shape.parameters[0].dimensions[1]


def maybe_transfer_with_static_ring(shape):
    if shape is None:
        return None
    return hlo.transfer_with_static_ring(shape)


class MaybePadder:
    def __init__(
        self, size, padding="end", split_size=None, interleaved_factor=None
    ) -> None:
        self.split_size = split_size
        self.size = size
        self.padding = padding
        self.interleaved_factor = interleaved_factor

    def __call__(self, weight, dim):
        if self.padding == "end":
            return maybe_pad_tensor(weight, dim, self.size, left=False)
        else:
            if weight is None:
                return weight
            assert self.padding == "interleaved", f"Invalid padding mode {self.padding}"
            assert self.interleaved_factor, "interleaved_factor is not provided"
            # when split_size is set, we first split the target weight at dim
            # into (split_size x ?), for example, to do interleaved padding on of KV weight
            # we first need to reshape it into (hidden, num_kv_head, d_head)
            # and then apply interleaved padding on num_kv_head
            weight_shapes = list(weight.shape)

            padded_shape = weight_shapes.copy()
            padded_shape[dim] = self.size

            new_size = self.size
            if self.split_size:
                assert weight_shapes[dim] % self.split_size == 0, (
                    f"shape on dim_{dim} {weight_shapes[dim]} cannot be evenly divisible by provided split_size {self.split_size}"
                )
                new_shape = (
                    weight_shapes[:dim]
                    + [self.split_size]
                    + [weight_shapes[dim] // self.split_size]
                    + weight_shapes[dim + 1 :]
                )
                weight = weight.view(new_shape)
                new_size = self.size // (weight_shapes[dim] // self.split_size)
            res = pad_interleaved(
                weight,
                dim,
                new_size,
                weight.shape[dim] // self.interleaved_factor,
                (new_size - weight.shape[dim]) // self.interleaved_factor,
            )
            return res.view(padded_shape)


class DecoderLayer:
    def __init__(
        self,
        tp_degree,
        n_positions,
        batch_size,
        attention_head_size,
        n_head,
        amp,
        n_kv_head=0,
        neuron_config=None,
        allow_pad=False,
        n_active_tokens=None,
        layer_num=None,
        is_unit_scale=False,
    ):
        super().__init__()
        self.pre_attn_ln_weight = None
        self.pre_attn_ln_bias = None
        self.attn_q_weight = None
        self.attn_q_bias = None
        self.attn_k_weight = None
        self.attn_k_bias = None
        self.attn_v_weight = None
        self.attn_v_bias = None
        self.attn_out_weight = None
        self.attn_out_bias = None
        self.post_attn_ln_weight = None
        self.post_attn_ln_bias = None
        self.pre_mlp_ln_weight = None
        self.pre_mlp_ln_bias = None
        self.mlp_in_weight = None
        self.mlp_in_bias = None
        self.mlp_out_weight = None
        self.mlp_out_bias = None
        self.post_mlp_ln_weight = None
        self.post_mlp_ln_bias = None
        self.attn_q_min = None
        self.attn_q_max = None
        self.attn_k_min = None
        self.attn_k_max = None
        self.attn_v_min = None
        self.attn_v_max = None
        self.attn_out_min = None
        self.attn_out_max = None
        self.mlp_in_min = None
        self.mlp_in_max = None
        self.mlp_out_min = None
        self.mlp_out_max = None
        # Create KV caches for each batch_size
        self.attn_k_cache = None
        self.attn_v_cache = None
        self.cache_shape = None
        self.tp_degree = tp_degree
        self.n_positions = n_positions
        self.n_head = n_head
        self.n_head_padded = None
        self.n_kv_head = n_kv_head
        self.batch_size = batch_size
        self.attention_head_size = (
            attention_head_size  # TODO: rename this to size_per_head
        )
        self.tp_degree = tp_degree
        self.amp = amp
        self.cache_dtype = dtypes.to_torch_dtype(amp)
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        self.extra_parameters = []
        self.allow_pad = allow_pad
        self.attn_out_sharding = 0
        self.attn_out_transposed = True
        self.mlp_out_sharding = 0
        self.mlp_out_transposed = True
        self.kv_replication = 1  # default value to denote weight replication factor
        self.layer_num = layer_num
        self.is_unit_scale = is_unit_scale
        self._cpu_compile = False

    def add_parameter(
        self, param, sharding=None, allow_pad=False, allow_transform=False
    ):
        self.extra_parameters.append((param, sharding, allow_pad, allow_transform))

    def add_pre_attention_layer_norm(self, weight, bias):
        self.pre_attn_ln_weight = weight
        self.pre_attn_ln_bias = bias

    def add_attention_query(self, weight, bias):
        self.attn_q_weight = weight
        self.attn_q_bias = bias

    def add_attention_key(self, weight, bias):
        self.attn_k_weight = weight
        self.attn_k_bias = bias

    def add_attention_value(self, weight, bias):
        self.attn_v_weight = weight
        self.attn_v_bias = bias

    def add_attention_output(
        self,
        weight,
        bias,
        sharding=0,
        transposed=True,
        out_feature_dim=None,
        contract_dims=None,
        pad=True,
    ):
        self.attn_out_weight = weight
        self.attn_out_bias = bias
        self.attn_out_sharding = sharding
        self.attn_out_transposed = transposed
        self.attn_out_feature_dim = out_feature_dim
        self.attn_out_contract_dims = contract_dims
        self.attn_out_pad = pad

    def add_pre_mlp_layer_norm(self, weight, bias):
        self.pre_mlp_ln_weight = weight
        self.pre_mlp_ln_bias = bias

    def add_mlp_input(self, weight, bias):
        self.mlp_in_weight = weight
        self.mlp_in_bias = bias

    def add_mlp_output(self, weight, bias, sharding=0, transposed=True):
        self.mlp_out_weight = weight
        self.mlp_out_bias = bias
        self.mlp_out_sharding = sharding
        self.mlp_out_transposed = transposed

    def to_neuron(self):
        # If we allow padding then we need to pad non-sharded QKV weight dimensions
        self.neuron_config.n_head_padded = self.n_head
        if self.allow_pad:
            # Hidden size padding
            _, hidden_size = self.attn_q_weight.shape
            n_heads = hidden_size // self.attention_head_size

            n_head_padded, n_kv_heads_padded = get_qkv_padding(
                n_heads, self.n_kv_head, self.tp_degree, self.neuron_config
            )
            self.n_head_padded = n_head_padded
            self.neuron_config.n_head_padded = self.n_head_padded

            hidden_size_padded = hidden_size_padded_qkv = (
                n_head_padded * self.attention_head_size
            )
            if (
                self.neuron_config.group_query_attention
                == constants.GQA.ALL_GATHER_HEADS
            ):
                qkv_maybe_pad = attn_out_maybe_pad = MaybePadder(
                    hidden_size_padded,
                    padding="interleaved",
                    split_size=n_heads,
                    interleaved_factor=self.n_kv_head,
                )
            else:
                qkv_maybe_pad = MaybePadder(hidden_size_padded_qkv)
                attn_out_maybe_pad = MaybePadder(hidden_size_padded)

                # Adjust padding strategy if we can use less K/V replication
                # with interleaved padding.
                extra_heads = n_head_padded - n_heads
                if (
                    self.n_head != self.n_kv_head
                    and self.neuron_config.group_query_attention
                    == constants.GQA.REPLICATED_HEADS
                    and self.tp_degree % self.n_kv_head == 0
                    and extra_heads % self.n_kv_head == 0
                    and extra_heads > 0
                ):
                    qkv_maybe_pad = MaybePadder(
                        hidden_size_padded_qkv,
                        padding="interleaved",
                        split_size=n_heads,
                        interleaved_factor=self.n_kv_head,
                    )
                    attn_out_maybe_pad = MaybePadder(
                        hidden_size_padded,
                        padding="interleaved",
                        split_size=n_heads,
                        interleaved_factor=self.n_kv_head,
                    )

            self.attn_q_weight = qkv_maybe_pad(self.attn_q_weight, dim=1)
            self.attn_q_bias = qkv_maybe_pad(self.attn_q_bias, dim=0)

            node_interleaving = False

            if n_kv_heads_padded != self.n_kv_head:
                if n_kv_heads_padded % self.n_kv_head == 0:
                    ratio = int(n_kv_heads_padded / self.n_kv_head)
                else:
                    ratio = int((n_kv_heads_padded - extra_heads) / self.n_kv_head)

                # Full replication: replicate KV heads to original Q heads and then do padding
                if n_head_padded == n_kv_heads_padded and extra_heads > 0:
                    ratio = int((n_kv_heads_padded - extra_heads) / self.n_kv_head)

                def repeat(weight):
                    if weight is None:
                        return weight
                    shape = weight.shape[:-1] + (
                        self.n_kv_head,
                        weight.shape[-1] // self.n_kv_head,
                    )
                    weight = weight.view(shape)
                    weight = torch.repeat_interleave(weight, repeats=ratio, dim=-2)
                    shape = weight.shape[:-2] + (weight.shape[-1] * weight.shape[-2],)
                    return weight.view(shape)

                def pad_kv_no_repeat(weight, pad_size):
                    if weight is None:
                        return weight
                    shape = weight.shape[:-1] + (
                        self.n_kv_head,
                        weight.shape[-1] // self.n_kv_head,
                    )
                    weight = weight.view(shape)
                    weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_size))
                    shape = weight.shape[:-2] + (weight.shape[-1] * weight.shape[-2],)
                    return weight.view(shape)

                if ratio == 0:
                    # in case no replication is needed, pad kv based on n_kv_heads_padded calculated above
                    self.attn_k_weight = pad_kv_no_repeat(
                        self.attn_k_weight, n_kv_heads_padded - self.n_kv_head
                    )
                    self.attn_v_weight = pad_kv_no_repeat(
                        self.attn_v_weight, n_kv_heads_padded - self.n_kv_head
                    )
                    self.attn_k_bias = pad_kv_no_repeat(
                        self.attn_k_bias, n_kv_heads_padded - self.n_kv_head
                    )
                    self.attn_v_bias = pad_kv_no_repeat(
                        self.attn_v_bias, n_kv_heads_padded - self.n_kv_head
                    )
                    self.n_kv_head = n_kv_heads_padded
                else:
                    self.attn_k_weight = repeat(self.attn_k_weight)
                    self.attn_v_weight = repeat(self.attn_v_weight)
                    self.attn_k_bias = repeat(self.attn_k_bias)
                    self.attn_v_bias = repeat(self.attn_v_bias)
                    self.n_kv_head *= ratio
                self.kv_replication = ratio
                # FIXME: As a workaround to get kv_replication info (after padding) in HLO construction
                self.neuron_config.kv_replication = self.kv_replication

            if self.n_head == self.n_kv_head:
                self.attn_k_weight = qkv_maybe_pad(self.attn_k_weight, dim=1)
                self.attn_k_bias = qkv_maybe_pad(self.attn_k_bias, dim=0)

                self.attn_v_weight = qkv_maybe_pad(self.attn_v_weight, dim=1)
                self.attn_v_bias = qkv_maybe_pad(self.attn_v_bias, dim=0)

            def interleave_by_node(tensor, dim, n_nodes):
                if tensor is None:
                    return tensor
                shape = tensor.shape
                assert shape[dim] % n_nodes == 0, (
                    f"cannot interleave across node for tensor shape {shape}"
                    f" and n_nodes {n_nodes}"
                )
                stride = constants.TRN1_WORLD_SIZE
                view_shape = (
                    (stride, shape[0] // stride, shape[1])
                    if dim == 0
                    else (shape[0], stride, shape[1] // stride)
                )
                return (
                    tensor.reshape(view_shape).permute(1, 0, 2).reshape(shape)
                    if dim == 0
                    else tensor.reshape(view_shape).permute(0, 2, 1).reshape(shape)
                )

            if node_interleaving:
                n_nodes = self.tp_degree // constants.TRN1_WORLD_SIZE
                self.attn_q_weight = interleave_by_node(
                    self.attn_q_weight, dim=1, n_nodes=n_nodes
                )
                self.attn_k_weight = interleave_by_node(
                    self.attn_k_weight, dim=1, n_nodes=n_nodes
                )
                self.attn_v_weight = interleave_by_node(
                    self.attn_v_weight, dim=1, n_nodes=n_nodes
                )
                self.attn_q_bias = interleave_by_node(
                    self.attn_q_bias, dim=0, n_nodes=n_nodes
                )
                self.attn_k_bias = interleave_by_node(
                    self.attn_k_bias, dim=0, n_nodes=n_nodes
                )
                self.attn_v_bias = interleave_by_node(
                    self.attn_v_bias, dim=0, n_nodes=n_nodes
                )

            if self.neuron_config and self.neuron_config.fuse_qkv:
                fused_qkv_weight = interleave_qkv(
                    self.attn_q_weight,
                    self.attn_k_weight,
                    self.attn_v_weight,
                    self.tp_degree,
                    dim=1,
                )
                if self.attn_q_bias is not None:
                    fused_qkv_bias = interleave_qkv(
                        self.attn_q_bias,
                        self.attn_k_bias,
                        self.attn_v_bias,
                        self.tp_degree,
                        dim=0,
                    )
                else:
                    fused_qkv_bias = None
                self.attn_k_weight = None
                self.attn_k_bias = None
                self.attn_v_weight = None
                self.attn_v_bias = None
            if self.attn_out_pad:
                self.attn_out_weight = attn_out_maybe_pad(
                    self.attn_out_weight, dim=self.attn_out_sharding
                )
            if node_interleaving:
                self.attn_out_weight = interleave_by_node(
                    self.attn_out_weight, dim=self.attn_out_sharding, n_nodes=n_nodes
                )
            # Intermediate MLP layer padding
            if self.mlp_in_weight is not None:
                _, intermediate_size = self.mlp_in_weight.shape
                intermediate_size_padded = round_up_to_divisor(
                    intermediate_size, self.tp_degree
                )
                maybe_pad = MaybePadder(intermediate_size_padded)

                self.mlp_in_weight = maybe_pad(self.mlp_in_weight, dim=1)
                self.mlp_in_bias = maybe_pad(self.mlp_in_bias, dim=0)
                if self.neuron_config.fuse_mlp:
                    intermediate_size = intermediate_size // 2
                    intermediate_size_padded = round_up_to_divisor(
                        intermediate_size, self.tp_degree
                    )
                    maybe_pad = MaybePadder(intermediate_size_padded)
                self.mlp_out_weight = maybe_pad(
                    self.mlp_out_weight, dim=self.mlp_out_sharding
                )

        if self.neuron_config and self.neuron_config.fused_rmsnorm_qkv:
            self.fused_pre_attn_ln_qkv_weight = (
                fused_qkv_weight.T
                * self.pre_attn_ln_weight.to(dtype=fused_qkv_weight.dtype)
            ).T

        maybe_manipulator = MaybeParallelTensorManipulator(
            self.tp_degree,
            on_cpu=self._cpu_compile,
        )
        maybe_duplicate = maybe_manipulator.duplicate
        maybe_shard_along = maybe_manipulator.shard_along
        maybe_primary_only = maybe_manipulator.primary_only
        maybe_shard_along_and_transform = maybe_manipulator.shard_along_and_transform
        self.pre_attn_ln_weight = maybe_duplicate(self.pre_attn_ln_weight)
        self.pre_attn_ln_bias = maybe_duplicate(self.pre_attn_ln_bias)
        qkv_weight_sharder = maybe_shard_along
        if self.neuron_config and self.neuron_config.fuse_qkv:
            self.attn_q_weight = qkv_weight_sharder(fused_qkv_weight, dim=1)
            self.attn_q_bias = maybe_shard_along(fused_qkv_bias, dim=0)
        else:
            self.attn_q_weight = qkv_weight_sharder(self.attn_q_weight, dim=1)
            self.attn_q_bias = maybe_shard_along(self.attn_q_bias, dim=0)
        self.attn_k_weight = qkv_weight_sharder(self.attn_k_weight, dim=1)
        self.attn_k_bias = maybe_shard_along(self.attn_k_bias, dim=0)
        self.attn_v_weight = qkv_weight_sharder(self.attn_v_weight, dim=1)
        self.attn_v_bias = maybe_shard_along(self.attn_v_bias, dim=0)
        self.attn_out_weight = maybe_shard_along(
            self.attn_out_weight, dim=self.attn_out_sharding
        )
        self.attn_out_bias = maybe_primary_only(self.attn_out_bias)
        self.post_attn_ln_weight = maybe_duplicate(self.post_attn_ln_weight)
        self.post_attn_ln_bias = maybe_duplicate(self.post_attn_ln_bias)
        self.pre_mlp_ln_weight = maybe_duplicate(self.pre_mlp_ln_weight)
        self.pre_mlp_ln_bias = maybe_duplicate(self.pre_mlp_ln_bias)
        if self.mlp_in_weight is not None:
            self.mlp_in_weight = maybe_shard_along_and_transform(self.mlp_in_weight, 1)
            self.mlp_in_bias = maybe_shard_along(self.mlp_in_bias, dim=0)
            self.mlp_out_weight = maybe_shard_along_and_transform(
                self.mlp_out_weight, dim=self.mlp_out_sharding
            )
            self.mlp_out_bias = maybe_primary_only(self.mlp_out_bias)
        self.post_mlp_ln_weight = maybe_duplicate(self.post_mlp_ln_weight)
        self.post_mlp_ln_bias = maybe_duplicate(self.post_mlp_ln_bias)

        extras = []
        for param, dim, allow_pad, allow_transform in self.extra_parameters:
            if allow_pad:
                size = round_up_to_divisor(param.shape[dim], self.tp_degree)
                param = maybe_pad_tensor(param, dim, size)

            if allow_transform:
                param = maybe_shard_along_and_transform(param, dim)
            else:
                param = maybe_manipulator.duplicate_or_shard_along(param, dim)

            extras.append(param)

        self.extra_parameters = extras
        self.init_caches()

    @property
    def shard_over_batch(self):
        return (
            self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        )

    def init_caches(self):
        n_heads_kv_cache = self.n_kv_head

        # When padding, compute the hidden size based on the padding. We must
        # allow the KV cache to be padded so it can be evenly divisible across
        # NeuronCores.
        if self.allow_pad and not self.shard_over_batch:
            n_heads_kv_cache = round_up_to_divisor(self.n_kv_head, self.tp_degree)
        # Select manipulator based on device
        if self._cpu_compile:
            manipulator = parallel.CPUTensorManipulator(self.tp_degree)
        else:
            manipulator = parallel.ParallelTensorManipulator(self.tp_degree)
        cpu_cache_shape = [
            self.n_positions,
            self.batch_size,
            n_heads_kv_cache,
            self.attention_head_size,
        ]
        self.cache_shape = [
            self.n_positions,
            self.batch_size,
            n_heads_kv_cache // self.tp_degree,
            self.attention_head_size,
        ]
        cpu_cache = torch.zeros(cpu_cache_shape, dtype=self.cache_dtype)
        assert (n_heads_kv_cache >= self.tp_degree) and (
            n_heads_kv_cache % self.tp_degree == 0
        ), (
            f"cannot shard along kv_heads dimension: n_kv_head={n_heads_kv_cache}, tp_degree={self.tp_degree}"
        )
        self.attn_k_cache = manipulator.shard_along(cpu_cache, dim=2)
        self.attn_v_cache = manipulator.shard_along(cpu_cache, dim=2)

    def assign_caches(self, layer):
        self.attn_k_cache = layer.attn_k_cache
        self.attn_v_cache = layer.attn_v_cache
        self.cache_shape = layer.cache_shape

    def all_parameters(self):
        return [
            self.pre_attn_ln_weight,
            self.pre_attn_ln_bias,
            self.attn_q_weight,
            self.attn_q_bias,
            self.attn_k_weight,
            self.attn_k_bias,
            self.attn_v_weight,
            self.attn_v_bias,
            self.attn_out_weight,
            self.attn_out_bias,
            self.post_attn_ln_weight,
            self.post_attn_ln_bias,
            self.pre_mlp_ln_weight,
            self.pre_mlp_ln_bias,
            self.mlp_in_weight,
            self.mlp_in_bias,
            self.mlp_out_weight,
            self.mlp_out_bias,
            self.post_mlp_ln_weight,
            self.post_mlp_ln_bias,
            *self.extra_parameters,
        ]

    def valid_parameters(self):
        return [par for par in self.all_parameters() if par is not None]

    def reset(self):
        for batch_size in self.batch_sizes:
            # CPU compilation sometimes returns tensors in a list, eg. [tensor(...), tensor(...)]
            if isinstance(self.attn_k_cache, list):
                self.attn_k_cache = torch.cat(self.attn_k_cache)
            zero_cache = torch.zeros(
                self.attn_k_cache.shape,
                dtype=self.attn_k_cache.dtype,
            )
            zero_cache = [zero_cache for _ in range(self.self.tp_degree)]
            if not self._cpu_compile:
                ops.parallel_write(self.attn_k_cache, zero_cache)
                ops.parallel_write(self.attn_v_cache, zero_cache)
            else:
                self.attn_k_cache = zero_cache
                self.attn_v_cache = zero_cache

    def assign_parameters(self, layer):
        self.pre_attn_ln_weight = layer.pre_attn_ln_weight
        self.pre_attn_ln_bias = layer.pre_attn_ln_bias
        self.attn_q_weight = layer.attn_q_weight
        self.attn_q_bias = layer.attn_q_bias
        self.attn_k_weight = layer.attn_k_weight
        self.attn_k_bias = layer.attn_k_bias
        self.attn_v_weight = layer.attn_v_weight
        self.attn_v_bias = layer.attn_v_bias
        self.attn_out_weight = layer.attn_out_weight
        self.attn_out_bias = layer.attn_out_bias
        self.post_attn_ln_weight = layer.post_attn_ln_weight
        self.post_attn_ln_bias = layer.post_attn_ln_bias
        self.pre_mlp_ln_weight = layer.pre_mlp_ln_weight
        self.pre_mlp_ln_bias = layer.pre_mlp_ln_bias
        self.mlp_in_weight = layer.mlp_in_weight
        self.mlp_in_bias = layer.mlp_in_bias
        self.mlp_out_weight = layer.mlp_out_weight
        self.mlp_out_bias = layer.mlp_out_bias
        self.post_mlp_ln_weight = layer.post_mlp_ln_weight
        self.post_mlp_ln_bias = layer.post_mlp_ln_bias
        self.attn_q_min = layer.attn_q_min
        self.attn_q_max = layer.attn_q_max
        self.attn_k_min = layer.attn_k_min
        self.attn_k_max = layer.attn_k_max
        self.attn_v_min = layer.attn_v_min
        self.attn_v_max = layer.attn_v_max
        self.attn_out_min = layer.attn_out_min
        self.attn_out_max = layer.attn_out_max
        self.mlp_in_min = layer.mlp_in_min
        self.mlp_in_max = layer.mlp_in_max
        self.mlp_out_min = layer.mlp_out_min
        self.mlp_out_max = layer.mlp_out_max
        self.extra_parameters = layer.extra_parameters


class MaybeParallelTensorManipulator:
    def __init__(self, tp_degree, on_cpu=False):
        self.use_cpu = on_cpu
        if on_cpu:
            self.manipulator = parallel.CPUTensorManipulator(tp_degree)
        else:
            self.manipulator = parallel.ParallelTensorManipulator(tp_degree)

    def duplicate(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.duplicate(tensor)

    def shard_along(self, tensor, dim):
        if tensor is None:
            return None
        return self.manipulator.shard_along(tensor, dim)

    def primary_only(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.primary_only(tensor)

    def duplicate_or_shard_along(self, tensor, dim):
        if dim is None:
            return self.duplicate(tensor)
        return self.shard_along(tensor, dim)

    def shard_along_and_transform(self, tensor, dim):
        if tensor is None:
            return None
        tensors = self.manipulator.shard_along_on_cpu(tensor, dim)
        if not self.use_cpu:
            tensors = ops.parallel_to_nc(tensors)
        return tensors


class DecoderParameterBuilder:
    def __init__(self, scribe, parameter_number):
        self.scribe = scribe
        self.parameter_number = parameter_number
        self.dtype_converter = compiler.DataTypeConverter()

    def from_tensor(self, tensor, dim_size=None):
        if tensor is None:
            return None
        # Tensor may be a list of tensors (e.g. [tensor(...)]) during CPU compilation flow
        if isinstance(tensor, list):
            tensor = tensor[0]
        name = self.dtype_converter.torch2name(tensor.dtype)
        dtype = getattr(self.scribe, name)
        sizes = list(tensor.shape)
        if dim_size is not None:
            for dim, size in dim_size.items():
                sizes[dim] = size
        param = dtype[sizes].Parameter(parameter_number=self.parameter_number)
        self.parameter_number += 1
        return param


class DecoderProgram:
    def __init__(
        self,
        neuron_config,
        layers,
        hlo_module,
        num_inputs,
        tp_degree,
        n_positions,
        batch_size,
        tag=None,
        num_exec_repetition=1,
        on_cpu=False,
    ):
        self.neuron_config = neuron_config
        self.layers = layers
        self.batch_size = batch_size
        self.n_positions = n_positions
        self.input_buffers = [
            compiler.gen_zero_input(hlo_module, idx) for idx in range(num_inputs)
        ]
        kernel_tag = f"seqlen{n_positions}-batch{batch_size}"
        if tag is not None:
            kernel_tag = f"{tag}-seqlen{n_positions}-batch{batch_size}"
        self.kernel = compiler.ParallelKernel(
            hlo_module,
            tp_degree,
            g_start_device_id=0,
            g_device_count=tp_degree,
            tag=kernel_tag,
            num_exec_repetition=num_exec_repetition,
        )
        self.n_active_tokens = read_n_active_tokens(hlo_module)
        self.tp_degree = tp_degree
        self.tag = tag
        self._cpu_compile = on_cpu
        # Select manipulator based on device
        if self._cpu_compile:
            self.manipulator = parallel.CPUTensorManipulator(tp_degree)
        else:
            self.manipulator = parallel.ParallelTensorManipulator(tp_degree)

    def setup(self, io_ring_cache_size):
        self.input_buffers = [
            self.manipulator.duplicate(buf) for buf in self.input_buffers
        ]
        self.logits_buffer = self.manipulator.duplicate(self.logits_buffer)
        self.kernel.load(io_ring_cache_size)

    def inputs_host_to_device(self, input_tensors):
        assert not (len(input_tensors) == 5 and len(self.input_buffers) == 6)
        for idx, (buf, tensor) in enumerate(zip(self.input_buffers, input_tensors)):
            tensor = tensor.to(buf.dtype)
            tensor = self.manipulator.duplicate_on_cpu(tensor)
            assert buf.shape == tensor[0].shape, (
                f"Copying tensor from host to device: buffer ({buf.shape}) and tensor ({tensor[0].shape}) have different shapes!"
            )
            if not self._cpu_compile:
                ops.parallel_write(buf, tensor)

    def run(self):
        raise NotImplementedError(DecoderProgram)

    def maybe_logits_device_to_host(self, return_ranks):
        if self.logits_buffer is not None:
            if self.tp_degree == self.self.tp_degree:
                logits = self.manipulator.unshard_along(self.logits_buffer, dim=0)
                if return_ranks > 0:
                    rank_size = logits.shape[0] // self.tp_degree
                    logits = logits[: rank_size * return_ranks]
                return logits
            else:
                return ops.parallel_cpu(self.logits_buffer)[0]
        else:
            return None

    def _fill_io_tensors(self, input_tensors, output_tensors, layers):
        end = self.n_positions
        for layer in layers:
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                cache_slice = self.manipulator.slice_on_nc(
                    cache, 0, start=0, end=end, step=1
                )
                input_tensors.append(cache_slice)
                output_tensors.append(cache_slice)
        for layer in layers:
            input_tensors.extend(layer.valid_parameters())


class DecoderProgramFullyUnrolled(DecoderProgram):
    def __init__(
        self,
        neuron_config,
        layers,
        hlo_module,
        num_inputs,
        tp_degree,
        n_positions,
        batch_size,
        tag=None,
        on_cpu=False,
    ):
        super().__init__(
            neuron_config,
            layers,
            hlo_module,
            num_inputs,
            tp_degree,
            n_positions,
            batch_size,
            tag=tag,
            on_cpu=on_cpu,
        )
        self.logits_buffer = compiler.gen_zero_output(hlo_module, 0)
        self.memory = None
        self.executor = None

    def setup(self, layers, pre_layer_params, ln_lm_head_params):
        super().setup(io_ring_cache_size=1)

        self.memory = self.kernel.build_memory()

        # Setup the memory with input and output buffers
        input_tensors = self.input_buffers
        output_tensors = [self.logits_buffer]
        self._fill_io_tensors(input_tensors, output_tensors, layers)
        input_tensors.extend(pre_layer_params)
        input_tensors.extend(ln_lm_head_params)
        self.memory.setup(input_tensors, output_tensors)

        # Warmup kernel to avoid unexpected initialization at runtime
        self.kernel.warmup()

    def run(self):
        self.kernel(self.memory)

    def enable_executor(self):
        input_tensors = [*self.input_buffers]
        output_tensors = [self.logits_buffer]
        self.executor = self.kernel.build_executor(
            self.memory, input_tensors, output_tensors
        )

    def execute(self, *inputs, return_ranks=-1):
        """
        Execute a kernel with using the optimized ParallelExecutor.

        Arguments:
            inputs: The set of CPU tensors to copy to each model
            return_ranks: The number of ranks to copy back to CPU
        """
        return self.executor(inputs, return_ranks)
