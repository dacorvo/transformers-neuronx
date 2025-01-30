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
import torch
import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from .compiler import ParallelKernel
from .constants import LAYOUT_BSH, LAYOUT_HSB
from .module import PretrainedModel
from .ops import init_neuron
from .utils import maybe_pad_tensor


# Mainly used to expose top level APIs to the model object for serialization
class NeuronModelBase(PretrainedModel):
    def __init__(self, chkpt_model_cls, *args, **kwargs):
        super().__init__()
        self.chkpt_model = chkpt_model_cls(*args, **kwargs)

    def load_state_dict_dir(self, pretrained_model_path):
        self.chkpt_model.load_state_dict_dir(pretrained_model_path)

    # top level api
    def load(self, directory):
        assert self.serialization_enabled(), (
            "serialization is not enabled for this model"
        )
        self._compiled_artifacts_directory = directory

    # top level api
    def compile(self, parallel_degree=None):
        kernels = self._get_all_kernels()
        neff_bytes_futures = dict()
        if parallel_degree is None:
            parallel_degree = len(kernels)
        with ProcessPoolExecutor(parallel_degree) as executor:
            for kernel in kernels:
                neff_bytes_futures[hash_hlo(kernel.hlo_module)] = executor.submit(
                    kernel.compile, kernel.num_exec_repetition
                )
            for kernel in kernels:
                kernel.neff_bytes = neff_bytes_futures[
                    hash_hlo(kernel.hlo_module)
                ].result()

    # top level api
    def setup(self):
        for nbs in self.nbs_objs:
            nbs.setup()

    # TODO: decouple hlo_generation from load weights so compile can be called before it
    def to_neuron(self):
        self.decoder_lm_head._cpu_compile = False
        init_neuron()
        self.load_weights()
        if hasattr(self, "_compiled_artifacts_directory"):
            if not os.path.isdir(self._compiled_artifacts_directory):
                raise FileNotFoundError(
                    f"Did not find directory: {self._compiled_artifacts_directory}."
                )
            for nbs_obj in self.nbs_objs:
                nbs_obj.set_neff_bytes(self._compiled_artifacts_directory)
        else:
            self.compile(parallel_degree=self.neuron_config.compilation_worker_count)
        self.setup()

    def save(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(
                f"Artifacts should be saved to a directory. "
                f"Found existing file: {directory}"
            )
        os.makedirs(directory, exist_ok=True)
        for i, nbs_obj in enumerate(self.nbs_objs):
            nbs_obj.save_compiler_artifacts(directory)

    def _get_all_kernels(self):
        all_kernels = []
        for nbs in self.nbs_objs:
            for kernel in nbs.get_all_kernels():
                all_kernels.append(kernel)
        return all_kernels

    # To enable serialization, have the model call this
    # function to register all nbs_obj of your model.
    # The nbs_obj must follow 2 rules:
    #   1. The nbs_obj must inherit from NeuronBaseSerializer.
    #   2. Since NeuronBaseSerializer is abstract, a nbs_obj.get_all_kernels()
    #      method should be implemented by the child class, which returns a
    #      list of all kernels which have NEFFs for that serialized object.
    def register_for_serialization(self, nbs_obj):
        assert issubclass(type(nbs_obj), NeuronBaseSerializer), (
            "The nbs_obj must inherit from NeuronBaseSerializer."
        )
        temp = getattr(self, "nbs_objs", [])
        nbs_obj.compiler_artifacts_path = None
        temp.append(nbs_obj)
        self.nbs_objs = temp

    def serialization_enabled(self):
        return getattr(self, "nbs_objs", None) is not None

    def profile(self, profile_dir, ntff_count_limit):
        kernels = self._get_all_kernels()

        for kernel in kernels:
            if isinstance(kernel, ParallelKernel):
                kernel.profile(profile_dir, ntff_count_limit)


class NeuronHloDecoderModel(NeuronModelBase):
    def reset(self):
        self.decoder_lm_head.reset()

    def decode(self, hidden, *args):
        return self.decoder_lm_head.forward(hidden, *args)

    def context(self, hidden, cache_ids, start_ids, last_token_id, *rest):
        return self.decoder_lm_head_for_context.forward(
            hidden, cache_ids, start_ids, last_token_id, *rest
        )

    def _prepare_for_par_ctx_rhs_padding(
        self, input_ids, cache_ids, start_ids=None, **kwargs
    ):
        """A helper to do rhs padding on prompt for parallel context encoding model
        i.e.
            input_ids = [[111, 222, 333]]
            context_length = 3

            if context bucket size is 4
            we will pad input_ids to [[111, 222, 333, 0]]

            last_token_id = 2 (used for generation to mark the last token is at index 2 instead 3)

        Note:
            - there is no change on start_ids with right padding.
            - cache_ids will be set to [0, 1, 2, 3] in self.forward()
        """
        batch_size, context_length = input_ids.shape

        block_tables = torch.tensor([0])
        context_lens = torch.tensor([0])

        # if last_token_id not used, simply set to 0
        if self.neuron_config.vectorize_last_token_id:
            last_token_id = torch.zeros(batch_size, dtype=torch.int32)
        else:
            last_token_id = torch.as_tensor([0], dtype=torch.int32)
        if context_length == 1:
            # token generation
            return input_ids, cache_ids, last_token_id, block_tables, context_lens

        estimate = self.neuron_config.n_positions

        if estimate:
            # when context length is larger than estimate, last_token_id=estimate-1
            if self.neuron_config.vectorize_last_token_id:
                last_token_id = cache_ids.max(dim=1).values
            else:
                last_token_id = torch.as_tensor(
                    [min(context_length - 1, estimate - 1)], dtype=torch.int32
                )
            if context_length < estimate:
                input_ids = maybe_pad_tensor(input_ids, 1, estimate, left=False)
                cache_ids = self._pad_cache_ids(
                    cache_ids, batch_size, context_length, estimate
                )

        return input_ids, cache_ids, last_token_id, block_tables, context_lens

    def _pad_cache_ids(self, cache_ids, batch_size, context_length, estimate):
        if self.neuron_config.use_2d_cache_ids:
            cache_ids = torch.arange(estimate, dtype=torch.int32)
            cache_ids = cache_ids.unsqueeze(0).expand(batch_size, estimate)
        else:
            if cache_ids is None:
                cache_ids = torch.arange(estimate, dtype=torch.int32)
            else:
                # Inputs: cache_ids = [16, 17], estimate = 512
                #
                # Process:
                # start_idx = 18, end_idx = 528 (= 512+16)
                # padded_elements =       [18, 19, ..., 511, 512, 513, ..., 525, 526, 527]
                # cache_ids_pad = [16, 17, 18, 19, ..., 511, 512, 513, ..., 525, 526, 527]
                # cache_ids =     [16, 17, 18, 19, ..., 511, 511, 511, ..., 511, 511, 511]
                start_idx = cache_ids[-1].item() + 1
                end_idx = estimate + start_idx - context_length
                pad_elements = torch.arange(start_idx, end_idx, dtype=torch.int32)
                cache_ids_pad = torch.concat([cache_ids, pad_elements], dim=0)
                cache_ids = torch.minimum(
                    cache_ids_pad, torch.tensor(estimate - 1, dtype=torch.int32)
                )
        return cache_ids

    def _prepare_for_continuous_batching(self, input_ids, cache_ids=None, seq_ids=None):
        n_seqs, n_active_tokens = input_ids.shape
        continuous_batching = (
            self.neuron_config and self.neuron_config.continuous_batching
        )

        if seq_ids is None or not continuous_batching:
            # static batching
            return input_ids, cache_ids, seq_ids

        batch_size = self.neuron_config.continuous_batching.batch_size_for_shared_caches

        if (n_active_tokens > 1) and cache_ids.flatten()[0].item() == 0:
            # context encoding
            n_active_seqs, n_active_tokens = input_ids.shape
            n_positions = self.neuron_config.n_positions
            assert n_active_seqs == cache_ids.shape[0], (
                f"invalid n_active_seqs ({n_active_seqs} vs {cache_ids.shape[0]})"
            )
            assert n_active_tokens <= n_positions, (
                f"invalid input prompt length ({n_active_tokens} <= {n_positions})"
            )
            cache_ids_pad = torch.zeros(
                n_active_seqs,
                n_positions,
                dtype=cache_ids.dtype,
                device="cpu",
            )
            for seq_id in range(n_active_seqs):
                cache_ids_pad[seq_id, :n_active_tokens] = cache_ids[
                    seq_id, :n_active_tokens
                ]
            return input_ids, cache_ids_pad, seq_ids

        # token generation - padding for naive continuous batching
        full_input_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
        full_cache_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
        full_seq_ids = torch.arange(batch_size, dtype=torch.int32)

        # vLLM v0.3.3 used to pass 1d seq_ids but starting with
        # v0.4.0 that is no longer the case. To ensure consistent behaviour
        # across versions, we flatten them before unsqueezing them.
        seq_ids_int64 = seq_ids.flatten().unsqueeze(-1).to(torch.int64)
        full_input_ids.scatter_(dim=0, index=seq_ids_int64, src=input_ids)
        full_cache_ids.scatter_(dim=0, index=seq_ids_int64, src=cache_ids)

        return full_input_ids, full_cache_ids, full_seq_ids

    def _preprocess(self, input_ids, start_ids=None, cache_ids=None, **kwargs):
        # enable dynamic batch size feature for continuous batching
        input_ids, cache_ids, new_start_ids = self._prepare_for_continuous_batching(
            input_ids, cache_ids, start_ids
        )

        # right pad the input_ids if neccessary
        input_ids, cache_ids, last_token_id, block_tables, context_lens = (
            self._prepare_for_par_ctx_rhs_padding(
                input_ids, cache_ids, start_ids, **kwargs
            )
        )
        start_ids = new_start_ids

        # note: this context_length is after right padded
        batch_size, context_length = input_ids.shape

        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)

        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)
            if self.neuron_config.use_2d_cache_ids:
                cache_ids = cache_ids.unsqueeze(0).expand(batch_size, context_length)

        return (
            input_ids,
            cache_ids,
            start_ids,
            last_token_id,
            block_tables,
            context_lens,
        )

    def _postprocess(self, input_ids, logits, start_ids):
        if start_ids is None or (
            self.neuron_config.output_all_logits and logits.shape[1] > 1
        ):
            return logits

        if not self.neuron_config.lhs_aligned or input_ids.shape[-1] > 1:
            return logits

        input_batch_size = start_ids.shape[0]
        seq_ids = start_ids.flatten()
        if torch.equal(seq_ids, torch.arange(input_batch_size)):
            logits = logits[:input_batch_size]
        else:
            logits = logits[seq_ids.to(torch.long)]

        return logits

    def _cast_logits(self, logits):
        # Cast logits to float32 or the dtype specified in the neuron config
        logits_dtype = torch.float32
        if self.neuron_config:
            if self.neuron_config.cast_logits_dtype is not None:
                logits_dtype = getattr(torch, self.neuron_config.cast_logits_dtype)
        return logits.to(logits_dtype)

    def _context_dynamic_batching(self, hidden, *args):
        is_bsh = (
            self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        )
        input_batch_size = (
            hidden.shape[0]
            if is_bsh or self.neuron_config.on_device_embedding
            else hidden.shape[2]
        )

        running_batch_size = 1
        if input_batch_size > running_batch_size:
            assert input_batch_size % running_batch_size == 0, (
                "input batch size ({input_batch_size}) not divisible by running batch size ({running_batch_size})"
            )
            n_iters = input_batch_size // running_batch_size
            all_logits = []
            cache_ids, start_ids, last_token_id = args[0], args[1], args[2]
            for iter_id in range(n_iters):
                start_idx = iter_id * running_batch_size
                end_idx = (iter_id + 1) * running_batch_size
                if is_bsh or self.neuron_config.on_device_embedding:
                    hidden_per_batch = hidden[start_idx:end_idx, ...]
                else:
                    hidden_per_batch = hidden[..., start_idx:end_idx]
                cache_ids_per_batch = cache_ids[start_idx:end_idx, :]
                start_ids_per_batch = start_ids[start_idx:end_idx]
                last_token_id_per_batch = last_token_id[start_idx:end_idx]
                logits_per_batch = self.context(
                    hidden_per_batch,
                    cache_ids_per_batch,
                    start_ids_per_batch,
                    last_token_id_per_batch,
                )
                all_logits.append(logits_per_batch)
            logits = torch.cat(all_logits, dim=-1)
        else:
            assert input_batch_size == running_batch_size, (
                "input batch size ({input_batch_size}) not equal to running batch size ({running_batch_size})"
            )
            logits = self.context(hidden, *args)
        return logits

    def _forward(self, hidden, *args):
        _, context_length, *_ = hidden.shape

        if context_length > 1:
            continuous_batching = (
                self.neuron_config and self.neuron_config.continuous_batching
            )
            if continuous_batching:
                logits = self._context_dynamic_batching(hidden, *args)
            else:
                logits = self.context(hidden, *args)
        else:
            logits = self.decode(hidden, *args)

        logits = self._cast_logits(logits)
        if self.neuron_config.output_all_logits and context_length > 1:
            logits = logits.permute(2, 1, 0)
        else:
            logits = logits[: self.config.vocab_size, -1, :]
            logits = logits.transpose(0, 1)
        return logits

    def forward(
        self,
        input_ids,
        cache_ids,
        start_ids,
    ):
        original_input_ids = input_ids
        padded_inputs, *rst = self._preprocess(
            input_ids, start_ids=start_ids, cache_ids=cache_ids
        )
        input_embeddings = self.chkpt_model.model.embed_tokens(padded_inputs)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            input_embeddings = input_embeddings.transpose(0, -1).contiguous()
        logits = self._forward(input_embeddings, *rst)
        return self._postprocess(original_input_ids, logits, start_ids=start_ids)


# Base class for all "Serializable Objects"
class NeuronBaseSerializer(ABC):
    def save_compiler_artifacts(self, path):
        for kernel in self.get_all_kernels():
            hlo_hash = hash_hlo(kernel.hlo_module)
            with open(os.path.join(path, hlo_hash), "wb") as f:
                assert kernel.neff_bytes is not None, (
                    "cannot save a model which has not been successfully compiled"
                )
                f.write(kernel.neff_bytes)

    def set_neff_bytes(self, directory):
        for kernel in self.get_all_kernels():
            hlo_hash = hash_hlo(kernel.hlo_module)
            try:
                with open(os.path.join(directory, hlo_hash), "rb") as f:
                    kernel.neff_bytes = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    (
                        "Could not find a matching NEFF for your HLO in this directory. "
                        "Ensure that the model you are trying to load is the same type and "
                        'has the same parameters as the one you saved or call "save" on '
                        "this model to reserialize it."
                    )
                )

    @abstractmethod
    def get_all_kernels(self):
        raise NotImplementedError(
            f"Class {type(self)} deriving from NeuronBaseSerializer must implement get_all_kernels"
        )


def hash_hlo(hlo_module):
    hash_gen = hashlib.sha256()
    message = hlo_module.SerializeToString()
    hash_gen.update(message)
    hash = str(hash_gen.hexdigest())[:20]
    return hash + ".neff"
