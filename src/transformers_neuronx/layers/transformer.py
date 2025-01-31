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
from transformers_neuronx import hlo
from ..config import Layout


def inputs(
    scribe,
    dtype,
    batch_size,
    n_active_tokens,
    hidden_size,
    neuron_config=None,
):
    """
    Defines the set of required inputs for all decoder models.

    For any model that requires *more* inputs than the required inputs produced
    by this function, the next parameters must be linearly allocated starting
    at 4. Additonal parameters must also define their own sequence slice
    dimensions (See below).

    Args:
        scribe: The PyHLO scribe object to write operations with.
        dtype: The data type of the hidden state.
        batch_size: The active batch size (may differ from cache batch size)
        n_active_tokens: The number of active tokens to process. During context
            prefill, this will be larger than 1. During autogregressive
            token generation this will be exactly equal to 1.
        hidden_size: The size of the hidden state.
        neuron_config: Optional configurations.

    Returns:
        hidden: The hidden state (Assumed to be embedded on CPU)
        cache_ids: The positions to update in the KV cache. This is 1d when
            using RHS-alignment since all batch lines update the same
            places in the KV cache. This is 2d when LHS-alignment since
            each batch line can update a different offset in the KV cache.
        start_ids: The offset into each batch line. When using
            LHS-alignment, this indicates the start offset. When using
            RHS-alignment, this indicates the batch line to update.
        last_token_id: An integer index (along the sequence dimenson) which
            indicates which is the last token. This is used in the language
            model head to slice the hidden state.
        sequence_slice_dimensions: The dimension of each input tensor which can
            be sliced during token generation.
    """
    s32 = scribe.s32

    if neuron_config and neuron_config.attention_layout == Layout.BSH:
        hidden_sizes = batch_size, n_active_tokens, hidden_size
    else:  # HASB LAyout
        hidden_sizes = hidden_size, n_active_tokens, batch_size

    hidden = dtype[hidden_sizes].Parameter(parameter_number=0)
    cache_2d = neuron_config and neuron_config.use_2d_cache_ids
    if cache_2d:
        position_sizes = batch_size, n_active_tokens
        cache_ids = s32[position_sizes].Parameter(parameter_number=1)  # 2d cache_ids
    else:
        cache_ids = s32[n_active_tokens].Parameter(parameter_number=1)  # 1d cache_ids

    start_ids = s32[batch_size].Parameter(parameter_number=2)

    block_table = None
    context_lens = None
    # Build parameters for last_token_id and others
    if cache_2d:
        # regular token gen
        last_token_id = s32[batch_size].Parameter(parameter_number=3)
    else:
        last_token_id = s32[1].Parameter(parameter_number=3)

    # add block tables and context lens parameters as single length tensors
    # when chunked prefill is not enabled. In this case, these inputs are not used
    # but are kept for consistency and are defined as length 1 tensors.
    if not block_table:
        block_table = s32[1].Parameter(parameter_number=4)
    if not context_lens:
        context_lens = s32[1].Parameter(parameter_number=5)

    sequence_slice_dimensions = (
        1,  # hidden        | In both HSB/BSH the sequence dim is 1
        1 if cache_2d else 0,  # cache_ids     | Sequence dim varies based on alignment
        None,  # start_ids     | Offset is per batch, no slicing required
        0 if cache_2d else None,  # last_token_id | Scalar, no slicing required
        None,  # block_table   | No sequence dim
        None,  # context_lens  | No sequence dim
    )

    return (
        hidden,
        cache_ids,
        start_ids,
        last_token_id,
        block_table,
        context_lens,
    ), sequence_slice_dimensions


def ln_lm_head(
    tp_degree,
    hidden,
    last_token_id,
    ln_f_weight,
    ln_f_bias,
    lm_head_weight,
    lm_head_bias,
    is_prefill=True,
    neuron_config=None,
):
    """
    Language model head with layer normalization.

    Context encoding network:
    n_active_tokens will be equal to context_length_estimate.
    In this case we slice the hidden input and compute the next token logits only for the last context token.

    Normal token gen network:
    n_active_tokens will be 1.
    No slicing required. Will return the next token logits for the current active token.

    Models: GPT2, OPT, GPT-J, GPTNeoX, BLOOM.

    logits = (layer_norm(H) @ W) + B
    """
    is_bsh = neuron_config and neuron_config.attention_layout == Layout.BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes

    if is_prefill:
        hidden = _dynamic_logits_slice(hidden, last_token_id, neuron_config)
        # slice the hidden input and compute the next token logits only for the last context token.
        n_active_tokens = 1

    if is_bsh:
        ln_hidden = hlo.layer_norm_bsh(
            hidden, ln_f_weight, ln_f_bias, neuron_config=None, tp_degree=tp_degree
        )
        ln_hidden = hlo.transpose210(ln_hidden)
    else:
        ln_hidden = hlo.layer_norm(
            hidden, ln_f_weight, ln_f_bias, neuron_config=None, tp_degree=tp_degree
        )
    ln_hidden = hlo.reshape(
        ln_hidden, shape=(hidden_size, n_active_tokens * batch_size)
    )

    logits = hlo.dot00(lm_head_weight, ln_hidden)
    if lm_head_bias is not None:
        lm_head_bias = hlo.broadcast(
            lm_head_bias, out_dim_size=logits.sizes, broadcast_dimensions=[0]
        )
        logits = hlo.add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    return hlo.reshape(logits, shape=(vocab_size, n_active_tokens, batch_size))


def rms_lm_head(
    tp_degree,
    hidden,
    last_token_id,
    rms_weight,
    lm_head_weight,
    lm_head_bias,
    is_prefill=True,
    eps=1e-6,
    neuron_config=None,
):
    """
    Language model head with rms normalization.

    Context encoding network:
    n_active_tokens will be equal to context_length_estimate.
    In this case we slice the hidden input and compute the next token logits only for the last context token.

    Normal token gen network:
    n_active_tokens will be 1.
    No slicing required. Will return the next token logits for the current active token.

    Models: LLaMa.

    logits = (rms_norm(H) @ W) + B
    """
    is_bsh = neuron_config and neuron_config.attention_layout == Layout.BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes
    dtype = hidden.dtype

    if is_prefill:
        hidden = _dynamic_logits_slice(hidden, last_token_id, neuron_config)
        # slice the hidden input and compute the next token logits only for the last context token.
        n_active_tokens = 1

    rms_hidden = (
        hlo.rms_norm(hidden, rms_weight, eps, neuron_config=None)
        if is_bsh
        else hlo.rms_norm(hidden, rms_weight, eps, dim=0, neuron_config=None)
    )

    if is_bsh:
        rms_hidden = hlo.transpose210(rms_hidden)
    rms_hidden = hlo.reshape(rms_hidden, (hidden_size, n_active_tokens * batch_size))
    logits = hlo.dot00(lm_head_weight, rms_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    return hlo.reshape(logits, (vocab_size, n_active_tokens, batch_size))


def _dynamic_logits_slice(hidden, last_token_id, neuron_config=None):
    is_bsh = neuron_config and neuron_config.attention_layout == Layout.BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes
    if neuron_config and neuron_config.lhs_aligned:
        if not is_bsh:
            hidden = hlo.transpose210(hidden)
        hidden = hlo.reshape(hidden, (batch_size * n_active_tokens, hidden_size))

        # [6,3,9] -> [(0,6),(1,3),(2,9)] -> [6+0*128,3+1*128,9+2*128] -> [6,131,265]
        # last_token_id + iota * n_active_tokens
        assert last_token_id.sizes[0] == batch_size, (
            f"vectorized last_token_id length ({last_token_id.sizes[0]}) is expected to equal to batch size ({batch_size})"
        )
        offset = hlo.iota(last_token_id.dtype, last_token_id.sizes, [0])
        offset = hlo.multiply(offset, n_active_tokens)
        last_token_id = hlo.add(last_token_id, offset)
        hidden = hlo.index_select(hidden, dim=0, index=last_token_id)
        hidden = hlo.reshape(hidden, (last_token_id.sizes[0], 1, hidden_size))
        if not is_bsh:
            hidden = hlo.transpose210(hidden)
    else:
        hidden = hlo.transpose102(hidden)
        hidden = hlo.index_select(hidden, dim=0, index=last_token_id)
        hidden = hlo.transpose102(hidden)
    return hidden
