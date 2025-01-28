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
import enum

# Size used to determine fused QKV operation.
FUSED_QKV_TP_FACTOR = 3

# KV sharding pad for flash decoding
KV_SHARD_PAD = 128

# Layout for attention
LAYOUT_BSH = "BSH"
LAYOUT_HSB = "HSB"
LAYOUT_SBH = "SBH"


class Layout(enum.Enum):
    HSB = "HSB"
    BSH = "BSH"
    SBH = "SBH"

    def __eq__(self, value):
        return super().__eq__(Layout(value))


# Group query attention sharding configurations
class GQA(enum.Enum):
    # [Default] Sharding over the heads splits entire (complete) K/V heads
    # onto the NeuronCores where the corresponding Q heads reside. This is
    # similar to traditional MHA except that the Q and K/V heads do not need
    # to be equal.
    #
    # This cannot be enabled when number of K/V heads cannot be evenly split
    # across the NeuronCores according to the tensor parallelism degree.
    SHARD_OVER_HEADS = "shard-over-heads"

    # This transforms a GQA attention mechanism into a traditional MHA mechanism
    # by replicating the K/V heads to evenly match the corresponding Q heads.
    # This consumes more memory than would otherwise be used with other sharding
    # mechanisms but avoids collective communications overheads.
    REPLICATED_HEADS = "replicated-heads"
