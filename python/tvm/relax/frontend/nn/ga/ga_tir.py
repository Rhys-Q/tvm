# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments,invalid-name
import enum
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import tvm
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.runtime import DataType
from tvm.script import tir as T
from tvm.target import Target

from .position_embedding import llama_rope_with_position_map, switch_rope_freq_func
from .tree_attn import (
    tree_attn,
    tree_attn_cpu,
    tree_attn_with_paged_kv_cache,
    tree_attn_with_paged_kv_cache_cpu,
)


def _var_cpu(dtype):
    return T.alloc_buffer((1,), dtype)


def get_max_num_threads_per_block(target: Target) -> int:
    """
    max(max_num_threads, max_threads_per_block); if latter does not exist, return max_num_threads.
    We add this method since some targets have both fields and `max_threads_per_block` is larger.
    """
    max_num_threads = target.max_num_threads
    max_threads_per_block = target.attrs.get("max_threads_per_block", None)
    if max_threads_per_block is None:
        return max_num_threads
    return max(max_num_threads, max_threads_per_block)


def check_thread_limits(target: Target, bdx: int, bdy: int, bdz: int, gdz: int):
    """
    Check whether max num threads exceeded given a target.

    Parameters
    ----------
    bdx: threadIdx.x
    bdy: threadIdx.y
    bdz: threadIdx.z
    gdz: blockIdx.z
    """
    max_num_threads_per_block = get_max_num_threads_per_block(target)

    assert (
        bdx * bdy * bdz <= max_num_threads_per_block
    ), f"{target.kind} max num threads exceeded: {bdx}*{bdy}*{bdz}>{max_num_threads_per_block}"

    if str(target.kind) == "webgpu":
        # https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxcomputeworkgroupsizez
        assert bdz <= 64, f"webgpu's threadIdx.z cannot exceed 64, but got bdz={bdz}"
        assert gdz == 1, f"webgpu's blockIdx.z should be 1, but got gdz={gdz}"


class AttnKind(enum.IntEnum):
    """The attention kind class.
    MHA denotes multi-head attention, multi-query attention or grouped query attention.
    MLA denotes multi-head latent attention.
    """

    MHA = 0
    MLA = 1


class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


def _ga_init_pop():
    """Return the TIR function that init the pop."""

    # pylint: disable=line-too-long
    # fmt: off
    @T.prim_func
    def tir_init_pop(
        var_pop: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        pop_size = T.int64()
        gen_len = T.int64()
        pop = T.match_buffer(var_pop, (pop_size, gen_len), "uint8")
        for pop_index, gen_index in T.grid(pop_size, gen_len):
            with T.block("init_pop"):
                T.reads()
                T.writes(pop[pop_index, gen_index])
                pop[pop_index, gen_index] =  tir.if_then_else(
                        tir.random("float32") > 0.5,  # 以 0.5 概率取 1
                        tir.const(1, "uint32"),
                        tir.const(0, "uint32")
                    )
    # fmt: on
    # pylint: enable=line-too-long

    return tir_init_pop

