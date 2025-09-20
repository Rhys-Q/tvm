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
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

try:
    import triton
    import triton.language as tl
except ImportError:
    pytestmark = pytest.skip("Triton is not available", allow_module_level=True)


@tvm.testing.requires_cuda
def test_tir_triton_integration():
    @triton.jit
    def add_kernel(
        x_ptr,  # *Pointer* to first input vector.
        y_ptr,  # *Pointer* to second input vector.
        output_ptr,  # *Pointer* to output vector.
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    ):
        """Triton vector add kernel from its tutorial."""
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    @I.ir_module
    class Module:
        @T.prim_func
        def add(x_handle: T.handle, y_handle: T.handle, output_handle: T.handle) -> None:
            T.func_attr({"global_symbol": "add"})
            m = T.int64()
            x = T.match_buffer(x_handle, (m,), "float32")
            y = T.match_buffer(y_handle, (m,), "float32")
            output = T.match_buffer(output_handle, (m,), "float32")
            with T.block("root"):
                T.reads(x[0:m], y[0:m])
                T.writes(output[0:m])
                BLOCK_SIZE = T.meta_var(64)
                T.call_kernel(
                    add_kernel,
                    (T.ceildiv(m, BLOCK_SIZE),),
                    x.data,
                    y.data,
                    output.data,
                    m,
                    BLOCK_SIZE,
                )

        @R.function
        def main(x: R.Tensor(("m",), "float32"), y: R.Tensor(("m",), "float32")):
            m = T.int64()
            with R.dataflow():
                output = R.call_tir(Module.add, [x, y], relax.TensorStructInfo((m,), "float32"))
                R.output(output)
            return output

    @I.ir_module
    class Parsed:
        @T.prim_func
        def add(x_handle: T.handle, y_handle: T.handle, output_handle: T.handle):
            m = T.int64()
            x = T.match_buffer(x_handle, (m,))
            y = T.match_buffer(y_handle, (m,))
            output = T.match_buffer(output_handle, (m,))
            with T.block("root"):
                T.reads(x[0:m], y[0:m])
                T.writes(output[0:m])
                T.call_packed(
                    "add_kernel",
                    x.data,
                    y.data,
                    output.data,
                    m,
                    128,
                    (m + T.int64(64) - T.int64(1)) // T.int64(64),
                )

    tvm.ir.assert_structural_equal(Module["add"], Parsed["add"])
    assert len(Module.get_attr("external_mods")) == 1

    device = tvm.cuda(0)
    x_nd = tvm.nd.array(np.random.rand(256).astype(np.float32), device)
    y_nd = tvm.nd.array(np.random.rand(256).astype(np.float32), device)
    output_np = x_nd.numpy() + y_nd.numpy()

    with tvm.target.Target("cuda"):
        lib = tvm.compile(Module)
        output_nd = tvm.runtime.vm.VirtualMachine(lib, device)["main"](x_nd, y_nd)
        tvm.testing.assert_allclose(output_nd.numpy(), output_np, rtol=1e-5)


@tvm.testing.requires_cuda
def test_tir_triton_searchsorted_integration():
    @triton.jit
    def _searchsorted_kernel(A, V, OUT,
                           N: tl.constexpr, iters: tl.constexpr,
                           strideA_batch, strideA_last,
                           strideV_batch, strideV_last,
                           strideO_batch, strideO_last,
                           right: tl.constexpr):
        """Triton searchsorted kernel."""
        batch_idx = tl.program_id(0)
        val_idx = tl.program_id(1)

        a_base = batch_idx * strideA_batch
        v_ptr = batch_idx * strideV_batch + val_idx * strideV_last
        out_ptr = batch_idx * strideO_batch + val_idx * strideO_last

        v = tl.load(V + v_ptr)

        lo = 0
        hi = N

        for _ in range(iters):
            mid = (lo + hi) // 2
            a_val = tl.load(A + a_base + mid * strideA_last,
                           mask=mid < N,
                           other=float("inf"))

            cond = tl.where(right, a_val <= v, a_val < v)
            lo = tl.where(cond, mid + 1, lo)
            hi = tl.where(cond, hi, mid)

        result = tl.minimum(tl.maximum(lo, 0), N)
        tl.store(OUT + out_ptr, result)

    @I.ir_module
    class Module:
        @T.prim_func
        def searchsorted(
            a_handle: T.handle, v_handle: T.handle, output_handle: T.handle
        ) -> None:
            T.func_attr({"global_symbol": "searchsorted"})
            batch_size = T.int64()
            n = T.int64()
            m = T.int64()
            a = T.match_buffer(a_handle, (batch_size, n), "float32")
            v = T.match_buffer(v_handle, (batch_size, m), "float32")
            output = T.match_buffer(output_handle, (batch_size, m), "int64")
            with T.block("root"):
                T.reads(a[0:batch_size, 0:n], v[0:batch_size, 0:m])
                T.writes(output[0:batch_size, 0:m])
                iters = T.meta_var(10)
                T.call_kernel(
                    _searchsorted_kernel,
                    (batch_size, m),
                    a.data,
                    v.data,
                    output.data,
                    n,
                    iters,
                    n,
                    1,
                    m,
                    1,
                    m,
                    1,
                    False,
                )

        @R.function
        def main(
            a: R.Tensor(("batch_size", "n"), "float32"),
            v: R.Tensor(("batch_size", "m"), "float32"),
        ):
            batch_size = T.int64()
            n = T.int64()
            m = T.int64()
            with R.dataflow():
                output = R.call_tir(
                    Module.searchsorted,
                    [a, v],
                    relax.TensorStructInfo((batch_size, m), "int64"),
                )
                R.output(output)
            return output

    @I.ir_module
    class Parsed:
        @T.prim_func
        def searchsorted(a_handle: T.handle, v_handle: T.handle, output_handle: T.handle):
            batch_size = T.int64()
            n = T.int64()
            m = T.int64()
            a = T.match_buffer(a_handle, (batch_size, n))
            v = T.match_buffer(v_handle, (batch_size, m))
            output = T.match_buffer(output_handle, (batch_size, m))
            with T.block("root"):
                T.reads(a[0:batch_size, 0:n], v[0:batch_size, 0:m])
                T.writes(output[0:batch_size, 0:m])
                T.call_packed(
                    "_searchsorted_kernel",
                    a.data,
                    v.data,
                    output.data,
                    n,
                    10,
                    n,
                    1,
                    m,
                    1,
                    m,
                    1,
                    False,
                    batch_size,
                    m,
                )

    tvm.ir.assert_structural_equal(Module["searchsorted"], Parsed["searchsorted"])
    assert len(Module.get_attr("external_mods")) == 1

    device = tvm.cuda(0)
    batch_size = 2
    n = 10
    m = 5

    # Create sorted array for searchsorted
    a_np = np.sort(np.random.rand(batch_size, n).astype(np.float32), axis=1)
    v_np = np.random.rand(batch_size, m).astype(np.float32)

    a_nd = tvm.nd.array(a_np, device)
    v_nd = tvm.nd.array(v_np, device)

    # Compute expected output using numpy searchsorted
    expected_output = np.searchsorted(a_np, v_np, side='left')

    with tvm.target.Target("cuda"):
        lib = tvm.compile(Module)
        output_nd = tvm.runtime.vm.VirtualMachine(lib, device)["main"](a_nd, v_nd)
        np.testing.assert_array_equal(output_nd.numpy(), expected_output)
