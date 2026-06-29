# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0.
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runnable raw-sum Event Tensor graph example.

This file intentionally mirrors ``docs/megakernel/tasks/fake_example.py``:
the example describes the paper's raw-sum task graph with ``graph_func``,
``call_device`` and ``ETensor``.  Static/dynamic megakernel details are owned by
``lower_event_tensor_graph``.
"""

import numpy as np

import tvm
from tvm.tirx.lang import (
    ETensor,
    Tensor,
    call_device,
    device_func,
    graph_func,
    lower_event_tensor_graph,
    sym_var,
)

ROW_TILE = 32
N_COLS = 128
K_PARTS = 4


class IRModule:
    @device_func
    def partial_sum(i: int, j: int, A: Tensor, B: Tensor):
        B[i * ROW_TILE : i * ROW_TILE + ROW_TILE, j] = sum(
            A[
                i * ROW_TILE : i * ROW_TILE + ROW_TILE,
                j * (N_COLS // K_PARTS) : j * (N_COLS // K_PARTS) + (N_COLS // K_PARTS),
            ]
        )

    @device_func
    def final_sum(i: int, B: Tensor, C: Tensor):
        C[i * ROW_TILE : i * ROW_TILE + ROW_TILE] = sum(
            B[i * ROW_TILE : i * ROW_TILE + ROW_TILE, :]
        )

    @graph_func
    def main_graph(A: Tensor(("n*32", N_COLS))) -> Tensor(("n*32",)):
        n = sym_var()
        E = ETensor((n,), wait_count=K_PARTS)
        B: Tensor((n * ROW_TILE, K_PARTS)) = call_device(
            IRModule.partial_sum,
            tile_num=(n, K_PARTS),
            args=[A],
            in_edges={},
            out_edges={E: "ij->i"},
        )
        C: Tensor((n * ROW_TILE,)) = call_device(
            IRModule.final_sum,
            tile_num=(n,),
            args=[B],
            in_edges={E: "i->i"},
            out_edges={},
        )
        return C


def trace_graph():
    return IRModule.main_graph(Tensor(("n*32", N_COLS)))


def build_static(n_tiles: int = 4):
    breakpoint()
    return lower_event_tensor_graph(
        trace_graph(),
        n_tiles=n_tiles,
        schedule="static",
        row_tile=ROW_TILE,
        n_cols=N_COLS,
    )


def build_dynamic(n_tiles: int = 4, *, early_push: bool = False):
    return lower_event_tensor_graph(
        trace_graph(),
        n_tiles=n_tiles,
        schedule="dynamic",
        row_tile=ROW_TILE,
        n_cols=N_COLS,
        early_push=early_push,
    )


def _compile(func):
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")


def _make_dynamic_queue(dev, n_tiles: int, *, early_push: bool):
    partial = 0
    final = 1
    capacity = n_tiles * (K_PARTS + 1)
    queue = np.zeros((capacity, 3), dtype="int32")
    tail = 0
    for tile in range(n_tiles):
        for part in range(K_PARTS):
            queue[tail] = (partial, tile, part)
            tail += 1
        if early_push:
            queue[tail] = (final, tile, 0)
            tail += 1
    return (
        tvm.runtime.tensor(queue, dev),
        tvm.runtime.tensor(np.array([0], dtype="int32"), dev),
        tvm.runtime.tensor(np.array([tail], dtype="int32"), dev),
        tvm.runtime.tensor(np.array([tail], dtype="int32"), dev),
        tvm.runtime.tensor(np.array([0], dtype="int32"), dev),
    )


def _run(func, dev, data, ref, *, early_push: bool = False):
    n_tiles = data.shape[0] // ROW_TILE
    rt_mod = _compile(func)
    a_dev = tvm.runtime.tensor(data, dev)
    y_dev = tvm.runtime.empty((data.shape[0],), dtype="float32", device=dev)
    e_dev = tvm.runtime.tensor(np.full((n_tiles,), K_PARTS, dtype="int32"), dev)
    p_dev = tvm.runtime.empty((data.shape[0], K_PARTS), dtype="float32", device=dev)
    if "dynamic" in func.__name__:
        queue_args = _make_dynamic_queue(dev, n_tiles, early_push=early_push)
        rt_mod(a_dev, y_dev, e_dev, p_dev, *queue_args)
    else:
        rt_mod(a_dev, y_dev, e_dev, p_dev)
    got = y_dev.numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
    print(f"{func.__name__}: max_diff={np.max(np.abs(got - ref)):.6f}")


def main():
    dev = tvm.device("cuda", 0)
    if not dev.exist:
        raise RuntimeError("CUDA device is required to run this example")

    n_tiles = 4
    rows = n_tiles * ROW_TILE
    data = np.linspace(0, 1, rows * N_COLS, dtype="float32").reshape(rows, N_COLS)
    ref = np.sum(data, axis=1)
    _run(build_static(n_tiles), dev, data, ref)
    _run(build_dynamic(n_tiles), dev, data, ref)
    _run(build_dynamic(n_tiles, early_push=True), dev, data, ref, early_push=True)


if __name__ == "__main__":
    main()
