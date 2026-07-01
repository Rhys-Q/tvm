# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0.
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for the lightweight Event Tensor graph frontend."""

import importlib.util
from pathlib import Path

import tvm
from tvm.script import tirx as T
from tvm.tirx.lang import (
    ETensor,
    Tensor,
    call_device,
    device_func,
    graph_func,
    lower_event_tensor_graph,
    sym_var,
)
from tvm.tirx.lang.megakernel import (
    _analyze_graph,
    _make_event_plan,
    _make_static_schedule_plan,
    _validate_static_raw_sum,
)


def _load_fake_example():
    path = Path("docs/megakernel/tasks/fake_example.py")
    spec = importlib.util.spec_from_file_location("fake_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cuda_source(func) -> str:
    mod = tvm.compile(
        tvm.IRModule({"main": func}),
        target=tvm.target.Target("cuda"),
        tir_pipeline="tirx",
    )
    return mod.mod.imports[0].inspect_source()


def test_fake_example_traces_event_tensor_graph():
    fake = _load_fake_example()
    graph = fake.trace_graph()
    assert [call.device_func.name for call in graph.calls] == [
        "partial_sum",
        "final_sum",
    ]
    assert next(iter(graph.calls[0].out_edges.values())) == "ij->i"
    assert next(iter(graph.calls[1].in_edges.values())) == "i->i"


def test_static_metadata_infers_wait_count_and_schedule():
    fake = _load_fake_example()
    metadata = _analyze_graph(fake.trace_graph())
    event = _validate_static_raw_sum(metadata)
    event_plan = _make_event_plan(event, {"n": 2})
    schedule_plan = _make_static_schedule_plan(metadata, {"n": 2}, num_workers=2)

    assert event_plan.shape == (2,)
    assert event_plan.wait_count == 4
    assert schedule_plan.task_record_layout == ("task_type", "linear_task_id")
    assert schedule_plan.queue_offsets == (0, 5, 10)
    assert schedule_plan.queue_tasks[0] == (0, 0)
    assert schedule_plan.queue_tasks[2] == (1, 0)
    assert schedule_plan.queue_tasks[-1] == (1, 1)


class InlineRawSum:
    @device_func
    @T.inline
    def partial_sum(i, j, A, B, tx):
        if tx < 2:
            row = i * 2 + tx
            B[row, j] = A[row, j * 2] + A[row, j * 2 + 1]

    @device_func
    @T.inline
    def final_sum(i, B, C, tx):
        if tx < 2:
            row = i * 2 + tx
            C[row] = B[row, 0] + B[row, 1]

    @graph_func
    def main_graph(A: Tensor(("n*2", 4))) -> Tensor(("n*2",)):
        n = sym_var()
        E = ETensor((n,), name="row_done")
        B = call_device(
            InlineRawSum.partial_sum,
            tile_num=(n, 2),
            args=[A],
            outputs=Tensor((n * 2, 2)),
            out_edges={E: "ij->i"},
        )
        return call_device(
            InlineRawSum.final_sum,
            tile_num=(n,),
            args=[B],
            outputs=Tensor((n * 2,)),
            in_edges={E: "i->i"},
        )


def test_static_lowering_accepts_inline_task_bodies():
    graph = InlineRawSum.main_graph(Tensor(("n*2", 4)))
    func = lower_event_tensor_graph(
        graph,
        n_tiles=2,
        schedule="static",
        row_tile=2,
        n_cols=4,
    )
    script = func.script()

    assert "T.ptx.atom_scalar" in script
    assert "T.ptx.ld_acquire" in script
    assert "A[row, phase * 2]" in script
    assert "Y[row] = P[row, 0] + P[row, 1]" in script


def test_fake_example_lowers_static_and_dynamic_event_tensor_graph():
    fake = _load_fake_example()

    static_src = _cuda_source(fake.build_static(1))
    assert "atom.release.gpu.global.add.s32" in static_src
    assert "ld.acquire.gpu.global.s32" in static_src

    dynamic_src = _cuda_source(fake.build_dynamic(1))
    assert "atom.release.gpu.global.add.s32" in dynamic_src
    assert "atomicCAS" in dynamic_src

    early_src = _cuda_source(fake.build_dynamic(1, early_push=True))
    assert "ld.acquire.gpu.global.s32" in early_src


if __name__ == "__main__":
    tvm.testing.main()
