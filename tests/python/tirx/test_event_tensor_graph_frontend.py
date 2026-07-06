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

import pytest

import tvm
from tvm.script import tirx as T
from tvm.tirx.lang import (
    ETensor,
    Tensor,
    call_device,
    device_func,
    estimate_static_task_profiler_buffer_size,
    graph_func,
    lower_event_tensor_graph,
    plan_static_event_tensor_graph,
    sym_var,
)
from tvm.tirx.lang.megakernel import (
    _analyze_graph,
    _make_event_plan,
    _make_static_schedule_plan,
    _validate_graph,
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
    metadata = _analyze_graph(fake.trace_graph(2))
    validated = _validate_graph(metadata)
    event_plan = _make_event_plan(metadata.events[0])
    schedule_plan = _make_static_schedule_plan(metadata, validated.dependency_plan, num_workers=2)

    assert event_plan.shape == (2,)
    assert event_plan.wait_count == 4
    assert schedule_plan.schedule_policy == "strided_global_task_id"
    assert schedule_plan.task_ranges == ((0, 0, 8), (1, 8, 10))
    assert schedule_plan.total_tasks == 10


class InlineRawSum:
    @device_func
    def partial_sum(i, j, A, B, tx):
        if tx < 2:
            row = i * 2 + tx
            B[row, j] = A[row, j * 2] + A[row, j * 2 + 1]

    @device_func
    def final_sum(i, B, C, tx):
        if tx < 2:
            row = i * 2 + tx
            C[row] = B[row, 0] + B[row, 1]

    @graph_func
    def main_graph(A: Tensor(("n*2", 4)), n_tiles: int | None = None) -> Tensor(("n*2",)):
        n = n_tiles if n_tiles is not None else sym_var()
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


class ThreeStageGraph:
    @device_func
    def stage0(i, A, B, tx):
        if tx < 1:
            B[i] = A[i] + T.float32(1)

    @device_func
    def stage1(i, B, C, tx):
        if tx < 1:
            C[i] = B[i] + T.float32(1)

    @device_func
    def stage2(i, C, D, tx):
        if tx < 1:
            D[i] = C[i] + T.float32(1)

    @graph_func
    def main_graph(A: Tensor(("n",), name="A"), n_tiles: int | None = None) -> Tensor(("n",)):
        n = n_tiles if n_tiles is not None else sym_var()
        e0 = ETensor((n,), name="e0")
        e1 = ETensor((n,), name="e1")
        B = call_device(
            ThreeStageGraph.stage0,
            tile_num=(n,),
            args=[A],
            outputs=Tensor((n,), name="B"),
            out_edges={e0: "i->i"},
            threads=1,
        )
        C = call_device(
            ThreeStageGraph.stage1,
            tile_num=(n,),
            args=[B],
            outputs=Tensor((n,), name="C"),
            in_edges={e0: "i->i"},
            out_edges={e1: "i->i"},
            threads=1,
        )
        return call_device(
            ThreeStageGraph.stage2,
            tile_num=(n,),
            args=[C],
            outputs=Tensor((n,), name="D"),
            in_edges={e1: "i->i"},
            threads=1,
        )


def test_static_lowering_accepts_inline_task_bodies():
    graph = InlineRawSum.main_graph(Tensor((4, 4)), 2)
    func = lower_event_tensor_graph(
        graph,
        schedule="static",
    )
    script = func.script()

    assert "T.ptx.atom_scalar" in script
    assert "T.ptx.ld_acquire" in script
    assert "for global_task_id in range" in script
    assert "if global_task_id < 4" in script
    assert "if global_task_id >= 4" in script
    assert "T.tvm_global_barrier_kinit" in script
    assert 'T.tvm_storage_sync("global", tx == 0, 4)' in script
    assert "input_0[row, j * 2]" in script
    assert "output_0[row] = intermediate_0_0[row, 0] + intermediate_0_0[row, 1]" in script

    signature = next(line for line in script.splitlines() if line.startswith("def "))
    assert "row_done_buf" not in signature
    assert "intermediate_0_0" not in signature
    assert "schedule_offsets" not in signature
    assert "schedule_tasks" not in signature
    assert 'row_done_buf = T.alloc_buffer((2,), "int32")' in script
    assert "intermediate_0_0 = T.alloc_buffer((4, 2))" in script
    assert "schedule_offsets" not in script
    assert "schedule_tasks" not in script


def test_static_lowering_can_profile_task_bodies():
    graph = InlineRawSum.main_graph(Tensor((4, 4)), 2)
    metadata = _analyze_graph(graph)
    _, schedule_plan, _ = plan_static_event_tensor_graph(metadata)
    profiler_buffer_size = estimate_static_task_profiler_buffer_size(schedule_plan)

    func = lower_event_tensor_graph(
        graph,
        schedule="static",
        profile_tasks=True,
        profiler_buffer_size=profiler_buffer_size,
    )
    script = func.script()

    signature = next(line for line in script.splitlines() if line.startswith("def "))
    assert "prof: T.Buffer" in signature
    assert "T.cuda.timer_init" in script
    assert "T.cuda.timer_start" in script
    assert "T.cuda.timer_end" in script
    assert "T.cuda.timer_finalize" in script
    assert "0, prof.data" in script
    assert "1, prof.data" in script


def test_static_lowering_rejects_dynamic_shape_graph():
    graph = InlineRawSum.main_graph(Tensor(("n*2", 4)))
    with pytest.raises(ValueError, match="requires concrete graph shapes"):
        lower_event_tensor_graph(graph, schedule="static")


def test_static_lowering_handles_generic_three_stage_graph():
    graph = ThreeStageGraph.main_graph(Tensor((2,), name="A"), 2)
    metadata = _analyze_graph(graph)
    event_plans, schedule_plan, _ = plan_static_event_tensor_graph(metadata, num_workers=1)

    assert [event.event_name for event in event_plans] == ["e0", "e1"]
    assert schedule_plan.task_ranges == ((0, 0, 2), (1, 2, 4), (2, 4, 6))
    assert schedule_plan.total_tasks == 6

    func = lower_event_tensor_graph(graph, schedule="static")
    script = func.script()
    assert "if global_task_id < 2" in script
    assert "if global_task_id >= 2 and global_task_id < 4" in script
    assert "if global_task_id >= 4" in script
    assert "e0_buf" in script
    assert "e1_buf" in script

    signature = next(line for line in script.splitlines() if line.startswith("def "))
    assert "e0_buf" not in signature
    assert "e1_buf" not in signature
    assert "schedule_offsets" not in signature
    assert "schedule_tasks" not in signature


def test_fake_example_lowers_static_event_tensor_graph():
    fake = _load_fake_example()

    static_src = _cuda_source(fake.build_static(1))
    assert "atom.release.gpu.global.add.s32" in static_src
    assert "ld.acquire.gpu.global.s32" in static_src
    assert "tvm_global_barrier_state" in static_src

    with pytest.raises(NotImplementedError, match="generic dynamic"):
        fake.build_dynamic(1)


if __name__ == "__main__":
    tvm.testing.main()
