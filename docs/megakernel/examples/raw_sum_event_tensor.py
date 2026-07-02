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
"""Raw-sum Event Tensor megakernel lowering walkthrough.

Run from the TVM repository root:

.. code-block:: bash

   PYTHONPATH=python:.local/python python docs/megakernel/examples/raw_sum_event_tensor.py

The example prints each lowering stage for a static Event Tensor graph:

1. trace ``@graph_func`` into graph calls,
2. analyze graph metadata,
3. infer Event Tensor wait_count and runtime resources,
4. build a static per-CTA task queue,
5. emit a TIRx PrimFunc,
6. compile to CUDA source and check synchronization primitives.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterable

import tvm
from tvm.script import tirx as T
from tvm.tirx.lang import (
    ETensor,
    Tensor,
    analyze_event_tensor_graph,
    call_device,
    device_func,
    graph_func,
    lower_event_tensor_graph,
    plan_static_event_tensor_graph,
    sym_var,
)

ROW_TILE = 32
N_COLS = 128
K_PARTS = 4


class RawSumGraph:
    """Raw row-sum graph with two tile task families."""

    @device_func
    def partial_sum(i, j, A, B, tx):
        """Compute one row tile and one column partition."""

        if tx < ROW_TILE:
            row = i * ROW_TILE + tx
            acc = T.float32(0)
            for c in T.serial(N_COLS // K_PARTS):
                acc = acc + A[row, j * (N_COLS // K_PARTS) + c]
            B[row, j] = acc

    @device_func
    def final_sum(i, B, C, tx):
        """Wait for all partial sums of a row tile, then reduce them."""

        if tx < ROW_TILE:
            row = i * ROW_TILE + tx
            acc = T.float32(0)
            for j in T.serial(K_PARTS):
                acc = acc + B[row, j]
            C[row] = acc

    @graph_func
    def main_graph(A: Tensor(("n*32", N_COLS))) -> Tensor(("n*32",)):
        n = sym_var()
        row_done = ETensor((n,), name="row_done")
        partial = call_device(
            RawSumGraph.partial_sum,
            tile_num=(n, K_PARTS),
            args=[A],
            outputs=Tensor((n * ROW_TILE, K_PARTS), name="partial"),
            out_edges={row_done: "ij->i"},
            threads=ROW_TILE,
        )
        return call_device(
            RawSumGraph.final_sum,
            tile_num=(n,),
            args=[partial],
            outputs=Tensor((n * ROW_TILE,), name="rowsum"),
            in_edges={row_done: "i->i"},
            threads=ROW_TILE,
        )


def trace_graph():
    """Trace the graph_func into an EventTensorGraph."""

    return RawSumGraph.main_graph(Tensor(("n*32", N_COLS), name="A"))


def build_static(n_tiles: int = 4):
    """Build the static Event Tensor megakernel PrimFunc."""

    return lower_event_tensor_graph(
        trace_graph(),
        n_tiles=n_tiles,
        schedule="static",
        row_tile=ROW_TILE,
        n_cols=N_COLS,
    )


def _compile_to_cuda_source(func) -> str:
    mod = tvm.IRModule({"main": func})
    rt_mod = tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")
    return rt_mod.mod.imports[0].inspect_source()


def _print_header(step: int, title: str) -> None:
    print(f"\n[{step}] {title}")
    print("-" * (len(title) + 5))


def _format_edges(edges: Iterable[tuple[object, object]]) -> str:
    items = []
    for event, edge_map in edges:
        name = getattr(event, "name", "")
        src = "".join(edge_map.source_axes)
        dst = "".join(edge_map.target_axes)
        items.append(f"{name or '<event>'}: {src}->{dst}")
    return ", ".join(items) if items else "-"


def _print_script_excerpt(script: str, *, lines: int = 80) -> None:
    excerpt = "\n".join(script.splitlines()[:lines])
    print(textwrap.indent(excerpt, "  "))
    if len(script.splitlines()) > lines:
        print(f"  ... ({len(script.splitlines()) - lines} more lines)")


def walkthrough_static_lowering(n_tiles: int = 4, *, show_script: bool = True) -> None:
    """Print the static Event Tensor lowering process step by step."""

    _print_header(1, "Trace graph_func")
    graph = trace_graph()
    for i, call in enumerate(graph.calls):
        print(
            f"  call[{i}] name={call.device_func.name} "
            f"tile_num={call.tile_num} outputs={getattr(call.outputs, 'shape', None)}"
        )
        print(f"    in_edges={call.in_edges or '-'}")
        print(f"    out_edges={call.out_edges or '-'}")

    _print_header(2, "Analyze GraphMetadata")
    metadata = analyze_event_tensor_graph(graph)
    print(f"  symbols={metadata.symbols}")
    print(f"  inputs={[input_spec.shape for input_spec in metadata.inputs]}")
    print(
        f"  outputs={[getattr(output, 'shape', None) for output in metadata.outputs]}"
    )
    for task in metadata.tasks:
        print(
            f"  task_type={task.task_type} name={task.name} "
            f"tile_axes={task.tile_axes} tile_shape={task.tile_shape} "
            f"inline={task.inline_body is not None}"
        )
        print(f"    in_edges={_format_edges(task.in_edges)}")
        print(f"    out_edges={_format_edges(task.out_edges)}")

    _print_header(3, "Infer EventPlan")
    event_plan, schedule_plan, runtime_plan = plan_static_event_tensor_graph(
        metadata, n_tiles=n_tiles
    )
    print(f"  event_name={event_plan.event_name}")
    print(f"  shape={event_plan.shape}")
    print(f"  wait_count={event_plan.wait_count}")
    print(f"  backing_buffer={event_plan.backing_buffer}")
    print(f"  init_policy={event_plan.init_policy}")

    _print_header(4, "Build StaticSchedulePlan")
    print(f"  num_workers={schedule_plan.num_workers}")
    print(f"  record_layout={schedule_plan.task_record_layout}")
    print(f"  queue_offsets={schedule_plan.queue_offsets}")
    print("  queue_tasks=(task_type, linear_task_id)")
    for worker in range(schedule_plan.num_workers):
        begin = schedule_plan.queue_offsets[worker]
        end = schedule_plan.queue_offsets[worker + 1]
        print(f"    worker[{worker}] {schedule_plan.queue_tasks[begin:end]}")

    _print_header(5, "Build RuntimeResourcePlan")
    print(f"  hidden_intermediates={runtime_plan.hidden_intermediates}")
    print(f"  event_buffers={runtime_plan.event_buffers}")
    print(f"  init_steps={runtime_plan.init_steps}")

    _print_header(6, "Emit static TIRx PrimFunc")
    prim_func = build_static(n_tiles)
    script = prim_func.script()
    print("  function_name=static_kernel")
    print("  parameters=A, Y, E_buf, P")
    if show_script:
        _print_script_excerpt(script)

    _print_header(7, "Compile and inspect CUDA source")
    cuda_src = _compile_to_cuda_source(prim_func)
    checks = {
        "release_notify": "atom.release.gpu.global.add.s32" in cuda_src,
        "acquire_wait": "ld.acquire.gpu.global.s32" in cuda_src,
        "single_kernel": "static_kernel_kernel" in cuda_src,
    }
    for name, ok in checks.items():
        print(f"  {name}={ok}")
    print("  CUDA source excerpt:")
    _print_script_excerpt(cuda_src, lines=40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=4)
    parser.add_argument(
        "--no-script",
        action="store_true",
        help="Only print summaries, not the emitted TIRx/CUDA excerpts.",
    )
    args = parser.parse_args()
    walkthrough_static_lowering(args.n_tiles, show_script=not args.no_script)


if __name__ == "__main__":
    main()
