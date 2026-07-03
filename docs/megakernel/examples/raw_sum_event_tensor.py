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
2. build graph metadata, the frontend IR used by the lowering path,
3. derive the task-family dependency plan,
4. plan concrete event resources for static emission,
5. build a deterministic static worker/task distribution,
6. plan hidden runtime resources,
7. emit a mixed TIRx PrimFunc whose public signature is graph inputs/outputs,
8. compile to CUDA source and check synchronization primitives,
9. run the compiled executable and compare against NumPy row sums.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterable

import numpy as np

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
    plan_event_tensor_dependencies,
    plan_static_event_tensor_graph,
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
    def main_graph(A: Tensor, n_tiles: int) -> Tensor:
        row_done = ETensor((n_tiles,), name="row_done")
        partial = call_device(
            RawSumGraph.partial_sum,
            tile_num=(n_tiles, K_PARTS),
            args=[A],
            outputs=Tensor((n_tiles * ROW_TILE, K_PARTS), name="partial"),
            out_edges={row_done: "ij->i"},
            threads=ROW_TILE,
        )
        return call_device(
            RawSumGraph.final_sum,
            tile_num=(n_tiles,),
            args=[partial],
            outputs=Tensor((n_tiles * ROW_TILE,), name="rowsum"),
            in_edges={row_done: "i->i"},
            threads=ROW_TILE,
        )


def trace_graph(n_tiles: int):
    """Trace the graph_func into an EventTensorGraph."""

    return RawSumGraph.main_graph(Tensor((n_tiles * ROW_TILE, N_COLS), name="A"), n_tiles)


def build_static(n_tiles: int = 4, *, num_workers: int | None = None):
    """Build the static Event Tensor megakernel PrimFunc."""

    return lower_event_tensor_graph(
        trace_graph(n_tiles),
        schedule="static",
        num_workers=num_workers,
    )


def _compile(func):
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")


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


def _format_worker_tasks(schedule_plan) -> list[tuple[tuple[int, int], ...]]:
    tasks_by_worker = [[] for _ in range(schedule_plan.num_workers)]
    for task_type, start, end in schedule_plan.task_ranges:
        for global_task_id in range(start, end):
            worker = global_task_id % schedule_plan.num_workers
            linear_task_id = global_task_id - start
            tasks_by_worker[worker].append((task_type, linear_task_id))
    return [tuple(tasks) for tasks in tasks_by_worker]


def _print_script_excerpt(script: str, *, lines: int = 80) -> None:
    excerpt = "\n".join(script.splitlines()[:lines])
    print(textwrap.indent(excerpt, "  "))
    if len(script.splitlines()) > lines:
        print(f"  ... ({len(script.splitlines()) - lines} more lines)")


def _run_and_check(rt_mod, n_tiles: int) -> None:
    dev = tvm.cuda(0)
    if not dev.exist:
        print("  skipped=True")
        print("  reason=CUDA device 0 is not available")
        return

    rng = np.random.default_rng(0)
    shape = (n_tiles * ROW_TILE, N_COLS)
    a_np = rng.standard_normal(shape).astype("float32")
    expected = a_np.sum(axis=1)

    a_tvm = tvm.runtime.tensor(a_np, device=dev)
    rowsum_tvm = tvm.runtime.tensor(np.zeros((n_tiles * ROW_TILE,), "float32"), device=dev)
    rt_mod(a_tvm, rowsum_tvm)

    actual = rowsum_tvm.numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("  skipped=False")
    print(f"  input_shape={shape}")
    print(f"  output_shape={actual.shape}")
    print("  allclose=True rtol=1e-5 atol=1e-5")


def walkthrough_static_lowering(
    n_tiles: int = 4,
    *,
    num_workers: int | None = None,
    show_script: bool = True,
    run: bool = True,
) -> None:
    """Print the static Event Tensor lowering process step by step."""

    _print_header(1, "Trace graph_func")
    graph = trace_graph(n_tiles)
    for i, call in enumerate(graph.calls):
        print(
            f"  call[{i}] name={call.device_func.name} "
            f"tile_num={call.tile_num} outputs={getattr(call.outputs, 'shape', None)}"
        )
        print(f"    in_edges={call.in_edges or '-'}")
        print(f"    out_edges={call.out_edges or '-'}")

    _print_header(2, "Build GraphMetadata frontend IR")
    metadata = analyze_event_tensor_graph(graph)
    print(f"  symbols={metadata.symbols}")
    print(f"  inputs={[input_spec.shape for input_spec in metadata.inputs]}")
    print(f"  outputs={[getattr(output, 'shape', None) for output in metadata.outputs]}")
    for task in metadata.tasks:
        print(
            f"  task_type={task.task_type} name={task.name} "
            f"tile_axes={task.tile_axes} tile_shape={task.tile_shape} "
            f"inline={task.inline_body is not None}"
        )
        print(f"    in_edges={_format_edges(task.in_edges)}")
        print(f"    out_edges={_format_edges(task.out_edges)}")

    _print_header(3, "Derive DependencyPlan")
    dependency_plan = plan_event_tensor_dependencies(metadata)
    print(f"  topo_order={dependency_plan.topo_order}")
    print(f"  family_edges={dependency_plan.family_edges}")

    _print_header(4, "Plan concrete static resources")
    event_plans, schedule_plan, runtime_plan = plan_static_event_tensor_graph(
        metadata, num_workers=num_workers
    )
    print(f"  concrete_n_tiles={n_tiles}")
    for event_plan in event_plans:
        print(f"  event_name={event_plan.event_name}")
        print(f"    shape={event_plan.shape}")
        print(f"    wait_count={event_plan.wait_count}")
        print(f"    backing_buffer={event_plan.backing_buffer}")
        print(f"    init_policy={event_plan.init_policy}")

    _print_header(5, "Build StaticSchedulePlan")
    print(f"  num_workers={schedule_plan.num_workers}")
    if num_workers is None:
        print("  num_workers_source=deterministic example default")
    else:
        print("  num_workers_source=explicit --num-workers")
    print(f"  schedule_policy={schedule_plan.schedule_policy}")
    print(f"  task_ranges={schedule_plan.task_ranges}")
    print(f"  total_tasks={schedule_plan.total_tasks}")
    print("  worker_tasks=(task_type, linear_task_id)")
    for worker, tasks in enumerate(_format_worker_tasks(schedule_plan)):
        print(f"    worker[{worker}] {tasks}")

    _print_header(6, "Build RuntimeResourcePlan")
    print(f"  hidden_intermediates={runtime_plan.hidden_intermediates}")
    print(f"  event_buffers={runtime_plan.event_buffers}")
    print(f"  init_steps={runtime_plan.init_steps}")

    _print_header(7, "Emit mixed static TIRx PrimFunc")
    prim_func = build_static(n_tiles, num_workers=num_workers)
    script = prim_func.script()
    print("  function_name=static_kernel")
    print("  parameters=graph inputs and outputs only")
    print("  hidden_resources=allocated in the host section before T.device_entry")
    if show_script:
        _print_script_excerpt(script)

    _print_header(8, "Compile and inspect CUDA source")
    rt_mod = _compile(prim_func)
    cuda_src = rt_mod.mod.imports[0].inspect_source()
    checks = {
        "release_notify": "atom.release.gpu.global.add.s32" in cuda_src,
        "acquire_wait": "ld.acquire.gpu.global.s32" in cuda_src,
        "runtime_init_barrier": "tvm_global_barrier_state" in cuda_src,
        "single_kernel": "static_kernel_kernel" in cuda_src,
    }
    for name, ok in checks.items():
        print(f"  {name}={ok}")
    print("  CUDA source excerpt:")
    _print_script_excerpt(cuda_src, lines=40)

    _print_header(9, "Run and compare against NumPy")
    if run:
        _run_and_check(rt_mod, n_tiles)
    else:
        print("  skipped=True")
        print("  reason=--no-run")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of persistent CTA workers. Defaults to a deterministic "
            "example value; for real kernels pass a hardware-aware value."
        ),
    )
    parser.add_argument(
        "--no-script",
        action="store_true",
        help="Only print summaries, not the emitted TIRx/CUDA excerpts.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip executable invocation and numerical comparison.",
    )
    args = parser.parse_args()
    walkthrough_static_lowering(
        args.n_tiles,
        num_workers=args.num_workers,
        show_script=not args.no_script,
        run=not args.no_run,
    )


if __name__ == "__main__":
    main()
