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
"""Lightweight Event Tensor megakernel graph frontend.

This module is intentionally small.  It records graph-level Event Tensor
constructs and lowers supported tiled graphs to TIRx PrimFuncs.  It is not a
general Relax/TIR graph compiler.
"""

from __future__ import annotations

import ast
import inspect
import keyword
import linecache
import operator
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from tvm.script import tirx as T


@dataclass(frozen=True)
class Symbol:
    """Symbolic integer used in graph shapes."""

    expr: str

    def __mul__(self, other):
        return Symbol(f"{self.expr}*{other}")

    def __rmul__(self, other):
        return Symbol(f"{other}*{self.expr}")

    def __add__(self, other):
        return Symbol(f"{self.expr}+{other}")

    def __radd__(self, other):
        return Symbol(f"{other}+{self.expr}")

    def __str__(self):
        return self.expr


@dataclass(frozen=True)
class Tensor:
    """Graph-level tensor placeholder."""

    shape: Any
    dtype: str = "float32"
    name: str = ""

    def __getitem__(self, _index):
        return self

    def __setitem__(self, _index, _value):
        return None


@dataclass(frozen=True)
class ETensor:
    """Graph-level Event Tensor placeholder."""

    shape: Any
    wait_count: Any = None
    name: str = ""
    dtype: str = "int32"


@dataclass(frozen=True)
class DeviceFunction:
    """Recorded tile task function."""

    fn: Callable

    @property
    def name(self) -> str:
        return getattr(self.fn, "__name__", "device_func")

    @property
    def is_inline(self) -> bool:
        return bool(getattr(self.fn, "__tvm_tirx_inline__", False))

    def __get__(self, _obj, _objtype=None):
        return self


@dataclass
class CallDevice:
    """A graph-level call_device node."""

    device_func: DeviceFunction
    tile_num: Any
    args: list[Any]
    outputs: Any = None
    in_edges: dict[ETensor, str] = field(default_factory=dict)
    out_edges: dict[ETensor, str] = field(default_factory=dict)
    name: str | None = None
    threads: int | None = None


@dataclass
class EventTensorGraph:
    """Recorded graph_func body."""

    calls: list[CallDevice]
    output: Any
    inputs: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class EdgeMap:
    """Affine axis projection used by Event Tensor edges."""

    source_axes: tuple[str, ...]
    target_axes: tuple[str, ...]

    @staticmethod
    def parse(spec: str) -> EdgeMap:
        text = spec.replace(" ", "")
        if "->" not in text:
            raise ValueError(f"Event edge map must contain '->', got {spec!r}")
        lhs, rhs = text.split("->", 1)
        if not lhs or not rhs:
            raise ValueError(f"Event edge map cannot be empty, got {spec!r}")
        source = _parse_axis_list(lhs)
        target = _parse_axis_list(rhs)
        missing = [axis for axis in target if axis not in source]
        if missing:
            raise ValueError(
                f"Event edge map target axes {missing} are not in source {source}"
            )
        return EdgeMap(tuple(source), tuple(target))

    def projected_extents(self, tile_shape: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(tile_shape) != len(self.source_axes):
            raise ValueError(
                "Event edge map rank does not match task tile rank: "
                f"{self.source_axes} vs {tile_shape}"
            )
        extent_by_axis = dict(zip(self.source_axes, tile_shape))
        return tuple(extent_by_axis[axis] for axis in self.target_axes)

    def uniform_wait_count(self, tile_shape: tuple[Any, ...]) -> Any:
        if len(tile_shape) != len(self.source_axes):
            raise ValueError(
                "Event edge map rank does not match task tile rank: "
                f"{self.source_axes} vs {tile_shape}"
            )
        product = 1
        for axis, extent in zip(self.source_axes, tile_shape):
            if axis not in self.target_axes:
                product = _mul_dim(product, extent)
        return product


@dataclass(frozen=True)
class TensorSpec:
    """Graph-visible tensor shape and dtype."""

    shape: Any
    dtype: str = "float32"
    name: str = ""


@dataclass(frozen=True)
class EventSpec:
    """Event Tensor metadata discovered during graph analysis."""

    event: ETensor
    shape: tuple[Any, ...]
    wait_count: Any
    name: str
    dtype: str = "int32"


@dataclass(frozen=True)
class TaskSpec:
    """Task family metadata used by megakernel lowering."""

    name: str
    task_type: int
    inline_body: Callable | None
    tile_axes: tuple[str, ...]
    tile_shape: tuple[Any, ...]
    inputs: tuple[Any, ...]
    outputs: Any
    in_edges: tuple[tuple[ETensor, EdgeMap], ...]
    out_edges: tuple[tuple[ETensor, EdgeMap], ...]
    threads: int | None


@dataclass(frozen=True)
class GraphMetadata:
    """Lowering metadata for a traced Event Tensor graph."""

    symbols: tuple[str, ...]
    inputs: tuple[TensorSpec, ...]
    input_values: tuple[Tensor, ...]
    outputs: tuple[Any, ...]
    tasks: tuple[TaskSpec, ...]
    events: tuple[EventSpec, ...]


@dataclass(frozen=True)
class DependencyPlan:
    """Task-family dependency graph used by generic megakernel lowering."""

    topo_order: tuple[int, ...]
    family_edges: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class ValidatedGraphMetadata:
    """Graph metadata after structural validation."""

    metadata: GraphMetadata
    dependency_plan: DependencyPlan


@dataclass(frozen=True)
class EventPlan:
    """Runtime Event Tensor resource plan."""

    event_name: str
    shape: tuple[int, ...]
    wait_count: int
    backing_buffer: str
    init_policy: str = "device"


@dataclass(frozen=True)
class StaticSchedulePlan:
    """Static per-CTA task queue plan."""

    num_workers: int
    queue_offsets: tuple[int, ...]
    queue_tasks: tuple[tuple[int, int], ...]
    task_record_layout: tuple[str, str] = ("task_type", "linear_task_id")
    queue_policy: str = "round_robin_topological"


@dataclass(frozen=True)
class RuntimeResourcePlan:
    """Hidden resources needed by the generated megakernel."""

    hidden_intermediates: tuple[TensorSpec, ...]
    event_buffers: tuple[EventPlan, ...]
    static_schedule_buffers: tuple[StaticSchedulePlan, ...]
    init_steps: tuple[str, ...]


def _parse_axis_list(text: str) -> list[str]:
    axes = list(text)
    if not all(axis.isalpha() and axis.islower() for axis in axes):
        raise ValueError(
            f"Only one-letter lowercase edge-map axes are supported, got {text!r}"
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"Event edge map axes must be unique, got {text!r}")
    return axes


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _tensor_list(value: Any) -> tuple[Tensor, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, tuple | list) else (value,)
    return tuple(item for item in values if isinstance(item, Tensor))


def _mul_dim(lhs: Any, rhs: Any) -> Any:
    if lhs == 1:
        return rhs
    if rhs == 1:
        return lhs
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs * rhs
    return Symbol(f"{lhs}*{rhs}")


def _add_dim(lhs: Any, rhs: Any) -> Any:
    if lhs == 0:
        return rhs
    if rhs == 0:
        return lhs
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs + rhs
    return Symbol(f"{lhs}+{rhs}")


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
}


def _eval_dim(value: Any, symbols: dict[str, int]) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, Symbol):
        value = value.expr
    if isinstance(value, str):
        node = ast.parse(value, mode="eval").body
        return int(_eval_dim_ast(node, symbols))
    raise TypeError(f"Cannot resolve symbolic dimension {value!r}")


def _eval_dim_ast(node: ast.AST, symbols: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in symbols:
        return int(symbols[node.id])
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported symbolic shape operator {ast.dump(node.op)}")
        return int(
            op(_eval_dim_ast(node.left, symbols), _eval_dim_ast(node.right, symbols))
        )
    raise ValueError(f"Unsupported symbolic shape expression {ast.dump(node)}")


def _resolve_shape(shape: Iterable[Any], symbols: dict[str, int]) -> tuple[int, ...]:
    return tuple(_eval_dim(dim, symbols) for dim in shape)


def _linear_task_count(shape: Iterable[int]) -> int:
    total = 1
    for extent in shape:
        total *= extent
    return total


def _shape_numel(shape: Iterable[int]) -> int:
    return _linear_task_count(shape)


def _unflatten_coords(linear: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords: list[int] = []
    for extent in reversed(shape):
        coords.append(linear % extent)
        linear //= extent
    return tuple(reversed(coords))


class _TraceContext:
    current: list[CallDevice] | None = None


class GraphFunction:
    """Callable graph function wrapper."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def __get__(self, _obj, _objtype=None):
        return self

    def trace(self, *args, **kwargs) -> EventTensorGraph:
        old = _TraceContext.current
        _TraceContext.current = []
        try:
            output = self.fn(*args, **kwargs)
            inputs = list(args) + list(kwargs.values())
            return EventTensorGraph(
                calls=_TraceContext.current, output=output, inputs=inputs
            )
        finally:
            _TraceContext.current = old

    def __call__(self, *args, **kwargs) -> EventTensorGraph:
        return self.trace(*args, **kwargs)


def device_func(fn: Callable) -> DeviceFunction:
    """Decorator marking a tile task function.

    The task body is automatically wrapped as a TIRx inline function so the
    megakernel emitter can expand it inside the task dispatch branch.
    """

    if not getattr(fn, "__tvm_tirx_inline__", False):
        fn = T.inline(fn)
    return DeviceFunction(fn)


def graph_func(fn: Callable) -> GraphFunction:
    """Decorator marking a graph function."""

    return GraphFunction(fn)


def sym_var(name: str = "n") -> Symbol:
    """Create a symbolic integer placeholder."""

    return Symbol(name)


def call_device(
    device_fn: DeviceFunction,
    *,
    tile_num,
    args,
    outputs=None,
    in_edges=None,
    in_edge=None,
    out_edges=None,
    name: str | None = None,
    threads: int | None = None,
) -> Tensor:
    """Record a device function launch in the active graph trace."""

    if not isinstance(device_fn, DeviceFunction):
        raise TypeError("call_device expects a @device_func-decorated function")
    in_edges = in_edges if in_edges is not None else in_edge
    result = outputs if outputs is not None else Tensor(shape=None)
    node = CallDevice(
        device_func=device_fn,
        tile_num=tile_num,
        args=list(args),
        outputs=result,
        in_edges=dict(in_edges or {}),
        out_edges=dict(out_edges or {}),
        name=name,
        threads=threads,
    )
    if _TraceContext.current is not None:
        _TraceContext.current.append(node)
    return result


def _analyze_graph(graph: EventTensorGraph) -> GraphMetadata:
    """Build lowering metadata from a traced graph_func."""

    task_specs: list[TaskSpec] = []
    inferred_event_wait_counts: dict[ETensor, Any] = {}
    explicit_event_wait_counts: dict[ETensor, Any] = {}
    symbols: set[str] = set()

    for task_type, call in enumerate(graph.calls):
        tile_shape = _as_tuple(call.tile_num)
        for dim in tile_shape:
            if isinstance(dim, Symbol):
                symbols.add(dim.expr)
        parsed_in_edges = tuple(
            (event, EdgeMap.parse(spec)) for event, spec in call.in_edges.items()
        )
        parsed_out_edges = tuple(
            (event, EdgeMap.parse(spec)) for event, spec in call.out_edges.items()
        )
        tile_axes = (
            parsed_out_edges[0][1].source_axes
            if parsed_out_edges
            else parsed_in_edges[0][1].source_axes
            if parsed_in_edges
            else tuple(chr(ord("i") + i) for i in range(len(tile_shape)))
        )
        for event, edge_map in parsed_out_edges:
            inferred = edge_map.uniform_wait_count(tile_shape)
            previous = inferred_event_wait_counts.get(event, 0)
            inferred_event_wait_counts[event] = _add_dim(previous, inferred)
            if event.wait_count is not None:
                explicit_event_wait_counts[event] = event.wait_count
        for event, _ in parsed_in_edges:
            if event.wait_count is not None:
                explicit_event_wait_counts.setdefault(event, event.wait_count)

        task_specs.append(
            TaskSpec(
                name=call.name or call.device_func.name,
                task_type=task_type,
                inline_body=call.device_func.fn if call.device_func.is_inline else None,
                tile_axes=tuple(tile_axes),
                tile_shape=tuple(tile_shape),
                inputs=tuple(call.args),
                outputs=call.outputs,
                in_edges=parsed_in_edges,
                out_edges=parsed_out_edges,
                threads=call.threads,
            )
        )

    events: list[EventSpec] = []
    seen_events: set[ETensor] = set()
    for task in task_specs:
        for event, _ in (*task.in_edges, *task.out_edges):
            if event in seen_events:
                continue
            seen_events.add(event)
            if event.dtype != "int32":
                raise ValueError("Event Tensor dtype must be int32")
            wait_count = explicit_event_wait_counts.get(
                event, inferred_event_wait_counts.get(event)
            )
            if wait_count is None:
                raise ValueError(
                    f"Cannot infer wait_count for Event Tensor {event.name or '<unnamed>'}"
                )
            events.append(
                EventSpec(
                    event=event,
                    shape=_as_tuple(event.shape),
                    wait_count=wait_count,
                    name=event.name or f"event_{len(events)}",
                    dtype=event.dtype,
                )
            )

    inputs = tuple(
        TensorSpec(arg.shape, arg.dtype, arg.name)
        for arg in graph.inputs
        if isinstance(arg, Tensor)
    )
    outputs = graph.output if isinstance(graph.output, tuple) else (graph.output,)
    return GraphMetadata(
        symbols=tuple(sorted(symbols)),
        inputs=inputs,
        input_values=tuple(arg for arg in graph.inputs if isinstance(arg, Tensor)),
        outputs=tuple(outputs),
        tasks=tuple(task_specs),
        events=tuple(events),
    )


def _validate_graph(metadata: GraphMetadata) -> ValidatedGraphMetadata:
    """Validate a generic Event Tensor graph and build task-family dependencies."""

    if not metadata.tasks:
        raise ValueError("Event Tensor graph must contain at least one call_device node")

    event_rank = {event.event: len(event.shape) for event in metadata.events}
    event_name = {event.event: event.name for event in metadata.events}
    producers: dict[ETensor, list[int]] = {}
    consumers: dict[ETensor, list[int]] = {}

    for task in metadata.tasks:
        if task.inline_body is None:
            raise ValueError(f"{task.name} must be a @device_func inline task body")
        if len(task.tile_shape) != len(task.tile_axes):
            raise ValueError(
                f"{task.name} tile rank does not match tile axes: "
                f"{task.tile_shape} vs {task.tile_axes}"
            )
        for event, edge_map in task.out_edges:
            if len(edge_map.target_axes) != event_rank[event]:
                raise ValueError(
                    f"{task.name} out edge to {event_name[event]} has rank "
                    f"{len(edge_map.target_axes)}, expected {event_rank[event]}"
                )
            producers.setdefault(event, []).append(task.task_type)
        for event, edge_map in task.in_edges:
            if len(edge_map.target_axes) != event_rank[event]:
                raise ValueError(
                    f"{task.name} in edge to {event_name[event]} has rank "
                    f"{len(edge_map.target_axes)}, expected {event_rank[event]}"
                )
            consumers.setdefault(event, []).append(task.task_type)

    edges: set[tuple[int, int, str]] = set()
    for event in event_rank:
        if event not in producers:
            raise ValueError(
                f"Event Tensor {event_name[event]} must have at least one producer edge"
            )
        for producer in producers.get(event, []):
            for consumer in consumers.get(event, []):
                if producer != consumer:
                    edges.add((producer, consumer, event_name[event]))

    topo_order = _topological_task_order(len(metadata.tasks), tuple(edges))
    return ValidatedGraphMetadata(
        metadata=metadata,
        dependency_plan=DependencyPlan(
            topo_order=topo_order,
            family_edges=tuple(sorted(edges)),
        ),
    )


def _topological_task_order(
    num_tasks: int, edges: tuple[tuple[int, int, str], ...]
) -> tuple[int, ...]:
    outgoing: dict[int, list[int]] = {task_type: [] for task_type in range(num_tasks)}
    indegree = [0] * num_tasks
    for src, dst, _ in edges:
        outgoing[src].append(dst)
        indegree[dst] += 1

    ready = [task_type for task_type, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        task_type = ready.pop(0)
        order.append(task_type)
        for dst in sorted(outgoing[task_type]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
                ready.sort()

    if len(order) != num_tasks:
        raise ValueError("Event Tensor task family DAG must be acyclic")
    return tuple(order)


def _make_event_plan(event: EventSpec, symbols: dict[str, int]) -> EventPlan:
    return EventPlan(
        event_name=event.name,
        shape=_resolve_shape(event.shape, symbols),
        wait_count=_eval_dim(event.wait_count, symbols),
        backing_buffer=f"{event.name}_buf",
    )


def _make_event_plans(
    metadata: GraphMetadata, symbols: dict[str, int]
) -> tuple[EventPlan, ...]:
    return tuple(_make_event_plan(event, symbols) for event in metadata.events)


def _make_static_schedule_plan(
    metadata: GraphMetadata,
    dependency_plan: DependencyPlan,
    symbols: dict[str, int],
    num_workers: int,
) -> StaticSchedulePlan:
    queue_tasks_by_worker: list[list[tuple[int, int]]] = [
        [] for _ in range(num_workers)
    ]
    records: list[tuple[int, int]] = []
    task_by_type = {task.task_type: task for task in metadata.tasks}
    for task_type in dependency_plan.topo_order:
        task = task_by_type[task_type]
        shape = _resolve_shape(task.tile_shape, symbols)
        for linear in range(_linear_task_count(shape)):
            records.append((task.task_type, linear))

    cursor = 0
    for record in records:
        queue_tasks_by_worker[cursor % num_workers].append(record)
        cursor += 1

    queue_offsets = [0]
    queue_tasks: list[tuple[int, int]] = []
    for tasks in queue_tasks_by_worker:
        queue_tasks.extend(tasks)
        queue_offsets.append(len(queue_tasks))
    return StaticSchedulePlan(
        num_workers=num_workers,
        queue_offsets=tuple(queue_offsets),
        queue_tasks=tuple(queue_tasks),
    )


def _make_runtime_plan(
    metadata: GraphMetadata,
    event_plans: tuple[EventPlan, ...],
    schedule_plan: StaticSchedulePlan,
) -> RuntimeResourcePlan:
    hidden_outputs = []
    public_outputs = {
        id(output) for output in metadata.outputs if isinstance(output, Tensor)
    }
    seen_hidden: set[int] = set()
    for task in metadata.tasks:
        for output in _tensor_list(task.outputs):
            if id(output) in public_outputs or id(output) in seen_hidden:
                continue
            seen_hidden.add(id(output))
            hidden_outputs.append(
                TensorSpec(output.shape, output.dtype, output.name or "intermediate")
            )
    return RuntimeResourcePlan(
        hidden_intermediates=tuple(hidden_outputs),
        event_buffers=event_plans,
        static_schedule_buffers=(),
        init_steps=tuple(
            f"device fill {event_plan.backing_buffer} with {event_plan.wait_count}"
            for event_plan in event_plans
        ),
    )


def analyze_event_tensor_graph(graph: EventTensorGraph) -> GraphMetadata:
    """Analyze a traced graph_func into Event Tensor lowering metadata."""

    return _analyze_graph(graph)


def _resolve_symbol_values(
    metadata: GraphMetadata,
    *,
    symbol_values: dict[str, int] | None = None,
    n_tiles: int | None = None,
) -> dict[str, int]:
    values = dict(symbol_values or {})
    if n_tiles is not None:
        values.setdefault("n", int(n_tiles))
    missing = [symbol for symbol in metadata.symbols if symbol not in values]
    if missing:
        raise ValueError(f"Missing values for symbolic shape variables: {missing}")
    return values


def plan_static_event_tensor_graph(
    metadata: GraphMetadata,
    *,
    n_tiles: int | None = None,
    symbol_values: dict[str, int] | None = None,
    num_workers: int | None = None,
) -> tuple[tuple[EventPlan, ...], StaticSchedulePlan, RuntimeResourcePlan]:
    """Build the static Event Tensor resource and schedule plans.

    This helper exposes the same planning path used by
    :func:`lower_event_tensor_graph`, so examples and tests can inspect each
    lowering stage without reimplementing planner details.
    """

    validated = _validate_graph(metadata)
    symbols = _resolve_symbol_values(
        metadata, symbol_values=symbol_values, n_tiles=n_tiles
    )
    event_plans = _make_event_plans(metadata, symbols)
    if num_workers is None:
        total_tasks = sum(
            _linear_task_count(_resolve_shape(task.tile_shape, symbols))
            for task in metadata.tasks
        )
        num_workers = min(4, max(1, total_tasks))
    schedule_plan = _make_static_schedule_plan(
        metadata, validated.dependency_plan, symbols, num_workers
    )
    runtime_plan = _make_runtime_plan(metadata, event_plans, schedule_plan)
    return event_plans, schedule_plan, runtime_plan


def lower_event_tensor_graph(
    graph: EventTensorGraph,
    *,
    n_tiles: int | None = None,
    symbol_values: dict[str, int] | None = None,
    schedule: str = "static",
    row_tile: int = 32,
    n_cols: int = 128,
    queue_policy: str = "centralized",
    early_push: bool = False,
    wait_backoff: int = 0,
    num_workers: int | None = None,
):
    """Lower a supported Event Tensor graph to a TIRx PrimFunc."""

    metadata = _analyze_graph(graph)
    validated = _validate_graph(metadata)
    symbols = _resolve_symbol_values(
        metadata, symbol_values=symbol_values, n_tiles=n_tiles
    )
    _event_plans, schedule_plan, runtime_plan = plan_static_event_tensor_graph(
        metadata,
        symbol_values=symbols,
        num_workers=num_workers,
    )
    if queue_policy != "centralized":
        raise ValueError("only centralized queue_policy is implemented")
    if schedule == "static":
        return _lower_static(
            metadata,
            validated.dependency_plan,
            runtime_plan,
            schedule_plan,
            symbols,
            wait_backoff,
        )
    if schedule == "dynamic":
        raise NotImplementedError("generic dynamic megakernel lowering is not implemented")
    raise ValueError(f"unknown Event Tensor schedule {schedule!r}")


def _build_static_mixed_source(
    metadata: GraphMetadata,
    dependency_plan: DependencyPlan,
    runtime_plan: RuntimeResourcePlan,
    schedule_plan: StaticSchedulePlan,
    symbols: dict[str, int],
    wait_backoff: int,
) -> tuple[str, dict[str, Any]]:
    buffer_names, params, allocations = _make_mixed_buffer_bindings(
        metadata, runtime_plan, schedule_plan, symbols
    )
    used_handle_names = set(buffer_names.values())
    event_handles = {
        event.event: _unique_name(f"E_{event_plan.event_name}", used_handle_names)
        for event, event_plan in zip(metadata.events, runtime_plan.event_buffers)
    }

    namespace: dict[str, Any] = {"T": T}
    lines: list[str] = ["@T.prim_func", f"def static_kernel({', '.join(params)}):"]

    def emit(indent: int, text: str) -> None:
        lines.append(f"{' ' * indent}{text}")

    for allocation in allocations:
        emit(4, allocation)

    emit(4, "T.device_entry()")
    emit(4, f"worker = T.cta_id([{schedule_plan.num_workers}])")
    threads = max([32, *(task.threads or 0 for task in metadata.tasks)])
    emit(4, f"tx = T.thread_id([{threads}])")

    for event, event_plan in zip(metadata.events, runtime_plan.event_buffers):
        handle = event_handles[event.event]
        storage = buffer_names[f"event:{event.name}"]
        emit(
            4,
            f'{handle} = T.event_tensor({_format_shape(event_plan.shape)}, '
            f"wait_count={event_plan.wait_count}, storage={storage}, "
            f'name="{event_plan.event_name}")',
        )

    _emit_static_runtime_init(
        emit,
        metadata,
        schedule_plan,
        event_handles,
        indent=4,
    )

    task_ranges = _static_task_ranges(metadata, dependency_plan, symbols)
    total_tasks = task_ranges[-1][2] if task_ranges else 0
    emit(
        4,
        f"for global_task_id in T.serial(worker, {total_tasks}, "
        f"step={schedule_plan.num_workers}):",
    )

    task_by_type = {task.task_type: task for task in metadata.tasks}
    for task_type, start, end in task_ranges:
        task = task_by_type[task_type]
        body_name = f"task_body_{task.task_type}"
        namespace[body_name] = task.inline_body
        emit(8, f"if {_format_static_task_range_condition(start, end, total_tasks)}:")
        emit(12, f"linear_task_id = global_task_id - {start}")
        _emit_task_branch(
            emit,
            task,
            body_name,
            buffer_names,
            event_handles,
            symbols,
            wait_backoff,
            indent=12,
        )

    source = "\n".join(lines) + "\n"
    return source, namespace


def _make_mixed_buffer_bindings(
    metadata: GraphMetadata,
    runtime_plan: RuntimeResourcePlan,
    _schedule_plan: StaticSchedulePlan,
    symbols: dict[str, int],
) -> tuple[dict[Any, str], list[str], list[str]]:
    used_names: set[str] = set()
    buffer_names: dict[Any, str] = {}
    params: list[str] = []
    allocations: list[str] = []

    def add_param_tensor(tensor: Tensor, fallback: str) -> None:
        if id(tensor) in buffer_names:
            return
        name = _unique_name(tensor.name or fallback, used_names)
        shape = _resolve_shape(_as_tuple(tensor.shape), symbols)
        buffer_names[id(tensor)] = name
        params.append(f'{name}: T.Buffer({_format_shape(shape)}, "{tensor.dtype}")')

    def add_hidden_tensor(tensor: Tensor, fallback: str) -> None:
        if id(tensor) in buffer_names:
            return
        name = _unique_name(tensor.name or fallback, used_names)
        shape = _resolve_shape(_as_tuple(tensor.shape), symbols)
        buffer_names[id(tensor)] = name
        allocations.append(
            f'{name} = T.alloc_buffer({_format_shape(shape)}, "{tensor.dtype}", scope="global")'
        )

    for index, tensor in enumerate(metadata.input_values):
        add_param_tensor(tensor, f"input_{index}")
    for index, tensor in enumerate(_tensor_list(metadata.outputs)):
        add_param_tensor(tensor, f"output_{index}")

    for event_plan in runtime_plan.event_buffers:
        name = _unique_name(event_plan.backing_buffer, used_names)
        buffer_names[f"event:{event_plan.event_name}"] = name
        allocations.append(
            f'{name} = T.alloc_buffer(({_shape_numel(event_plan.shape)},), '
            '"int32", scope="global")'
        )

    for task in metadata.tasks:
        for index, tensor in enumerate(_tensor_list(task.outputs)):
            add_hidden_tensor(
                tensor, tensor.name or f"intermediate_{task.task_type}_{index}"
            )

    return buffer_names, params, allocations


def _emit_static_runtime_init(
    emit: Callable[[int, str], None],
    metadata: GraphMetadata,
    schedule_plan: StaticSchedulePlan,
    event_handles: dict[ETensor, str],
    *,
    indent: int,
) -> None:
    emit(indent, "T.evaluate(T.tvm_global_barrier_kinit())")
    emit(indent, "if worker == 0:")
    emit(indent + 4, "if tx == 0:")
    for event in metadata.events:
        emit(indent + 8, f"T.event_init({event_handles[event.event]})")

    emit(
        indent,
        f'T.tvm_storage_sync("global", tx == 0, {schedule_plan.num_workers})',
    )


def _static_task_ranges(
    metadata: GraphMetadata,
    dependency_plan: DependencyPlan,
    symbols: dict[str, int],
) -> tuple[tuple[int, int, int], ...]:
    task_by_type = {task.task_type: task for task in metadata.tasks}
    start = 0
    ranges: list[tuple[int, int, int]] = []
    for task_type in dependency_plan.topo_order:
        task = task_by_type[task_type]
        count = _linear_task_count(_resolve_shape(task.tile_shape, symbols))
        end = start + count
        ranges.append((task_type, start, end))
        start = end
    return tuple(ranges)


def _format_static_task_range_condition(start: int, end: int, total: int) -> str:
    if start == 0 and end == total:
        return "True"
    if start == 0:
        return f"global_task_id < {end}"
    if end == total:
        return f"global_task_id >= {start}"
    return f"T.And(global_task_id >= {start}, global_task_id < {end})"


def _emit_task_branch(
    emit: Callable[[int, str], None],
    task: TaskSpec,
    body_name: str,
    buffer_names: dict[Any, str],
    event_handles: dict[ETensor, str],
    symbols: dict[str, int],
    wait_backoff: int,
    *,
    indent: int,
) -> None:
    resolved_shape = _resolve_shape(task.tile_shape, symbols)
    axis_vars = _emit_unflatten_coords(emit, task, resolved_shape, indent)

    if task.in_edges:
        emit(indent, "if tx == 0:")
        for event, edge_map in task.in_edges:
            event_index = _format_event_index(edge_map, axis_vars)
            emit(
                indent + 4,
                f"T.event_wait({event_handles[event]}, {event_index}, "
                f"backoff={wait_backoff})",
            )
        emit(indent, 'T.tvm_storage_sync("shared")')

    call_args = _task_call_args(task, axis_vars, buffer_names)
    _validate_task_signature(task, call_args)
    emit(indent, f"{body_name}({', '.join(call_args)})")

    if task.out_edges:
        emit(indent, 'T.tvm_storage_sync("shared")')
        emit(indent, "if tx == 0:")
        for event, edge_map in task.out_edges:
            event_index = _format_event_index(edge_map, axis_vars)
            emit(
                indent + 4,
                f"T.evaluate(T.event_notify({event_handles[event]}, {event_index}))",
            )


def _emit_unflatten_coords(
    emit: Callable[[int, str], None],
    task: TaskSpec,
    shape: tuple[int, ...],
    indent: int,
) -> dict[str, str]:
    axis_vars: dict[str, str] = {}
    if not shape:
        return axis_vars
    for index, axis in enumerate(task.tile_axes):
        var_name = _sanitize_identifier(axis)
        stride = _linear_task_count(shape[index + 1 :]) if index + 1 < len(shape) else 1
        if len(shape) == 1:
            expr = "linear_task_id"
        elif index == len(shape) - 1:
            expr = f"linear_task_id % {shape[index]}"
        elif index == 0:
            expr = f"linear_task_id // {stride}"
        else:
            expr = f"(linear_task_id // {stride}) % {shape[index]}"
        emit(indent, f"{var_name} = {expr}")
        axis_vars[axis] = var_name
    return axis_vars


def _format_event_index(edge_map: EdgeMap, axis_vars: dict[str, str]) -> str:
    indices = tuple(axis_vars[axis] for axis in edge_map.target_axes)
    if len(indices) == 1:
        return indices[0]
    return f"({', '.join(indices)})"


def _task_call_args(
    task: TaskSpec, axis_vars: dict[str, str], buffer_names: dict[Any, str]
) -> list[str]:
    args = [axis_vars[axis] for axis in task.tile_axes]
    for value in task.inputs:
        args.append(_format_task_value(value, buffer_names))
    for value in _tensor_list(task.outputs):
        args.append(_format_task_value(value, buffer_names))

    signature = inspect.signature(task.inline_body)
    parameter_count = len(signature.parameters)
    if parameter_count == len(args) + 1:
        args.append("tx")
    return args


def _format_task_value(value: Any, buffer_names: dict[Any, str]) -> str:
    if isinstance(value, Tensor):
        return buffer_names[id(value)]
    if isinstance(value, int | float):
        return repr(value)
    raise TypeError(f"Unsupported call_device argument {value!r}; expected Tensor or scalar")


def _validate_task_signature(task: TaskSpec, call_args: list[str]) -> None:
    signature = inspect.signature(task.inline_body)
    parameter_count = len(signature.parameters)
    if parameter_count != len(call_args):
        raise ValueError(
            f"{task.name} expects {parameter_count} arguments after lowering, "
            f"but generic megakernel would pass {len(call_args)}"
        )


def _sanitize_identifier(name: str) -> str:
    candidate = re.sub(r"\W", "_", str(name or "value"))
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = f"v_{candidate}"
    return candidate


def _unique_name(name: str, used_names: set[str]) -> str:
    base = _sanitize_identifier(name)
    candidate = base
    index = 0
    while candidate in used_names:
        index += 1
        candidate = f"{base}_{index}"
    used_names.add(candidate)
    return candidate


def _format_shape(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return f"({', '.join(str(dim) for dim in shape)})"


def _lower_static(
    metadata: GraphMetadata,
    dependency_plan: DependencyPlan,
    runtime_plan: RuntimeResourcePlan,
    schedule_plan: StaticSchedulePlan,
    symbols: dict[str, int],
    wait_backoff: int,
):
    source, namespace = _build_static_mixed_source(
        metadata,
        dependency_plan,
        runtime_plan,
        schedule_plan,
        symbols,
        wait_backoff,
    )
    filename = "<tirx_megakernel_static>"
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)  # pylint: disable=exec-used
    return namespace["static_kernel"]


__all__ = [
    "CallDevice",
    "DependencyPlan",
    "DeviceFunction",
    "ETensor",
    "EdgeMap",
    "EventPlan",
    "EventSpec",
    "EventTensorGraph",
    "GraphFunction",
    "GraphMetadata",
    "RuntimeResourcePlan",
    "StaticSchedulePlan",
    "Symbol",
    "TaskSpec",
    "Tensor",
    "TensorSpec",
    "ValidatedGraphMetadata",
    "analyze_event_tensor_graph",
    "call_device",
    "device_func",
    "graph_func",
    "lower_event_tensor_graph",
    "plan_static_event_tensor_graph",
    "sym_var",
]
