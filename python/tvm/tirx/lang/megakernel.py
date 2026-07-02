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
import operator
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
    outputs: tuple[Any, ...]
    tasks: tuple[TaskSpec, ...]
    events: tuple[EventSpec, ...]


@dataclass(frozen=True)
class EventPlan:
    """Runtime Event Tensor resource plan."""

    event_name: str
    shape: tuple[int, ...]
    wait_count: int
    backing_buffer: str
    init_policy: str = "host"


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


def _mul_dim(lhs: Any, rhs: Any) -> Any:
    if lhs == 1:
        return rhs
    if rhs == 1:
        return lhs
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs * rhs
    return Symbol(f"{lhs}*{rhs}")


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
    event_wait_counts: dict[ETensor, Any] = {}
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
            explicit = event.wait_count
            if explicit is not None and str(explicit) != str(inferred):
                inferred = explicit
            previous = event_wait_counts.get(event)
            if previous is not None and str(previous) != str(inferred):
                raise ValueError(
                    f"Conflicting wait_count for Event Tensor {event.name or '<unnamed>'}: "
                    f"{previous} vs {inferred}"
                )
            event_wait_counts[event] = inferred
        for event, _ in parsed_in_edges:
            if event.wait_count is not None:
                event_wait_counts.setdefault(event, event.wait_count)

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
            wait_count = event_wait_counts.get(event)
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
        outputs=tuple(outputs),
        tasks=tuple(task_specs),
        events=tuple(events),
    )


def _validate_static_raw_sum(metadata: GraphMetadata) -> EventSpec:
    if len(metadata.tasks) != 2:
        raise ValueError("lowering expects exactly two call_device nodes")
    producer, consumer = metadata.tasks
    if len(producer.out_edges) != 1 or len(consumer.in_edges) != 1:
        raise ValueError("lowering expects one Event Tensor edge")
    event, out_map = producer.out_edges[0]
    in_event, in_map = consumer.in_edges[0]
    if event != in_event:
        raise ValueError("producer and consumer must use the same Event Tensor")
    if out_map != EdgeMap.parse("ij->i"):
        raise ValueError("producer edge must be 'ij->i'")
    if in_map != EdgeMap.parse("i->i"):
        raise ValueError("consumer edge must be 'i->i'")
    if not metadata.events:
        raise ValueError("lowering expects one Event Tensor")
    return metadata.events[0]


def _make_event_plan(event: EventSpec, symbols: dict[str, int]) -> EventPlan:
    return EventPlan(
        event_name=event.name,
        shape=_resolve_shape(event.shape, symbols),
        wait_count=_eval_dim(event.wait_count, symbols),
        backing_buffer=f"{event.name}_buf",
    )


def _make_static_schedule_plan(
    metadata: GraphMetadata, symbols: dict[str, int], num_workers: int
) -> StaticSchedulePlan:
    queue_tasks_by_worker: list[list[tuple[int, int]]] = [
        [] for _ in range(num_workers)
    ]
    records: list[tuple[int, int]] = []
    if len(metadata.tasks) == 2 and metadata.tasks[0].out_edges:
        producer, consumer = metadata.tasks
        producer_shape = _resolve_shape(producer.tile_shape, symbols)
        consumer_shape = _resolve_shape(consumer.tile_shape, symbols)
        if len(producer_shape) == 2 and len(consumer_shape) == 1:
            for i in range(producer_shape[0]):
                for j in range(producer_shape[1]):
                    records.append((producer.task_type, i * producer_shape[1] + j))
                records.append((consumer.task_type, i))
    if not records:
        for task in metadata.tasks:
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
    metadata: GraphMetadata, event_plan: EventPlan, schedule_plan: StaticSchedulePlan
) -> RuntimeResourcePlan:
    hidden_outputs = []
    if len(metadata.tasks) >= 1 and isinstance(metadata.tasks[0].outputs, Tensor):
        output = metadata.tasks[0].outputs
        hidden_outputs.append(
            TensorSpec(output.shape, output.dtype, output.name or "intermediate")
        )
    return RuntimeResourcePlan(
        hidden_intermediates=tuple(hidden_outputs),
        event_buffers=(event_plan,),
        static_schedule_buffers=(schedule_plan,),
        init_steps=(f"fill {event_plan.backing_buffer} with {event_plan.wait_count}",),
    )


def analyze_event_tensor_graph(graph: EventTensorGraph) -> GraphMetadata:
    """Analyze a traced graph_func into Event Tensor lowering metadata."""

    return _analyze_graph(graph)


def plan_static_event_tensor_graph(
    metadata: GraphMetadata,
    *,
    n_tiles: int,
) -> tuple[EventPlan, StaticSchedulePlan, RuntimeResourcePlan]:
    """Build the static Event Tensor resource and schedule plans.

    This helper exposes the same planning path used by
    :func:`lower_event_tensor_graph`, so examples and tests can inspect each
    lowering stage without reimplementing planner details.
    """

    event = _validate_static_raw_sum(metadata)
    symbols = {"n": int(n_tiles)}
    event_plan = _make_event_plan(event, symbols)
    num_workers = min(4, max(1, n_tiles * (event_plan.wait_count + 1)))
    schedule_plan = _make_static_schedule_plan(metadata, symbols, num_workers)
    runtime_plan = _make_runtime_plan(metadata, event_plan, schedule_plan)
    return event_plan, schedule_plan, runtime_plan


def lower_event_tensor_graph(
    graph: EventTensorGraph,
    *,
    n_tiles: int,  # 4
    schedule: str,
    row_tile: int = 32,
    n_cols: int = 128,
    queue_policy: str = "centralized",
    early_push: bool = False,
    wait_backoff: int = 0,
):
    """Lower a supported Event Tensor graph to a TIRx PrimFunc."""

    metadata = _analyze_graph(graph)
    event_plan, schedule_plan, runtime_plan = plan_static_event_tensor_graph(
        metadata, n_tiles=n_tiles
    )
    k_parts = event_plan.wait_count  # 4
    rows = n_tiles * row_tile  # 128
    if n_cols % k_parts != 0:
        raise ValueError("lowering requires n_cols divisible by wait_count")
    if queue_policy != "centralized":
        raise ValueError("only centralized queue_policy is implemented")
    if schedule == "static":
        return _lower_static(
            metadata,
            runtime_plan,
            n_tiles,
            rows,
            row_tile,
            n_cols,
            k_parts,
            wait_backoff,
        )
    if schedule == "dynamic":
        return _lower_dynamic(
            n_tiles, rows, row_tile, n_cols, k_parts, early_push, wait_backoff
        )
    raise ValueError(f"unknown Event Tensor schedule {schedule!r}")


def _lower_static(
    metadata: GraphMetadata,
    runtime_plan: RuntimeResourcePlan,
    n_tiles: int,  # 4
    rows: int,  # 128
    row_tile: int,  # 32
    n_cols: int,  # 128
    k_parts: int,  # 4
    wait_backoff: int,  # 0
):
    part_cols = n_cols // k_parts  # 4
    schedule_plan = runtime_plan.static_schedule_buffers[0]
    event_plan = runtime_plan.event_buffers[0]
    producer, consumer = metadata.tasks
    producer_body = producer.inline_body
    consumer_body = consumer.inline_body
    producer_arg_count = (
        len(inspect.signature(producer_body).parameters)
        if producer_body is not None
        else 0
    )
    consumer_arg_count = (
        len(inspect.signature(consumer_body).parameters)
        if consumer_body is not None
        else 0
    )
    producer_has_tx = producer_arg_count == 5
    consumer_has_tx = consumer_arg_count == 4
    if producer_body is not None and producer_arg_count not in (4, 5):
        raise ValueError(
            f"{producer.name} must accept (i, j, input, output[, tx]); "
            f"got {producer_arg_count} parameters"
        )
    if consumer_body is not None and consumer_arg_count not in (3, 4):
        raise ValueError(
            f"{consumer.name} must accept (i, input, output[, tx]); "
            f"got {consumer_arg_count} parameters"
        )
    workers = schedule_plan.num_workers  # 4
    event_shape = event_plan.shape
    event_wait_count = event_plan.wait_count
    event_name = event_plan.event_name
    task_threads = [task.threads for task in metadata.tasks if task.threads is not None]
    threads = max([32, row_tile, *task_threads])  # 32
    tasks_per_tile = k_parts + 1  # 5
    task_count = n_tiles * tasks_per_tile  # 20

    @T.prim_func
    def static_kernel(
        A: T.Buffer((rows, n_cols), "float32"),
        Y: T.Buffer((rows,), "float32"),
        E_buf: T.Buffer((n_tiles,), "int32"),
        P: T.Buffer((rows, k_parts), "float32"),
    ):
        T.device_entry()
        worker = T.cta_id([workers])
        tx = T.thread_id([threads])
        E = T.event_tensor(
            event_shape, wait_count=event_wait_count, storage=E_buf, name=event_name
        )
        for task in T.serial(worker, task_count, step=workers):
            i = task // tasks_per_tile
            phase = task % tasks_per_tile
            if phase < k_parts:
                if producer_body is not None:
                    if producer_has_tx:
                        producer_body(i, phase, A, P, tx)
                    else:
                        producer_body(i, phase, A, P)
                else:
                    if tx < row_tile:
                        row = i * row_tile + tx
                        acc = T.float32(0)
                        for c in T.serial(part_cols):
                            acc = acc + A[row, phase * part_cols + c]
                        P[row, phase] = acc
                T.tvm_storage_sync("shared")
                if tx == 0:
                    T.evaluate(T.event_notify(E, i))
            else:
                if tx == 0:
                    T.event_wait(E, i, backoff=wait_backoff)
                T.tvm_storage_sync("shared")
                if consumer_body is not None:
                    if consumer_has_tx:
                        consumer_body(i, P, Y, tx)
                    else:
                        consumer_body(i, P, Y)
                else:
                    if tx < row_tile:
                        row = i * row_tile + tx
                        acc = T.float32(0)
                        for j in T.serial(k_parts):
                            acc = acc + P[row, j]
                        Y[row] = acc

    return static_kernel


def _lower_dynamic(
    n_tiles: int,
    rows: int,
    row_tile: int,
    n_cols: int,
    k_parts: int,
    early_push: bool,
    wait_backoff: int,
):
    part_cols = n_cols // k_parts
    queue_capacity = n_tiles * (k_parts + 1)
    workers = min(4, max(1, queue_capacity))
    threads = max(32, row_tile)

    @T.prim_func
    def dynamic_kernel(
        A: T.Buffer((rows, n_cols), "float32"),
        Y: T.Buffer((rows,), "float32"),
        E_buf: T.Buffer((n_tiles,), "int32"),
        P: T.Buffer((rows, k_parts), "float32"),
        Q_storage: T.Buffer((queue_capacity, 3), "int32"),
        Q_head: T.Buffer((1,), "int32"),
        Q_tail: T.Buffer((1,), "int32"),
        Q_pending: T.Buffer((1,), "int32"),
        Q_lock: T.Buffer((1,), "int32"),
    ):
        T.device_entry()
        T.cta_id([workers])
        tx = T.thread_id([threads])
        E = T.event_tensor(
            (n_tiles,), wait_count=k_parts, storage=E_buf, name="row_done"
        )
        Q_storage[0, 0] = Q_storage[0, 0]
        Q_tail[0] = Q_tail[0]
        Q_pending[0] = Q_pending[0]
        task_pos = T.alloc_buffer((1,), "int32", scope="shared")
        task_i = T.alloc_buffer((1,), "int32", scope="shared")
        task_j = T.alloc_buffer((1,), "int32", scope="shared")
        while True:
            if tx == 0:
                while T.cuda.atomic_cas(Q_lock.data, T.int32(0), T.int32(1)) != 0:
                    T.cuda.nano_sleep(64)
                task_pos[0] = Q_head[0]
                if task_pos[0] < n_tiles * k_parts:
                    Q_head[0] = task_pos[0] + 1
                    task_i[0] = task_pos[0] // k_parts
                    task_j[0] = task_pos[0] % k_parts
                else:
                    task_i[0] = -1
                    task_j[0] = 0
                T.evaluate(
                    T.ptx.atom_scalar(
                        Q_lock.data,
                        T.int32(0),
                        sem="release",
                        scope="gpu",
                        space="global",
                        op="exch",
                        ptx_type="b32",
                    )
                )
            T.tvm_storage_sync("shared")
            if task_i[0] == -1:
                break
            if tx < row_tile:
                row = task_i[0] * row_tile + tx
                acc = T.float32(0)
                for c in T.serial(part_cols):
                    acc = acc + A[row, task_j[0] * part_cols + c]
                P[row, task_j[0]] = acc
            T.tvm_storage_sync("shared")
            ready = T.alloc_buffer((1,), "int32", scope="shared")
            if tx == 0:
                T.cuda.thread_fence()
                if T.event_notify(E, task_i[0]):
                    ready[0] = 1
                else:
                    ready[0] = 0
            T.tvm_storage_sync("shared")
            if ready[0] != 0:
                if early_push:
                    if tx == 0:
                        T.event_wait(E, task_i[0], backoff=wait_backoff)
                    T.tvm_storage_sync("shared")
                if tx < row_tile:
                    row = task_i[0] * row_tile + tx
                    acc = T.float32(0)
                    for jj in T.serial(k_parts):
                        acc = acc + P[row, jj]
                    Y[row] = acc
            T.tvm_storage_sync("shared")

    return dynamic_kernel


__all__ = [
    "CallDevice",
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
    "analyze_event_tensor_graph",
    "call_device",
    "device_func",
    "graph_func",
    "lower_event_tensor_graph",
    "plan_static_event_tensor_graph",
    "sym_var",
]
