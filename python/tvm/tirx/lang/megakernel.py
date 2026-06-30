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

from collections.abc import Callable
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

    def __getitem__(self, _index):
        return self

    def __setitem__(self, _index, _value):
        return None


@dataclass(frozen=True)
class ETensor:
    """Graph-level Event Tensor placeholder."""

    shape: Any
    wait_count: Any
    name: str = ""


@dataclass(frozen=True)
class DeviceFunction:
    """Recorded tile task function."""

    fn: Callable

    @property
    def name(self) -> str:
        return self.fn.__name__

    def __get__(self, _obj, _objtype=None):
        return self


@dataclass
class CallDevice:
    """A graph-level call_device node."""

    device_func: DeviceFunction
    tile_num: Any
    args: list[Any]
    in_edges: dict[ETensor, str] = field(default_factory=dict)
    out_edges: dict[ETensor, str] = field(default_factory=dict)


@dataclass
class EventTensorGraph:
    """Recorded graph_func body."""

    calls: list[CallDevice]
    output: Any


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
            return EventTensorGraph(calls=_TraceContext.current, output=output)
        finally:
            _TraceContext.current = old

    def __call__(self, *args, **kwargs) -> EventTensorGraph:
        return self.trace(*args, **kwargs)


def device_func(fn: Callable) -> DeviceFunction:
    """Decorator marking a tile task function."""

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
    in_edges=None,
    in_edge=None,
    out_edges=None,
) -> Tensor:
    """Record a device function launch in the active graph trace."""

    if not isinstance(device_fn, DeviceFunction):
        raise TypeError("call_device expects a @device_func-decorated function")
    in_edges = in_edges if in_edges is not None else in_edge
    node = CallDevice(
        device_func=device_fn,
        tile_num=tile_num,
        args=list(args),
        in_edges=dict(in_edges or {}),
        out_edges=dict(out_edges or {}),
    )
    if _TraceContext.current is not None:
        _TraceContext.current.append(node)
    return Tensor(shape=None)


def _validate_graph(graph: EventTensorGraph) -> ETensor:
    if len(graph.calls) != 2:
        raise ValueError("lowering expects exactly two call_device nodes")
    producer, consumer = graph.calls
    if len(producer.out_edges) != 1 or len(consumer.in_edges) != 1:
        raise ValueError("lowering expects one Event Tensor edge")
    event = next(iter(producer.out_edges))
    if event not in consumer.in_edges:
        raise ValueError("producer and consumer must use the same Event Tensor")
    if producer.out_edges[event].replace(" ", "") != "ij->i":
        raise ValueError("producer edge must be 'ij->i'")
    if consumer.in_edges[event].replace(" ", "") != "i->i":
        raise ValueError("consumer edge must be 'i->i'")
    return event


def lower_event_tensor_graph(
    graph: EventTensorGraph,
    *,
    n_tiles: int, # 4
    schedule: str,
    row_tile: int = 32,
    n_cols: int = 128,
    queue_policy: str = "centralized",
    early_push: bool = False,
    wait_backoff: int = 0,
):
    """Lower a supported Event Tensor graph to a TIRx PrimFunc."""

    event = _validate_graph(graph)
    k_parts = int(event.wait_count) # 4
    rows = n_tiles * row_tile # 128
    if n_cols % k_parts != 0:
        raise ValueError("lowering requires n_cols divisible by wait_count")
    if queue_policy != "centralized":
        raise ValueError("only centralized queue_policy is implemented")
    if schedule == "static":
        return _lower_static(n_tiles, rows, row_tile, n_cols, k_parts, wait_backoff)
    if schedule == "dynamic":
        return _lower_dynamic(
            n_tiles, rows, row_tile, n_cols, k_parts, early_push, wait_backoff
        )
    raise ValueError(f"unknown Event Tensor schedule {schedule!r}")


def _lower_static(
    n_tiles: int, # 4
    rows: int, # 128
    row_tile: int, # 32
    n_cols: int, # 128
    k_parts: int,# 4
    wait_backoff: int, # 0
):
    part_cols = n_cols // k_parts # 4
    workers = min(4, max(1, n_tiles * (k_parts + 1))) # 4
    threads = max(32, row_tile) # 32
    tasks_per_tile = k_parts + 1 # 5
    task_count = n_tiles * tasks_per_tile # 20

    @T.prim_func
    def static_kernel(
        A: T.Buffer((rows, n_cols), "float32"),
        Y: T.Buffer((rows,), "float32"),
        E_buf: T.Buffer((n_tiles,), "int32"),
        P: T.Buffer((rows, k_parts), "float32"),
    ):
        T.device_entry()
        worker = T.cta_id([workers]) # 4个cta，worker是cta id
        tx = T.thread_id([threads]) # 每个cta 32个thread，tx是thread id
        E = T.event_tensor((n_tiles,), wait_count=k_parts, storage=E_buf, name="row_done")
        for task in T.serial(worker, task_count, step=workers):
            i = task // tasks_per_tile
            phase = task % tasks_per_tile
            if phase < k_parts:
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
        E = T.event_tensor((n_tiles,), wait_count=k_parts, storage=E_buf, name="row_done")
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
    "EventTensorGraph",
    "GraphFunction",
    "Symbol",
    "Tensor",
    "call_device",
    "device_func",
    "graph_func",
    "lower_event_tensor_graph",
    "sym_var",
]
