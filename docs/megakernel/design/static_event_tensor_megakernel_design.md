<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Static Event Tensor Megakernel Design

本文只设计 static schedule。目标是把用户写的 tile-level graph 编译成一个通用
TIRx megakernel，而不是为 raw sum 预置 `_lower_static` 这类固定 PrimFunc。

用户侧代码应接近 `docs/megakernel/tasks/fake_example.py`：

- `graph_func` 描述完整 megakernel 的 tile task graph。
- `device_func` 描述一个 tile task 的计算体。
- `call_device` 创建一组 task，并用 Event Tensor edge 描述依赖。
- 框架负责 task graph 分析、Event Tensor counter 推导、workspace 规划、static
  schedule 生成和 TIRx PrimFunc emit。

本文不覆盖 dynamic ready queue。后续 dynamic lowering 可以复用本文的 GraphIR、
TaskIR、EventIR 和 device task body emitter。

## 1. 当前实现需要替换的部分

`python/tvm/tirx/lang/megakernel.py` 目前的问题不是缺少几个参数，而是抽象层次不对：

- `device_func` 只保存 Python callable，函数体没有被 trace、解析或降低。
- `Tensor.__getitem__` 返回自身，`Tensor.__setitem__` 是 no-op，所以用户写的 tile
  计算没有 IR 表达。
- `_validate_graph` 只接受两个 `call_device`，且 edge string 必须是 `ij->i` 和
  `i->i`。
- `_lower_static` 手写了 `A -> P -> Y` raw-sum PrimFunc，忽略 graph 中的
  `device_func`、tensor shape、dtype、参数和输出。
- Event Tensor backing buffer 的初始化、重复 launch、memory ordering 还没有成为
  框架 contract。

新的 static 框架应删除“识别固定 graph 然后返回固定 kernel”的路径，改为：

```text
user graph_func
  -> GraphIR
  -> TaskIR + EventIR + WorkspacePlan
  -> StaticSchedule
  -> TIRx PrimFunc
```

## 2. 用户编程模型

### 2.1 Graph function

`graph_func` 是完整 megakernel 的 tile graph 描述，也是用户看到的对外接口。
用户传递 graph input tensors，并从它获得 graph output tensors。它的源码只描述
tile task 之间的 DAG 关系，不描述 host launch、workspace 分配或 event 初始化细节。

系统对 `graph_func` 的 lowering 结果是一个 TIRx `PrimFunc` megakernel。这个
PrimFunc 是纯 device kernel：

- 包含 `T.device_entry()`。
- 不包含 host 控制逻辑。
- 不发起子 kernel launch。
- 不在 kernel 内动态分配 global memory。
- 使用调用方传入的 workspace、Event Tensor storage 和 static task queue。

实现上，`@graph_func` 可以复用 TIRx script/PrimFunc 的 builder 能力，但它不是让用户
手写完整 `@T.prim_func`。用户写的是 graph DSL；系统解析 graph DSL 后生成 TIRx
PrimFunc。

示例：

```python
class IRModule:
    @device_func
    def partial_sum(i: int, j: int, A: Tensor, B: Tensor):
        B[i * 32 : i * 32 + 32, j] = sum(
            A[i * 32 : i * 32 + 32, j * 32 : j * 32 + 32]
        )

    @device_func
    def final_sum(i: int, B: Tensor, C: Tensor):
        C[i * 32 : i * 32 + 32] = sum(B[i * 32 : i * 32 + 32, :])

    @graph_func
    def main_graph(A: Tensor(("n*32", 128))) -> Tensor(("n*32",)):
        n = sym_var("n")
        E = ETensor((n,), name="row_done")
        B = call_device(
            IRModule.partial_sum,
            tile_num=(n, 4),
            args=[A],
            outputs=Tensor((n * 32, 4), "float32"),
            out_edges={E: "ij->i"},
        )
        C = call_device(
            IRModule.final_sum,
            tile_num=(n,),
            args=[B],
            outputs=Tensor((n * 32,), "float32"),
            in_edges={E: "i->i"},
        )
        return C
```

这里新增 `outputs=` 是正式 API。`B: Tensor(...) = call_device(...)` 这种局部变量
annotation 可以作为源码解析 sugar，但不能作为唯一接口，因为 Python 运行时 trace
无法可靠获取局部变量 annotation。

### 2.2 Device function

`device_func` 是 tile task body。它不是 host function，也不是独立 kernel launch。
lowering 后它会 inline 到 megakernel 的 task dispatch/body 中。

第一版支持两种 device body 来源：

1. Tensor DSL device function。
   - 解析 Python source AST。
   - 支持 tensor slice load/store、标量表达式、`sum(slice)` 形式 reduction。
   - 生成框架内部 TaskIR，再 emit TIRx loops。

2. TIRx inline device function。
   - 用户用 `T.inline` 或等价 decorator 写 TIRx script body。
   - 框架把 tile coordinates、input/output buffers 绑定进去，并 inline 该 body。

Tensor DSL 是为了让 raw-sum 这类 `fake_example.py` 写法自然可用。TIRx inline 是
通用逃生口，避免第一版 DSL 试图覆盖所有 Python。

### 2.3 call_device

`call_device` 创建一个 task family：

```python
call_device(
    fn,
    tile_num,
    args,
    outputs=None,
    in_edges=None,
    out_edges=None,
    name=None,
    threads=None,
)
```

语义：

- `tile_num` 是 task domain shape。`tile_num=(n, 4)` 表示 task coordinate
  `(i, j)`，其中 `0 <= i < n`，`0 <= j < 4`。
- `args` 是已有 tensors，来自 graph input 或上游 `call_device` output。
- `outputs` 声明此 task family 产生的新 tensor。单输出返回 `Tensor`，多输出返回
  tuple。
- `in_edges` 描述 task body 前需要等待的 Event Tensor。
- `out_edges` 描述 task body 完成后需要通知的 Event Tensor。
- `threads` 是可选 task-local CTA 线程数；缺省由 lowering config 或 device_func
  schedule hint 决定。

`device_func` 参数绑定规则：

- 前缀标量参数绑定 task coordinates。
- `args` 绑定已有 tensor 参数。
- `outputs` 绑定剩余 tensor 参数。

以 `partial_sum(i, j, A, B)` 为例，`i,j` 来自 task coordinate，`A` 来自
`args=[A]`，`B` 来自 `outputs=Tensor(...)`。

## 3. 内部 IR

### 3.1 GraphIR

Trace `graph_func` 后得到：

```text
GraphIR {
  symbols: [n, ...]
  inputs: [TensorSpec]
  outputs: [TensorValue]
  calls: [CallNode]
  events: [EventSpec]
}
```

`TensorSpec`：

```text
TensorSpec {
  name
  shape: [PrimExpr]
  dtype
  storage_role: input | output | intermediate
}
```

`CallNode`：

```text
CallNode {
  name
  task_type
  device_func
  tile_axes: [Axis]
  tile_shape: [PrimExpr]
  inputs: [TensorValue]
  outputs: [TensorValue]
  in_edges: [EventUse]
  out_edges: [EventUse]
  threads
}
```

`EventSpec`：

```text
EventSpec {
  name
  shape: [PrimExpr]
  dtype: int32
  init_policy
  storage
}
```

### 3.2 TaskIR

`device_func` lowering 产生 TaskIR：

```text
TaskIR {
  params: tile axes + tensor buffers
  body: Stmt
  thread_extent
  completion_scope: cta
}
```

TaskIR 可以来自 Tensor DSL，也可以来自 TIRx inline body。它必须满足：

- 不能包含 `T.device_entry()`。
- 不能 launch 新 kernel。
- 不能创建新的 global storage。
- 可以使用 local/shared allocation、TIRx loops、tile primitives 和 CUDA intrinsics。
- 对外可见写入只能写 task outputs 或显式标注允许写的 buffers。

### 3.3 Event edge

Event edge 使用 `IndexMap` 表示，而不是 lowering 时临时解析字符串。

用户层允许 string shorthand：

```python
out_edges={E: "ij->i"}
in_edges={E: "i->i"}
```

内部等价为：

```text
producer_map: (i, j) -> (i)
consumer_map: (i) -> (i)
```

正式 API 也应支持 callable/IndexMap，避免多字符 axis 名和复杂 affine map 受限：

```python
out_edges={E: lambda i, j: (i,)}
```

第一版 edge map 限制为 affine map，输入只能引用 task coordinate，输出 rank 必须等于
Event Tensor rank。

## 4. Event Tensor 语义

Event Tensor 是 global `int32` counter tensor。每个元素代表一个 dependency event。

对 static schedule，Event Tensor 的核心操作是：

```python
T.event_wait(E, event_indices)
T.event_notify(E, event_indices)
```

语义：

- producer task 完成可见写入后，对对应 event counter 执行 atomic decrement。
- consumer task 执行前等待对应 event counter 变为 0。
- counter 初始值等于映射到该 event 元素的 producer task 数。

### 4.1 wait_count 推导

框架不应要求用户手写 `wait_count=4` 才能正确 lowering。`wait_count` 应从 graph edge
推导：

```text
count_E[e] =
  sum over producer edges p:
    number of task coordinates d in producer_domain(p)
    such that producer_map_p(d) == e
```

raw sum：

```text
producer domain: (i, j), 0 <= i < n, 0 <= j < 4
producer map: (i, j) -> i
count_E[i] = 4
```

第一版支持 uniform wait count，即每个 event element 的 count 是同一个 `PrimExpr`。
如果推导结果非 uniform，有两种后续扩展：

- 生成单独 host/event-init kernel 填充 per-element count。
- 在 megakernel prologue 中填充 event storage，并用 init flag 同步。

static 第一版建议只接受 uniform count。这样 raw-sum、tiled matmul split-K、regular
stencil 等场景都能覆盖，且实现边界清楚。

### 4.2 storage 和初始化

正式 Event Tensor storage 位于 global memory。原因是 static megakernel 中 producer
和 consumer 可能在不同 CTA/SM。

低层 PrimFunc ABI 显式接收 event storage：

```text
E_row_done: int32[num_events]
```

高层 framework wrapper 负责在 launch 前初始化：

```text
fill(E_row_done, wait_count)
launch(megakernel)
```

这样可以避免在单个 CUDA kernel 内实现 grid-wide init barrier。若调用方直接使用
PrimFunc，则必须按照 `WorkspaceSpec` 初始化 event storage。框架应在返回值里暴露：

```text
LoweredMegakernel {
  prim_func
  workspace_spec
  init_plan
}
```

为了兼容现有 `lower_event_tensor_graph(...)->PrimFunc` 风格，可以提供：

```python
lowered = lower_event_tensor_graph(..., schedule="static")
func = lowered.prim_func
```

或者在过渡期让 `LoweredMegakernel` 代理 `.script()` 和 TVM compile 所需接口。

### 4.3 memory ordering

默认 correctness-first lowering：

Producer side：

```python
emit_task_body()
T.cuda.thread_fence()          # all participating writer threads
T.tvm_storage_sync("shared")   # CTA-local completion
if tx == 0:
    T.evaluate(T.event_notify(E, event_idx))
```

Consumer side：

```python
T.event_wait(E, event_idx)     # default: all participating consumer threads wait
T.tvm_storage_sync("shared")
emit_task_body()
```

说明：

- `event_notify` 使用 release atomic decrement。
- `event_wait` 使用 acquire load 轮询。
- 默认让所有 consumer threads wait，避免只有 leader acquire 后其他线程直接读 global
  data 的 memory-model 空洞。
- producer task 如果只有单线程写入，可以通过 schedule hint 省略全 CTA fence/barrier。
- 后续优化可以实现 leader-wait、cooperative-group wait 或 warp-level wait，但必须由
  明确 memory proof 支撑。

## 5. Static schedule lowering

static schedule 不使用 GPU ready queue。它使用 host 或编译期预先生成的 per-CTA
static task queue。megakernel 启动时按硬件 core/SM 数量选择 `num_workers`，启动
对应数量的 CTA；每个 CTA 只消费自己的静态队列。

因此 static megakernel 的总体形态是一个 worker loop：

```text
worker = cta_id
pos = queue_offsets[worker]
end = queue_offsets[worker + 1]
while pos < end:
    task = queue_tasks[pos]
    wait task dependencies
    run task body
    notify task outputs
    pos += 1
```

这里的 queue 是静态的：任务内容、任务顺序和每个 CTA 的任务区间在 launch 前已经确定。
kernel 内不会根据 event ready 状态 push 新任务。

### 5.1 Static task queue ABI

第一版使用显式 task queue buffer 作为 PrimFunc 参数：

```text
Q_offsets: int32[num_workers + 1]
Q_tasks:   int32[num_tasks, task_record_fields]
```

`Q_offsets[w]` 和 `Q_offsets[w + 1]` 给出 CTA `w` 的 task record 半开区间。

`Q_tasks` 的每条 task record 至少包含：

```text
field 0: task_type
field 1..R: flattened or padded task coordinates
```

`R = max_rank(call.tile_num)`。rank 较低的 task family 使用 0 padding。也可以选择
`linear_task_id` 方案：

```text
field 0: task_type
field 1: linear_task_id
```

然后在 kernel 内根据对应 task family 的 `tile_shape` unflatten。第一版推荐
`task_type + linear_task_id`，因为 record 更短，且避免为每个 task 存多个坐标。

static queue 的生成位置：

- 如果 task 数和 shape 全是编译期常量，可以由 lowering 直接生成常量 queue。
- 如果包含 `n` 这类 runtime symbol，则由 framework launcher 在 launch 前根据实际
  shape 生成 `Q_offsets/Q_tasks` 并传给 megakernel。

无论哪种方式，生成的 TIRx PrimFunc 本身都是纯 kernel。

### 5.2 Queue construction policy

默认 static planner 按 call-node DAG 的 topological order 生成队列：

```text
for call in topological_order(GraphIR.calls):
    enumerate all tasks in call domain
    assign each task to a CTA queue
```

分配策略第一版采用 round-robin 或 block partition：

```text
target_worker = task_linear_id % num_workers
```

每个 CTA queue 中的任务顺序必须保持 stage order：所有 stage-0 任务排在该 CTA 的
stage-1 任务之前，依此类推。这样 static queue 是 materialized 的，但它仍然来自通用
DAG，不是 raw-sum 特化。

### 5.3 Kernel shape

生成的 TIRx 结构：

```python
@T.prim_func
def megakernel(..., Q_offsets: T.Buffer(...), Q_tasks: T.Buffer(...)):
    T.device_entry()
    worker = T.cta_id([num_workers])
    tx = T.thread_id([threads])

    E0 = T.event_tensor(E0_shape, wait_count=..., storage=E0_storage, name="...")
    E1 = T.event_tensor(E1_shape, wait_count=..., storage=E1_storage, name="...")

    pos = Q_offsets[worker]
    end = Q_offsets[worker + 1]
    while pos < end:
        task_type = Q_tasks[pos, 0]
        linear = Q_tasks[pos, 1]

        if task_type == TASK_0:
            coords = unflatten(linear, tile_shape_0)
            wait_input_events(task_0, coords)
            emit_task_0(coords)
            notify_output_events(task_0, coords)

        elif task_type == TASK_1:
            coords = unflatten(linear, tile_shape_1)
            wait_input_events(task_1, coords)
            emit_task_1(coords)
            notify_output_events(task_1, coords)

        pos = pos + 1
```

这就是 static 版本的“每个 CTA 一个 task queue”。和 dynamic queue 的区别是：

- static queue 不在 kernel 内 push/pop。
- 没有 global queue lock。
- 没有 `pending_tasks` 退出条件。
- 每个 CTA 的 `begin/end` 已经预先确定。

### 5.2 Deadlock freedom

static schedule 必须保证不会出现所有 CTA 都在等待尚未执行的 producer task。

默认 per-CTA queue schedule 的规则：

- `CallNode` DAG 必须无环。
- queue construction 按 `CallNode` DAG 的 topological stage 顺序追加任务。
- 同一个 CTA queue 内，producer stage 的任务必须排在 consumer stage 的任务之前。
- 不允许把一个 CTA 的 consumer task 排在该 CTA 尚未执行的 upstream producer stage
  之前。
- consumer task 可以等待其他 CTA queue 中尚未完成的 producer task；其他 CTA 不会因此
  被阻塞，因为它们有自己的 queue。

因此 raw sum 中 `final_sum(i)` 可能等待其他 CTA 还未完成的 `partial_sum(i, j)`，但
不会等待排在自己本地队列后面的 producer task。

planner 必须验证每个 CTA queue 是按 DAG stage 约束构造的；否则可能出现所有 CTA 都在
等待尚未执行的 producer task，从而死锁。

### 5.3 Task coordinate lowering

对 rank-k task domain：

```text
tile_shape = [s0, s1, ..., s{k-1}]
linear in [0, product(tile_shape))
```

unflatten：

```text
coord[k-1] = linear % s{k-1}
linear = linear // s{k-1}
...
coord[0] = linear % s0
```

这些表达式用 TIRx `FloorDiv`/`FloorMod` 生成。坐标变量绑定到 device_func 的 tile
axis 参数。

### 5.4 Wait/notify insertion

对每个 `CallNode`：

```text
for edge in in_edges:
    event_idx = edge.map(coords)
    T.event_wait(edge.event, event_idx, backoff=config.wait_backoff)
T.tvm_storage_sync("shared")

emit TaskIR body

if call may write cross-CTA-visible data:
    T.cuda.thread_fence()
T.tvm_storage_sync("shared")

if tx == 0:
    for edge in out_edges:
        event_idx = edge.map(coords)
        T.evaluate(T.event_notify(edge.event, event_idx))
```

`event_notify` 只执行一次。`event_wait` 默认由参与 task 的所有线程执行。

### 5.5 Multiple input/output events

一个 task 可以等待多个 Event Tensor，也可以通知多个 Event Tensor：

```python
call_device(
    fn,
    tile_num=(m, n),
    in_edges={E0: "ij->i", E1: "ij->j"},
    out_edges={E2: "ij->ij"},
)
```

lowering 顺序：

- 所有 `in_edges` wait 完成后才执行 body。
- body 完成和 fence/barrier 后通知所有 `out_edges`。
- 如果多个 producer family notify 同一个 Event Tensor，wait count 是它们贡献之和。

## 6. Device function lowering

### 6.1 Tensor DSL path

Tensor DSL 不应依赖 `Tensor.__getitem__` 返回自身。新的 trace/lowering 需要 source
AST 或 proxy IR。

推荐实现：

1. `@device_func` 保存 `inspect.getsource(fn)`、signature 和 closure constants。
2. lowering 时创建 symbolic tile vars 和 TensorProxy。
3. 用 AST visitor 转换函数体为 TaskIR。
4. TaskIR emitter 生成 TIRx statements。

第一版 Tensor DSL 支持集合：

- tensor slice load：`A[rows, cols]`
- tensor slice store：`B[rows, j] = expr`
- scalar tensor load/store：`A[i, j]`
- affine index expression：`i * 32 + tx`、`j * 32 + c`
- `sum(tensor_slice)` reduction
- dtype cast 和常量

raw-sum `partial_sum` 可降低为：

```python
if tx < ROW_TILE:
    row = i * ROW_TILE + tx
    acc = T.float32(0)
    for c in T.serial(PART_COLS):
        acc = acc + A[row, j * PART_COLS + c]
    B[row, j] = acc
```

`final_sum` 可降低为：

```python
if tx < ROW_TILE:
    row = i * ROW_TILE + tx
    acc = T.float32(0)
    for j in T.serial(K_PARTS):
        acc = acc + B[row, j]
    C[row] = acc
```

这里 row mapping 和 reduction loop 来自 Tensor DSL，不来自 hard-coded raw-sum
lowerer。

### 6.2 TIRx inline path

复杂 task 可以直接写成 TIRx inline body：

```python
@device_func
@T.inline
def my_task(i: T.int32, A: T.Buffer(...), B: T.Buffer(...), tx: T.int32):
    ...
```

框架负责：

- 绑定 tile coordinate 到 `i` 等参数。
- 绑定 graph tensor 到 TIRx Buffer。
- 提供 `tx`、`worker` 等 execution context，或按约定由 inline body 调用
  `T.thread_id`。
- inline body 前后统一插入 event wait/notify。

第一版建议框架统一创建 `tx = T.thread_id([threads])`，并把 `tx` 作为可选特殊参数传给
device_func，避免每个 task body 重复声明 thread extent。

## 7. Workspace plan

static megakernel 需要三类 storage：

1. 用户输入/输出 tensors。
2. intermediate tensors，例如 raw sum 的 `B/P`。
3. Event Tensor backing buffers。
4. static task queue buffers。

第一版全部使用显式 PrimFunc buffer 参数：

```text
main(A, C, B_workspace, E_row_done, Q_offsets, Q_tasks)
```

`WorkspaceSpec` 描述这些额外参数：

```text
WorkspaceSpec {
  intermediates: [
    {name: "B", shape: (n * 32, 4), dtype: "float32"}
  ]
  events: [
    {name: "row_done", shape: (n,), dtype: "int32", init: 4}
  ]
  static_queues: [
    {
      offsets: {shape: (num_workers + 1,), dtype: "int32"},
      tasks: {shape: (num_tasks, 2), dtype: "int32"},
      fields: ["task_type", "linear_task_id"]
    }
  ]
}
```

高层 wrapper 可以根据 `WorkspaceSpec` 自动分配和初始化。低层用户也可以手动传入。

优化方向：

- 将生命周期不重叠的 intermediate 复用到同一 workspace。
- 对只在同 CTA 内生产消费的 temporary 使用 shared memory。
- 对 event storage 和 small intermediates 做 packed workspace，减少 ABI 参数数量。
- 对 static queue 使用 packed int32 workspace，避免 ABI 参数过多。

但第一版应优先保持 ABI 清晰，避免 workspace offset 和 aliasing 干扰 correctness。

## 8. Lowering pipeline

### 8.1 Python frontend

```text
GraphFunction.trace(args)
  -> creates TensorValue for graph inputs
  -> executes graph_func in trace mode
  -> call_device appends CallNode
  -> returns GraphIR
```

Trace mode 中的 `Tensor` 必须是 graph value，不是 no-op placeholder：

- 保存 `TensorSpec`。
- 记录 producer/consumer。
- 支持传入 downstream call。
- 不在 graph trace 阶段解释 tensor indexing。

`device_func` body 不在 graph trace 阶段执行；它在 lowering 阶段用 symbolic tile vars
转换为 TaskIR。

### 8.2 Graph analysis

输入 GraphIR 后执行：

1. Resolve symbols and tensor specs.
2. Parse `tile_num` to task domains.
3. Parse edge maps to IndexMap.
4. Build CallNode dependency DAG from Event Tensor producers/consumers.
5. Infer each Event Tensor's shape and wait count.
6. Validate dtype, edge rank, affine map, no cycle, and uniform wait count.
7. Build WorkspaceSpec.

### 8.3 Static schedule construction

```text
topo_calls = topological_sort(CallNode DAG)
queue_builder = StaticQueueBuilder(num_workers)
for stage_id, call in enumerate(topo_calls):
    for linear_task_id in enumerate_task_domain(call.tile_shape):
        worker = assign_worker(call, linear_task_id, num_workers)
        queue_builder.append(worker, task_type=call.task_type, linear_task_id=linear_task_id)
Q_offsets, Q_tasks = queue_builder.finish()
```

第一版 static schedule 正式使用 materialized per-CTA task queues。对于 compile-time
constant task domain，`Q_offsets/Q_tasks` 可以作为常量或预填 workspace；对于 runtime
shape，framework launcher 在 launch 前生成 queue buffers。

可选 schedule config：

```python
StaticScheduleConfig(
    num_workers="auto",
    threads=32,
    wait_backoff=0,
    consumer_wait="all_threads",
    producer_notify="leader",
    queue_policy="round_robin",
)
```

### 8.4 TIRx emit

Emit 顺序：

1. Build PrimFunc signature from graph inputs, graph outputs, and WorkspaceSpec.
2. Emit `T.device_entry()`。
3. Emit `worker = T.cta_id([num_workers])`。
4. Emit `tx = T.thread_id([threads])`。
5. Emit `T.event_tensor(..., storage=...)` handles。
6. Emit per-CTA queue cursor:
   - `pos = Q_offsets[worker]`
   - `end = Q_offsets[worker + 1]`
7. Emit `while pos < end` loop。
8. Load `task_type` and `linear_task_id` from `Q_tasks`。
9. Emit dispatch branch for each task type:
   - unflatten task coordinate according to that task type's tile shape
   - emit input waits
   - emit TaskIR body
   - emit completion fence/barrier
   - emit output notifies
10. Increment `pos`。
11. Return TIRx PrimFunc.

## 9. Raw-sum lowering example

Graph:

```text
partial_sum: domain (n, 4), output B[n*32, 4], notify E[i]
final_sum:   domain (n),    input B, output C[n*32], wait E[i]
```

Inferred event:

```text
E.shape = (n,)
E.init = 4
E.storage = int32[n]
```

Generated static skeleton:

```python
@T.prim_func
def main(
    A: T.Buffer((n * 32, 128), "float32"),
    C: T.Buffer((n * 32,), "float32"),
    B: T.Buffer((n * 32, 4), "float32"),
    E_buf: T.Buffer((n,), "int32"),
    Q_offsets: T.Buffer((num_workers + 1,), "int32"),
    Q_tasks: T.Buffer((n * 5, 2), "int32"),
):
    T.device_entry()
    worker = T.cta_id([num_workers])
    tx = T.thread_id([32])
    E = T.event_tensor((n,), wait_count=4, storage=E_buf, name="row_done")

    pos = Q_offsets[worker]
    end = Q_offsets[worker + 1]
    while pos < end:
        task_type = Q_tasks[pos, 0]
        linear = Q_tasks[pos, 1]

        if task_type == TASK_PARTIAL:
            i = linear // 4
            j = linear % 4
            if tx < 32:
                row = i * 32 + tx
                acc = T.float32(0)
                for c in T.serial(32):
                    acc = acc + A[row, j * 32 + c]
                B[row, j] = acc
            T.cuda.thread_fence()
            T.tvm_storage_sync("shared")
            if tx == 0:
                T.evaluate(T.event_notify(E, i))

        elif task_type == TASK_FINAL:
            i = linear
            T.event_wait(E, i)
            T.tvm_storage_sync("shared")
            if tx < 32:
                row = i * 32 + tx
                acc = T.float32(0)
                for j in T.serial(4):
                    acc = acc + B[row, j]
                C[row] = acc

        pos = pos + 1
```

这个 skeleton 仍然是 raw sum 的结果，但它应由通用 pipeline 生成：

- `Q_tasks` 来自两个 `CallNode` 的 task domain。
- `n * 4` 和 `n` 来自 `tile_num`，用于 queue construction 和 unflatten。
- `i,j` 来自 unflatten。
- partial/final body 来自 `device_func` TaskIR。
- `E.init=4` 来自 `ij->i` producer multiplicity。
- wait/notify 来自 `in_edges/out_edges`。
- `while pos < end` 来自 per-CTA static queue ABI。

## 10. Validation rules

第一版 static lowering 应拒绝以下情况：

- `CallNode` dependency graph 有环。
- Event Tensor dtype 不是 `int32`。
- Event Tensor edge map 输出 rank 和 event rank 不一致。
- edge map 不是 affine map。
- wait count 不是 uniform `PrimExpr`。
- `device_func` body 写入未声明 output 或未授权 buffer。
- intermediate tensor shape/dtype 无法确定。
- task body 要求的 thread extent 和 call/device config 冲突。
- task domain 含无法 lower 为 TIRx `PrimExpr` 的 Python object。
- generated per-CTA queue 不满足 DAG stage order。
- `Q_offsets/Q_tasks` 的容量和 task record field 数与 schedule plan 不一致。

这些拒绝必须给出 graph node、event name、edge map 的错误信息，不能退回到固定模板。

## 11. Testing plan

### 11.1 Unit tests

- `graph_func` trace 能生成 GraphIR：
  - call 数量、task domains、tensor specs、event specs。
- edge map parser：
  - `"ij->i"`、`"ij->ij"`、callable IndexMap。
- wait count inference：
  - raw sum `ij->i` 得到 4。
  - 多 producer edge 得到 sum。
  - non-uniform case 报错。
- static queue planner：
  - multi-stage DAG 被 materialize 成 per-CTA queues。
  - 每个 CTA queue 保持 stage order。
  - `Q_offsets` 覆盖全部 task record 且没有重叠。
  - cycle 报错。
- device_func Tensor DSL：
  - raw-sum partial/final body 生成预期 TIRx stmt 结构。

### 11.2 Codegen tests

- raw-sum static graph 编译为 CUDA source。
- source 包含 global release atomic 和 acquire load。
- source 不应依赖 `_lower_static` 中固定的 `A/Y/P` 名称和固定两节点限制。
- 修改 device_func body 后，生成 source 随之变化。

### 11.3 Runtime tests

- raw sum 单 tile、多 tile correctness。
- 不同 `n` correctness。
- 多 Event Tensor DAG correctness。
- 重复 launch correctness：framework wrapper 必须重新初始化 event storage。
- 不同 `num_workers` correctness：每个 CTA 消费自己的 static queue。

## 12. Implementation milestones

1. GraphIR cleanup。
   - 让 `Tensor` 成为真实 graph value。
   - `call_device` 支持 `outputs=`。
   - edge map 解析为内部 IndexMap。

2. Event analysis。
   - producer/consumer collection。
   - uniform wait count inference。
   - WorkspaceSpec 生成。

3. Device body lowering。
   - 先支持 Tensor DSL raw-sum subset。
   - 同时保留 TIRx inline path。

4. Static TIRx emitter。
   - per-CTA static queue ABI。
   - `while pos < end` worker loop。
   - task-type dispatch。
   - generic wait/body/notify 插入。

5. Runtime wrapper。
   - 根据 WorkspaceSpec 分配 intermediate/event buffers。
   - launch 前初始化 event storage。
   - launch 前生成并填充 `Q_offsets/Q_tasks`。
   - 暴露低层 PrimFunc 和高层 runnable helper。

6. 删除或废弃固定 `_lower_static`。
   - 用新 static pipeline 支撑 fake/raw-sum example。
   - 保留旧路径只能作为临时 compatibility shim，并在测试中确保 device_func body
     真正影响生成代码。

## 13. 非目标

第一版 static 框架不解决：

- GPU dynamic ready queue。
- data-dependent task creation。
- non-uniform event wait count 的 device-side 初始化。
- 自动从 Relax graph 推导 Event Tensor graph。
- 任意 Python device_func 语义。
- 自动性能最优的 task placement。

这些都可以在 GraphIR、TaskIR 和 EventIR 稳定后继续扩展。

## 14. 核心结论

static megakernel 的通用性来自三点：

1. `device_func` 必须 lowering 成 TaskIR/TIRx body，而不是被忽略。
2. Event Tensor counter 必须从 `call_device` edge map 和 task domain 推导，而不是为
   raw sum 手写。
3. TIRx PrimFunc 必须由 call-node DAG 生成 per-CTA static task queues，并用统一的
   `while pos < end` worker loop dispatch task，而不是为 `_lower_static` 预置两段计算。

只要这三点成立，用户就可以写接近 `fake_example.py` 的 tile graph，由框架完成 static
megakernel 的其余部分。
