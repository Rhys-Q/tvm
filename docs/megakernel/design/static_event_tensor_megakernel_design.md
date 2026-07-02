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

本文只设计 static schedule 下的 Event Tensor megakernel。核心目标是：

- 用户用 `@graph_func` 写 tile-level DAG。
- 用户用 `@device_func` 写 tile task body。
- lowering 后生成一个 host+device mixed PrimFunc。
- host 部分负责准备 hidden runtime resources。
- device 部分是一个 fused megakernel，所有 device task body 都 inline 到同一个 task
  dispatch loop 中。

这不是把 raw sum 特化成一个预置 PrimFunc；也不是让用户手写完整 `@T.prim_func`。
`@graph_func` 是用户-facing 的完整计算入口，lowering 负责把它变成可编译执行的
TIRx host+device program。

## 1. Overall Lowering Shape

用户看到的接口：

```python
@graph_func
def main_graph(A: Tensor(...), B: Tensor(...)) -> Tensor(...):
    ...
    return D
```

lowering 后的概念结构：

```text
main_graph_host_device(A, B, D):
  # host region
  allocate hidden intermediate buffers
  allocate/init Event Tensor backing buffers
  build or bind static schedule data
  launch/fall into generated device megakernel region

  # device region
  worker = cta_id
  tx = thread_id
  while tile_scheduler.valid(worker):
      task_type, linear_task_id = tile_scheduler.get_task(worker)
      if task_type == TASK_0:
          wait input events
          inline device_func_0(...)
          notify output events
      elif task_type == TASK_1:
          wait input events
          inline device_func_1(...)
          notify output events
      tile_scheduler.next_tile(worker)
```

TIRx 后续 pipeline 再负责 host/device split、packed API、storage rewrite、buffer
flatten 等常规 lowering。megakernel framework 不重新实现这些已有 pass。

## 2. Programming Model

### 2.1 graph_func

`@graph_func` 描述完整 tile DAG。它是用户对外接口，只描述：

- graph input/output tensors。
- task families。
- task family 的 tile domain。
- task family 之间的 Event Tensor dependency。

`@graph_func` 不描述：

- hidden intermediate allocation。
- Event Tensor backing storage allocation。
- Event Tensor 初始化。
- static queue buffer layout。
- CUDA launch ABI。

这些由 lowering 的 host region 自动生成。

示例：

```python
@graph_func
def main_graph(A: Tensor((n * ROW_TILE, N_COLS))) -> Tensor((n * ROW_TILE,)):
    E = ETensor((n,), name="row_done")
    B = call_device(
        partial_sum,
        tile_num=(n, K_PARTS),
        args=[A],
        outputs=Tensor((n * ROW_TILE, K_PARTS), "float32"),
        out_edges={E: "ij->i"},
    )
    C = call_device(
        final_sum,
        tile_num=(n,),
        args=[B],
        outputs=Tensor((n * ROW_TILE,), "float32"),
        in_edges={E: "i->i"},
    )
    return C
```

### 2.2 device_func

`@device_func` 是 tile task body。它会自动把 Python 函数包装成 TIRx inline body：

```python
@device_func
def partial_sum(
    i: T.int32,
    j: T.int32,
    A: T.Buffer(...),
    B: T.Buffer(...),
    tx: T.int32,
):
    ...
```

语义：

- `@device_func` 登记 task body 和 graph-level 元信息。
- `@device_func` 内部应用 TIRx inline 包装，使 body 可在 megakernel dispatch branch 中展开。
- lowering 不解析 Python Tensor DSL。
- lowering 不从 `sum(A[...])` 或 Python slice 自动生成 task body。
- emitter 在 megakernel dispatch branch 中调用 inline body，由 TIRx parser 展开。

`device_func` 不是 host function，也不是独立 CUDA kernel。它最终只是 fused
megakernel 中的一个 task body fragment。

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

含义：

- `fn` 是 `@device_func` task body。
- `tile_num` 是 task domain。例如 `(n, 4)` 表示 task coordinate `(i, j)`。
- `args` 是 graph input 或上游 tensor。
- `outputs` 是该 task family 产生的 tensor。
- `in_edges` 是 task 执行前要等待的 Event Tensor。
- `out_edges` 是 task 完成后要通知的 Event Tensor。

## 3. Lowering Metadata

lowering 需要 Python 侧 metadata，但不需要新建计算 IR。计算 body 已经由 `@device_func`
表达。metadata 只记录 graph、event、schedule 和 hidden resource 信息。

### 3.1 GraphMetadata

```text
GraphMetadata {
  symbols: [n, ...]
  inputs: [TensorSpec]
  outputs: [TensorValue]
  tasks: [TaskSpec]
  events: [EventSpec]
}
```

```text
TaskSpec {
  name
  task_type
  inline_body
  tile_axes
  tile_shape
  inputs
  outputs
  in_edges
  out_edges
  threads
}
```

`inline_body` 是 `@device_func` 自动包装后的 TIRx inline function reference。emitter 用 tile coordinate、
buffer bindings 和 execution context 调用它。

### 3.2 EventPlan

```text
EventPlan {
  event_name
  shape
  wait_count
  backing_buffer
  init_policy
}
```

EventPlan 描述 Event Tensor 的 global counter tensor 以及初始化方式。它不是普通
storage planning；它表达 task dependency counter 的 runtime contract。

### 3.3 StaticSchedulePlan

```text
StaticSchedulePlan {
  num_workers
  task_record_layout = ["task_type", "linear_task_id"]
  queue_offsets
  queue_tasks
  queue_policy
}
```

static schedule 是 launch 前确定的。它可以由 host region 生成，也可以在完全静态时由
emitter 公式化或常量化。

### 3.4 RuntimeResourcePlan

```text
RuntimeResourcePlan {
  hidden_intermediates
  event_buffers
  static_schedule_buffers
  init_steps
}
```

这个 plan 只指导 host region 生成代码。它不替代 TIRx 现有 storage planning pass。
TIRx kernel 内的 local/shared temporary 仍由已有 pass 处理。

## 4. Event Tensor

Event Tensor 是 global `int32` counter tensor。每个元素代表一个 dependency event。

### 4.1 wait_count inference

`wait_count` 的含义是：某个 event 元素需要收到多少个 producer notify，consumer 才能
继续。

raw sum：

```python
E = ETensor((n,), name="row_done")

B = call_device(
    partial_sum,
    tile_num=(n, 4),
    out_edges={E: "ij->i"},
)

C = call_device(
    final_sum,
    tile_num=(n,),
    in_edges={E: "i->i"},
)
```

`partial_sum` 的 producer domain 是：

```text
(i, j), 0 <= i < n, 0 <= j < 4
```

edge map 是：

```text
(i, j) -> i
```

固定某个 `i`，有 4 个 producer task 会 notify `E[i]`：

```text
partial_sum(i, 0)
partial_sum(i, 1)
partial_sum(i, 2)
partial_sum(i, 3)
```

所以：

```text
wait_count(E[i]) = 4
```

host region 初始化：

```text
E_buf[:] = 4
```

device megakernel 中：

```text
partial_sum(i, j) 完成后 notify E[i]
final_sum(i) 执行前 wait E[i] == 0
```

### 4.2 Memory ordering

producer side：

```python
inline_task_body()
T.cuda.thread_fence()
T.tvm_storage_sync("shared")
if tx == 0:
    T.evaluate(T.event_notify(E, event_idx))
```

consumer side：

```python
T.event_wait(E, event_idx)
T.tvm_storage_sync("shared")
inline_task_body()
```

`event_notify` 使用 release atomic decrement；`event_wait` 使用 acquire load 轮询。

## 5. Static Schedule Lowering

static schedule 使用 per-CTA static task queue。每个 CTA 有自己的 `[begin, end)` task
record 区间。

### 5.1 Task record

真实 queue 不存 Python 对象，只存整数：

```text
Q_tasks[pos, 0] = task_type
Q_tasks[pos, 1] = linear_task_id
```

raw sum 中：

```text
TASK_PARTIAL = 0
TASK_FINAL = 1
```

`partial_sum(i, j)` 的 tile shape 是 `(n, 4)`：

```text
linear_task_id = i * 4 + j
i = linear_task_id // 4
j = linear_task_id % 4
```

`final_sum(i)` 的 tile shape 是 `(n,)`：

```text
linear_task_id = i
```

### 5.2 Queue buffers

```text
Q_offsets: int32[num_workers + 1]
Q_tasks:   int32[num_tasks, 2]
```

CTA `w` 执行：

```text
begin = Q_offsets[w]
end   = Q_offsets[w + 1]
Q_tasks[begin:end]
```

这些 buffers 可以是：

- host region launch 前生成并填充。
- 完全静态情况下由 emitter 硬编码或公式化。
- 未来优化中放进 packed runtime workspace。

### 5.3 Tile scheduler abstraction

`tile_scheduler` 是 per-CTA static queue 的轻量 wrapper，不是 dynamic scheduler。

```python
pos = Q_offsets[worker]
end = Q_offsets[worker + 1]
while pos < end:
    task_type = Q_tasks[pos, 0]
    linear = Q_tasks[pos, 1]
    ...
    pos += 1
```

也可以在 TIRx 中用 `@T.inline` helper 包装成：

```python
tile_scheduler = init_tile_scheduler(worker)
while tile_scheduler.valid():
    task_type, linear = tile_scheduler.get_task()
    ...
    tile_scheduler.next_tile()
```

两者语义相同。

### 5.4 Deadlock freedom

static queue 必须按 DAG topological order 构造。raw sum 中：

```text
partial_sum -> final_sum
```

因此每个 CTA queue 内，partial stage 必须排在 final stage 前。

否则可能所有 CTA 都先执行 final task，然后全部等待还没执行的 partial task，导致死锁。

### 5.5 Device megakernel shape

```python
@T.prim_func
def generated_device_megakernel(...):
    T.device_entry()
    worker = T.cta_id([num_workers])
    tx = T.thread_id([threads])

    pos = Q_offsets[worker]
    end = Q_offsets[worker + 1]
    while pos < end:
        task_type = Q_tasks[pos, 0]
        linear = Q_tasks[pos, 1]

        if task_type == TASK_PARTIAL:
            i = linear // 4
            j = linear % 4
            inline partial_sum(i, j, A, B_hidden, tx)
            notify E[i]

        elif task_type == TASK_FINAL:
            i = linear
            wait E[i]
            inline final_sum(i, B_hidden, C, tx)

        pos = pos + 1
```

实际产物不是单独暴露这个 kernel API，而是 host+device mixed PrimFunc 的 device region。

## 6. Host Region and Runtime Resources

`@graph_func` lowering 后不是一个纯 device kernel，而是 host+device mixed PrimFunc。

以 raw sum 为例，用户接口是：

```python
main_graph(A) -> C
```

lowering 后的 host region 负责：

```text
allocate output C
allocate hidden intermediate B
allocate Event Tensor backing buffer E_buf
initialize E_buf with wait_count = 4
build or bind Q_offsets/Q_tasks
enter/launch generated device megakernel
return C
```

### 6.1 B is hidden intermediate

`B` 是跨 task、跨 CTA 的 intermediate：

```text
partial_sum writes B
final_sum reads B
```

如果 producer 和 consumer 在不同 CTA，`B` 不能放在 shared memory。它需要 global
storage。但它不是用户参数，应由 host region 作为 hidden buffer 准备。

### 6.2 E_buf is hidden event storage

`E_buf` 是 Event Tensor 的 backing counter storage。它不是用户参数，也不是普通临时
buffer，因为每次 launch 前要按 wait_count 初始化。

raw sum：

```text
E_buf: int32[n]
E_buf[:] = 4
```

也可以选择在 device megakernel prologue 中初始化 `E_buf`，但需要额外 init flag/epoch
协议来避免其他 CTA 在初始化完成前开始 notify/wait。第一版推荐 host region 初始化。

### 6.3 Q_offsets/Q_tasks are schedule data

`Q_offsets/Q_tasks` 表示 static schedule。它们可以作为 hidden schedule buffers，也可以
在完全静态时硬编码。

通用第一版建议 host region 准备它们，因为：

- `n` 可能是 runtime shape。
- `num_workers` 可能依赖实际 target。
- schedule policy 可能随 target 改变。
- 不想为每个 schedule 重新编译 device kernel。

## 7. Matmul + Epilogue Example

用户 graph：

```python
@graph_func
def main_graph(A: Tensor((M, K)), B: Tensor((K, N))) -> Tensor((M, N)):
    E = ETensor((M // BLK_M, N // BLK_N), wait_count=1)
    C = call_device(
        matmul,
        tile_num=(M // BLK_M, N // BLK_N),
        args=[A, B],
        out_edges={E: "ij->ij"},
    )

    D = call_device(
        epilogue,
        tile_num=(M // BLK_M, N // BLK_N),
        args=[C],
        in_edges={E: "ij->ij"},
    )
    return D
```

generated device megakernel 的核心 dispatch：

```python
while tile_scheduler.valid():
    task_type, linear = tile_scheduler.get_task()

    if task_type == TASK_MATMUL:
        i, j = unflatten(linear, (M // BLK_M, N // BLK_N))
        inline matmul(i, j, A, B, C_hidden, tx)
        notify E[i, j]

    elif task_type == TASK_EPILOGUE:
        i, j = unflatten(linear, (M // BLK_M, N // BLK_N))
        wait E[i, j]
        inline epilogue(i, j, C_hidden, D, tx)

    tile_scheduler.next_tile()
```

host region 准备：

- hidden global intermediate `C_hidden`。
- hidden event storage `E_buf`。
- event init value `1`。
- static queues for `TASK_MATMUL` and `TASK_EPILOGUE`。

## 8. Lowering Pipeline

### 8.1 Frontend trace

```text
GraphFunction.trace(args)
  -> create TensorValue for graph inputs
  -> execute graph_func in trace mode
  -> call_device appends TaskSpec
  -> return GraphMetadata
```

`device_func` body 不在 trace 阶段执行。trace 阶段只记录 TIRx inline body reference。

### 8.2 Analysis

```text
resolve tensor specs
parse tile domains
parse Event Tensor edge maps
infer wait_count
build dependency DAG
validate acyclic graph
```

### 8.3 Static schedule

```text
topo_tasks = topological_sort(task family DAG)
for task_family in topo_tasks:
    enumerate task domain
    assign task records to per-CTA queues
```

Output:

```text
StaticSchedulePlan {
  num_workers
  Q_offsets
  Q_tasks
}
```

### 8.4 Host+device emit

Emitter 生成一个 host+device mixed PrimFunc：

```text
host region:
  allocate hidden buffers
  initialize event buffers
  prepare schedule buffers

device region:
  T.device_entry()
  fused persistent task loop
```

后续 TIRx pipeline 负责 split host/device 和常规 codegen。

## 9. Validation Rules

第一版应拒绝：

- `device_func` body 不能被 TIRx inline parser 接受。
- task family DAG 有环。
- Event Tensor dtype 不是 `int32`。
- edge map 输出 rank 和 Event Tensor rank 不一致。
- edge map 不是 affine map。
- wait_count 不能推导为 uniform `PrimExpr`。
- per-CTA static queue 不满足 DAG topological order。
- hidden intermediate shape/dtype 无法确定。
- task body thread extent 和 task family config 冲突。

## 10. Testing Plan

### 10.1 Unit tests

- graph trace 能生成 GraphMetadata。
- `@device_func` body 能被 task dispatch branch 调用并展开。
- edge map parser 支持 `"ij->i"` 和 `"ij->ij"`。
- raw sum wait_count 推导为 4。
- static queue planner 生成 per-CTA queues，并保持 topological order。

### 10.2 Codegen tests

- raw sum static graph 生成 host+device mixed PrimFunc。
- matmul + epilogue static graph 生成 host+device mixed PrimFunc。
- device source 包含 release atomic notify 和 acquire wait。
- 生成代码不依赖 `_lower_static` 中固定的 raw-sum 模板。

### 10.3 Runtime tests

- raw sum 单 tile、多 tile correctness。
- repeated launch correctness：host region 每次正确初始化 Event Tensor。
- 不同 `num_workers` correctness。
- matmul + epilogue correctness。

## 11. Implementation Milestones

1. Frontend metadata。
   - `Tensor` 成为 graph value。
   - `call_device` 记录 TaskSpec。
   - `@device_func` 自动包装 TIRx inline body。

2. Event analysis。
   - edge map 解析。
   - uniform wait_count inference。

3. Static scheduler。
   - per-CTA queue construction。
   - task record layout `[task_type, linear_task_id]`。
   - topological-order validation。

4. Host+device emitter。
   - host region hidden resource allocation/init。
   - device region fused persistent task loop。
   - task-type dispatch and inline body calls。
   - wait/notify insertion。

5. Integration with TIRx pipeline。
   - host/device split。
   - storage rewrite and buffer flatten。
   - CUDA codegen tests。

## 12. Non-goals

第一版不支持：

- dynamic ready queue。
- data-dependent task creation。
- arbitrary Python Tensor DSL device body。
- non-uniform per-element wait_count initialization inside device kernel。
- automatic Relax graph to Event Tensor graph extraction。
- optimal task placement or cost-model scheduling。

## 13. Summary

正确的 static Event Tensor megakernel 设计是：

```text
@graph_func
  -> host+device mixed PrimFunc
       host region:
         prepare hidden resources and static schedule
       device region:
         one fused megakernel
         all @device_func bodies inline into dispatch loop
```

用户只写 tile DAG 和 inline task body；framework 负责 hidden resource、event counter、
static queue 和 fused megakernel generation。
