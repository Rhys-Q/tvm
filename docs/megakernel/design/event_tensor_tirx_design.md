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

# 基于 TIRx 的 Event Tensor 设计方案

## 1. 目标

本文设计如何在 TVM TIRx 中实现论文
`docs/megakernel/2604.13327v2.pdf` 的 Event Tensor 抽象，并支持
`docs/megakernel/tasks/fake_example.py` 中的 raw sum graph 编译为单个
megakernel。设计目标是：

- 用 Event Tensor 表达 tile task 之间的细粒度依赖。
- 支持 static schedule：任务队列编译期或 host 侧预先确定。
- 支持 dynamic schedule：GPU 端 ready queue 根据 Event Tensor 触发后继任务。
- 将 `graph_func`、`device_func`、`call_device` 映射到 TIRx 可实现的 API 和 lowering。
- 明确 Event Tensor 的存储、同步、运行时状态和测试验收方式。

本文不要求第一版自动从任意 Relax/TIR graph 推导 Event Tensor graph。第一版以显式
Event Tensor graph 输入为边界，先打通 task graph 到 persistent TIRx megakernel 的设计。

## 2. 编程模型

Event Tensor 程序由三层概念组成。

### 2.1 Graph Function

`graph_func` 描述完整 megakernel 的 tile-level dataflow graph。它不是普通逐 op
host launch graph，而是一个待 lowering 的设备端 persistent kernel 模板。

在 TIRx 中，`graph_func` lowering 为一个 `@T.prim_func`：

- 函数参数是 input/output tensor 以及必要的 workspace。
- 函数体包含 `T.device_entry()`。
- kernel 内部创建 Event Tensor、task queue、intermediate tensor。
- kernel 内部执行 persistent loop，并 dispatch 到不同 tile task。

### 2.2 Device Function

`device_func` 表示 tile task body。每个 task 由 task coordinate 标识，例如
`partial_sum(i, j)` 或 `final_sum(i)`。task body 可以用三种方式在 TIRx 中表达：

- `T.inline` helper：适合第一版 raw sum demo。
- private TIRx PrimFunc：适合较复杂 tile primitive，lowering 前 inline 到 fused kernel。
- dispatch 分支内直接写 task body：适合生成代码最简单的原型。

第一版推荐使用 `T.inline` 或 dispatch 分支内代码，避免新增独立 device function 调用约定。

### 2.3 Call Device

`call_device(fn, tile_num, args, in_edges, out_edges)` 表示创建一组 tile tasks。

Lowering 时为每个 `call_device` 分配一个 `task_type`，并记录：

- `tile_num`：task grid shape。
- `fn`：task body。
- `in_edges`：task 执行前需要等待的 Event Tensor。
- `out_edges`：task 完成后需要通知的 Event Tensor。
- coordinate mapping：从 task coordinate 到 event coordinate 的映射。

例如：

```python
out_edges={E: "ij->i"}
in_edges={E: "i->i"}
```

表示 `partial_sum(i, j)` 完成后通知 `E[i]`，`final_sum(i)` 执行前等待 `E[i]`。

## 3. Event Tensor 语义

Event Tensor 是多维 `int32` counter tensor。每个元素代表一个事件，事件的 counter
记录该事件还需要等待多少 producer task 完成。

### 3.1 API

TIRx script 暴露以下 API：

```python
E = T.event_tensor(
    shape,
    wait_count,
    dtype="int32",
    scope="global",
    name=None,
    storage=None,
)

T.event_init(E)
T.event_reset(E, indices, value)
T.event_wait(E, indices, backoff=0)
is_triggered = T.event_notify(E, indices)
```

语义如下：

- `T.event_tensor` 声明一个 Event Tensor handle。
- `shape` 是 event space shape，允许包含 symbolic shape。
- `wait_count` 是默认初始 counter，可以是常量或 `PrimExpr`。
- `storage` 是可选 backing buffer。正式 global Event Tensor 必须传入 global buffer；
  不传时只允许作为 single-CTA shared-memory prototype。
- `T.event_init(E)` 初始化所有 event 元素为默认 `wait_count`。
- `T.event_reset(E, indices, value)` 设置单个 event counter，用于 data-dependent wait count。
- `T.event_wait(E, indices)` 等待目标 counter 到 0。
- `T.event_notify(E, indices)` 对目标 counter 做 atomic decrement，并返回是否触发事件。

`event_notify` 的返回值定义为 `old_value == 1`。返回 true 的 task 是最后一个 producer，
dynamic schedule 可用它 push 后继 consumer tasks。

### 3.2 存储位置

正式语义要求 Event Tensor 存放在 global memory。

原因：

- Event Tensor 的目标是表达跨 CTA/SM 的 tile task 依赖。
- static schedule 中一个 SM 的 consumer 可能等待另一个 SM 的 producer。
- dynamic schedule 中任意 SM 都可能 notify 或消费同一个 event。
- shared memory 只在单 CTA 内可见，不能作为跨 SM 依赖的正确实现。

CUDA codegen 不允许 kernel 内部动态分配 global memory，因此第一版正式 lowering 使用由调用方
传入的 global backing buffer：

```text
E_storage: int32[num_events]
```

多维 event index flatten 到线性 index：

```text
flat = (((i0 * shape1) + i1) * shape2 + i2) ...
```

不显式传入 `storage` 时，TIRx helper 可以分配 shared memory 支撑 single-CTA demo，但文档
和最终实现不把 shared memory 作为 Event Tensor 的正式语义存储。shared memory 只能作为
局部优化，例如同 CTA 内 event 或 global counter 的 cache，但必须保持 global 语义等价。

### 3.3 同步方式

Event Tensor 同步需要满足 producer 写入对 consumer 可见：

- producer task 先写普通 tensor/intermediate buffer。
- producer task 执行 `event_notify`。
- consumer task 的 `event_wait` 返回后才能读 producer 输出。

Lowering 约定：

- `event_notify` 使用 release 语义的 atomic decrement。
- `event_wait` 使用 acquire 语义 load 轮询 counter。
- counter 到 0 后，consumer 可见 producer 在 notify 前完成的写入。

CUDA lowering 形态：

```text
old = atomic_add_release(event_ptr, -1)
is_triggered = (old == 1)

while load_acquire(event_ptr) != 0:
    optional_backoff()
```

`event_reset` 和 `event_init` 必须发生在任何相关 producer/consumer task 访问 event
之前。多 CTA 初始化时需要使用单 CTA 初始化加发布标志，或由 host/workspace 初始化完成后
再 launch megakernel。第一版建议在 kernel prologue 中由单 CTA 初始化，并通过 global
start flag 让其他 CTA 等待初始化完成。

## 4. Ready Queue 语义

Dynamic schedule 需要 GPU 端 ready queue。TIRx script 暴露以下 API：

```python
Q = T.ready_queue(
    capacity,
    task_fields,
    scope="global",
    storage=None,
    head=None,
    tail=None,
    pending=None,
    lock=None,
)
T.queue_push(Q, task_type, *coords)
ok, task_type, *coords = T.queue_pop(Q)
live = T.queue_live(Q)
T.queue_finish(Q)
```

第一版 queue descriptor 全部为 `int32`：

```text
storage[capacity, task_fields]
head
tail
pending_tasks
lock
```

语义：

- `queue_push` 写入 task descriptor，并增加 `pending_tasks`。
- `queue_pop` 从 queue 中取一个 task descriptor，成功时返回 `ok=True`。
- `queue_finish` 在 task body 和所有后继 push 完成后减少 `pending_tasks`。
- `queue_live` 返回未完成 task 数，persistent loop 使用 `queue_live(Q) > 0` 判断退出。
- `lock` 是第一版 centralized queue 的互斥锁，用 atomicCAS 获取和释放；后续可替换为
  lock-free 或 per-SM queue。

第一版使用 centralized global queue，优点是实现简单和语义清楚，缺点是 lock 会限制高并发
性能。后续可优化为 per-SM queue、work stealing、分片 queue 或 lock-free queue，但不改变
上层 Event Tensor graph 语义。

## 5. Static Schedule Lowering

Static schedule 在编译期或 host 侧预先生成每个 SM 的 task queue。每个 CTA/SM 在
megakernel 内执行自己的 persistent loop。

### 5.1 Lowering 输入

输入是 tile-level graph：

```text
TaskGrid {
  task_type
  tile_num
  device_func
  in_edges
  out_edges
}
```

每条 edge 包含：

```text
event tensor
producer/consumer task type
coordinate map
```

### 5.2 Static Task Queue

生成：

```text
static_tasks[num_tasks] = {
  task_type,
  coord_offset,
}

static_coords[num_coords]

static_queues[sm_count] = {
  begin,
  end,
}
```

最简单策略是按 logical schedule round-robin 分配给 SM。后续可以使用 cost model 或
polyhedral schedule 优化分配。

### 5.3 Kernel 形态

```python
@T.prim_func
def megakernel(A, C, workspace):
    T.device_entry()
    sm = T.cta_id([SM_COUNT])
    tx = T.thread_id([THREADS])

    E = T.event_tensor((...), wait_count=..., storage=E_storage)
    T.event_init(E)
    T.grid_init_barrier()

    task_pos = static_queues[sm].begin
    while task_pos < static_queues[sm].end:
        task_type, coords = load_static_task(task_pos)

        if task_type == PARTIAL:
            partial_sum(coords...)
            T.event_notify(E, event_index_from_partial(coords))

        elif task_type == FINAL:
            T.event_wait(E, event_index_from_final(coords))
            final_sum(coords...)

        task_pos += 1
```

实际 pass 应按统一规则插入 wait/notify：

- 对 `in_edges`：task body 前插入 `event_wait`。
- 对 `out_edges`：task body 后插入 `event_notify`。

static schedule 的特点：

- 调度开销低。
- 适合 regular workload。
- 如果 consumer 被排到较早位置，可能 spin wait；但其他 SM 仍可继续执行自己的 queue。
- 对 data-dependent workload 只能保守化处理，或改用 dynamic schedule。

## 6. Dynamic Schedule Lowering

Dynamic schedule 不预先固定每个 SM 的完整任务序列，而是在 GPU 端根据 ready queue 动态取任务。

### 6.1 基本策略

- 初始化 source tasks 到 ready queue。
- 每个 CTA/SM 在 persistent loop 中 pop task。
- task 执行完后 notify out events。
- 当 event counter 到 0 时，push 对应 consumer tasks。
- `pending_tasks` 到 0 后所有 CTA/SM 退出。

### 6.2 Kernel 形态

```python
@T.prim_func
def megakernel_dynamic(A, C, workspace):
    T.device_entry()
    sm = T.cta_id([SM_COUNT])
    tx = T.thread_id([THREADS])

    E = T.event_tensor((...), wait_count=..., storage=E_storage)
    Q = T.ready_queue(
        capacity=...,
        task_fields=...,
        storage=Q_storage,
        head=Q_head,
        tail=Q_tail,
        pending=Q_pending,
        lock=Q_lock,
    )

    if sm == 0 and tx == 0:
        T.event_init(E)
        for each source task:
            T.queue_push(Q, task_type, coords...)
        T.publish_init_done()

    T.wait_init_done()

    while T.queue_live(Q) > 0:
        ok, task_type, coords = T.queue_pop(Q)
        if ok:
            if task_type == PARTIAL:
                partial_sum(coords...)
                if T.event_notify(E, event_index_from_partial(coords)):
                    T.queue_push(Q, FINAL, consumer_coords...)
                T.queue_finish(Q)

            elif task_type == FINAL:
                final_sum(coords...)
                T.queue_finish(Q)
```

Dynamic schedule 中 consumer task 是由最后一个 producer 的 notify 触发入队，因此正常情况
下 consumer task body 前不需要再次 wait。为了支持 early-push 优化，可以允许 consumer
提前入队，但这种模式必须在 consumer body 前保留 `event_wait`。

### 6.3 Early Push 优化

论文中的 early-push 策略用于隐藏 scheduler push 开销：

- 不等 producer task 完成后才 push consumer。
- 当所有 producer task 已经 dispatch 到 SM 后，可以提前 push consumer。
- consumer 执行前仍然执行 `event_wait`，保证数据依赖正确。

第一版实现可不启用 early-push。文档和接口要预留该模式：

```text
trigger policy:
  complete_push: notify counter 到 0 时 push
  early_push: producer dispatch counter 到 0 时 push，consumer 执行前 wait
```

## 7. Raw Sum 示例

任务文件中的 graph：

```python
class IRModule:
    @device_func
    def partial_sum(i: int, j: int, A: Tensor, B: Tensor):
        B[i*32: i*32 + 32, j] = sum(A[i*32: i*32 + 32, j*32:j*32+32])

    @device_func
    def final_sum(i: int, B: Tensor, C: Tensor):
        C[i*32: i*32+32] = sum(B[i*32: i*32+32, :])

    @graph_func
    def main_graph(A: Tensor(("n*32", 128))) -> Tensor(("n*32",)):
        n = sym_var()
        E = ETensor((n,), wait_count=4)
        B = call_device(
            IRModule.partial_sum,
            tile_num=(n, 4),
            args=[A],
            in_edge={},
            out_edges={E: "ij->i"},
        )
        C = call_device(
            IRModule.final_sum,
            tile_num=(n,),
            args=[B],
            in_edges={E: "i->i"},
            out_edges={},
        )
        return C
```

### 7.1 Event Tensor

```text
E.shape = (n,)
E.wait_count = 4
```

每个 `E[i]` 等待四个 producer：

```text
partial_sum(i, 0)
partial_sum(i, 1)
partial_sum(i, 2)
partial_sum(i, 3)
```

当四个 producer 都完成后，`final_sum(i)` 可以执行。

### 7.2 Intermediate Tensor

`B` 是跨 task intermediate tensor：

```text
B.shape = (n * 32, 4)
```

正式实现中 `B` 需要 global/workspace allocation，因为 `partial_sum` 和 `final_sum` 可在
不同 CTA/SM 上执行。single-CTA demo 可以把 `B` 放 shared memory，但这不是通用语义。

### 7.3 Static Lowering 伪代码

```python
@T.prim_func
def raw_sum_static(A, C, workspace):
    T.device_entry()
    sm = T.cta_id([SM_COUNT])
    tx = T.thread_id([THREADS])

    E = T.event_tensor((n,), wait_count=4, storage=E_storage)
    B = B_workspace

    T.event_init(E)
    T.grid_init_barrier()

    while static_scheduler_valid(sm):
        task_type, i, j = static_scheduler_get(sm)

        if task_type == PARTIAL:
            partial_sum(i, j, A, B)
            T.event_notify(E, i)

        elif task_type == FINAL:
            T.event_wait(E, i)
            final_sum(i, B, C)

        static_scheduler_next(sm)
```

### 7.4 Dynamic Lowering 伪代码

```python
@T.prim_func
def raw_sum_dynamic(A, C, workspace):
    T.device_entry()
    sm = T.cta_id([SM_COUNT])
    tx = T.thread_id([THREADS])

    E = T.event_tensor((n,), wait_count=4, storage=E_storage)
    B = B_workspace
    Q = T.ready_queue(
        capacity=n * 5,
        task_fields=3,
        storage=Q_storage,
        head=Q_head,
        tail=Q_tail,
        pending=Q_pending,
        lock=Q_lock,
    )

    if sm == 0 and tx == 0:
        T.event_init(E)
        for i in T.serial(n):
            for j in T.serial(4):
                T.queue_push(Q, PARTIAL, i, j)
        T.publish_init_done()

    T.wait_init_done()

    while T.queue_live(Q) > 0:
        ok, task_type, i, j = T.queue_pop(Q)
        if ok:
            if task_type == PARTIAL:
                partial_sum(i, j, A, B)
                if T.event_notify(E, i):
                    T.queue_push(Q, FINAL, i, 0)
                T.queue_finish(Q)

            elif task_type == FINAL:
                final_sum(i, B, C)
                T.queue_finish(Q)
```

## 8. TIRx 实现路线

### 8.1 前端表示

新增或规范化以下 Python-side object：

```text
EventTensor {
  shape
  wait_count
  dtype
  scope
  name
  backing_buffer
}

ReadyQueue {
  capacity
  task_fields
  scope
  storage
  head
  tail
  pending_tasks
  lock
}
```

第一版可以把 Event Tensor 和 ReadyQueue 表达为带 annotation 的 `Buffer`，由 lowering pass
识别。长期更合适的方式是独立 TIRx IR node，因为 verifier、printer、结构相等和错误诊断
会更清晰。

### 8.2 LowerEventTensor

新增 `tirx.transform.LowerEventTensor()`：

- 为每个 Event Tensor 创建 global `int32` backing buffer。
- 处理 symbolic shape 的 numel 计算。
- flatten 多维 event index。
- lowering `event_init` 到初始化 loop。
- lowering `event_reset` 到 counter store。
- lowering `event_wait` 到 acquire load spin loop。
- lowering `event_notify` 到 release atomic decrement 和 `old == 1`。

### 8.3 LowerReadyQueue

新增 `tirx.transform.LowerReadyQueue()`：

- 创建 global queue storage、head、tail、pending。
- lowering `queue_push` 到 lock acquire、descriptor store、tail/pending update、lock release。
- lowering `queue_pop` 到 lock acquire、head/tail check、head update、lock release。
- lowering `queue_finish` 到 pending decrement。
- lowering `queue_live` 到 acquire load pending。

### 8.4 Graph-to-Megakernel Pass

新增 graph-level transformation，输入是显式 Event Tensor graph：

```text
LowerEventTensorGraph(schedule="static" | "dynamic")
```

static 模式：

- 枚举所有 task instances。
- 生成 static task queue。
- 生成 dispatch loop。
- 插入 wait/body/notify。

dynamic 模式：

- 生成 source task 初始化逻辑。
- 生成 ready queue persistent loop。
- 在 notify trigger 处插入后继 task push。
- 支持 complete-push，预留 early-push。

## 9. 校验和错误处理

Verifier 应检查：

- Event Tensor dtype 必须是 `int32`。
- Event Tensor 正式 scope 必须是 `global`。
- `event_wait/notify/reset` 的 indices 数量必须等于 event rank。
- `wait_count` 非负。
- 每个 event 的 `wait_count` 与 producer 数一致，除非用户显式标记为 runtime init。
- dynamic queue 的 `task_fields` 与 push/pop 解包数量一致。
- `queue_finish` 必须在成功 pop 的 task 路径上执行。
- dynamic schedule 的 queue capacity 必须能容纳最坏情况下的 live task，或插入 overflow assert。
- 跨 task intermediate tensor 不能放在 shared memory，除非证明所有 producer/consumer 在同一 CTA。

## 10. 测试计划

### 10.1 Script 和 IR 测试

- 构造 `T.event_tensor`、`T.event_init`、`T.event_wait`、`T.event_notify`。
- 检查 index rank mismatch 报错。
- 检查 dtype/scope 限制报错。
- 检查 `T.ready_queue` 的 field 数校验。
- 检查 script printer roundtrip。

### 10.2 Lowering 测试

- `event_init` lowering 为 global counter 初始化。
- `event_notify` lowering 为 release atomic decrement。
- `event_wait` lowering 为 acquire load spin wait。
- `queue_push/pop/finish/live` lowering 为 global queue 操作。
- dynamic complete-push 中只有最后一个 producer push consumer。

### 10.3 Raw Sum 运行测试

新增或完善：

```text
docs/megakernel/examples/raw_sum_event_tensor.py
tests/python/tirx/test_event_tensor_raw_sum.py
```

测试场景：

- static schedule raw sum。
- dynamic schedule raw sum。
- `n=1`、`n=8`、非整除边界形状。
- 对比 NumPy `A.sum(axis=1)`。
- dynamic 下检查每个 `final_sum(i)` 只执行一次。

### 10.4 环境说明

当前本地 TIRx CUDA 测试需要 sm_100a。若机器不具备该硬件，相关测试会 skip。最终验收应在
sm_100a CUDA 环境运行：

```bash
export PYTHONPATH="$(pwd)/python:$(pwd)/.local/python"
python -m pytest tests/python/tirx/test_event_tensor.py -q -rs
python -m pytest tests/python/tirx-base/test_event_tensor_megakernel.py -q -rs
python docs/megakernel/examples/raw_sum_event_tensor.py
```

## 11. 分阶段交付

第一阶段：Event Tensor 最小闭环

- 支持 global Event Tensor backing buffer。
- 支持 init/reset/wait/notify。
- 支持 static raw sum。
- 完成 lowering 和基础 verifier。

第二阶段：Dynamic Ready Queue

- 支持 global ready queue。
- 支持 complete-push dynamic schedule。
- 支持 dynamic raw sum。
- 增加 queue overflow/assert 和 pending liveness 校验。

第三阶段：Graph-to-Megakernel 自动化

- 支持从 `graph_func`/`device_func`/`call_device` 表示生成 TIRx megakernel。
- 支持 einsum-like coordinate mapping。
- 支持 symbolic shape。
- 支持 data-dependent event reset 和 task triggering。

第四阶段：性能优化

- early-push。
- per-SM queue / work stealing。
- backoff 策略。
- event storage 分片。
- shared-memory cache 或 warp-level cooperative wait/notify。

## 12. 设计结论

Event Tensor 在 TIRx 中应作为一等依赖同步抽象，而不是普通 host-side task graph。核心实现是
把 event 元素 lowering 为 global `int32` counter，并用 release atomic decrement 与 acquire
spin wait 表达 producer-consumer 依赖。static schedule 通过预计算 per-SM queue 最小化调度
开销；dynamic schedule 通过 GPU ready queue 在运行时触发后继 task，支持 shape 和
data-dependent dynamism。

对 `fake_example.py` 的 raw sum，`partial_sum(i, j)` 到 `final_sum(i)` 的依赖可完整表达为
`E[i]` 的 wait-count counter：四个 partial task 分别 notify，同一个 row 的 final task 在
counter 到 0 后执行。该设计能覆盖任务要求的 static 和 dynamic 两种 megakernel 编译运行路径。
