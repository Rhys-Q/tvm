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

# 基于 TIRx PrimFunc 的 Event Tensor 设计方案

## 1. 背景和目标

Event Tensor 是 dynamic megakernel 中表达细粒度 task 依赖的抽象。论文
`docs/megakernel/2604.13327v2.pdf` 中的核心语义是：一个多维事件数组的每个元素
维护 wait count，producer task 完成后通知事件，consumer task 在事件满足后执行。

本设计选择 **不新增 graph function、TaskGraph、`mk.device_func`、`call_device` 或
`mk.workspace`**。第一版直接把 megakernel 写成普通 TIRx `PrimFunc`：

- task 计算用 `T.inline` 或普通 TIRx 代码片段表达。
- static schedule 用 `PrimFunc` 内的 schedule table、persistent loop 和 `if/switch`
  dispatch 表达。
- dynamic schedule 用 `PrimFunc` 内的 ready queue、event counter 和 persistent loop
  表达。
- Event Tensor 只作为 TIRx script/IR 中的依赖计数器抽象，后续 lowering 到 buffer、
  atomic、load/store 和 `while`。

这样做的好处是最小化新抽象：TIRx 现有 `PrimFunc` 已经能表达 host/device 混合代码、
`T.device_entry()` device region、buffer、控制流、内联 helper、CUDA atomic 和后端
intrinsic。Event Tensor 不需要承担 task graph 前端职责。

第一版目标：

- 在 TIRx `PrimFunc` 中支持 `T.event_tensor`、`T.event_wait`、`T.event_notify`。
- 支持 static schedule 和 dynamic schedule 两种写法。
- 用论文中的 row/raw sum 作为 demo。
- 保持与现有 TIRx lowering pipeline、script builder、printer 和测试风格一致。

## 2. 用户层 API

### 2.1 Event Tensor API

新增 TIRx script builder API，挂在 `tvm.script.tirx as T` 下：

```python
E = T.event_tensor(shape, init, dtype="int32", scope="global", name="row_done")
T.event_init(E)
T.event_reset(E, indices, value)
T.event_wait(E, indices)
is_last = T.event_notify(E, indices)
```

语义：

- `T.event_tensor(shape, init, ...)` 声明一个事件计数器 tensor。
- `shape` 是 event space 的形状，允许 `PrimExpr`。
- `init` 是默认初始计数，允许常量或 `PrimExpr`。
- `scope` 第一版只支持 `"global"`，因为跨 SM 依赖需要全局可见。
- `T.event_init(E)` 按 `init` 初始化整个 Event Tensor。
- `T.event_reset(E, idx, value)` 写入计数器，主要用于 kernel prologue 初始化。
- `T.event_wait(E, idx)` 等待 `E[idx] == 0`。
- `T.event_notify(E, idx)` 对 `E[idx]` 做 atomic decrement，并返回 decrement 后是否为
  0。返回 `True` 的执行者是最后一个 producer，可用于 dynamic schedule 触发后继 task。

第一版约束：

- Event Tensor 元素类型固定为 `int32`。
- 不支持同一 kernel 内多轮 reset/reuse 一个 event 元素。
- `wait` 和 `notify` 的索引必须与 `shape` 维度一致。
- event storage 默认由 lowering pass 生成，不作为用户可见普通 buffer 参数。

### 2.2 Inline Task 写法

不新增 `@mk.device_func`。task body 使用现有 `T.inline` 或直接写在 dispatch 分支里：

```python
@T.inline
def partial_sum(A, P, m_tile: T.int32, k_part: T.int32):
    ...


@T.inline
def final_sum(P, Y, m_tile: T.int32):
    ...
```

如果 task 很复杂，也可以保留为 private `PrimFunc`，后续通过现有 private function
inline pass 或普通 call lowering 处理。Event Tensor 设计本身不规定 task 表达方式。

### 2.3 Dynamic Queue API

dynamic schedule 第一版可以有两种实现方式：

- 最小实现：用户直接用普通 `T.alloc_buffer`、`T.cuda.atomic_add`、`T.ptx.atom_scalar`
  等写 queue。
- 便利实现：新增少量 `T.ready_queue`、`T.queue_push`、`T.queue_pop` helper。

建议第一版先做便利 API，但将其定义为普通 TIRx 内建 op，而不是 graph-level API：

```python
Q = T.ready_queue(capacity, task_fields=3, scope="global")
T.queue_push(Q, op_id, coord0, coord1)
ok, op_id, coord0, coord1 = T.queue_pop(Q)
live = T.queue_live(Q)
T.queue_finish(Q)
```

这些 API 只负责队列 lowering，不负责推导 task graph。

## 3. PrimFunc 代码形态

### 3.1 Static Schedule

static schedule 完全写在 `PrimFunc` 的 device region 中。schedule table 可以是：

- 编译期生成的 constant buffer。
- host 侧作为参数传入的 buffer。
- demo 第一版中直接用 loop 计算 `op_id/m/k`，避免额外常量表。

代码形态：

```python
@T.prim_func
def row_sum_static(A: T.Buffer((M, N), "float32"), Y: T.Buffer((M,), "float32")):
    T.device_entry()
    sm = T.cta_id([SM_COUNT])
    tx = T.thread_id([NUM_THREADS])

    E = T.event_tensor((M_TILES,), init=K_PARTS, name="row_done")
    P = T.alloc_buffer((M_TILES, K_PARTS), "float32", scope="global")

    T.event_init(E)

    for slot in T.serial(tasks_per_sm):
        op_id = ...
        m_tile = ...
        k_part = ...
        if op_id == PARTIAL:
            partial_sum(A, P, m_tile, k_part)
            T.event_notify(E, m_tile)
        else:
            T.event_wait(E, m_tile)
            final_sum(P, Y, m_tile)
```

`T.event_init(E)` 是 script 便利函数，可 lowering 成对所有 event 元素写入 `init`。也可
由 `T.event_tensor` 在 lowering 时自动插入初始化；第一版建议显式调用，避免隐藏
控制流和初始化位置。

### 3.2 Dynamic Schedule

dynamic schedule 也写在 `PrimFunc` 中。Event Tensor 只提供最后 producer 判定，是否
push 后继 task 由用户代码决定：

```python
@T.prim_func
def row_sum_dynamic(A: T.Buffer((M, N), "float32"), Y: T.Buffer((M,), "float32")):
    T.device_entry()
    sm = T.cta_id([SM_COUNT])
    tx = T.thread_id([NUM_THREADS])

    E = T.event_tensor((M_TILES,), init=K_PARTS, name="row_done")
    P = T.alloc_buffer((M_TILES, K_PARTS), "float32", scope="global")
    Q = T.ready_queue(capacity=M_TILES * K_PARTS + M_TILES, task_fields=3)

    T.event_init(E)
    if sm == 0 and tx == 0:
        for m in T.serial(M_TILES):
            for k in T.serial(K_PARTS):
                T.queue_push(Q, PARTIAL, m, k)

    while T.queue_live(Q) > 0:
        ok, op_id, m_tile, k_part = T.queue_pop(Q)
        if ok:
            if op_id == PARTIAL:
                partial_sum(A, P, m_tile, k_part)
                if T.event_notify(E, m_tile):
                    T.queue_push(Q, FINAL, m_tile, 0)
                T.queue_finish(Q)
            else:
                final_sum(P, Y, m_tile)
                T.queue_finish(Q)
```

实际实现中 `queue_live` 和 `queue_finish` 的精确定义需要避免 race。推荐语义：

- 每个 `queue_push` 增加 `pending_tasks`。
- 每个成功 pop 不改变 `pending_tasks`。
- task 完成后调用 `queue_finish` 减少 `pending_tasks`。
- loop 条件读取 `pending_tasks > 0`。

source task 初始化完成后需要一个 device-wide 可见的同步点。第一版可要求只用单 CTA
初始化并通过 global flag + acquire/release 让其他 CTA 等待，而不是依赖 CUDA grid
barrier。

## 4. IR 和 Lowering

### 4.1 新增 IR 节点

新增的 IR 应是 TIRx statement/expression 级别，而不是 graph-level function：

- `EventTensorNode`
  - 描述 event handle，字段包括 `shape`、`init`、`dtype`、`scope`、`name`。
  - Python 侧对象表现类似 Buffer handle，但只允许 event op 使用。
- `EventInitStmt`
  - 初始化整个 event tensor。
- `EventResetStmt`
  - 初始化单个 event 元素。
- `EventWaitStmt`
  - 等待单个 event 元素为 0。
- `EventNotify`
  - 表达式，返回 `bool`，表示当前 notify 是否将计数器减到 0。

如果为了少增节点，也可以将 `event_init/reset/wait/notify` 先实现为 `tirx.Call` 到
builtin op，再由 `LowerEventTensor` 识别。长期看，独立节点更利于 verifier、printer
和结构相等测试。

### 4.2 LowerEventTensor Pass

新增 `tirx.transform.LowerEventTensor()`，插入在 `LowerTIRx` 之前或作为
`LowerTIRx` 早期子步骤。职责：

- 为每个 `EventTensor` 分配 backing buffer：
  - `int32[event_numel]`。
  - 第一版使用 global workspace。
- 将多维 event index flatten 成线性 index。
- `EventInitStmt` lowering：
  - 生成初始化 loop。
  - 对 dynamic schedule 的 cross-CTA start flag 插入必要 acquire/release。
- `EventResetStmt` lowering：
  - `event_buf[flat_idx] = value`。
- `EventWaitStmt` lowering：
  - 生成 `while T.ptx.ld_acquire(event_buf + idx) != 0: pass`。
  - 后续可加入 backoff。
- `EventNotify` lowering：
  - 生成 atomic decrement。
  - 返回 `old_value == 1`。
  - 使用 release 语义保证 producer 写入对 consumer 可见。

### 4.3 Queue Lowering

若实现便利 queue API，新增 `LowerReadyQueue()`：

- `T.ready_queue(capacity, task_fields)` 分配：
  - `queue_storage[capacity, task_fields]`
  - `head`
  - `tail`
  - `pending_tasks`
- `queue_push`：
  - atomic add tail，写 task descriptor，再 release 增加 `pending_tasks`。
- `queue_pop`：
  - atomic add head 获取 slot。
  - 若 head 超过当前 tail，返回 `ok=False` 或回退重试。
- `queue_finish`：
  - release decrement `pending_tasks`。

第一版可以限制 queue capacity 静态可知，并在 verifier 中要求所有 fields 为 `int32`。

### 4.4 Verifier

新增 verifier 检查：

- `event_wait/notify/reset` 的 index 维度匹配 event shape。
- Event Tensor 只能在 `T.device_entry()` device region 中使用。
- `scope` 第一版只能是 `"global"`。
- `EventNotify` 的返回值只能作为 `PrimExpr(bool)` 使用，不能被 store 成非 bool。
- dynamic queue API 只能在 device region 中使用。
- `ready_queue` 的 task field 数和 `queue_push/pop` 参数个数一致。

## 5. Row/Raw Sum Demo

目标计算：

```text
Y[m] = sum_n A[m, n]
```

任务划分：

- `partial_sum(m_tile, k_part)` 计算一段 K/N 维度上的 partial sum，写入
  `P[m_tile, k_part]`。
- `final_sum(m_tile)` 等待该 row tile 的所有 partial 完成，读取 `P[m_tile, :]` 并写
  `Y[m_tile]`。

Event Tensor：

```text
E.shape = (M_TILES,)
E.init = K_PARTS
```

依赖逻辑在 PrimFunc 中显式表达：

```python
partial_sum(A, P, m, k)
if T.event_notify(E, m):
    # dynamic schedule: last producer releases final_sum
    T.queue_push(Q, FINAL, m, 0)
```

static schedule 中 final task 显式等待：

```python
T.event_wait(E, m)
final_sum(P, Y, m)
```

dynamic schedule 中 final task 只由最后一个 notify push，因此可以不再 wait；为了调试
和稳健性，也可保留 `T.event_wait(E, m)`，但第一版建议不保留，避免重复 spin。

## 6. 测试和验收标准

### 6.1 IR 和 Script 测试

新增测试：

- `tests/python/tirx/test_event_tensor.py`
  - 构造 `T.event_tensor`、`T.event_init`、`T.event_wait`、`T.event_notify`。
  - 检查 parser/printer roundtrip。
  - 检查结构相等和 script 输出。
- `tests/python/tirx/test_ready_queue.py`
  - 构造 `T.ready_queue`、`T.queue_push`、`T.queue_pop`。
  - 检查 task field 数不匹配时报错。

### 6.2 Lowering 测试

新增 transform 测试：

- `tests/python/tirx/transform/test_lower_event_tensor.py`
  - `event_init` lowering 到 `int32` buffer 初始化 loop。
  - `event_wait` lowering 到 acquire load + while。
  - `event_notify` lowering 到 atomic decrement，并返回 `old == 1`。
- `tests/python/tirx/transform/test_lower_ready_queue.py`
  - queue storage、head/tail/pending allocation。
  - push/pop/finish lowering 到 atomic 和 buffer access。

### 6.3 Demo 和 Runtime 测试

新增 row/raw sum demo：

- static 版本：`tests/python/tirx/test_event_tensor_row_sum_static.py`
- dynamic 版本：`tests/python/tirx/test_event_tensor_row_sum_dynamic.py`

验证：

- 对比 NumPy `A.sum(axis=1)`。
- 覆盖整除和非整除形状：
  - `M=128, N=1024, K_PARTS=4`
  - `M=257, N=1000, K_PARTS=8`
- 对 dynamic 版本检查 final task 数等于 `M_TILES`，避免重复 push。

最小命令：

```bash
export PYTHONPATH="$(pwd)/python:$(pwd)/.local/python"
python -m pytest tests/python/tirx/test_event_tensor.py -xvs
python -m pytest tests/python/tirx/transform/test_lower_event_tensor.py -xvs
```

若修改 C++ IR 或 codegen，需要运行：

```bash
cmake --build build --parallel
python -m pytest tests/python/tirx/transform/ -xvs
```

## 7. 分阶段实现

### 阶段 1：Event Tensor 最小闭环

- 新增 Event Tensor script API 和 IR 表达。
- 实现 `LowerEventTensor`。
- 实现 static row/raw sum demo。
- 不实现 ready queue helper，dynamic schedule 可先手写 queue buffer。

### 阶段 2：Ready Queue Helper

- 新增 `T.ready_queue`、`T.queue_push`、`T.queue_pop`、`T.queue_finish`。
- 实现 `LowerReadyQueue`。
- 实现 dynamic row/raw sum demo。

### 阶段 3：工程化和性能

- 增加 backoff、acquire/release 语义选择、queue capacity 检查。
- 支持 per-SM queue 或分片 queue。
- 支持 event storage 与 workspace memory planner 结合。
- 增加复杂 megakernel demo，例如 MoE-like dispatch。

## 8. 开放问题和默认选择

- Event Tensor 是否作为独立 IR 节点还是 builtin call：第一版建议独立节点，调试和验证更清楚。
- `T.event_tensor` 是否自动初始化：第一版要求显式 `T.event_init(E)`。
- dynamic queue 是否必须作为 API：第一阶段不必须，第二阶段作为便利 helper。
- static schedule 是否由 pass 自动生成：第一版不做，用户在 `PrimFunc` 中显式写 schedule
  loop；后续可以新增优化 pass 生成 schedule table。
- `P` 这样的跨 task 中间 buffer 不需要 `mk.workspace`，直接用现有 buffer/alloc 表达；
  如果 global-scope `AllocBuffer` 的 lowering 不满足需求，再扩展现有 buffer allocation，
  不新增 megakernel workspace API。

第一版默认：纯 `PrimFunc` 承载；只新增 Event Tensor 语义；queue helper 延后；row/raw
sum 先跑 static schedule，再补 dynamic schedule。
