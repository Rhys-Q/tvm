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
"""Runnable Event Tensor raw-sum example with NumPy precision checking.

This is a small single-CTA example for the current Event Tensor prototype.
The logical Event Tensor / ReadyQueue API uses ``scope="global"``, while the
current implementation backs it with shared memory so it can run through the
existing CUDA backend.
"""

import numpy as np

import tvm
from tvm.script import tirx as T


M = 4
N = 8
K_PARTS = 2
PARTIAL = 0
FINAL = 1


@T.prim_func
def row_sum_static(A: T.Buffer((M, N), "float32"), Y: T.Buffer((M,), "float32")):
    T.device_entry()
    T.cta_id([1])
    tx = T.thread_id([32])
    E = T.event_tensor((M,), init=K_PARTS, name="row_done")
    P = T.alloc_buffer((M, K_PARTS), "float32", scope="shared")
    if tx == 0:
        T.event_init(E)
        for m in T.serial(M):
            for k in T.serial(K_PARTS):
                acc = T.float32(0)
                for n in T.serial(N // K_PARTS):
                    acc = acc + A[m, k * (N // K_PARTS) + n]
                P[m, k] = acc
                T.evaluate(T.event_notify(E, m))
        for m in T.serial(M):
            T.event_wait(E, m)
            acc = T.float32(0)
            for k in T.serial(K_PARTS):
                acc = acc + P[m, k]
            Y[m] = acc


@T.prim_func
def row_sum_dynamic(A: T.Buffer((M, N), "float32"), Y: T.Buffer((M,), "float32")):
    T.device_entry()
    T.cta_id([1])
    tx = T.thread_id([32])
    E = T.event_tensor((M,), init=K_PARTS, name="row_done")
    P = T.alloc_buffer((M, K_PARTS), "float32", scope="shared")
    Q = T.ready_queue(M * K_PARTS + M, task_fields=3)
    if tx == 0:
        T.event_init(E)
        for m in T.serial(M):
            for k in T.serial(K_PARTS):
                T.queue_push(Q, PARTIAL, m, k)
        while T.queue_live(Q) > 0:
            ok, op, m, k = T.queue_pop(Q)
            if ok:
                if op == PARTIAL:
                    acc = T.float32(0)
                    for n in T.serial(N // K_PARTS):
                        acc = acc + A[m, k * (N // K_PARTS) + n]
                    P[m, k] = acc
                    if T.event_notify(E, m):
                        T.queue_push(Q, FINAL, m, 0)
                    T.queue_finish(Q)
                else:
                    acc = T.float32(0)
                    for kk in T.serial(K_PARTS):
                        acc = acc + P[m, kk]
                    Y[m] = acc
                    T.queue_finish(Q)


def _build(func):
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")


def _run_one(name, func, dev, data, ref):
    rt_mod = _build(func)
    a_dev = tvm.runtime.tensor(data, dev)
    y_dev = tvm.runtime.empty((M,), dtype="float32", device=dev)
    rt_mod(a_dev, y_dev)
    got = y_dev.numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
    print(f"{name}:")
    print(f"  got      = {got}")
    print(f"  expected = {ref}")
    print(f"  max_diff = {np.max(np.abs(got - ref))}")


def main():
    dev = tvm.device("cuda", 0)
    if not dev.exist:
        raise RuntimeError("CUDA device is required to run this example")

    data = np.linspace(0, 1, M * N, dtype="float32").reshape(M, N)
    ref = np.sum(data, axis=1)
    _run_one("row_sum_static", row_sum_static, dev, data, ref)
    _run_one("row_sum_dynamic", row_sum_dynamic, dev, data, ref)


if __name__ == "__main__":
    main()
