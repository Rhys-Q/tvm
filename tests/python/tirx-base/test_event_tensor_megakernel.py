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
"""Event Tensor and ready-queue megakernel smoke tests."""

import pytest

import tvm
from tvm.script import tirx as T

M = 4
N = 8
K_PARTS = 2
PARTIAL = 0
FINAL = 1


def _cuda_source(func) -> str:
    mod = tvm.compile(
        tvm.IRModule({"main": func}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
    )
    return mod.mod.imports[0].inspect_source()


def test_ready_queue_helper_codegen():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        Q = T.ready_queue(4, task_fields=3)
        if tx == 0:
            T.queue_push(Q, 1, 2, 3)
            while T.queue_live(Q) > 0:
                ok, op, x, y = T.queue_pop(Q)
                if ok:
                    A[0] = op + x + y
                    T.queue_finish(Q)

    script = main.script()
    assert "while" in script
    assert "buffer[v_1, 0] = 1" in script
    assert "buffer_3 = buffer_3 - 1" in script
    assert "main_kernel" in _cuda_source(main)


def test_ready_queue_rejects_bad_field_count():
    with pytest.raises(Exception, match="task fields"):
        @T.prim_func
        def main() -> None:
            T.device_entry()
            T.cta_id([1])
            Q = T.ready_queue(4, task_fields=2)
            T.queue_push(Q, 1, 2, 3)

        main.script()


def test_event_wait_backoff_codegen():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        E = T.event_tensor((1,), init=1)
        if tx == 0:
            T.event_init(E)
            T.evaluate(T.event_notify(E, 0))
            T.event_wait(E, 0, backoff=64)
            A[0] = 1

    src = _cuda_source(main)
    assert "atom.release.cta.shared.add.s32" in src
    assert "ld.acquire.cta.shared.s32" in src
    assert "__nanosleep" in src or "tvm_builtin_cuda_nano_sleep" in src


def test_raw_sum_static_shape_codegen():
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

    src = _cuda_source(row_sum_static)
    assert "atom.release.cta.shared.add.s32" in src
    assert "ld.acquire.cta.shared.s32" in src
    assert "for (int m" in src


def test_raw_sum_dynamic_shape_codegen():
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

    src = _cuda_source(row_sum_dynamic)
    assert "atom.release.cta.shared.add.s32" in src
    assert "while (1)" in src
    assert "op_ptr[0] == 0" in src


if __name__ == "__main__":
    tvm.testing.main()
