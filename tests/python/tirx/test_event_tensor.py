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
"""Tests for TIRx Event Tensor script helpers."""

import tvm
from tvm.script import tirx as T


def test_event_tensor_script_expands_to_tirx_ir():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        E = T.event_tensor((1,), init=1, name="done")
        if tx == 0:
            T.event_init(E)
            if T.event_notify(E, 0):
                A[0] = 1
            T.event_wait(E, 0)

    script = main.script()
    assert "T.ptx.atom_scalar" in script
    assert "T.ptx.ld_acquire" in script
    assert "while" in script


def test_event_tensor_codegen_contains_atomic_and_acquire_load():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        E = T.event_tensor((1,), init=1, name="done")
        if tx == 0:
            T.event_init(E)
            if T.event_notify(E, 0):
                A[0] = 1
            T.event_wait(E, 0)

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "atom.release.cta.shared.add.s32" in src
    assert "ld.acquire.cta.shared.s32" in src


if __name__ == "__main__":
    tvm.testing.main()
