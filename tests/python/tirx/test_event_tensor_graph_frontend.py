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
"""Tests for the lightweight Event Tensor graph frontend."""

import importlib.util
from pathlib import Path

import tvm


def _load_fake_example():
    path = Path("docs/megakernel/tasks/fake_example.py")
    spec = importlib.util.spec_from_file_location("fake_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cuda_source(func) -> str:
    mod = tvm.compile(
        tvm.IRModule({"main": func}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
    )
    return mod.mod.imports[0].inspect_source()


def test_fake_example_traces_event_tensor_graph():
    fake = _load_fake_example()
    graph = fake.trace_graph()
    assert [call.device_func.name for call in graph.calls] == ["partial_sum", "final_sum"]
    assert next(iter(graph.calls[0].out_edges.values())) == "ij->i"
    assert next(iter(graph.calls[1].in_edges.values())) == "i->i"


def test_fake_example_lowers_static_and_dynamic_event_tensor_graph():
    fake = _load_fake_example()

    static_src = _cuda_source(fake.build_static(1))
    assert "atom.release.gpu.global.add.s32" in static_src
    assert "ld.acquire.gpu.global.s32" in static_src

    dynamic_src = _cuda_source(fake.build_dynamic(1))
    assert "atom.release.gpu.global.add.s32" in dynamic_src
    assert "atomicCAS" in dynamic_src

    early_src = _cuda_source(fake.build_dynamic(1, early_push=True))
    assert "ld.acquire.gpu.global.s32" in early_src


if __name__ == "__main__":
    tvm.testing.main()
