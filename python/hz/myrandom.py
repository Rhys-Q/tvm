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
# pylint: disable=missing-docstring
import subprocess
from pathlib import Path

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.relax.transform import AttachExternModules
from typing import List, Tuple
from tvm.runtime import NDArray, ShapeTuple, ndarray

def _infer_random(size, dtype):  # pylint: disable=invalid-name
    return nn.Tensor.placeholder(shape=size, dtype=dtype)


class Random(nn.Module):
    def __init__(self):
        self.source = Path(__file__).parent / "myrandom.cu"
        self.ext_mod = None

    def _get_ext_mod(self):
        if self.ext_mod is None:
            compile_options = nn.SourceModule.get_compile_options(source_format="cu")
            compile_options.append("-lcurand")
            self.ext_mod = nn.SourceModule(
                {
                    "relax_random_cuda": _infer_random,
                },
                source_code=self.source,
                compile_options=compile_options,
                source_format="cu",
                compiler="nvcc",
            )
            nn.add_extern(self.ext_mod)
        return self.ext_mod

    def forward(
        self, size, dtype, low=0, high=1
    ):  # pylint: disable=invalid-name
        if dtype == "float32":
            return self._get_ext_mod()["relax_random_cuda"](size, dtype)
        elif dtype == "int32":
            random_value = self._get_ext_mod()["relax_random_cuda"](size, "float32")
            value = random_value * (high - low) + low
            return value.astype("int32")
        else:
            raise NotImplementedError


def test_extern_source():
    from hz.pipeline.pipeline import _pipeline
    class TestModule(nn.Module):
        def __init__(self):
            self.random = Random()
            self.shape = [2,4]
            pass

        def main(
            self
        ):  # pylint: disable=invalid-name
            return self.random(self.shape , "int32", 0 ,9)

    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "main": {
                
            },
        },
        allow_extern=True,
    )
    # mod = AttachExternModules(ext_mods)(mod)  # pylint: disable=not-callable
    compiled = tvm.runtime.relax_vm.VirtualMachine(
        relax.build(mod, target="cuda", pipeline=_pipeline(ext_mods)),
        device=tvm.cuda(),
    )
    for i in range(5):
        compiled.set_input("main")
        compiled.invoke_stateful("main")
        tvm_res = compiled.get_outputs("main").numpy()
        print(tvm_res)


if __name__ == "__main__":
    test_extern_source()