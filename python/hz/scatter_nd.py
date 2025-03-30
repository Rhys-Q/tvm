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
import tempfile
from pathlib import Path

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.relax.transform import AttachExternModules


def _infer_scatter_nd(data, indices, updates):  # pylint: disable=invalid-name
    # assert isinstance(data, nn.Tensor)
    # assert isinstance(indices, nn.Tensor)
    # assert isinstance(updates, nn.Tensor)
    # # breakpoint()
    # # assert isinstance(mode, int)
    # assert updates.ndim == 1
    # assert updates.shape[0] == indices.shape[0]
    # assert updates.dtype == data.dtype

    # assert indices.ndim == 2
    # assert indices.shape[1] == data.ndim
    # assert indices.dtype == "int32"

    return nn.Tensor.placeholder(shape=data.shape, dtype=data.dtype)

    # Given updates with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), indices with shape
    # (M, Y_0, ..., Y_{K-1}), and output copied from data with shape (X_0, X_1, ..., X_{N-1}),
    # scatter_nd computes

    # .. code-block::

    #     output[indices[0, y_0, ..., y_{K-1}],
    #            ...,
    #            indices[M-1, y_0, ..., y_{K-1}],
    #            x_M,
    #            ...,
    #            x_{N-1}
    #           ] = f(output[...], updates[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}])


class ScatterNd(nn.Module):
    def __init__(self):
        self.source = Path(__file__).parent / "scatter_nd.cu"
        self.ext_mod = None

    def _get_ext_mod(self):
        if self.ext_mod is None:
            self.ext_mod = nn.SourceModule(
                {
                    "relax_scatter_nd_cuda": _infer_scatter_nd,
                },
                source_code=self.source,
                source_format="cu",
                compiler="nvcc",
            )
            nn.add_extern(self.ext_mod)
        return self.ext_mod

    def forward(
        self, data: nn.Tensor, indices: nn.Tensor, updates: nn.Tensor
    ):  # pylint: disable=invalid-name
        return self._get_ext_mod()["relax_scatter_nd_cuda"](data, indices, updates)


def test_extern_source():
    source = Path(__file__).parent / "scatter_nd.cu"

    class TestModule(nn.Module):
        def __init__(self):
            self.scatter_nd = ScatterNd()
            pass

        def main(
            self, data: nn.Tensor, indices: nn.Tensor, updates: nn.Tensor
        ):  # pylint: disable=invalid-name
            return self.scatter_nd(data, indices, updates)

    data_shape = [128, 4]
    indices_shape = [2, 10]
    updates_shape = [10]
    mod, _, ext_mods = TestModule().export_tvm(
        spec={
            "main": {
                "data": spec.Tensor(data_shape, "float32"),
                "indices": spec.Tensor(indices_shape, "int32"),
                "updates": spec.Tensor(updates_shape, "float32"),
            },
        },
        allow_extern=True,
    )
    mod = AttachExternModules(ext_mods)(mod)  # pylint: disable=not-callable
    compiled = tvm.runtime.relax_vm.VirtualMachine(
        relax.build(mod, target="cuda"),
        device=tvm.cuda(),
    )
    data = tvm.nd.array(
        10 * np.random.uniform(size=data_shape).astype("float32"), device=tvm.cuda()
    )
    updates = tvm.nd.array(
        np.random.uniform(size=updates_shape).astype("float32"), device=tvm.cuda()
    )
    indices = tvm.nd.array(
        np.random.randint(
            low=0,
            high=4,
            size=indices_shape,
        ).astype("int32"),
        device=tvm.cuda(),
    )
    compiled.set_input("main", data, indices, updates)
    compiled.invoke_stateful("main")
    tvm_res = compiled.get_outputs("main").numpy()

    def scatter_nd_numpy(data, indices, updates):
        assert indices.ndim == 2
        for d in range(indices.shape[1]):
            index = indices[:, d].tolist()
            data[*index] = updates[d]
        return data

    numpy_res = scatter_nd_numpy(data.numpy(), indices.numpy(), updates.numpy())
    np.testing.assert_allclose(numpy_res, tvm_res)


if __name__ == "__main__":
    test_extern_source()