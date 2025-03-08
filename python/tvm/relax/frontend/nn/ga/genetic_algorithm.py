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

"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments,invalid-name
from typing import List

import tvm
from tvm import relax as rx
from tvm.relax.frontend import nn as nn
from tvm.relax.frontend.nn import Object


class GeneticAlgorithm(Object):  # pylint: disable=too-few-public-methods
    """The genetic algorithm."""

    extern_mods: List[tvm.runtime.Module] = []

    def __init__(  # pylint: disable=too-many-locals
        self,
        _expr,
        _name: str = "genetic_algorithm",
    ) -> None:
        """Create a paged KV cache object with FlashInfer kernels.
        """
        super().__init__(
            _expr=_expr,
            _name=_name,
        )


    def run(self,):
        bb = rx.BlockBuilder.current()
        result = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.genetic_algorithm_run",
                [
                    self._expr,
                ],
                out_sinfo=[
                    rx.TensorStructInfo((1,), "float32"),
                ],
            )
        )
        result = nn.Tensor(_expr=bb.emit(result)).reshape(1,)
        return result

class DefaultGeneticAlgorithm(GeneticAlgorithm):
    """The default genetic algorithm."""

    def __init__(self):
        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.genetic_algorithm_create",
                [],
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name="default_genetic_algorithm",
        )