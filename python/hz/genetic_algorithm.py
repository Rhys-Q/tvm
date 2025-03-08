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
# pylint: disable=redefined-builtin, wildcard-import
"""genetic algorithm model"""

from tvm import relax
from tvm.relax.frontend import nn

from tvm.relax.frontend.nn.ga.genetic_algorithm import GeneticAlgorithm, DefaultGeneticAlgorithm

class GeneticAlgorithmConfig():
    pass

class GeneticAlgorithmModel(nn.Module):
    def __init__(self, config: GeneticAlgorithmConfig):
        pass

    def forward(self, ga: GeneticAlgorithm):
        res = ga.run()
        return res 
    
class GeneticAlgorithmWarp(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GeneticAlgorithmConfig):
        self.model = GeneticAlgorithmModel(config)


    def run(self, ga: GeneticAlgorithm):
        return self.model(ga)

    def create_genetic_algorithm(self) -> GeneticAlgorithm:
        return DefaultGeneticAlgorithm()

    def get_default_spec(self):
        mod_spec = {
            "run": {
                "ga": nn.spec.Object(object_type=GeneticAlgorithm),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "create_genetic_algorithm":{
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            }
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
