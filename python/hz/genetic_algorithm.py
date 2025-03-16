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
from tvm.contrib import random

from tvm.relax.frontend.nn.ga.genetic_algorithm import GeneticAlgorithm, DefaultGeneticAlgorithm

class GeneticAlgorithmConfig():
    def __init__(self, num_pop: int = 1000,gene_len = 100 , num_gene = 4, gene_indices=[],  save_good_rate = 0.1, cross_rate = 0.8, mutate_rate = 0.1):
        self.num_pop = num_pop
        self.num_good = int(self.num_pop * save_good_rate)
        self.num_cross = int(self.num_pop * cross_rate)
        self.num_mutate = int(self.num_pop * mutate_rate)
        self.gene_len = gene_len
        self.num_gene = num_gene
        self.gene_indices = gene_indices
        self.pop_dtype = "uint8"
        assert self.num_pop == self.num_good + self.num_cross + self.num_mutate

class GeneticAlgorithmModel(nn.Module):
    def __init__(self, config: GeneticAlgorithmConfig):
        pass

    def forward(self, ga: GeneticAlgorithm):
        res = ga.run()
        return res 
    
class GeneticAlgorithmWarp(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GeneticAlgorithmConfig):
        self.model = GeneticAlgorithmModel(config)
        self.config = config
    
    def _cal_fitness(self, pop: nn.Tensor, pop_indices: nn.Tensor):
        num_pop = pop.shape[0]
        fitness = nn.op.zeros([num_pop], dtype="float32")
        return fitness
    
    def _cross(self, pop: nn.Tensor, fitness: nn.Tensor):
        return nn.zeros([self.config.num_cross, pop.shape[1]], dtype=self.config.pop_dtype)

    def _mutate(self, pop: nn.Tensor):
        return nn.zeros([self.config.num_mutate, pop.shape[1]], dtype=self.config.pop_dtype)
    def random_init(self):
        return nn.tensor_expr_op(random.randint, "random", [0, 1, [self.config.num_pop, self.config.num_gene], self.config.pop_dtype])
    def forward(self, pop: nn.Tensor):
        # 1. cal the fitness
        fitness = self._cal_fitness(pop) # shape is [num_pop]
        # 2. select the best
        values, indices = nn.op.topk(fitness, k=self.config.num_good, axis=0, largest=False) # shape is [num_good]
        pop_good = nn.op.take(pop, indices) # [num_good, gene_len]
        # 3. cross
        pop_cross = self._cross(pop_good, fitness)
        # 3. mutate
        pop_mutate = self._mutate(pop_cross)
        pop = nn.op.concat([pop_good, pop_cross, pop_mutate], axis=0)
        return pop, values
    def create_genetic_algorithm(self) -> GeneticAlgorithm:
        return DefaultGeneticAlgorithm()

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "pop": nn.spec.Tensor([self.config.num_pop, self.config.gene_len], self.config.pop_dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "random_init": {
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
