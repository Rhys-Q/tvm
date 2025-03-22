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
import math
from tvm import relax
from tvm.relax.frontend import nn
from tvm.contrib import random
import numpy as np

from tvm.relax.frontend.nn.ga.genetic_algorithm import GeneticAlgorithm, DefaultGeneticAlgorithm
import logging

logger = logging.getLogger(__name__)

def calculate_binary_bits(min_value, max_value, precision):
    # 计算总范围内的可能取值数量
    range_size = (max_value - min_value) / precision + 1
    
    # 计算需要的二进制位数
    num_bits = math.ceil(math.log2(range_size))
    
    return num_bits

def calculate_all_gene_len(min_values, max_values, precisions):
    assert len(min_values) == len(max_values) == len(precisions)
    gene_lens = []
    for idx in range(len(min_values)):
        gene_lens.append(calculate_binary_bits(min_values[idx], max_values[idx], precisions[idx]))
    
    return gene_lens

class GeneticAlgorithmConfig():
    def __init__(self, num_pop: int = 1000,min_values = [0,0,0,0] , max_values = [100,100,100,100] , eps = [1e-5,1e-5,1e-5,1e-5], save_good_rate = 0.1, cross_rate = 0.8, mutate_rate = 0.1, mutate_point = 2):
        gene_lens = calculate_all_gene_len(min_values, max_values, eps)
        for lens in gene_lens:
            if lens > 32:
                assert False, "not support gene_lens > 32"
        logger.info("gene_lens: %s" % gene_lens)
        self.gene_split = gene_lens.copy()
        self.gene_lens = gene_lens
        self.min_values = min_values
        self.max_values = max_values
        for i in range(1, len(gene_lens)):
            self.gene_split[i] += self.gene_split[i-1]
        self.gene_split.pop()
        
        logger.info("gene_split: %s" % self.gene_split)
        self.num_var = len(gene_lens)
        self.all_gene_len = sum(gene_lens)

        self.num_pop = num_pop
        self.num_good = int(self.num_pop * save_good_rate)
        self.num_cross = int(self.num_pop * cross_rate)
        self.num_mutate = int(self.num_pop * mutate_rate)
        self.mutate_point = mutate_point
        self.eps = eps
        self.pop_dtype = "uint8"
        assert self.num_pop == self.num_good + self.num_cross + self.num_mutate
    
class GeneticAlgorithmWarp(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GeneticAlgorithmConfig):
        self.config = config

    def _target_func(self, decoded_vars: nn.Tensor):
        target_var = nn.Tensor.from_const(np.array([1,2,3,4], dtype=np.float32))
        y = decoded_vars - target_var
        y = y*y
        y = nn.op.sum(y, axis=1)
        return y
    def _cal_fitness(self, pop: nn.Tensor):
        decode_var = self._decode(pop)
        return self._target_func(decode_var)

    def _decode(self, pop: nn.Tensor):
        vars = nn.op.split(pop, self.config.gene_split, axis=1)
        assert len(vars) == self.config.num_var
        decode_vars = []
        for i in range(self.config.num_var):
            powers_of_two = 2 ** np.arange(self.config.gene_lens[i])[::-1]
            powers_of_two = nn.Tensor.from_const(powers_of_two.astype("int32"))
            int_values = vars[i].astype("int32") * powers_of_two
            int_values = nn.op.sum(int_values, axis=1)
            float_values = int_values.astype("float32") / (2 ** self.config.gene_lens[i] -1 ) * (self.config.max_values[i] - self.config.min_values[i]) + self.config.min_values[i]
            float_values = nn.op.unsqueeze(float_values, dim=1)
            decode_vars.append(float_values)
        return nn.op.concat(decode_vars, dim=1)

    
    def _cross(self, pop: nn.Tensor, fitness: nn.Tensor):
        assert fitness.ndim == 1
        sum_fitness = nn.op.sum(fitness)
        fitness = fitness / sum_fitness
        fitness = nn.op.cumsum(fitness)
        rand_val_1 = nn.tensor_expr_op(random.uniform, "random.uniform", [0, 1, [self.config.num_cross,1 ]])
        selected_index_1 = nn.op.argmax(rand_val_1 <= fitness, axis=1)
        cross_1 = nn.op.take(pop, selected_index_1, axis=0) 
        rand_val_2 = nn.tensor_expr_op(random.uniform, "random.uniform", [0, 1, [self.config.num_cross,1 ]])
        selected_index_2 = nn.op.argmax(rand_val_2 <= fitness, axis=1)
        cross_2 = nn.op.take(pop, selected_index_2, axis=0)
        cross_index = nn.tensor_expr_op(random.randint, "random", [0, pop.shape[1], [cross_1.shape[0], 1]])
        index = nn.op.arange(0, pop.shape[1])
        cross = nn.op.where(index < cross_index, cross_1, cross_2)
        return cross

    def _mutate(self, pop: nn.Tensor):
        mutate_index = nn.tensor_expr_op(random.randint, "random", [0, pop.shape[0], [self.config.num_mutate]])
        modify_index_1 = nn.op.arange(0, self.config.num_mutate, dtype="int32")
        modify_index_1 = nn.op.repeat(modify_index_1,self.config.mutate_point)
        modify_index_1 = nn.op.reshape(modify_index_1, [self.config.num_mutate* self.config.mutate_point, 1])
        modify_index_2 = nn.tensor_expr_op(random.randint, "random", [0, pop.shape[1], [self.config.num_mutate* self.config.mutate_point, 1]])
        modify_index = nn.op.concat([modify_index_1, modify_index_2], dim=1)
        
        modify_value = nn.tensor_expr_op(random.randint, "random", [0, 2, [self.config.num_mutate* self.config.mutate_point], pop.dtype])
        mutate_pop = nn.op.take(pop, mutate_index, axis=0)
        mutate_pop = nn.op.scatter_nd(mutate_pop, modify_index, modify_value)
        return mutate_pop
    def random_init(self):
        return nn.tensor_expr_op(random.randint, "random", [0, 2, [self.config.num_pop, self.config.all_gene_len], self.config.pop_dtype])
    def forward(self, pop: nn.Tensor):
        # 1. cal the fitness
        fitness = self._cal_fitness(pop) # shape is [num_pop]
        # 2. select the best
        values, indices = nn.op.topk(fitness, k=self.config.num_good, axis=0, largest=False) # shape is [num_good]
        pop_good = nn.op.take(pop, indices, axis=0) # [num_good, gene_len]
        # 3. cross
        pop_cross = self._cross(pop_good, fitness)
        # 3. mutate
        pop_mutate = self._mutate(pop_cross)
        pop = nn.op.concat([pop_good, pop_cross, pop_mutate], dim=0)
        return pop, values
    def create_genetic_algorithm(self) -> GeneticAlgorithm:
        return DefaultGeneticAlgorithm()

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "pop": nn.spec.Tensor([self.config.num_pop, self.config.all_gene_len], self.config.pop_dtype),
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
