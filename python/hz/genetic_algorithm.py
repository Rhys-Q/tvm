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
from .scatter_nd import ScatterNd
from tvm.relax.frontend.nn.ga.genetic_algorithm import GeneticAlgorithm, DefaultGeneticAlgorithm
import logging
from typing import List
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
    def __init__(self, num_pop: int = 1000,min_values = [0,0,0,0] , max_values = [100,100,100,100] , eps = [1e-5,1e-5,1e-5,1e-5], save_good_rate = 0.1, cross_rate = 0.8, mutate_rate = 0.1, mutate_point = 20):
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
        self.pop_dtype = "int32"
        assert self.num_pop == self.num_good + self.num_cross + self.num_mutate
    
class GeneticAlgorithmWarp(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GeneticAlgorithmConfig):
        self.config = config
        gene_len = config.gene_lens[0]
        for glen in config.gene_lens:
            assert gene_len == glen
        self.same_gene_len = True
        self.scatter_nd = ScatterNd()

    def _target_func(self, decoded_vars: nn.Tensor):
        target_var = nn.Tensor.from_const(np.array([2, 3,5,19], dtype=np.float32))
        target_var = nn.op.reshape(target_var, [1, 4])
        target_var = nn.op.repeat(target_var, self.config.num_pop, axis=0)
        y = decoded_vars - target_var
        y = y*y
        if y.shape[1] > 1:
            y = nn.op.sum(y, axis=1)
        else:
            y = nn.op.squeeze(y, axis=1)
        return y
    def _cal_fitness(self, pop_vars: List[nn.Tensor]):
        decode_var = self._decode(pop_vars)
        return self._target_func(decode_var)

    def _decode(self, vars: List[nn.Tensor]):
        if isinstance(vars, nn.Tensor):
            if self.config.num_var == 1:
                vars = [vars]
            else:
                vars = nn.op.split(vars, self.config.gene_split, axis=1)
        else:
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

    def _encode(self, vars: nn.Tensor) -> nn.Tensor:
        # Split input tensor into individual variables if necessary
        if self.config.num_var == 1:
            var_list = [vars]
        else:
            var_list = nn.op.split(vars, self.config.num_var, axis=1)
    
        encoded_bits = []
        for i in range(self.config.num_var):
            # Extract current variable
            var_i = var_list[i] # [1000, 1]
            min_val = self.config.min_values[i]
            max_val = self.config.max_values[i]
            gene_len = self.config.gene_lens[i]

            # Step 1: Clip values to valid range
            clipped_var = nn.op.clip(var_i, min_val, max_val) # [1000, 1]
            
            # Step 2: Normalize to [0, 1] and scale to integer range
            normalized = (clipped_var - min_val) / (max_val - min_val)
            scaled = normalized * (2 ** gene_len - 1)
            int_val = nn.op.round(scaled).astype("int32")

            # Step 3: Generate binary representation
            # Create powers of two tensor (MSB first)
            powers = 2 ** np.arange(gene_len-1, -1, -1, dtype="int32")
            powers_tensor = nn.Tensor.from_const(powers.reshape(1, -1)) # [1, gene_len]

            # Calculate binary bits using bitwise operations
            expanded_int = nn.op.reshape(int_val, (-1, 1))  # Ensure 2D tensor [1000, 1]
            div = nn.op.divide(expanded_int, powers_tensor)
            div = nn.op.floor(div)
            bits = nn.op.mod(div, nn.Tensor.from_const(2))

            # Step 4: Convert to float32 and store
            encoded_bits.append(bits.astype(self.config.pop_dtype))

        # Concatenate all binary segments
        return nn.op.concat(encoded_bits, dim=1)

    
    def _cross(self, pop_vars: List[nn.Tensor], fitness: nn.Tensor):
        assert fitness.ndim == 1
        assert self.config.num_cross % 2 == 0
        num_cross_single = self.config.num_cross // 2
        assert self.same_gene_len
        pop = nn.op.concat(pop_vars, dim=1) # [1000, 96]
        num_var = self.config.num_var # 4
        pop = nn.op.reshape(pop, [ self.config.num_pop, num_var, -1]) # [1000, 4, 24]


        sum_fitness = nn.op.sum(fitness) # [1000]
        fitness = fitness / sum_fitness
        fitness = nn.op.cumsum(fitness)
        rand_val_1 = nn.tensor_expr_op(random.uniform, "random.uniform", [0, 1, [self.config.num_cross,1 ]]) # [800,1 ]
        flag1 = rand_val_1 <= fitness # [800, 1000]
        flag1 = nn.op.astype(flag1, "uint8")
        selected_index_1 = nn.op.argmax(flag1, axis=1) #[800]
        cross = nn.op.take(pop, selected_index_1, axis=0)  #[800, 4, 24]
        cross_1, cross_2 = nn.op.split(cross, 2, axis=0)

        cross_index = nn.tensor_expr_op(random.randint, "random", [0, cross_2.shape[2], [cross_1.shape[0],cross_1.shape[1], cross_1.shape[2]]]) #[400, 4, 4]
        # index = nn.op.arange(0, cross_1.shape[2]) #[24]
        # dd =  index < cross_index # [400, 4, 24]
        cross_pop1 = nn.op.where(cross_index > 0, cross_1, cross_2)
        cross_pop2 = nn.op.where(cross_index <= 0, cross_1, cross_2)
        cross_pop = nn.op.concat([cross_pop1, cross_pop2], dim=0) # [800, 4, 24]
        cross_pop = nn.op.reshape(cross_pop, [cross_pop.shape[0], -1]) # [800, 96]
        return cross_pop

    def debug_cross(self, pop_vars: nn.Tensor, fitness: nn.Tensor):
        assert fitness.ndim == 1
        sum_fitness = nn.op.sum(fitness)
        fitness = fitness / sum_fitness
        fitness = nn.op.cumsum(fitness)
        rand_val_1 = nn.tensor_expr_op(random.uniform, "random.uniform", [0, 1, [self.config.num_cross,1 ]])
        flag1 = rand_val_1 <= fitness
        flag1 = nn.op.astype(flag1, "uint8")
        selected_index_1 = nn.op.argmax(flag1, axis=1)
        cross_1 = nn.op.take(pop, selected_index_1, axis=0) 
        rand_val_2 = nn.tensor_expr_op(random.uniform, "random.uniform", [0, 1, [self.config.num_cross,1 ]])
        flag2 = rand_val_2 <= fitness
        flag2 = nn.op.astype(flag2, "uint8")
        selected_index_2 = nn.op.argmax(flag2, axis=1)
        cross_2 = nn.op.take(pop, selected_index_2, axis=0)
        cross_index = nn.tensor_expr_op(random.randint, "random", [0, pop.shape[1], [cross_1.shape[0], 1]])
        index = nn.op.arange(0, pop.shape[1])
        dd = index < cross_index
        cross = nn.op.where(index < cross_index, cross_1, cross_2)
        return cross, cross_1, cross_2, cross_index, selected_index_1, selected_index_2, dd

    def _mutate(self, pop_vars: List[nn.Tensor]):
        assert self.same_gene_len
        pop = nn.op.concat(pop_vars, dim=1) # [1000, 96]
        num_var = self.config.num_var # 4
        pop = nn.op.reshape(pop, [ self.config.num_pop, num_var, -1]) # [1000, 4, 24]
        mutate_index = nn.tensor_expr_op(random.randint, "random", [0, pop.shape[0], [self.config.num_mutate]]) # [num_mutate]
        mutate_pop = nn.op.take(pop, mutate_index, axis=0)

        modify_index_1 = nn.op.reshape(nn.op.arange(0, self.config.num_mutate, dtype="int32"), [1, self.config.num_mutate,1 ,1 ]) # [1, num_mutate, 1, 1]
        modify_index_1 = nn.op.repeat(modify_index_1, num_var, axis=2) # [1, num_mutate, num_var,1 ]
        modify_index_1 = nn.op.repeat(modify_index_1, self.config.mutate_point, axis=3) # [1, num_mutate, num_var,mutate_point]

        modify_index_2 = nn.op.reshape(nn.op.arange(0, num_var, dtype="int32"),  [1, 1, num_var, 1]) # [1, 1, num_var, 1]
        modify_index_2 = nn.op.repeat(modify_index_2, self.config.num_mutate, axis=1) # [1, num_mutate, num_var, 1]
        modify_index_2 = nn.op.repeat(modify_index_2, self.config.mutate_point, axis=3) # [1, num_mutate, num_var, mutate_point]

        modify_index_3 = nn.tensor_expr_op(random.randint, "random", [0, pop.shape[-1], [1, self.config.num_mutate, num_var, self.config.mutate_point]]) # [1, num_mutate, num_var, mutate_point]
        modify_index = nn.op.concat([modify_index_1, modify_index_2, modify_index_3], dim=0) # [3, num_mutate, num_var, mutate_point]
        
        modify_value = nn.tensor_expr_op(random.randint, "random", [0, 2, [self.config.num_mutate, num_var,  self.config.mutate_point], pop.dtype])
        
        mutate_pop = self.scatter_nd(mutate_pop, modify_index, modify_value) # [1000, 4, 24]
        mutate_pop = nn.op.reshape(mutate_pop, [self.config.num_mutate, -1])
        return mutate_pop

    def _mutate_diff(self, pop_vars: List[nn.Tensor]):
        assert self.same_gene_len
        num_var = self.config.num_var
        decoded_vars = self._decode(pop_vars) # [1000, 4]
        mutate_index1 = nn.tensor_expr_op(random.randint, "random", [0, decoded_vars.shape[0], [self.config.num_mutate]]) # [num_mutate]
        mutate_index2 = nn.tensor_expr_op(random.randint, "random", [0, decoded_vars.shape[0], [self.config.num_mutate]]) # [num_mutate]
        mutate_index3 = nn.tensor_expr_op(random.randint, "random", [0, decoded_vars.shape[0], [self.config.num_mutate]]) # [num_mutate]

        mutate_vars1 = nn.op.take(decoded_vars, mutate_index1, axis=0)
        mutate_vars2 = nn.op.take(decoded_vars, mutate_index2, axis=0)
        mutate_vars3 = nn.op.take(decoded_vars, mutate_index3, axis=0)

        rand_val = nn.tensor_expr_op(random.uniform, "random.uniform", [0, 1, [self.config.num_mutate,num_var ]]) # [100,4 ]

        new_vars = mutate_vars1 + (mutate_vars2 - mutate_vars3)* 0.01  + rand_val

        new_vars = nn.op.clip(new_vars, self.config.min_values[0], self.config.max_values[0])
        mutate_pop = self._encode(new_vars)
        return mutate_pop
    def random_init(self):
        return nn.tensor_expr_op(random.randint, "random", [0, 2, [self.config.num_pop, self.config.all_gene_len], self.config.pop_dtype])
    def forward(self, pop: nn.Tensor):
        # 1. cal the fitness
        if self.config.num_var == 1:
            pop_vars = [pop]
        else:
            pop_vars = nn.op.split(pop, self.config.gene_split, axis=1)
        fitness = self._cal_fitness(pop_vars) # shape is [num_pop]
        # 2. select the best
        values, indices = nn.op.topk(fitness, k=self.config.num_good, axis=0, largest=False) # shape is [num_good]
        pop_good = nn.op.take(pop, indices, axis=0) # [num_good, gene_len]
        # 3. cross
        pop_cross = self._cross(pop_vars, fitness)
        # 3. mutate
        pop_mutate = self._mutate_diff(pop_vars)
        pop = nn.op.concat([pop_good, pop_cross, pop_mutate], dim=0)
        # best
        best_value = nn.op.take(values, nn.reshape(nn.Tensor.from_const(0), [1]).astype("int32"), axis=0)
        best_gen = nn.op.take(pop_good, nn.reshape(nn.Tensor.from_const(0), [1]).astype("int32"), axis=0)
        best_var = self._decode(best_gen)
        return pop, pop_good, values, best_var, best_value 
    def create_genetic_algorithm(self) -> GeneticAlgorithm:
        return DefaultGeneticAlgorithm()

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "pop": nn.spec.Tensor([self.config.num_pop, self.config.all_gene_len], self.config.pop_dtype),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "random_init": {
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            # "debug_cross": {
            #     "pop": nn.spec.Tensor([self.config.num_pop, self.config.all_gene_len], self.config.pop_dtype),
            #     "fitness": nn.spec.Tensor([self.config.num_pop ], "float32"),
            #     "$": {
            #         "param_mode": "none",
            #         "effect_mode": "none",
            #     },
            # },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
