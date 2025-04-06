// /*
//  * Licensed to the Apache Software Foundation (ASF) under one
//  * or more contributor license agreements.  See the NOTICE file
//  * distributed with this work for additional information
//  * regarding copyright ownership.  The ASF licenses this file
//  * to you under the Apache License, Version 2.0 (the
//  * "License"); you may not use this file except in compliance
//  * with the License.  You may obtain a copy of the License at
//  *
//  *   http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing,
//  * software distributed under the License is distributed on an
//  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//  * KIND, either express or implied.  See the License for the
//  * specific language governing permissions and limitations
//  * under the License.
//  */
// /*!
//  * \file src/runtime/relax_vm/paged_kv_cache.cc
//  * \brief Runtime paged KV cache object for language models.
//  */
// #include <tvm/runtime/logging.h>
// #include <tvm/runtime/object.h>
// #include <tvm/runtime/registry.h>

// #include "color_state.h"
// namespace tvm {
// namespace runtime {
// namespace relax_vm {

// class GeneticAlgorithmObj : public Object {
//  public:
//   virtual GeneticAlgorithmOutputObj Run() = 0;

//   static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
//   static constexpr const char* _type_key = "relax.vm.GeneticAlgorithm";
//   TVM_DECLARE_BASE_OBJECT_INFO(GeneticAlgorithmObj, Object)
// };

// class GeneticAlgorithm : public ObjectRef {
//  public:
//   TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GeneticAlgorithm, ObjectRef, GeneticAlgorithmObj);
// };

// class GeneticAlgorithmOutputObj : public Object {
//  public:
//   std::vector<float> best_gene_;
//   float fitness_;
//   float time_used_ms_;
//   GeneticAlgorithmOutputObj(NDArray best_gene, float fitness, float time_used_ms) {
//     auto best_gene_cpu = best_gene.CopyTo(DLDevice{kDLCPU, 0});
//     float* best_gene_host = static_cast<float*>(best_gene_cpu->data);

//     this->best_gene_ = std::vector<float>(best_gene.Shape()[0], 0);
//     for (int i = 0; i < best_gene.Shape()[0]; i++) {
//       this->best_gene_[i] = best_gene_host[i];
//     }
//     this->fitness_ = fitness;
//     this->time_used_ms_ = time_used_ms;
//   }
//   static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
//   static constexpr const char* _type_key = "relax.vm.GeneticAlgorithmOutput";
//   TVM_DECLARE_BASE_OBJECT_INFO(GeneticAlgorithmOutputObj, Object)
// };

// class GeneticAlgorithmOutput : public ObjectRef {
//  public:
//   TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GeneticAlgorithmOutput, ObjectRef,
//                                         GeneticAlgorithmOutputObj);
// };
// class GeneticAlgorithmImplObj : public GeneticAlgorithmObj {
//  public:
//   GeneticAlgorithmImplObj() {}
//   GeneticAlgorithmOutputObj Run() override {
//     // init the pop
//     auto pop_init = f_init_(pop_size_, num_gen_, x_min_, x_max_);
//     gene_indices_ = f_encode_(pop_init, pop_, x_min_, x_max_, eps_);
//     float best_fitness = std::numeric_limits<float>::max();
//     for (int i = 0; i < max_epoch_; i++) {
//       // cal the fitness
//       auto pop_decoded = f_decode_(pop_, gene_indices_, x_min_, x_max_, eps_);
//       auto fitness = f_fitness_(pop_decoded);

//       f_update_best_(best_gene_, fitness, pop_decoded, best_fitness);

//       // selection
//       auto selected = f_selection_(pop_, fitness);
//       // crossover
//       auto crossover = f_crossover_(pop_, fitness, crossover_rate_);
//       // mutation
//       auto mutation = f_mutation_(pop_, mutation_rate_);

//       f_merge_(pop_, selected, crossover, mutation);
//     }
//     GeneticAlgorithmOutputObj ret = GeneticAlgorithmOutputObj(best_gene_, 0, 0);
//     return ret;
//   }

//   int pop_size_;
//   int num_gen_;
//   int max_epoch_;
//   double mutation_rate_;
//   double crossover_rate_;
//   std::vector<float> x_min_;
//   std::vector<float> x_max_;
//   std::vector<float> eps_;
//   NDArray pop_;           // shape is [pop_size, len_gen]
//   NDArray gene_indices_;  // shape is [num_gen]
//   NDArray best_gene_;

//   ColorState color_state_;

//   PackedFunc f_init_;    // init the pop
//   PackedFunc f_decode_;  // decode the pop
//   PackedFunc f_encode_;  // encode the pop
//   PackedFunc f_update_best_;
//   PackedFunc f_fitness_;    // fitness of the pop
//   PackedFunc f_crossover_;  // crossover the pop
//   PackedFunc f_mutation_;   // mutation the pop
//   PackedFunc f_selection_;  // selection the pop
//   PackedFunc f_merge_;      // merge the pop
//   PackedFunc f_avalanche_;  // avalanche the pop

//   static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
//   static constexpr const char* _type_key = "relax.vm.GeneticAlgorithmImpl";
//   TVM_DECLARE_FINAL_OBJECT_INFO(GeneticAlgorithmImplObj, GeneticAlgorithmObj);
// };

// TVM_REGISTER_OBJECT_TYPE(GeneticAlgorithmImplObj);
// TVM_REGISTER_OBJECT_TYPE(GeneticAlgorithmOutputObj);

// TVM_REGISTER_GLOBAL("vm.builtin.genetic_algorithm_create")
//     .set_body([](TVMArgs args, TVMRetValue* rv) {
//       ObjectPtr<GeneticAlgorithmImplObj> n = make_object<GeneticAlgorithmImplObj>();
//       *rv = GeneticAlgorithm(std::move(n));
//     });
// TVM_REGISTER_GLOBAL("vm.builtin.genetic_algorithm_run")
//     .set_body_typed([](GeneticAlgorithm ga, NDArray o_data) { ga->Run(); });
// }  // namespace relax_vm
// }  // namespace runtime
// }  // namespace tvm