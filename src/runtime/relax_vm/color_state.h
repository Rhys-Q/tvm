/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RUNTIME_RELAX_VM_COLOR_STATE_H_
#define TVM_RUNTIME_RELAX_VM_COLOR_STATE_H_
#include <tvm/runtime/ndarray.h>

#include "tvm/runtime/object.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

/*! \brief The class of state. */
class ColorStateObj : public Object {
 public:
  std::vector<NDArray> reflict_lines;  // the size is num_mat, each element's shape is [num_con, 31]
  std::vector<NDArray> reflict_line_cons;  // the size is num_mat

  std::vector<NDArray> k_datas;      // the size is num_mat, each element's shape is [num_con, 31]
  std::vector<NDArray> k_data_cons;  // the size is num_mat

  std::vector<NDArray> w_datas;      // the size is num_mat, each element's shape is [num_con, 31]
  std::vector<NDArray> w_data_cons;  // the size is num_mat

  NDArray reflict_line_black;
  NDArray reflict_line_con_black;

  NDArray color_rank;  // shape is [31, num_mat]

  ColorStateObj(std::string path, int system_id, std::vector<int> mat_ids);
  ~ColorStateObj();

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.ColorState";
  TVM_DECLARE_BASE_OBJECT_INFO(ColorStateObj, Object)
};

class ColorState : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ColorState, ObjectRef, ColorStateObj);
};
}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
#endif