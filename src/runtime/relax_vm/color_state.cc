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
#include "color_state.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>  // 用于std::iota
#include <string>
namespace tvm {
namespace runtime {
namespace relax_vm {
TVM_REGISTER_OBJECT_TYPE(ColorStateObj);

std::pair<NDArray, NDArray> LoadColorData(std::string path, std::string type, int system_id,
                                          int mat_id, Device dev) {
  std::filesystem::path base_path = path;
  std::filesystem::path data_path;
  if (type == "K1") {
    std::filesystem::path data_path =
        base_path / std::to_string(system_id) / "K1" / ("K1_" + std::to_string(mat_id) + ".txt");
  } else if (type == "W") {
    std::filesystem::path data_path =
        base_path / std::to_string(system_id) / "W" / ("W_" + std::to_string(mat_id) + ".txt");
  } else if (type == "reflict_line") {
    std::filesystem::path data_path =
        base_path / std::to_string(system_id) / "ALLDATA" / (std::to_string(mat_id) + ".txt");
  } else {
    LOG(FATAL) << "type error: " << type;
  }

  std::ifstream in;
  in.precision(15);
  in.open(data_path, std::ios::in);
  if (!in.is_open()) {
    LOG(FATAL) << "data path: " << data_path << " not found";
  } else {
    int num_con = 0;
    in >> num_con;
    std::vector data_host = std::vector<float>(num_con * 31);
    std::vector data_ind_host = std::vector<float>(num_con);
    for (int i = 0; i < num_con; i++) {
      in >> data_host[i];
    }

    for (int i = 0; i < num_con; i++) {
      for (int j = 0; j < 31; j++) {
        in >> data_host[i * 31 + j];
      }
    }
    in.close();

    // copy to ndarray
    auto data_device = NDArray::Empty({num_con, 31}, DataType::Float(32), dev);
    auto data_ind_device = NDArray::Empty({num_con}, DataType::Float(32), dev);

    data_device.CopyFromBytes(data_host.data(), num_con * 31 * sizeof(float));
    data_ind_device.CopyFromBytes(data_ind_host.data(), num_con * sizeof(float));

    return std::make_pair(data_device, data_ind_device);
  }
}
std::vector<int> getSortIndices(const std::vector<float>& data) {
  std::vector<int> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);  // 初始化索引0,1,2,...
  std::sort(indices.begin(), indices.end(), [&](int a, int b) { return data[a] > data[b]; });
  return indices;
}
NDArray CalColorRank(std::vector<NDArray> reflict_lines) {
  int num_mat = reflict_lines.size();

  // extract the last line
  std::vector<std::vector<float>> last_reflict_lines = std::vector<std::vector<float>>();
  std::vector<int> color_rank = std::vector<int>();

  for (int j = 0; j < 31; j++) {
    last_reflict_lines.push_back(std::vector<float>(num_mat, 0));
    for (int i = 0; i < num_mat; i++) {
      int num_con = reflict_lines[i].Shape()[0];
      float* reflict_lines_data = static_cast<float*>(reflict_lines[i]->data);
      last_reflict_lines[j][i] = reflict_lines_data[(num_con - 1) * 31 + j];
    }
    auto index = getSortIndices(last_reflict_lines[j]);
    color_rank.insert(color_rank.end(), index.begin(), index.end());
  }
  NDArray color_rank_device = NDArray::Empty({31, num_mat}, DataType::Int(32), Device{kDLCPU, 0});

  color_rank_device.CopyFromBytes(color_rank.data(), num_mat * 31 * sizeof(int));
  return color_rank_device;
}

ColorStateObj::ColorStateObj(std::string path, int system_id, std::vector<int> mat_ids) {
  Device device = Device{kDLCPU, 0};
  this->reflict_lines = std::vector<NDArray>();
  this->reflict_line_cons = std::vector<NDArray>();
  this->w_datas = std::vector<NDArray>();
  this->w_data_cons = std::vector<NDArray>();
  this->k_datas = std::vector<NDArray>();
  this->k_data_cons = std::vector<NDArray>();
  for (int mat_id : mat_ids) {
    std::pair<NDArray, NDArray> K1_data = LoadColorData(path, "K1", system_id, mat_id, device);
    std::pair<NDArray, NDArray> W_data = LoadColorData(path, "W", system_id, mat_id, device);
    std::pair<NDArray, NDArray> reflice_line =
        LoadColorData(path, "reflict_line", system_id, mat_id, device);
    this->k_datas.push_back(K1_data.first);
    this->k_data_cons.push_back(K1_data.second);
    this->w_datas.push_back(W_data.first);
    this->w_data_cons.push_back(W_data.second);
    this->reflict_lines.push_back(reflice_line.first);
    this->reflict_line_cons.push_back(reflice_line.second);
  }
  // black data
  std::pair<NDArray, NDArray> black_data =
      LoadColorData(path, "reflict_line", system_id, 1, device);
  this->reflict_line_black = black_data.first;
  this->reflict_line_con_black = black_data.second;
  this->color_rank = CalColorRank(this->reflict_lines);
}
}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm