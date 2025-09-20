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
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "./helper_cuda_kernels.h"

namespace tvm {
namespace runtime {
namespace curand {

__global__ void KernelFp32ToFp16(const float* src, half* dst, int num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < num) {
    dst[idx] = src[idx];
  }
}

void ConvertFp32toFp16(const void* _src, void* _dst, int64_t num) {
  const float* src = static_cast<const float*>(_src);
  half* dst = static_cast<half*>(_dst);
  KernelFp32ToFp16<<<(num + 255) / 256, 256>>>(src, dst, num);
}

__global__ void KernelInitCurandStates(curandState* states, unsigned long seed, int64_t num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < num) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

void InitCurandStates(void* _states, unsigned long seed, int64_t num) {
  curandState* states = static_cast<curandState*>(_states);
  KernelInitCurandStates<<<(num + 255) / 256, 256>>>(states, seed, num);
}

template<typename T>
__global__ void KernelGenerateRandInt(curandState* states, T* output, int64_t size,
                                      T low, T high, int64_t num_states) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    // Use round-robin assignment of states to threads
    int state_idx = idx % num_states;
    curandState localState = states[state_idx];

    // Generate uniform random float and convert to integer range
    float rand_val = curand_uniform(&localState);
    T result = low + static_cast<T>(rand_val * (high - low));

    // Ensure result is within bounds
    if (result >= high) result = high - 1;
    output[idx] = result;

    // Update the state
    states[state_idx] = localState;
  }
}

void GenerateRandIntKernelImpl(void* _states, void* _output, int64_t size,
                          int64_t low, int64_t high, DLDataType dtype) {
  curandState* states = static_cast<curandState*>(_states);

  // Calculate number of states (assume 65536 as default)
  int64_t num_states = 65536;  // This should match CUDARandomEngine::max_states_

  dim3 blocks((size + 255) / 256);
  dim3 threads(256);

  if (dtype.code == kDLInt && dtype.bits == 32) {
    int32_t* output = static_cast<int32_t*>(_output);
    KernelGenerateRandInt<<<blocks, threads>>>(states, output, size,
                                               static_cast<int32_t>(low),
                                               static_cast<int32_t>(high),
                                               num_states);
  } else if (dtype.code == kDLInt && dtype.bits == 16) {
    int16_t* output = static_cast<int16_t*>(_output);
    KernelGenerateRandInt<<<blocks, threads>>>(states, output, size,
                                               static_cast<int16_t>(low),
                                               static_cast<int16_t>(high),
                                               num_states);
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    int8_t* output = static_cast<int8_t*>(_output);
    KernelGenerateRandInt<<<blocks, threads>>>(states, output, size,
                                               static_cast<int8_t>(low),
                                               static_cast<int8_t>(high),
                                               num_states);
  } else if (dtype.code == kDLUInt && dtype.bits == 32) {
    uint32_t* output = static_cast<uint32_t*>(_output);
    KernelGenerateRandInt<<<blocks, threads>>>(states, output, size,
                                               static_cast<uint32_t>(low),
                                               static_cast<uint32_t>(high),
                                               num_states);
  } else if (dtype.code == kDLUInt && dtype.bits == 16) {
    uint16_t* output = static_cast<uint16_t*>(_output);
    KernelGenerateRandInt<<<blocks, threads>>>(states, output, size,
                                               static_cast<uint16_t>(low),
                                               static_cast<uint16_t>(high),
                                               num_states);
  } else if (dtype.code == kDLUInt && dtype.bits == 8) {
    uint8_t* output = static_cast<uint8_t*>(_output);
    KernelGenerateRandInt<<<blocks, threads>>>(states, output, size,
                                               static_cast<uint8_t>(low),
                                               static_cast<uint8_t>(high),
                                               num_states);
  }
}

template<typename T>
__global__ void KernelGenerateUniform(curandState* states, T* output, int64_t size, int64_t num_states) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    // Use round-robin assignment of states to threads
    int state_idx = idx % num_states;
    curandState localState = states[state_idx];

    // Generate uniform random float [0,1)
    float rand_val = curand_uniform(&localState);
    output[idx] = static_cast<T>(rand_val);

    // Update the state
    states[state_idx] = localState;
  }
}

void GenerateUniformKernelImpl(void* _states, void* _output, int64_t size, DLDataType dtype) {
  curandState* states = static_cast<curandState*>(_states);

  // Calculate number of states (assume 65536 as default)
  int64_t num_states = 65536;  // This should match CUDARandomEngine::max_states_

  dim3 blocks((size + 255) / 256);
  dim3 threads(256);

  if (dtype.code == kDLFloat && dtype.bits == 32) {
    float* output = static_cast<float*>(_output);
    KernelGenerateUniform<<<blocks, threads>>>(states, output, size, num_states);
  } else if (dtype.code == kDLFloat && dtype.bits == 64) {
    double* output = static_cast<double*>(_output);
    KernelGenerateUniform<<<blocks, threads>>>(states, output, size, num_states);
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    half* output = static_cast<half*>(_output);
    KernelGenerateUniform<<<blocks, threads>>>(states, output, size, num_states);
  }
}

}  // namespace curand
}  // namespace runtime
}  // namespace tvm
