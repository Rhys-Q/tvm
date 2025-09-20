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
#include <curand.h>
#include <dmlc/thread_local.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>

#include "../../cuda/cuda_common.h"
#include "./helper_cuda_kernels.h"

namespace tvm {
namespace runtime {
namespace curand {

#define TVM_CURAND_CALL(func)                                    \
  {                                                              \
    curandStatus_t e = (func);                                   \
    ICHECK(e == CURAND_STATUS_SUCCESS) << "cuRAND error: " << e; \
  }

class CURandGenerator {
 public:
  CURandGenerator() { TVM_CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); }
  ~CURandGenerator() { TVM_CURAND_CALL(curandDestroyGenerator(gen)); }

  void Generate32bit(void* ptr, int64_t n) {
    TVM_CURAND_CALL(curandGenerateNormal(gen, static_cast<float*>(ptr), n, 0.0f, 5.0f));
    cudaDeviceSynchronize();
  }

  void Generate64bit(void* ptr, int64_t n) {
    TVM_CURAND_CALL(curandGenerateNormalDouble(gen, static_cast<double*>(ptr), n, 0.0f, 5.0f));
  }

  curandGenerator_t gen;
};

/*!
 * \brief CUDA Random Engine for CUDA Graph compatible random number generation
 */
class CUDARandomEngine {
 public:
  CUDARandomEngine() : initialized_(false), device_states_(nullptr), max_states_(0) {}

  ~CUDARandomEngine() { Cleanup(); }

  /*!
   * \brief Initialize the CUDA random engine (called during model initialization)
   * \param seed Random seed to use
   */
  void Init(unsigned long seed = 0);

  /*!
   * \brief Cleanup allocated resources
   */
  void Cleanup();

  /*!
   * \brief Check if the engine is initialized
   * \return true if initialized, false otherwise
   */
  bool IsInitialized() const { return initialized_; }

  /*!
   * \brief Generate random integers using CUDA Graph compatible approach
   * \param output Output tensor data pointer
   * \param size Number of elements to generate
   * \param low Lower bound (inclusive)
   * \param high Upper bound (exclusive)
   * \param dtype Data type of the output
   */
  void GenerateRandIntKernel(void* output, int64_t size, int64_t low, int64_t high, DLDataType dtype);

 private:
  bool initialized_;
  void* device_states_;  // curandState array on device
  int64_t max_states_;   // Maximum number of states allocated
};

struct CUDARandomThreadLocalEntry {
  CUDARandomEngine cuda_random_engine;
  static CUDARandomThreadLocalEntry* ThreadLocal();
};

typedef dmlc::ThreadLocalStore<CUDARandomThreadLocalEntry> CUDARandomThreadLocalStore;

CUDARandomThreadLocalEntry* CUDARandomThreadLocalEntry::ThreadLocal() {
  return CUDARandomThreadLocalStore::Get();
}

DeviceAPI* GetCUDADeviceAPI() {
  auto func = tvm::ffi::Function::GetGlobalRequired("device_api.cuda");
  void* ret = func().cast<void*>();
  runtime::DeviceAPI* cuda_api = static_cast<runtime::DeviceAPI*>(ret);
  return cuda_api;
}

int64_t GetTensorSize(DLTensor* tensor) {
  int64_t tensor_size = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    tensor_size *= tensor->shape[i];
  }
  return tensor_size;
}

struct DeferredFunc {
 public:
  explicit DeferredFunc(std::function<void()> func) : func_(func) {}
  ~DeferredFunc() { func_(); }

 private:
  std::function<void()> func_;
};

// Implementation of CUDARandomEngine methods
void CUDARandomEngine::Init(unsigned long seed) {
  if (initialized_) {
    return;  // Already initialized
  }

  // Set up device states for CUDA Graph compatible random generation
  max_states_ = 65536;  // Configurable number of states
  size_t state_size = max_states_ * sizeof(curandState);

  CUDA_CALL(cudaMalloc(&device_states_, state_size));

  // Initialize curandState array (this is the only Host API call needed)
  InitCurandStates(device_states_, seed, max_states_);

  // Synchronize to ensure initialization is complete
  CUDA_CALL(cudaDeviceSynchronize());

  initialized_ = true;
}

void CUDARandomEngine::Cleanup() {
  if (initialized_ && device_states_) {
    CUDA_CALL(cudaFree(device_states_));
    device_states_ = nullptr;
  }
  initialized_ = false;
  max_states_ = 0;
}

void CUDARandomEngine::GenerateRandIntKernel(void* output, int64_t size,
                                            int64_t low, int64_t high, DLDataType dtype) {
  ICHECK(initialized_) << "CUDARandomEngine not initialized. Call Init() first.";
  ICHECK(device_states_) << "Device states not allocated";

  // Call the CUDA Graph compatible kernel
  GenerateRandIntKernelImpl(device_states_, output, size, low, high, dtype);
}

void RandomFill(DLTensor* tensor) {
  static DeviceAPI* cuda_api = GetCUDADeviceAPI();
  CHECK(tensor->device.device_type == DLDeviceType::kDLCUDA)
      << "ValueError: cuRAND only works on CUDA devices";
  int64_t tensor_size = GetTensorSize(tensor);
  int64_t actual_size = tensor_size % 2 == 0 ? tensor_size : tensor_size + 1;
  if (tensor->dtype.code == DLDataTypeCode::kDLFloat && tensor->dtype.bits == 16) {
    // curand only works for size % 2 = 0
    void* data = cuda_api->AllocWorkspace(tensor->device, actual_size * sizeof(float));
    {
      DeferredFunc defer([data, tensor]() { cuda_api->FreeWorkspace(tensor->device, data); });
      CURandGenerator().Generate32bit(data, actual_size);
      ConvertFp32toFp16(/*src=*/data, /*dst=*/tensor->data, /*num=*/tensor_size);
    }
  } else if (tensor->dtype.code == DLDataTypeCode::kDLFloat && tensor->dtype.bits == 32) {
    if (tensor_size % 2 == 1) {
      void* data = cuda_api->AllocWorkspace(tensor->device, actual_size * sizeof(float));
      DeferredFunc defer([data, tensor]() { cuda_api->FreeWorkspace(tensor->device, data); });
      CURandGenerator().Generate32bit(data, actual_size);
      cudaMemcpy(tensor->data, data, tensor_size * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
      CURandGenerator().Generate32bit(tensor->data, actual_size);
    }
  } else if (tensor->dtype.code == DLDataTypeCode::kDLFloat && tensor->dtype.bits == 64) {
    if (tensor_size % 2 == 1) {
      void* data = cuda_api->AllocWorkspace(tensor->device, actual_size * sizeof(double));
      DeferredFunc defer([data, tensor]() { cuda_api->FreeWorkspace(tensor->device, data); });
      CURandGenerator().Generate64bit(data, actual_size);
      cudaMemcpy(tensor->data, data, tensor_size * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
      CURandGenerator().Generate64bit(tensor->data, actual_size);
    }
  } else {
    LOG(FATAL) << "ValueError: Unsupported dtype: " << tensor->dtype;
  }
  // TVMSynchronize(tensor->device.device_type, tensor->device.device_type, nullptr);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.contrib.curand.RandomFill", RandomFill)
      .def_packed("runtime.contrib.curand.Init",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    CUDARandomThreadLocalEntry* entry = CUDARandomThreadLocalEntry::ThreadLocal();
                    unsigned long seed = args.size() > 0 ? args[0].cast<unsigned long>() : 0;
                    entry->cuda_random_engine.Init(seed);
                  })
      .def_packed("runtime.contrib.curand.RandInt",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    CUDARandomThreadLocalEntry* entry = CUDARandomThreadLocalEntry::ThreadLocal();
                    int64_t low = args[0].cast<int64_t>();
                    int64_t high = args[1].cast<int64_t>();
                    auto out = args[2].cast<DLTensor*>();

                    ICHECK(out->device.device_type == DLDeviceType::kDLCUDA)
                        << "CUDARandomEngine only works on CUDA devices";

                    int64_t tensor_size = GetTensorSize(out);
                    entry->cuda_random_engine.GenerateRandIntKernel(out->data, tensor_size, low, high, out->dtype);
                  });
});

}  // namespace curand
}  // namespace runtime
}  // namespace tvm
