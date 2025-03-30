#include <cuda_fp16.h>
#include <dmlc/parameter.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#define CUDA_CALL(func)                                       \
  {                                                           \
    cudaError_t e = (func);                                   \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                 \
  }

__device__ __inline__ half& operator+=(half& x, half y) {  // NOLINT
  x = __hadd(x, y);
  return x;
}

#if !(__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
__device__ __inline__ half atomicAdd(half* address, half val) {
  unsigned int* aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));  // NOLINT
  unsigned int old = *aligned, assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);  // NOLINT
#if __CUDACC_VER_MAJOR__ >= 9
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + float(val));  // NOLINT
    unsigned short sum_as_us = __half_as_ushort(sum);                                    // NOLINT
#else
    unsigned short sum_as_us = __float2half_rn(__half2float(old_as_us) + float(val));  // NOLINT
#endif
    unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)  // NOLINT
                                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

__device__ __inline__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;                      // NOLINT
  if (val == 0.0) return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif  // !(__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))

__device__ __inline__ int64_t atomicAdd(int64_t* address, int64_t val) {
  return atomicAdd((unsigned long long*)(address), (unsigned long long)(val));  // NOLINT
}

enum class ScatterNDOpType : int {
  Update = 0,
  Add = 1,
  // Sub = 2,
  // Min = 3,
  // Max = 4,
};

template <typename DataT, ScatterNDOpType Op>
struct LeftUpdate {
  __device__ __inline__ void operator()(DataT* out, DataT val);
};

template <typename DataT>
struct LeftUpdate<DataT, ScatterNDOpType::Update> {
  __device__ __inline__ void operator()(DataT* out, DataT val) { *out = val; }
};

template <typename DataT>
struct LeftUpdate<DataT, ScatterNDOpType::Add> {
  __device__ __inline__ void operator()(DataT* out, DataT val) { atomicAdd(out, val); }
};

// TODO(chengfan.jia): Improve the performance of this impl
template <typename DataT, typename IndicesT, ScatterNDOpType Op>
__global__ void scatter_nd_kernel_large_update(const DataT* data, const int64_t* data_shape,
                                               const IndicesT* indices, const DataT* updates,
                                               DataT* out, int m, int fused_shape,
                                               int fused_indices_dimension,
                                               int fused_updates_dimension,
                                               bool enable_single_value) {
  auto update_f = LeftUpdate<DataT, Op>();
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < fused_shape) {
    out[index] = __ldg(data + index);
  }

  __syncthreads();

  int bdim_x = 1 + (fused_updates_dimension - 1) / blockDim.x;
  if (blockIdx.x < bdim_x) {
    for (int i = 0; i < fused_indices_dimension; i++) {
      if (index < fused_updates_dimension) {
        int offset = fused_updates_dimension;
        int idx = index;
#pragma unroll
        for (int l = m - 1; l >= 0; l--) {
          idx += offset * __ldg(indices + i + l * fused_indices_dimension);
          offset *= __ldg(data_shape + l);
        }
        update_f(out + idx,
                 __ldg(updates + (enable_single_value ? 0 : i) * fused_updates_dimension + index));
      }
    }
  }
}

template <typename DataT>
__global__ void update_kernel(const DataT* data, DataT* out, int fused_shape, int vectorize_size) {
  int index = blockIdx.x * blockDim.x * vectorize_size + threadIdx.x * vectorize_size;
  if (index + vectorize_size <= fused_shape) {
    ((uint4*)(out + index))[0] = ((uint4*)(data + index))[0];  // NOLINT
  } else if (index < fused_shape) {
    for (size_t i = 0; i < (fused_shape & (vectorize_size - 1)); i++) {
      out[index + i] = (data + index + i)[0];
    }
  }
}

template <typename DataT, typename IndicesT, ScatterNDOpType Op>
__global__ void scatter_nd_kernel(const int64_t* data_shape, const IndicesT* indices,
                                  const DataT* updates, DataT* out, int m, int fused_shape,
                                  int fused_indices_dimension, int fused_updates_dimension,
                                  int slice, bool enable_single_value) {
  auto update_f = LeftUpdate<DataT, Op>();
  int offset = fused_updates_dimension;
  int idx = 0;
#pragma unroll
  for (int l = m - 1; l >= 0; l--) {
    idx += offset * __ldg(indices + blockIdx.x + l * fused_indices_dimension);
    offset *= __ldg(data_shape + l);
  }

#pragma unroll
  for (int i = 0; i < slice; i++) {
    int tx = threadIdx.x * slice + i;
    if (tx < fused_updates_dimension) {
      update_f(
          out + idx + tx,
          __ldg(updates + (enable_single_value ? 0 : blockIdx.x) * fused_updates_dimension + tx));
    }
  }
}

template <typename DataT, typename IndicesT>
void scatter_nd_cuda(DLTensor* data, DLTensor* indices, DLTensor* updates, DLTensor* out,
                     void* data_shape_buffer, int mode, int fused_shape,
                     int fused_indices_dimension, int fused_updates_dimension,
                     bool enable_single_value, TVMStreamHandle stream) {
  int tdim = std::min(1024 /* max_threads */, fused_updates_dimension);
  int bdim = 1 + (fused_shape - 1) / tdim;
  if (fused_updates_dimension / tdim >= bdim / 2) {
    ICHECK(bdim > 0 && tdim > 0);
    if (mode == 0) {
      scatter_nd_kernel_large_update<DataT, IndicesT, ScatterNDOpType::Update>
          <<<bdim, tdim, 0, static_cast<cudaStream_t>(stream)>>>(
              static_cast<DataT*>(data->data), static_cast<int64_t*>(data_shape_buffer),
              static_cast<IndicesT*>(indices->data), static_cast<DataT*>(updates->data),
              static_cast<DataT*>(out->data), indices->shape[0], fused_shape,
              fused_indices_dimension, fused_updates_dimension, enable_single_value);
    } else {
      scatter_nd_kernel_large_update<DataT, IndicesT, ScatterNDOpType::Add>
          <<<bdim, tdim, 0, static_cast<cudaStream_t>(stream)>>>(
              static_cast<DataT*>(data->data), static_cast<int64_t*>(data_shape_buffer),
              static_cast<IndicesT*>(indices->data), static_cast<DataT*>(updates->data),
              static_cast<DataT*>(out->data), indices->shape[0], fused_shape,
              fused_indices_dimension, fused_updates_dimension, enable_single_value);
    }
  } else {
    dim3 block(1024);
    int vectorize_size = 128 / static_cast<int>(data->dtype.bits);
    dim3 grid((fused_shape + block.x * vectorize_size - 1) / (block.x * vectorize_size));
    update_kernel<DataT><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<DataT*>(data->data), static_cast<DataT*>(out->data), fused_shape,
        vectorize_size);
    bdim = fused_indices_dimension;
    if (bdim > 0) {
      ICHECK(tdim > 0);
      int slice = 1 + (fused_updates_dimension - 1) / tdim;
      if (mode == 0) {
        scatter_nd_kernel<DataT, IndicesT, ScatterNDOpType::Update>
            <<<bdim, tdim, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<int64_t*>(data_shape_buffer), static_cast<IndicesT*>(indices->data),
                static_cast<DataT*>(updates->data), static_cast<DataT*>(out->data),
                indices->shape[0], fused_shape, fused_indices_dimension, fused_updates_dimension,
                slice, enable_single_value);
      } else {
        scatter_nd_kernel<DataT, IndicesT, ScatterNDOpType::Add>
            <<<bdim, tdim, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<int64_t*>(data_shape_buffer), static_cast<IndicesT*>(indices->data),
                static_cast<DataT*>(updates->data), static_cast<DataT*>(out->data),
                indices->shape[0], fused_shape, fused_indices_dimension, fused_updates_dimension,
                slice, enable_single_value);
      }
    }
  }
}

using namespace tvm;
using namespace tvm::runtime;

namespace {
int external_dynamic_scatter_nd_cuda(DLTensor* data, DLTensor* indices, DLTensor* updates,
                                     DLTensor* out) {
  // We combine all the indices dimensions but the first one into a single
  // dimension so we can iterate it in single loop instead of an arbitrary
  // number of loops. We do the same thing for all the update dimensions.
  // int mode = (int)(mode_arg.v_int64);
  int mode = 0;
  size_t fused_indices_dimension = 1;
  for (int i = 1; i < indices->ndim; i++) {
    fused_indices_dimension *= indices->shape[i];
  }

  int fused_updates_dimension = 1;
  for (int i = indices->ndim - 1; i < updates->ndim; i++) {
    fused_updates_dimension *= updates->shape[i];
  }

  size_t fused_shape = 1;
  for (int i = 0; i < data->ndim; i++) {
    fused_shape *= data->shape[i];
  }

  // Get sum dimension for each update inside data.
  size_t data_updates_dimension = 1;
  for (int i = indices->shape[0]; i < data->ndim; i++) {
    data_updates_dimension *= data->shape[i];
  }

  // Get total dimension of updates.
  size_t updates_dimension = 1;
  for (int i = 0; i < updates->ndim; i++) {
    updates_dimension *= updates->shape[i];
  }

  // Normally total dimension of updates should be equal to the sum dimension for each update
  // inside data multiply by the fused indices dimension. Or total dimension of updates should
  // be exactly equal to the sum dimension for each update inside data. If total dimension of
  // updates exactly equal to the sum dimension for each update inside data while fused indices
  // dimension larger than one, the update value should be the same for each update with no need
  // of indexing inside the update value. enable_single_value stands for such case.

  ICHECK(fused_indices_dimension * data_updates_dimension == updates_dimension ||
         data_updates_dimension == updates_dimension);
  bool enable_single_value =
      data_updates_dimension == updates_dimension && fused_indices_dimension > 1;

  if (enable_single_value) {
    fused_updates_dimension = updates_dimension;
  }

  Device cuda_dev = {DLDeviceType::kDLCUDA};
  auto cuda_device_api = tvm::runtime::DeviceAPI::Get(cuda_dev);
  TVMStreamHandle stream = cuda_device_api->GetCurrentStream(cuda_dev);
  int64_t data_shape_shape[1] = {data->ndim};
  DLDataType data_shape_dtype = {DLDataTypeCode::kDLInt, 64, 1};
  auto data_shape_buffer =
      MemoryManager::GetOrCreateAllocator(cuda_dev, tvm::runtime::memory::kPooled)
          ->Empty({data->ndim}, data_shape_dtype, cuda_dev);
  std::unique_ptr<DLTensor> dltensor_from(new DLTensor(
      {data->shape, DLDevice{DLDeviceType::kDLCPU}, 1, data_shape_dtype, data_shape_shape}));
  std::unique_ptr<DLTensor> dltensor_to(new DLTensor(
      {data_shape_buffer->data, data_shape_buffer->device, 1, data_shape_dtype, data_shape_shape}));
  NDArray::CopyFromTo(dltensor_from.get(), dltensor_to.get(), stream);

  if (TypeMatch(indices->dtype, DLDataTypeCode::kDLInt, 32)) {
    if (TypeMatch(data->dtype, DLDataTypeCode::kDLFloat, 16)) {
      scatter_nd_cuda<half, int32_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                     fused_shape, fused_indices_dimension, fused_updates_dimension,
                                     enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLFloat, 32)) {
      scatter_nd_cuda<float, int32_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                      fused_shape, fused_indices_dimension, fused_updates_dimension,
                                      enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLFloat, 64)) {
      scatter_nd_cuda<double, int32_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                       fused_shape, fused_indices_dimension,
                                       fused_updates_dimension, enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLInt, 32)) {
      scatter_nd_cuda<int32_t, int32_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                        fused_shape, fused_indices_dimension,
                                        fused_updates_dimension, enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLInt, 64)) {
      scatter_nd_cuda<int64_t, int32_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                        fused_shape, fused_indices_dimension,
                                        fused_updates_dimension, enable_single_value, stream);
    } else {
      LOG(FATAL) << "Not implemented for this data type: "
                 << runtime::DLDataType2String(data->dtype);
    }
  } else if (TypeMatch(indices->dtype, DLDataTypeCode::kDLInt, 64)) {
    if (TypeMatch(data->dtype, DLDataTypeCode::kDLFloat, 16)) {
      scatter_nd_cuda<half, int64_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                     fused_shape, fused_indices_dimension, fused_updates_dimension,
                                     enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLFloat, 32)) {
      scatter_nd_cuda<float, int64_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                      fused_shape, fused_indices_dimension, fused_updates_dimension,
                                      enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLFloat, 64)) {
      scatter_nd_cuda<double, int64_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                       fused_shape, fused_indices_dimension,
                                       fused_updates_dimension, enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLInt, 32)) {
      scatter_nd_cuda<int32_t, int64_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                        fused_shape, fused_indices_dimension,
                                        fused_updates_dimension, enable_single_value, stream);
    } else if (TypeMatch(data->dtype, DLDataTypeCode::kDLInt, 64)) {
      scatter_nd_cuda<int64_t, int64_t>(data, indices, updates, out, data_shape_buffer->data, mode,
                                        fused_shape, fused_indices_dimension,
                                        fused_updates_dimension, enable_single_value, stream);
    } else {
      LOG(FATAL) << "Not implemented for this data type: "
                 << runtime::DLDataType2String(data->dtype);
    }
  } else {
    LOG(FATAL) << "Indices must be int32_t or int64_t, but get: "
               << runtime::DLDataType2String(indices->dtype);
  }
  cudaError_t state = cudaGetLastError();
  ICHECK(state == cudaSuccess) << cudaGetErrorString(state);
  CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  return 0;
}
}  // namespace

TVM_DLL_EXPORT_TYPED_FUNC(relax_scatter_nd_cuda, external_dynamic_scatter_nd_cuda);