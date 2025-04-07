#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <dmlc/parameter.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
using namespace tvm;
using namespace tvm::runtime;
#define CUDA_CALL(func)                                       \
  {                                                           \
    cudaError_t e = (func);                                   \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                 \
  }

class RandomState {
 public:
  /*!
   * \brief Creates a RandomEngine using a default seed.
   */
  RandomState() {
    LOG(INFO) << "init random state";
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  }
  ~RandomState() {
    LOG(INFO) << "destory random state";
    curandDestroyGenerator(gen);
  }

  curandGenerator_t gen;
};

struct MyRandomThreadLocalEntry {
  RandomState state;
  static MyRandomThreadLocalEntry* ThreadLocal();
};

typedef dmlc::ThreadLocalStore<MyRandomThreadLocalEntry> MyRandomThreadLocalStore;

MyRandomThreadLocalEntry* MyRandomThreadLocalEntry::ThreadLocal() {
  return MyRandomThreadLocalStore::Get();
}

int RandomFunction(Array<tvm::Integer> shape, DLTensor* out) {
  using namespace tvm::runtime;

  // 1. 构造 shape 向量
  std::vector<int64_t> shape_vec;
  int64_t total = 1;
  for (const auto& s : shape) {
    shape_vec.push_back(s->value);
    total *= s->value;
  }

  // 3. 使用 curand 填充
  float* data_ptr = static_cast<float*>(out->data);
  auto entry = MyRandomThreadLocalEntry::ThreadLocal();

  curandGenerateUniform(entry->state.gen, data_ptr, total);
  return 0;
}

namespace {
int external_random_cuda(Array<Integer> size, String out_dtype, DLTensor* out) {
  RandomFunction(size, out);

  return 0;
}
}  // namespace

TVM_DLL_EXPORT_TYPED_FUNC(relax_random_cuda, external_random_cuda);