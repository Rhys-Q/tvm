#ifndef TVM_RUNTIME_CONTRIB_CURAND_LOAD_NUMPY_BIN_H_
#define TVM_RUNTIME_CONTRIB_CURAND_LOAD_NUMPY_BIN_H_
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace tvm {
namespace runtime {
namespace curand {
/**
 * 从二进制文件加载 NumPy 数组数据
 * 对应 Python dump_numpy_random 函数保存的格式
 */
class NumpyBinLoader {
 public:
  struct ArrayInfo {
    std::string dtype;          // 数据类型字符串
    std::vector<size_t> shape;  // 数组形状
    std::vector<char> data;     // 原始数据
    size_t total_elements;      // 总元素数量

    // 计算总元素数量
    void calculate_total_elements() {
      total_elements = 1;
      for (size_t dim : shape) {
        total_elements *= dim;
      }
    }
  };

  /**
   * 加载二进制文件
   * @param filepath 文件路径
   * @return ArrayInfo 包含数组信息的结构体
   */
  static ArrayInfo* load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("无法打开文件: " + filepath);
    }

    ArrayInfo* info = new ArrayInfo();

    // 1. 读取数据类型字符串长度
    uint32_t dtype_len;
    file.read(reinterpret_cast<char*>(&dtype_len), sizeof(uint32_t));
    if (file.gcount() != sizeof(uint32_t)) {
      throw std::runtime_error("读取数据类型长度失败");
    }

    // 2. 读取数据类型字符串
    info->dtype.resize(dtype_len);
    file.read(&info->dtype[0], dtype_len);
    if (file.gcount() != dtype_len) {
      throw std::runtime_error("读取数据类型字符串失败");
    }

    // 3. 读取维度数量
    uint32_t num_dims;
    file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
    if (file.gcount() != sizeof(uint32_t)) {
      throw std::runtime_error("读取维度数量失败");
    }

    // 4. 读取每个维度的大小
    info->shape.resize(num_dims);
    for (uint32_t i = 0; i < num_dims; i++) {
      uint32_t dim_size;
      file.read(reinterpret_cast<char*>(&dim_size), sizeof(uint32_t));
      if (file.gcount() != sizeof(uint32_t)) {
        throw std::runtime_error("读取维度大小失败");
      }
      info->shape[i] = dim_size;
    }

    // 5. 计算总元素数量和数据类型大小
    info->calculate_total_elements();
    size_t element_size = get_element_size(info->dtype);
    size_t total_bytes = info->total_elements * element_size;

    // 6. 读取数据
    info->data.resize(total_bytes);
    file.read(info->data.data(), total_bytes);
    if (file.gcount() != total_bytes) {
      throw std::runtime_error("读取数据失败");
    }

    file.close();
    return info;
  }

  /**
   * 根据数据类型字符串获取元素大小
   * @param dtype 数据类型字符串
   * @return 元素大小（字节）
   */
  static size_t get_element_size(const std::string& dtype) {
    if (dtype == "float32" || dtype == "<f4") return sizeof(float);
    if (dtype == "float64" || dtype == "<f8") return sizeof(double);
    if (dtype == "int32" || dtype == "<i4") return sizeof(int32_t);
    if (dtype == "int64" || dtype == "<i8") return sizeof(int64_t);
    if (dtype == "uint32" || dtype == "<u4") return sizeof(uint32_t);
    if (dtype == "uint64" || dtype == "<u8") return sizeof(uint64_t);
    if (dtype == "int8" || dtype == "<i1") return sizeof(int8_t);
    if (dtype == "uint8" || dtype == "<u1") return sizeof(uint8_t);
    if (dtype == "int16" || dtype == "<i2") return sizeof(int16_t);
    if (dtype == "uint16" || dtype == "<u2") return sizeof(uint16_t);

    throw std::runtime_error("不支持的数据类型: " + dtype);
  }

  /**
   * 获取类型化的数据指针
   * @tparam T 数据类型
   * @param info 数组信息
   * @return 类型化的数据指针
   */
  template <typename T>
  static T* get_data(ArrayInfo& info) {
    size_t expected_size = get_element_size(info.dtype);
    if (sizeof(T) != expected_size) {
      throw std::runtime_error("数据类型不匹配");
    }
    return reinterpret_cast<T*>(info.data.data());
  }

  /**
   * 获取常量类型化的数据指针
   * @tparam T 数据类型
   * @param info 数组信息
   * @return 常量类型化的数据指针
   */
  template <typename T>
  static const T* get_data(const ArrayInfo& info) {
    size_t expected_size = get_element_size(info.dtype);
    if (sizeof(T) != expected_size) {
      throw std::runtime_error("数据类型不匹配");
    }
    return reinterpret_cast<const T*>(info.data.data());
  }

  /**
   * 打印数组信息
   * @param info 数组信息
   */
  static void print_info(const ArrayInfo& info) {
    std::cout << "数据类型: " << info.dtype << std::endl;
    std::cout << "形状: [";
    for (size_t i = 0; i < info.shape.size(); i++) {
      std::cout << info.shape[i];
      if (i < info.shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "总元素数量: " << info.total_elements << std::endl;
    std::cout << "元素大小: " << get_element_size(info.dtype) << " 字节" << std::endl;
    std::cout << "总数据大小: " << info.data.size() << " 字节" << std::endl;
  }
};

#endif  // LOAD_NUMPY_BIN_H

}  // namespace curand
}  // namespace runtime
}  // namespace tvm