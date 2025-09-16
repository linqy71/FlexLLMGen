// direct_io.cc
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pybind11/stl.h>

torch::Tensor read_file_direct(std::string filename, 
                              std::vector<int64_t> shape, 
                              torch::Dtype dtype) {
    // 1. 首先用标准IO读取文件头信息
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // 读取NumPy头部标识和版本(8字节)
    char magic[8];
    file.read(magic, 8);
    
    // 读取头部长度(2字节)
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(uint16_t));
    
    // 计算数据开始位置
    size_t data_offset = 8 + 2 + header_len; // 魔数(6)+版本(2)+头长(2)+头内容
    file.close();
    
    // 2. 使用O_DIRECT读取数据部分
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        perror("open");
        throw std::runtime_error("Failed to open file with O_DIRECT");
    }
    
    // 计算数据大小
    size_t numel = 1;
    for (auto &dim : shape) {
        numel *= dim;
    }
    size_t elem_size = torch::elementSize(dtype);
    size_t data_size = numel * elem_size;
    
    // 计算对齐读取的位置和大小
    size_t align = 4096;  // 扇区大小，O_DIRECT的最小对齐要求
    
    // 将数据起始位置向下对齐到扇区边界
    size_t aligned_offset = (data_offset / align) * align;
    size_t prefix_size = data_offset - aligned_offset;
    
    // 计算需要读取的总大小(包括前缀和数据)
    size_t read_size = prefix_size + data_size;
    size_t aligned_size = ((read_size + align - 1) / align) * align;
    
    // 分配对齐内存
    void* aligned_buffer;
    if (posix_memalign(&aligned_buffer, align, aligned_size) != 0) {
        close(fd);
        throw std::runtime_error("Failed to allocate aligned memory");
    }
    
    // 定位到对齐的文件位置
    if (lseek(fd, aligned_offset, SEEK_SET) == -1) {
        perror("lseek");
        close(fd);
        free(aligned_buffer);
        throw std::runtime_error("Failed to seek in file");
    }
    
    // 读取数据(包含文件头部分的余留)
    ssize_t bytes_read = read(fd, aligned_buffer, aligned_size);
    close(fd);
    
    if (bytes_read < 0) {
        perror("read");
        free(aligned_buffer);
        throw std::runtime_error("Failed to read file");
    }
    
    // 从读取的数据中跳过前缀，创建张量
    auto options = torch::TensorOptions().dtype(dtype);
    void* data_ptr = static_cast<char*>(aligned_buffer) + prefix_size;
    
    // 创建张量并拷贝数据(以释放aligned_buffer)
    torch::Tensor tensor = torch::from_blob(
        data_ptr, 
        shape, 
        [aligned_buffer](void *p) { free(aligned_buffer); }, 
        options
    );
    
    return tensor.clone();
}

// // 将张量数据使用直接IO写入文件
// bool write_file_direct(torch::Tensor tensor, 
//                        std::string filename) {
//     // 确保张量是连续的
//     tensor = tensor.contiguous();
    
//     // 获取张量数据
//     void* tensor_data = tensor.data_ptr();
//     size_t elem_size = torch::elementSize(tensor.dtype());
//     size_t data_size = tensor.numel() * elem_size;
    
//     // 对齐大小 (4KB对齐)
//     size_t align = 4096;
//     size_t aligned_size = ((data_size + align - 1) / align) * align;
    
//     // 创建对齐的临时缓冲区
//     void* aligned_buffer;
//     if (posix_memalign(&aligned_buffer, align, aligned_size) != 0) {
//         std::cerr << "Failed to allocate aligned memory" << std::endl;
//         return false;
//     }
    
//     // 填充对齐缓冲区(将张量数据复制到对齐内存)
//     memcpy(aligned_buffer, tensor_data, data_size);
//     if (aligned_size > data_size) {
//         // 如果有额外空间，填充零
//         memset(static_cast<char*>(aligned_buffer) + data_size, 0, aligned_size - data_size);
//     }
    
//     // 使用O_DIRECT打开文件
//     int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
//     if (fd == -1) {
//         perror("open");
//         free(aligned_buffer);
//         throw std::runtime_error("Failed to open file for writing with O_DIRECT");
//     }
    
//     // 写入数据
//     ssize_t bytes_written = write(fd, aligned_buffer, aligned_size);
    
//     // 同步并关闭文件
//     fsync(fd);
//     close(fd);
//     free(aligned_buffer);
    
//     if (bytes_written < 0) {
//         perror("write");
//         throw std::runtime_error("Failed to write data with O_DIRECT");
//     }
    
//     // 确保写入了足够的字节
//     if (static_cast<size_t>(bytes_written) < data_size) {
//         std::cerr << "Warning: Only wrote " << bytes_written 
//                   << " bytes out of " << data_size << std::endl;
//         return false;
//     }
    
//     return true;
// }

PYBIND11_MODULE(direct_io, m) {
    m.def("read_file_direct", &read_file_direct, 
          "Read file with O_DIRECT and return as tensor");
    
    // m.def("write_file_direct", &write_file_direct, 
    //       "Write tensor to file with O_DIRECT",
    //       py::arg("tensor"), py::arg("filename"));
}


