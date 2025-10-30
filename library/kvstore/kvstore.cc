#include "kvstore.h"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <pybind11/stl.h>
#include <omp.h>

KVStore::KVStore(){
    this->allocated = false;
    this->kv_meta = new std::unordered_map<uint64_t, FileOffsetInfo>();
    this->persisted = false;
}

KVStore::~KVStore(){
    if (this->allocated) {
        for (int i = 0; i < this->num_layers; ++i)
        {
            delete [] this->key_cache[i];
            delete [] this->value_cache[i];
        }
        delete kv_meta;
        delete this->queried_key;
        delete this->queried_value;
        this->allocated = false;
    }
}

void KVStore::alloc(
    int num_layers,
    int num_attention_heads,
    int num_key_value_heads, 
    int head_dim,
    int max_length
) {
    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->head_dim = head_dim;
    this->max_length = max_length;

    for (int i = 0; i < this->num_layers; i++){
        DTYPE * k = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
        DTYPE * v = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];

        memset(k, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        memset(v, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

        this->key_cache.push_back(k);
        this->value_cache.push_back(v);
    }

    this->queried_key = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
    this->queried_value = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    //this->layer_stats.assign(num_layers, LayerStats());

}

void KVStore::clear() {
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for(int i = 0; i < this->num_layers; ++i){
        memset(this->key_cache[i], 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        memset(this->value_cache[i], 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    }

    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
}

void KVStore::fill(
    int layer_id,
    torch::Tensor k, 
    torch::Tensor v)
{
    DTYPE * key = this->key_cache[layer_id];
    DTYPE * value = this->value_cache[layer_id];

    int seq_len = k.size(1);
    this->offload_len = seq_len;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->num_key_value_heads; ++i){
        memcpy(key + i * this->max_length * this->head_dim, static_cast<DTYPE *>(k.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(DTYPE));
        memcpy(value + i * this->max_length * this->head_dim, static_cast<DTYPE *>(v.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(DTYPE));

    }
}

uint64_t KVStore::get_meta_id(int token_id, int layer_id, int head_id){
    return layer_id * this->num_key_value_heads * this->max_length + head_id * this->max_length + token_id;
}

void KVStore::persist_meta(int prefix_id) {
    std::string meta_file = this->store_path + "/" + std::to_string(prefix_id) + ".meta";
    std::ofstream file(meta_file, std::ios::app);
    std::stringstream ss;
    for (const auto &[id, meta] : *kv_meta) {
        ss << id << " " << meta.file_index << " " << meta.offset << "\n";
    }
    file.write(ss.str().data(), ss.str().size());
    file.flush();
    file.close();
}

void KVStore::recover_meta(std::string path, int prefix_id) {
    this->store_path = path;
    std::string meta_file = this->store_path + "/" + std::to_string(prefix_id) + ".meta";
    std::ifstream file(meta_file);
    assert(this->file_meta.size() == 0);

    file.seekg(0, std::ios::beg);
    int meta_id;
    uint64_t file_index, offset;
    int lines = 0;
    while(file >> meta_id >> file_index >> offset) {
        kv_meta->insert({meta_id, FileOffsetInfo(file_index, offset)});
    }
    file.close();
    this->persisted = true;
}

// void KVStore::write_to_storage(
//     std::string path,
//     int prefix_id,
//     int layer_id,
//     const std::vector<std::vector<int>>& strategy 
// ) {

//     this->store_path = path;
//     DTYPE * k = this->key_cache[layer_id];
//     DTYPE * v = this->value_cache[layer_id];

//     std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + ".bin";
//     int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_DIRECT | O_APPEND, 0644);
//     if (fd == -1) {
//         perror("open");
//         return;
//     }
//     off_t current_size = lseek(fd, 0, SEEK_END);
    
//     const size_t alignment = 512;
//     const size_t dtype_size = sizeof(DTYPE);

//     size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value
//     size_t total_size = entry_size * this->offload_len * this->num_key_value_heads;
//     // Ensure total_size is aligned
//     if (total_size % alignment != 0) {
//         total_size = ((total_size / alignment) + 1) * alignment;
//     }

//     DTYPE* aligned_buffer;
//     posix_memalign((void**)&aligned_buffer, alignment, total_size);
//     memset(aligned_buffer, 0, total_size);

//     for (int i = 0; i < this->num_key_value_heads; i++){
//         const std::vector<int>& head_strategy = strategy[i];
//         size_t head_entries = head_strategy.size();

//         DTYPE* head_key = k + i * this->max_length * this->head_dim;
//         DTYPE* head_value = v + i * this->max_length * this->head_dim;

//         for (size_t j = 0; j < head_entries; j++) {
//             int idx = head_strategy[j];

//             DTYPE* cur_key = head_key + idx * this->head_dim;
//             DTYPE* cur_value = head_value + idx * this->head_dim;

//             // Calculate aligned offsets
//             size_t buffer_offset = i * entry_size * this->offload_len + j * entry_size;
//             DTYPE* key_dest = aligned_buffer + (buffer_offset / dtype_size);
//             DTYPE* value_dest = key_dest + this->head_dim;

//             // Verify alignment of destination pointers
//             assert(reinterpret_cast<uintptr_t>(key_dest) % alignment == 0);
//             assert(reinterpret_cast<uintptr_t>(value_dest) % alignment == 0);

//             // copy key
//             memcpy(key_dest, cur_key, this->head_dim * sizeof(DTYPE));
//             // copy value
//             memcpy(value_dest, cur_value, this->head_dim * sizeof(DTYPE));

//             // record meta
//             size_t file_offset = current_size + buffer_offset;
//             uint64_t meta_id = get_meta_id(idx, layer_id, i);
//             kv_meta->insert({meta_id, uint64_t(file_offset)});
//         }
//     }
    
//     ssize_t written = write(fd, aligned_buffer, total_size);
//     free(aligned_buffer);
//     close(fd);
//     this->persisted = true;

// }
void KVStore::write_to_layer_promote_file(
    std::string path, 
    int prefix_id, 
    int layer_id, 
    const std::vector<std::vector<int>>& strategy
) {
    this->store_path = path;
    DTYPE * k = this->key_cache[layer_id];
    DTYPE * v = this->value_cache[layer_id];

    int seq_len = this->offload_len;
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value

    std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id) + "_part0.bin";
    std::ofstream file(file_name, std::ios::app | std::ios::binary);  //std::ios::app 所有层所有头所有token的数据保存在一个文件里面

    for (int i = 0; i < this->num_key_value_heads; i++){
        const std::vector<int>& head_strategy = strategy[i];
        size_t head_entries = head_strategy.size();
        size_t total_size = head_entries * entry_size;


        DTYPE* head_key = k + i * this->max_length * this->head_dim;  //当前注意力头的数据存放的位置
        DTYPE* head_value = v + i * this->max_length * this->head_dim;

        for (size_t j = 0; j < head_entries; j++) {
            int idx = head_strategy[j];

            std::streampos head_base_offset = file.tellp();
            std::streampos entry_offset = head_base_offset;
            DTYPE* cur_key = head_key + idx * this->head_dim; //当前token的数据位置
            DTYPE* cur_value = head_value + idx * this->head_dim;
            
            file.write(reinterpret_cast<char*>(cur_key), this->head_dim * sizeof(DTYPE));
            file.write(reinterpret_cast<char*>(cur_value), this->head_dim * sizeof(DTYPE));
            uint64_t meta_id = get_meta_id(idx, layer_id, i);
            FileOffsetInfo file_offset_info(uint64_t(0), uint64_t(entry_offset));
            kv_meta->insert({meta_id, file_offset_info});
        }
    }
    file.flush();
    file.close();
    this->persisted = true;
}

void KVStore::write_to_layer_file(
    std::string path, 
    int prefix_id, 
    int layer_id, 
    const std::vector<std::vector<int>>& strategy
) {
    this->store_path = path;
    DTYPE * k = this->key_cache[layer_id];
    DTYPE * v = this->value_cache[layer_id];

    const int tokens_per_file = 4096; // max tokens per file

    std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id) + ".bin";
    std::ofstream file(file_name, std::ios::app | std::ios::binary);  //std::ios::app 所有层所有头所有token的数据保存在一个文件里面
    
    // 准备按token范围分片存储 所有层所有头的一部分kv保存在一个文件，

    int seq_len = this->offload_len;
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value


    for (int i = 0; i < this->num_key_value_heads; i++){
        const std::vector<int>& head_strategy = strategy[i];
        size_t head_entries = head_strategy.size();
        size_t total_size = head_entries * entry_size;
        // std::streampos head_base_offset = file.tellp();

        // DTYPE* key_value = new DTYPE[head_entries * this->head_dim * 2]; // key + value
        // memset(key_value, 0, head_entries * this->head_dim * 2 * sizeof(DTYPE));

        DTYPE* head_key = k + i * this->max_length * this->head_dim;  //当前注意力头的数据存放的位置
        DTYPE* head_value = v + i * this->max_length * this->head_dim;

        for (size_t j = 0; j < head_entries; j++) {
            int idx = head_strategy[j];

            std::streampos head_base_offset = file.tellp();
            //std::streampos entry_offset = head_base_offset + entry_size;
            std::streampos entry_offset = head_base_offset;
            DTYPE* cur_key = head_key + idx * this->head_dim; //当前token的数据位置
            DTYPE* cur_value = head_value + idx * this->head_dim;
            
            file.write(reinterpret_cast<char*>(cur_key), this->head_dim * sizeof(DTYPE));
            file.write(reinterpret_cast<char*>(cur_value), this->head_dim * sizeof(DTYPE));
            // // copy key
            // memcpy(key_value + j * 2 * this->head_dim, cur_key, this->head_dim * sizeof(DTYPE));
            // // copy value
            // memcpy(key_value + j * 2 * this->head_dim + this->head_dim, cur_value, this->head_dim * sizeof(DTYPE));

            // // record meta
            // std::streampos entry_offset = head_base_offset + j * entry_size;
            uint64_t meta_id = get_meta_id(idx, layer_id, i);
            FileOffsetInfo file_offset_info(layer_id, uint64_t(entry_offset));
            kv_meta->insert({meta_id, file_offset_info});
            // kv_meta->insert({meta_id, uint64_t(entry_offset)});
        }

        // file.write(reinterpret_cast<const char*>(key_value), total_size);
        // delete[] key_value;
        
    }
    file.flush();
    file.close();
    this->persisted = true;
}


void KVStore::write_to_file(
    std::string path,
    int prefix_id,
    int layer_id, 
    const std::vector<std::vector<int>>& strategy 
) {
    this->store_path = path;
    DTYPE * k = this->key_cache[layer_id];
    DTYPE * v = this->value_cache[layer_id];

    const int tokens_per_file = 128; // max tokens per file

    // std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + ".bin";
    // std::ofstream file(file_name, std::ios::app | std::ios::binary);  //std::ios::app 所有层所有头所有token的数据保存在一个文件里面
    
    // 准备按token范围分片存储 所有层所有头的一部分kv保存在一个文件，

    int seq_len = this->offload_len;
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value

    //DTYPE * key_value = new DTYPE[seq_len * this->head_dim * 2];
    //memset(key_value, 0, seq_len * this->head_dim * 2 * sizeof(DTYPE));
    int num_files = (seq_len + tokens_per_file - 1) / tokens_per_file;
    std::vector<std::ofstream> files(num_files);
    // 打开/创建 所有分片文件
    for (int file_index = 0; file_index < num_files; ++file_index) {
        std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_part" + std::to_string(file_index) + ".bin";
        files[file_index].open(file_name, std::ios::binary | std::ios::app);
        if (!files[file_index].is_open()) {
            std::cerr << "Failed to open file: " << file_name << std::endl;
            return;
        }
    }
    
    int file_index = 0;
    int current_token_nums = 0;

    for (int i = 0; i < this->num_key_value_heads; i++){
        const std::vector<int>& head_strategy = strategy[i];
        size_t head_entries = head_strategy.size();
        size_t total_size = head_entries * entry_size;
        // std::streampos head_base_offset = file.tellp();

        // DTYPE* key_value = new DTYPE[head_entries * this->head_dim * 2]; // key + value
        // memset(key_value, 0, head_entries * this->head_dim * 2 * sizeof(DTYPE));

        DTYPE* head_key = k + i * this->max_length * this->head_dim;  //当前注意力头的数据存放的位置
        DTYPE* head_value = v + i * this->max_length * this->head_dim;

        for (size_t j = 0; j < head_entries; j++) {
            int idx = head_strategy[j];
            int file_index = j / tokens_per_file;

            std::streampos head_base_offset = files[file_index].tellp();
            //std::streampos entry_offset = head_base_offset + entry_size;
            std::streampos entry_offset = head_base_offset;
            DTYPE* cur_key = head_key + idx * this->head_dim; //当前token的数据位置
            DTYPE* cur_value = head_value + idx * this->head_dim;
            
            files[file_index].write(reinterpret_cast<char*>(cur_key), this->head_dim * sizeof(DTYPE));
            files[file_index].write(reinterpret_cast<char*>(cur_value), this->head_dim * sizeof(DTYPE));
            // // copy key
            // memcpy(key_value + j * 2 * this->head_dim, cur_key, this->head_dim * sizeof(DTYPE));
            // // copy value
            // memcpy(key_value + j * 2 * this->head_dim + this->head_dim, cur_value, this->head_dim * sizeof(DTYPE));

            // // record meta
            // std::streampos entry_offset = head_base_offset + j * entry_size;
            uint64_t meta_id = get_meta_id(idx, layer_id, i);
            FileOffsetInfo file_offset_info(file_index, uint64_t(entry_offset));
            kv_meta->insert({meta_id, file_offset_info});
            // kv_meta->insert({meta_id, uint64_t(entry_offset)});
        }

        // file.write(reinterpret_cast<const char*>(key_value), total_size);
        // delete[] key_value;
        
    }
    for(auto &file : files){
        file.flush();
        file.close();
    }
    this->persisted = true;
}


void KVStore::collect_queried_key_value(
    int prefix_id,
    int layer_id, 
    torch::Tensor ind_pt, 
    torch::Tensor nnz_pt
) {
    //memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    //memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());

    int attn_group = this->num_attention_heads / this->num_key_value_heads;

    
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);

    //handle persisted=false case; collect kv from memory
    if (this->persisted == false) {
      int stride = this->max_length * this->head_dim; // 一个头完整数据的长度
      DTYPE* key = this->key_cache[layer_id];
      DTYPE* value = this->value_cache[layer_id];
      for (int i = 0; i < this->num_key_value_heads; i++) {
        auto head_ind = ind + i * this->max_length;
        int num_indices = nnz[i];
        auto queried_key_ptr = this->queried_key + i * stride; // 当前头要查询的数据位置
        auto queried_value_ptr = this->queried_value + i * stride;
        auto key_ptr = key + i * stride; // 当前头完整数据位置
        auto value_ptr = value + i * stride;
        for (int j = 0; j < num_indices; j++) {
          auto cur_ind = head_ind[j];
          memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
          memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
        }
      }
      return;
    }
    #pragma omp parallel for schedule(static) num_threads(32)
    for(int i = 0; i < this->num_key_value_heads; i++){
        std::vector<int> token_order;
        std::vector<std::tuple<uint64_t, uint64_t, int>> content; // offset, length in bytes

        auto head_ind = ind + i * this->max_length;
        //std::set<std::pair<uint64_t, uint64_t>> queried_meta_offset;
        std::set<std::tuple<uint64_t, uint64_t, int>> queried_meta_offset;
        int num_indices = nnz[i];
        for (int j = 0; j < num_indices; j++) {
            uint64_t meta_id = get_meta_id(head_ind[j], layer_id, i);
            FileOffsetInfo& info = kv_meta->at(meta_id);
            uint64_t file_index = info.file_index;
            uint64_t offset = info.offset;
            queried_meta_offset.insert({file_index, offset, j});
        }

        // merge contiguous offset
        if (!queried_meta_offset.empty()) {
            auto it = queried_meta_offset.begin();
            // uint64_t start_file_index = (*it).first; uint64_t start_offset = (*it).second;
            // uint64_t prev_file_index = (*it).first; uint64_t prev_offset = (*it).second;
            auto [start_file_index, start_offset, key_index] = *it; 
            uint64_t prev_file_index = start_file_index;
            uint64_t prev_offset = start_offset;
            token_order.emplace_back(key_index);
            int count = 1;
            ++it;

            for (; it != queried_meta_offset.end(); ++it) {
                //uint64_t cur_file_index = (*it).first; uint64_t cur_offset = (*it).second;
                auto [cur_file_index, cur_offset, cur_key_index] = *it; 
                token_order.emplace_back(cur_key_index);
                if (cur_file_index == prev_file_index && cur_offset == prev_offset + entry_size) {
                    count++;
                } else {
                    content.emplace_back(start_file_index, start_offset, count * entry_size);
                    start_file_index = cur_file_index;
                    start_offset = cur_offset;
                    count = 1;
                }
                prev_file_index = cur_file_index, prev_offset = cur_offset;
            }
            content.emplace_back(start_file_index, start_offset, count * entry_size);
        }

        // load from storage
        //this->load_key_value(content, prefix_id, layer_id, i);
        //this->load_key_value_from_file(content, prefix_id, layer_id, i);

        //this->load_key_value(content, token_order, prefix_id, layer_id, i);
        this->load_key_value_from_file(content, token_order, prefix_id, layer_id, i);

        //this->load_key_value_from_file_concurrent(content, token_order, prefix_id, layer_id, i);
    }
}
void KVStore::merge_collect_queried_key_value(
    int prefix_id,
    int layer_id, 
    torch::Tensor ind_pt, 
    torch::Tensor nnz_pt
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());

    int attn_group = this->num_attention_heads / this->num_key_value_heads;

    uint64_t io_size = 4096;
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);

    //handle persisted=false case; collect kv from memory
    if (this->persisted == false) {
      int stride = this->max_length * this->head_dim; // 一个头完整数据的长度
      DTYPE* key = this->key_cache[layer_id];
      DTYPE* value = this->value_cache[layer_id];
      for (int i = 0; i < this->num_key_value_heads; i++) {
        auto head_ind = ind + i * this->max_length;
        int num_indices = nnz[i];
        auto queried_key_ptr = this->queried_key + i * stride; // 当前头要查询的数据位置
        auto queried_value_ptr = this->queried_value + i * stride;
        auto key_ptr = key + i * stride; // 当前头完整数据位置
        auto value_ptr = value + i * stride;
        for (int j = 0; j < num_indices; j++) {
          auto cur_ind = head_ind[j];
          memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
          memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
        }
      }
      return;
    }

    // std::vector<int> token_order;
    // std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>> content; //(file_index, start_offset, length, bitmap)

    #pragma omp parallel for schedule(static) num_threads(32)
    for(int i = 0; i < this->num_key_value_heads; i++){
        std::vector<int> token_order;
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>> content; //(file_index, start_offset, length, bitmap)
        auto head_ind  = ind + i * this->max_length;
        // std::set<std::tuple<uint64_t, uint64_t, int>> queried_meta_offset;

        // int num_indices = nnz[i];

        // for (int j = 0; j < num_indices; j++) {
        //     uint64_t meta_id = get_meta_id(head_ind[j], layer_id, i);
        //     FileOffsetInfo& info = kv_meta->at(meta_id);
        //     uint64_t file_index = info.file_index;
        //     uint64_t offset = info.offset;
        //     queried_meta_offset.insert({file_index, offset, j});
        // }
        int num_indices = nnz[i];
        std::vector<std::tuple<uint64_t, uint64_t, int>> queried_meta_offset;
        queried_meta_offset.reserve(num_indices); // 预分配内存
        for (int j = 0; j < num_indices; j++) {
            uint64_t meta_id = get_meta_id(head_ind[j], layer_id, i);
            FileOffsetInfo& info = kv_meta->at(meta_id);
            uint64_t file_index = info.file_index;
            uint64_t offset = info.offset;
            queried_meta_offset.emplace_back(file_index, offset, j);
        }
        std::sort(queried_meta_offset.begin(), queried_meta_offset.end());
        
        if(!queried_meta_offset.empty()){
            auto [start_file_index, start_offset, token_index] = *queried_meta_offset.begin();
            token_order.emplace_back(token_index);
            uint64_t prev_file_index = start_file_index;  uint64_t prev_offset = start_offset;
            
            int count = 1;
            for(auto it = next(queried_meta_offset.begin()); it != queried_meta_offset.end(); it++){
                auto [cur_file_index, cur_offset, token_index] = *it;
                token_order.emplace_back(token_index);

                if(prev_file_index == cur_file_index && prev_offset + entry_size == cur_offset){
                    count++;
                }else{
                    //new seg: [start_file_index, start_offset, count * entry_size]
                    if(content.empty()){
                        int bitmap = -1;
                        if(count * entry_size <= io_size) bitmap = (1<<count) - 1;
                        content.emplace_back(start_file_index, start_offset, count * entry_size, bitmap);
                    }else{
                        auto &[last_file_index, last_start_offset, last_length, last_bitmap] = content.back();
                        uint64_t last_end_offset = last_start_offset + last_length;
                        uint64_t gap = start_offset - last_end_offset;
                        if(last_file_index == start_file_index && last_length + gap + count * entry_size <= io_size){
                            int start_bit = last_length / entry_size + gap / entry_size;
                            int end_bit = start_bit + count;
                            for(int b = start_bit; b < end_bit; b++) last_bitmap |= (1 << b);

                            last_length += gap + count * entry_size;
                        }else{
                            int bitmap = -1;
                            if(count * entry_size <= io_size) bitmap = (1<<count) - 1;
                            content.emplace_back(start_file_index, start_offset, count * entry_size, bitmap);
                        }
                    }
                    start_file_index = cur_file_index;
                    start_offset = cur_offset;
                    count = 1;
                }
                prev_file_index = cur_file_index, prev_offset = cur_offset;
            }
            int bitmap = -1;
            if(count * entry_size <= io_size) bitmap = (1 << count) - 1;
            content.emplace_back(start_file_index, start_offset, count * entry_size, bitmap);
        }
        //this->load_key_value(content, token_order, prefix_id, layer_id, i);
        //this->load_key_value_from_file(content, token_order, prefix_id, layer_id, i);
        //this->merge_load_key_value(content, token_order, prefix_id, layer_id, i);
        this->merge_load_key_value_from_file(content, token_order, prefix_id, layer_id, i);
        token_order.clear();
        content.clear();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "[TIMER] compute and IO for layer " << layer_id 
              << " took " << duration.count()  << " seconds." << std::endl;
}

void analyze_content_segments(const std::vector<std::tuple<uint64_t, uint64_t, int>>& content, int layer_id, int head_id,
                                std::vector<LayerStats> &layer_stats) {
    if (content.empty()) {
        std::cout << "Content is empty." << std::endl;
        return;
    }

    std::cout << "[Analysis Result] Layer ID: " << layer_id << ", Head ID: " << head_id << std::endl;

    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> file_segments;
    for (const auto& [file_index, offset, length] : content) {
        //std::cout<<"FileIndex: " << file_index << ", Offset: " << offset << ", Length: " << length << std::endl;
        file_segments[file_index].emplace_back(offset, offset + length);
    }

    size_t total_segments = 0;
    uint64_t total_continuous_length = 0;
    uint64_t max_continuous_length = 0;
    uint64_t max_gap = 0;

    uint64_t max_file_index = 0;

    for (auto& [file_index, segments] : file_segments) {

        max_file_index = std::max(max_file_index, file_index);

        if (segments.empty()) continue;
        // 按offset排序
        std::vector<std::pair<uint64_t, uint64_t>>&sorted_segments = segments;
        std::sort(sorted_segments.begin(), sorted_segments.end());

        uint64_t current_start = sorted_segments[0].first;
        uint64_t current_end = sorted_segments[0].second;
        size_t continuous_segment_length = 0;

        for (size_t i = 1; i < sorted_segments.size(); ++i) {
            uint64_t next_start = sorted_segments[i].first;
            uint64_t next_end = sorted_segments[i].second;

            if (next_start == current_end) {
                // 连续
                current_end = next_end;
            } else {
                // 统计当前连续区间
                uint64_t continuous_length = current_end - current_start;
                continuous_segment_length += continuous_length;
                total_continuous_length += continuous_length;
                if (continuous_length > max_continuous_length) {
                    max_continuous_length = continuous_length;
                }
                // 统计gap
                uint64_t gap = next_start - current_end;
                if (gap > max_gap) {
                    max_gap = gap;
                }
                // 开始新的连续段
                current_start = next_start;
                current_end = next_end;
            }
        }
        // 最后一个连续区间
        uint64_t last_continuous_length = current_end - current_start;
        total_continuous_length += last_continuous_length;
        continuous_segment_length += last_continuous_length;

        if (last_continuous_length > max_continuous_length) {
            max_continuous_length = last_continuous_length;
        }
        total_segments += sorted_segments.size();

        std::cout << "[FileIndex " << file_index << "] Segments: " << sorted_segments.size() << ", Avg Continuous Length: "
                  << static_cast<double>(continuous_segment_length) / sorted_segments.size() << " bytes" << std::endl;
    }

    double avg_continuous_length = total_segments > 0
        ? static_cast<double>(total_continuous_length) / total_segments
        : 0.0;

    //layer_stats[layer_id].update_len(total_segments, avg_continuous_length);
    std::cout << "Total Segments: " << total_segments << std::endl;
    std::cout << "Total Continuous Length (sum of merged segments): " << total_continuous_length << " bytes" << std::endl;
    std::cout << "Max Continuous Length: " << max_continuous_length << " bytes" << std::endl;
    //std::cout << "Max Gap between Segments: " << max_gap << " bytes" << std::endl;
    std::cout << "Average Continuous Length: " << avg_continuous_length << " bytes" << std::endl;
    std::cout<< "=====================================================================================================" << std::endl;

    std::cout << "Segments in max FileIndex " << max_file_index << ":" << std::endl;
    uint64_t cur = 0;
    for(auto &[l,r]:file_segments[max_file_index]){
        if(l != cur && cur > 0) {
            std::cout << ">>> GAP >>>\t[" << cur << " -> " << l 
                      << ")\tgap=" << (l - cur) << " bytes\t*** MISSING DATA ***" << "\t\t";
        }
        std::cout << "=== SEG ===\t[" << l << " -> " << r 
                  << ")\tsize=" << (r - l) << " bytes\t--- CONTINUOUS ---" << std::endl;
        cur = r;
    }
    std::cout<< "=====================================================================================================" << std::endl;  


    auto seg = file_segments[max_file_index];
    for(int i=0;i<seg.size();i++){
        auto [l,r] = seg[i];
        if(i>0){
            auto [li,ri] = seg[i-1];
            if(l == li || r == ri){
                std::cout << "ERROR: Duplicate segment detected!" << std::endl
                          << "  File Index:\t\t" << max_file_index << std::endl
                          << "  Current Segment [" << i << "]:\t[" << l << " -> " << r << ")" << std::endl
                          << "  Previous Segment [" << (i-1) << "]:\t[" << li << " -> " << ri << ")" << std::endl
                          << "  Conflict:\t\t" << (l == li ? "Same start offset" : "Same end offset") << std::endl
                          << "=====================================================================================================" << std::endl;
            }
        }
    }

}

void merge_analyze_content_segments(
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>>& content, 
    int layer_id, 
    int head_id,
    std::vector<LayerStats> &layer_stats)
{
    if (content.empty()) {
        std::cout << "Content is empty." << std::endl;
        return;
    }
    std::cout << "[Analysis Result] Layer ID: " << layer_id << ", Head ID: " << head_id << std::endl;
    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> file_segments;
    for (const auto& [file_index, offset, length, bitmap] : content) {
        //std::cout<<"FileIndex: " << file_index << ", Offset: " << offset << ", Length: " << length << std::endl;
        file_segments[file_index].emplace_back(offset, offset + length);
    }

    size_t total_segments = 0;
    uint64_t total_continuous_length = 0;
    uint64_t max_continuous_length = 0;
    uint64_t max_gap = 0;

    uint64_t max_file_index = 0;

    for (auto& [file_index, segments] : file_segments) {

        max_file_index = std::max(max_file_index, file_index);

        if (segments.empty()) continue;
        // 按offset排序
        std::vector<std::pair<uint64_t, uint64_t>>&sorted_segments = segments;
        std::sort(sorted_segments.begin(), sorted_segments.end());

        uint64_t current_start = sorted_segments[0].first;
        uint64_t current_end = sorted_segments[0].second;
        size_t continuous_segment_length = 0;

        for (size_t i = 1; i < sorted_segments.size(); ++i) {
            uint64_t next_start = sorted_segments[i].first;
            uint64_t next_end = sorted_segments[i].second;

            if (next_start == current_end) {
                // 连续
                current_end = next_end;
            } else {
                // 统计当前连续区间
                uint64_t continuous_length = current_end - current_start;
                continuous_segment_length += continuous_length;
                total_continuous_length += continuous_length;
                if (continuous_length > max_continuous_length) {
                    max_continuous_length = continuous_length;
                }
                // 统计gap
                uint64_t gap = next_start - current_end;
                if (gap > max_gap) {
                    max_gap = gap;
                }
                // 开始新的连续段
                current_start = next_start;
                current_end = next_end;
            }
        }
        // 最后一个连续区间
        uint64_t last_continuous_length = current_end - current_start;
        total_continuous_length += last_continuous_length;
        continuous_segment_length += last_continuous_length;

        if (last_continuous_length > max_continuous_length) {
            max_continuous_length = last_continuous_length;
        }
        total_segments += sorted_segments.size();

        std::cout << "[FileIndex " << file_index << "] Segments: " << sorted_segments.size() << ", Avg Continuous Length: "
                  << static_cast<double>(continuous_segment_length) / sorted_segments.size() << " bytes" << std::endl;
    }

    double avg_continuous_length = total_segments > 0
        ? static_cast<double>(total_continuous_length) / total_segments
        : 0.0;

    //layer_stats[layer_id].update_len(total_segments, avg_continuous_length);
    std::cout << "Total Segments: " << total_segments << std::endl;
    std::cout << "Total Continuous Length (sum of merged segments): " << total_continuous_length << " bytes" << std::endl;
    std::cout << "Max Continuous Length: " << max_continuous_length << " bytes" << std::endl;
    //std::cout << "Max Gap between Segments: " << max_gap << " bytes" << std::endl;
    std::cout << "Average Continuous Length: " << avg_continuous_length << " bytes" << std::endl;
    std::cout<< "=====================================================================================================" << std::endl;  

    std::cout << "Segments in max FileIndex " << max_file_index << ":" << std::endl;
    uint64_t cur = 0;
    for(auto &[l,r]:file_segments[max_file_index]){
        if(l != cur && cur > 0) {
            std::cout << ">>> GAP >>>\t[" << cur << " -> " << l 
                      << ")\tgap=" << (l - cur) << " bytes\t*** MISSING DATA ***" << "\t\t";
        }
        std::cout << "=== SEG ===\t[" << l << " -> " << r 
                  << ")\tsize=" << (r - l) << " bytes\t--- CONTINUOUS ---" << std::endl;
        cur = r;
    }
    std::cout<< "=====================================================================================================" << std::endl;  


    auto seg = file_segments[max_file_index];
    for(int i=0;i<seg.size();i++){
        auto [l,r] = seg[i];
        if(i>0){
            auto [li,ri] = seg[i-1];
            if(l == li || r == ri){
                std::cout << "ERROR: Duplicate segment detected!" << std::endl
                          << "  File Index:\t\t" << max_file_index << std::endl
                          << "  Current Segment [" << i << "]:\t[" << l << " -> " << r << ")" << std::endl
                          << "  Previous Segment [" << (i-1) << "]:\t[" << li << " -> " << ri << ")" << std::endl
                          << "  Conflict:\t\t" << (l == li ? "Same start offset" : "Same end offset") << std::endl
                          << "=====================================================================================================" << std::endl;
            }
        }
    }

}

void KVStore::load_key_value(
    std::vector<std::tuple<uint64_t, uint64_t, int>> &content,
    std::vector<int> &token_order,
    int prefix_id,
    int layer_id,
    int head_id)
{
    //if (layer_id == 1 && head_id == 0){
    if(head_id == 0 || head_id == 1 || head_id == 2){  
        //analyze_content_segments(content, layer_id, head_id, this->layer_stats);
    }
    const size_t alignment = 4096; // Should match filesystem block size
    const size_t dtype_size = sizeof(DTYPE);
    size_t entry_size = 2 * this->head_dim * dtype_size; // key + value

    uint64_t cur_file_index = std::numeric_limits<uint64_t>::max();
    int fd = -1; // 文件描述符
    int count = 0;

    for (const auto& [file_index, offset, length] : content) {
        // 如果文件分片改变，则切换文件描述符
        if (file_index != cur_file_index) {
            if (fd != -1) {
                close(fd);
            }
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_part" + std::to_string(file_index) + ".bin";
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(file_index) + ".bin";
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_layer" + std::to_string(layer_id) + ".bin";
            fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
            if (fd == -1) {
                perror(("open failed for " + file_name).c_str());
                //continue; // 跳过这个损坏或不存在的文件
            }
            cur_file_index = file_index;
        }

        // Calculate aligned parameters for the read
        size_t aligned_offset = (offset / alignment) * alignment;
        size_t aligned_length = ((offset % alignment) + length + alignment - 1) / alignment * alignment;
        size_t read_offset = offset - aligned_offset;
        
        // Allocate aligned buffer for the read
        DTYPE* aligned_buffer;
        if (posix_memalign((void**)&aligned_buffer, alignment, aligned_length) != 0) {
            close(fd);
            throw std::runtime_error("Failed to allocate aligned buffer");
        }
        // Perform aligned read
        if (lseek(fd, aligned_offset, SEEK_SET) == -1) {
            perror("lseek");
            free(aligned_buffer);
            continue;
        }

        ssize_t bytes_read = read(fd, aligned_buffer, aligned_length);
        if (bytes_read == -1) {
            perror("read");
            free(aligned_buffer);
            continue;
        }
        // Verify we got enough data
        if (static_cast<size_t>(bytes_read) < (read_offset + length)) {
            free(aligned_buffer);
            throw std::runtime_error("Read returned insufficient data");
        }
        // Process the actual data we need (starting at read_offset, length bytes)
        char* data_start = reinterpret_cast<char*>(aligned_buffer) + read_offset;
        int num_entries = length / entry_size;

        for (int i = 0; i < num_entries; i++) {
            DTYPE* entry = reinterpret_cast<DTYPE*>(data_start + i * entry_size);
            DTYPE* cur_key = entry;
            DTYPE* cur_value = entry + this->head_dim;

            // 写入 queried_key 和 queried_value 中对应 head_id 和 position
            int token_index = token_order[count];
            //int key_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
            //int value_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
            int key_offset = head_id * this->max_length * this->head_dim + token_index * this->head_dim;
            int value_offset = head_id * this->max_length * this->head_dim + token_index * this->head_dim;
            memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
            memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

            count++;
        }
        free(aligned_buffer);
    }

    if (fd != -1) {
        close(fd);
    }
}

void KVStore::load_key_value_from_file(
    std::vector<std::tuple<uint64_t,uint64_t, int>>& content,
    std::vector<int> &token_order,
    int prefix_id,
    int layer_id,
    int head_id)
{
    //auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间
    // std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + ".bin";
    // std::ifstream file(file_name, std::ios::binary);
    // if (!file.is_open()) {
    //     std::cerr << "Failed to open file: " << file_name << std::endl;
    //     return;
    // }
    //////// ananlyze
    //if (layer_id == 1 && head_id == 0){

    if (head_id == 0 || head_id == 1 || head_id == 2){  
        //analyze_content_segments(content, layer_id, head_id, this->layer_stats);
    }


    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value
    int count = 0;

    uint64_t cur_file_index = std::numeric_limits<uint64_t>::max();
    std::ifstream file;

    for (const auto& [file_index, offset, length] : content) {
        // change file
        if(file_index != cur_file_index){
            if (file.is_open()) file.close();
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_part" + std::to_string(file_index) + ".bin";
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(file_index) + ".bin";
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_layer" + std::to_string(layer_id) + ".bin";
            file.open(file_name, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << file_name << std::endl;
                continue;
            }
            cur_file_index = file_index;
        }
        
        int num_entries = length / entry_size;

        std::vector<char> buffer(length);
        file.seekg(offset, std::ios::beg);
        file.read(buffer.data(), length);

        for (int i = 0; i < num_entries; i++) {
            DTYPE* entry = reinterpret_cast<DTYPE*>(buffer.data() + i * entry_size);
            DTYPE* cur_key = entry;
            DTYPE* cur_value = entry + this->head_dim;

            // 写入 queried_key 和 queried_value 中对应 head_id 和 position
            // int key_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
            // int value_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
            int token_index = token_order[count];
            int key_offset = head_id * this->max_length * this->head_dim + token_index * this->head_dim;
            int value_offset = head_id * this->max_length * this->head_dim + token_index * this->head_dim;

            memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
            memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

            count++;
        }
    }

    if (file.is_open()) file.close();
    //auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    //double elapsed = std::chrono::duration<double>(end - start).count();
    // printf("[KVStore::load_key_value_from_file] elapsed time: %.6f seconds\n", elapsed);
}

void KVStore::load_key_value_from_file_concurrent(
    std::vector<std::tuple<uint64_t,uint64_t, int>>& content,
    std::vector<int> &token_order,
    int prefix_id,
    int layer_id,
    int head_id)
{
    // Currently no-op; placeholder for potential concurrent loading implementation
    if (content.empty()) return;

    using TaskIdx = size_t;
    const size_t dtype_size = sizeof(DTYPE);
    const size_t entry_size = 2 * this->head_dim * dtype_size; // key + value


    struct Task{
        uint64_t file_index;
        uint64_t offset;
        uint64_t length;
        int start_pos;
    };
    std::vector<Task> tasks;
    tasks.reserve(content.size());

    int cum_entries = 0;
    for (const auto &seg : content) {
        uint64_t file_index;
        uint64_t offset;
        int length;
        std::tie(file_index, offset, length) = seg;
        int entries = static_cast<int>(length / entry_size);
        tasks.push_back(Task{file_index, offset, length, cum_entries});
        cum_entries += entries;
    }
    if (tasks.empty()) return;

    // Determine number of workers
    unsigned int hw = std::thread::hardware_concurrency();
    int num_workers = hw > 0 ? std::min<unsigned int>(hw, static_cast<unsigned int>(tasks.size())) : std::min<int>(4, static_cast<int>(tasks.size()));
    if (num_workers <= 0) num_workers = 1;
    
    num_workers = 4;

    std::atomic<TaskIdx> next_idx(0);

    auto worker = [&](int worker_id){
        // Per-worker fd cache to avoid frequent open/close
        std::unordered_map<uint64_t, int> fd_cache;

        std::vector<char> local_buf; // will be resized per task as needed

        while(true){
            TaskIdx idx = next_idx.fetch_add(1);
            if (idx >= tasks.size()) break;
            const Task &t = tasks[idx];

            //std::string file_name= this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id) + "_part" + std::to_string(t.file_index) + ".bin";
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id) + ".bin";

            int fd = -1;
            auto it = fd_cache.find(t.file_index);
            if (it != fd_cache.end()) {
                fd = it->second;
            } else {
                fd = open(file_name.c_str(), O_RDONLY);
                if (fd == -1) {
                    perror(("open failed for " + file_name).c_str());
                    continue;
                }
                fd_cache[t.file_index] = fd;
            }
            if ((int)local_buf.size() < t.length) local_buf.resize(t.length);
            ssize_t got = pread(fd, local_buf.data(), t.length, static_cast<off_t>(t.offset));

            if (got <= 0) {
                perror("pread failed");
                continue;
            }
            int num_entries = t.length / static_cast<int>(entry_size);

            for (int e = 0; e < num_entries; ++e) {
                DTYPE* entry_ptr = reinterpret_cast<DTYPE*>(local_buf.data() + e * entry_size);
                DTYPE* cur_key = entry_ptr;
                DTYPE* cur_value = entry_ptr + this->head_dim;

                int token_pos_in_order = token_order[t.start_pos + e]; // unique write slot
                size_t key_offset = static_cast<size_t>(head_id) * this->max_length * this->head_dim + static_cast<size_t>(token_pos_in_order) * this->head_dim;

                // memcpy into queried buffers (no locks required: non-overlapping)
                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + key_offset, cur_value, this->head_dim * sizeof(DTYPE));
            }
        }
        // close cached fds
        for (auto &p : fd_cache) {
            if (p.second != -1) close(p.second);
        }
    };
    // spawn workers
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (int w = 0; w < num_workers; ++w) workers.emplace_back(worker, w);
    for (auto &th : workers) if (th.joinable()) th.join();
    return;
}


void KVStore::merge_load_key_value(
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>> &content,
    std::vector<int> &token_order,
    int prefix_id,
    int layer_id,
    int head_id)
{
    if(head_id == 0 || head_id == 1 || head_id == 2){  
        //merge_analyze_content_segments(content, layer_id, head_id, this->layer_stats);
    }
    const size_t alignment = 4096; // Should match filesystem block size
    const size_t dtype_size = sizeof(DTYPE);
    size_t entry_size = 2 * this->head_dim * dtype_size; // key + value

    uint64_t cur_file_index = std::numeric_limits<uint64_t>::max();
    int fd = -1; // 文件描述符
    int count = 0;

    for(const auto& [file_index, offset, length, bitmap] : content) {
        // 如果文件分片改变，则切换文件描述符
        if (file_index != cur_file_index) {
            if (fd != -1) {
                close(fd);
            }
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_part" + std::to_string(file_index) + ".bin";
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(file_index) + ".bin";
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_layer" + std::to_string(layer_id) + ".bin";
            fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
            if (fd == -1) {
                perror(("open failed for " + file_name).c_str());
                //continue; // 跳过这个损坏或不存在的文件
            }
            cur_file_index = file_index;
        }
        // Calculate aligned parameters for the read
        size_t aligned_offset = (offset / alignment) * alignment;
        size_t aligned_length = ((offset % alignment) + length + alignment - 1) / alignment * alignment;
        size_t read_offset = offset - aligned_offset;
        
        // Allocate aligned buffer for the read
        DTYPE* aligned_buffer;
        if (posix_memalign((void**)&aligned_buffer, alignment, aligned_length) != 0) {
            close(fd);
            throw std::runtime_error("Failed to allocate aligned buffer");
        }
        // Perform aligned read
        if (lseek(fd, aligned_offset, SEEK_SET) == -1) {
            perror("lseek");
            free(aligned_buffer);
            continue;
        }

        ssize_t bytes_read = read(fd, aligned_buffer, aligned_length);
        if (bytes_read == -1) {
            perror("read");
            free(aligned_buffer);
            continue;
        }
        // Verify we got enough data
        if (static_cast<size_t>(bytes_read) < (read_offset + length)) {
            free(aligned_buffer);
            throw std::runtime_error("Read returned insufficient data");
        }
        // Process the actual data we need (starting at read_offset, length bytes)
        char* data_start = reinterpret_cast<char*>(aligned_buffer) + read_offset;
        

        if(bitmap == -1){ 
            int num_entries = length / entry_size;
            for (int i = 0; i < num_entries; i++) {
                DTYPE* entry = reinterpret_cast<DTYPE*>(data_start + i * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;

                // 写入 queried_key 和 queried_value 中对应 head_id 和 position
                int token_index = token_order[count];
                //int key_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
                //int value_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
                int key_offset = head_id * this->max_length * this->head_dim + token_index * this->head_dim;
                int value_offset = head_id * this->max_length * this->head_dim + token_index * this->head_dim;
                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

                count++;
            }
        }else{
            int tmp_bitmap = bitmap;
            while(tmp_bitmap != 0){
                int pos = __builtin_ffs(tmp_bitmap) - 1;
                tmp_bitmap &= (tmp_bitmap - 1);

                DTYPE* entry = reinterpret_cast<DTYPE*>(data_start + pos * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;

                int key_index = token_order[count];
                int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                int value_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;

                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

                count++;
            }
        }
        free(aligned_buffer);
    }
        
    if (fd != -1) {
        close(fd);
    }
}

void KVStore::merge_load_key_value_from_file(
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>>& content,
    std::vector<int> &token_order,
    int prefix_id,
    int layer_id,
    int head_id
) {
    if(head_id == 0 || head_id == 5){
        //merge_analyze_content_segments(content, layer_id, head_id, this->layer_stats);
    }
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value
    int count = 0;

    uint64_t cur_file_index = std::numeric_limits<uint64_t>::max();
    std::ifstream file;

    for(const auto& [file_index, start_offset, length, bitmap] : content){
        // change file
        if(file_index != cur_file_index){
            if (file.is_open()) file.close();
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_part" + std::to_string(file_index) + ".bin";
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(file_index) + ".bin";
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_layer" + std::to_string(layer_id) + ".bin";
            file.open(file_name, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << file_name << std::endl;
                continue;
            }
            cur_file_index = file_index;
        }

        int num_entries = length / entry_size;
        std::vector<char> buffer(length);
        file.seekg(start_offset, std::ios::beg);
        file.read(buffer.data(), length);

        if(bitmap == -1){
            for (int i = 0; i < num_entries; i++) {
                DTYPE* entry = reinterpret_cast<DTYPE*>(buffer.data() + i * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;

                // 写入 queried_key 和 queried_value 中对应 head_id 和 position
                // int key_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
                // int value_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
                int key_index = token_order[count];
                int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                int value_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;

                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

                count++;
            }
        }else{
            int tmp_bitmap = bitmap;
            while(tmp_bitmap != 0){
                int pos = __builtin_ffs(tmp_bitmap) - 1;
                tmp_bitmap &= (tmp_bitmap - 1);
                
                DTYPE* entry = reinterpret_cast<DTYPE*>(buffer.data() + pos * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;
                
                int key_index = token_order[count];
                int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                int value_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;

                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

                count++;
            }
        }
    }
    if (file.is_open()) file.close();
}

void KVStore::promote_persist(std::string path, int prefix_id, int layer_id, const torch::Tensor &promote_token_info){
    if (promote_token_info.numel() == 0) {
        return; // 没有需要提升的token，直接返回
    }

    this->store_path = path;
    const size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);

    // 1. 按源文件对要提升的token进行分组
    //    map<source_file_index, vector<pair<meta_id, source_offset>>>
    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> promotions_by_file;
    auto accessor = promote_token_info.accessor<long, 2>();
    for (int i = 0; i < accessor.size(0); ++i) {
        int head_id = accessor[i][0];
        int token_id = accessor[i][1];
        
        uint64_t meta_id = get_meta_id(token_id, layer_id, head_id);
        
        if (kv_meta->count(meta_id)) {
            const FileOffsetInfo& info = kv_meta->at(meta_id);
            promotions_by_file[info.file_index].push_back({meta_id, info.offset});
        }
    }
    // 2. 遍历每个源文件，执行读写和元数据更新操作
    for (auto const& [source_file_index, promotions] : promotions_by_file) {
        uint64_t dest_file_index = source_file_index + 1;

        std::string source_file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(source_file_index) + ".bin";
        std::string dest_file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(dest_file_index) + ".bin";

        std::ifstream source_file(source_file_name, std::ios::binary);
        // 使用追加模式打开目标文件
        std::ofstream dest_file(dest_file_name, std::ios::binary | std::ios::app);

        if (!source_file.is_open() || !dest_file.is_open()) {
            std::cerr << "Error opening source or destination file for promotion." << std::endl;
            continue;
        }

        std::vector<char> buffer(entry_size);

        for (const auto& promotion_info : promotions) {
            uint64_t meta_id = promotion_info.first;
            uint64_t source_offset = promotion_info.second;

            // a. 从源文件读取KV数据
            source_file.seekg(source_offset, std::ios::beg);
            source_file.read(buffer.data(), entry_size);

            // b. 获取目标文件的当前末尾位置，作为新的offset
            std::streampos new_offset = dest_file.tellp();

            // c. 将数据追加写入到目标文件
            dest_file.write(buffer.data(), entry_size);

            // d. 更新内存中的kv_meta信息
            (*kv_meta)[meta_id] = FileOffsetInfo(dest_file_index, static_cast<uint64_t>(new_offset));
        }

        source_file.close();
        dest_file.flush();
        dest_file.close();
    }
    //overwrite_persist_meta(prefix_id);
}

torch::Tensor KVStore::to_tensor(DTYPE *start, int length)
{
    // auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    torch::Tensor tensor = torch::from_blob(start, {length, this->head_dim}, options);
    return tensor;
}

torch::Tensor KVStore::get_queried_key_cache()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    // auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(this->queried_key, {this->num_key_value_heads, this->max_length, this->head_dim}, options);
    return tensor;
}

torch::Tensor KVStore::get_queried_value_cache()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    // auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(this->queried_value, {this->num_key_value_heads, this->max_length, this->head_dim}, options);
    return tensor;
}

void KVStore::concurrent_merge_collect_queried_key_value(
    int prefix_id,
    int layer_id, 
    torch::Tensor ind_pt, 
    torch::Tensor nnz_pt
){
    auto start_time = std::chrono::high_resolution_clock::now();

    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());

    int attn_group = this->num_attention_heads / this->num_key_value_heads;

    uint64_t io_size = 4096;
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    //handle persisted=false case; collect kv from memory
    if (this->persisted == false) {
      int stride = this->max_length * this->head_dim; // 一个头完整数据的长度
      DTYPE* key = this->key_cache[layer_id];
      DTYPE* value = this->value_cache[layer_id];
      for (int i = 0; i < this->num_key_value_heads; i++) {
        auto head_ind = ind + i * this->max_length;
        int num_indices = nnz[i];
        auto queried_key_ptr = this->queried_key + i * stride; // 当前头要查询的数据位置
        auto queried_value_ptr = this->queried_value + i * stride;
        auto key_ptr = key + i * stride; // 当前头完整数据位置
        auto value_ptr = value + i * stride;
        for (int j = 0; j < num_indices; j++) {
          auto cur_ind = head_ind[j];
          memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
          memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
        }
      }
      return;
    }

    std::vector<std::vector<IOTask>> tasks_per_head(this->num_key_value_heads);
    
    int tot_task = 0;
    uint64_t max_file_index = 0;
    #pragma omp parallel for schedule(static) num_threads(36)
    for(int i = 0; i < this->num_key_value_heads; i++){
        auto head_ind = ind + i * this->max_length;
        int num_indices = nnz[i];

        std::vector<std::tuple<uint64_t, uint64_t, int>> queried_meta_offset;
        queried_meta_offset.reserve(num_indices); // 预分配内存

        for (int j = 0; j < num_indices; j++) {
            uint64_t meta_id = get_meta_id(head_ind[j], layer_id, i);
            FileOffsetInfo& info = kv_meta->at(meta_id);
            uint64_t file_index = info.file_index;
            uint64_t offset = info.offset;
            queried_meta_offset.emplace_back(file_index, offset, j);

            max_file_index = std::max(max_file_index, file_index);
        }
        std::sort(queried_meta_offset.begin(), queried_meta_offset.end());

        std::vector<IOTask> local_task;
        
    
        if(!queried_meta_offset.empty()){
            auto [start_file_index, start_offset, token_index] = *queried_meta_offset.begin();
            uint64_t prev_file_index = start_file_index;  uint64_t prev_offset = start_offset;
            
            std::vector<int> seg_tokens;  seg_tokens.push_back(token_index);

            int count = 1;
            for(auto it = next(queried_meta_offset.begin()); it != queried_meta_offset.end(); it++){
                auto [cur_file_index, cur_offset, token_index] = *it;

                if(prev_file_index == cur_file_index && prev_offset + entry_size == cur_offset){
                    count++;
                    seg_tokens.push_back(token_index);
                }else{
                    if(local_task.empty()){
                        int bitmap = -1;
                        if(count * entry_size <= io_size) bitmap = (1<<count) - 1;
                        
                        IOTask t;
                        t.file_index = start_file_index; t.offset = start_offset;
                        t.length = count * entry_size; t.bitmap = bitmap;
                        t.head_id = i;

                        t.token_indices = std::move(seg_tokens);
                        seg_tokens.clear();

                        local_task.push_back(std::move(t));
                    }else{
                        auto &last_task = local_task.back();
                        uint64_t &last_file_index = last_task.file_index, last_start_offset = last_task.offset;
                        uint64_t &last_length = last_task.length, last_end_offset = last_start_offset + last_length;
                        int &last_bitmap = last_task.bitmap;
                        uint64_t  gap = start_offset - last_end_offset;  
                        if(last_file_index == start_file_index && last_length + gap + count * entry_size <= io_size){
                            int start_bit = last_length / entry_size + gap / entry_size;
                            int end_bit = start_bit + count;
                            for(int b = start_bit; b < end_bit; b++) last_task.bitmap |= (1 << b);

                            last_task.length += gap + count * entry_size;

                            last_task.token_indices.insert(last_task.token_indices.end(), seg_tokens.begin(),seg_tokens.end());
                            seg_tokens.clear();
                        
                        }else{
                            int bitmap = -1;
                            if(count * entry_size <= io_size) bitmap = (1<<count) - 1;
                            IOTask t;
                            t.file_index = prev_file_index; t.offset = start_offset;
                            t.length = count * entry_size; t.bitmap = bitmap;
                            t.head_id = i;

                            t.token_indices = std::move(seg_tokens);
                            seg_tokens.clear();

                            local_task.push_back(std::move(t));
                            
                        }
                    }
                    start_file_index = cur_file_index;
                    start_offset = cur_offset;
                    count = 1;
                    seg_tokens.push_back(token_index);
                }
                prev_file_index = cur_file_index, prev_offset = cur_offset;
            }
            int bitmap = -1;
            if(count * entry_size <= io_size) bitmap = (1<<count) - 1;
            IOTask t;
            t.file_index = prev_file_index; t.offset = start_offset;
            t.length = count * entry_size; t.bitmap = bitmap;
            t.head_id = i;

            t.token_indices = std::move(seg_tokens);
            seg_tokens.clear();

            local_task.push_back(std::move(t));
        }

        tasks_per_head[i] = std::move(local_task);
        tot_task += tasks_per_head[i].size();
    }
    std::vector<IOTask> all_tasks;
    all_tasks.reserve(tot_task);
    for(int i=0;i<this->num_key_value_heads;i++){
        for(auto &t : tasks_per_head[i]){
            all_tasks.push_back(std::move(t));
        }
    }

    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end_time - start_time;
    // std::cout << "[TIMER] compute for layer " << layer_id 
    //           << " took " << duration.count()  << " seconds." << std::endl;

    this->concurrent_merge_load_key_value_from_file(all_tasks, max_file_index, prefix_id, layer_id);
}

void KVStore::concurrent_merge_load_key_value_from_file(
    std::vector<IOTask> &all_tasks,
    uint64_t max_file_index,
    int prefix_id,
    int layer_id
) {
    if(all_tasks.empty()) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value

    std::vector<int> fd_table(max_file_index + 1, -1);
    for(uint64_t i = 0; i <= max_file_index; i++){
        std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id) + "_part" + std::to_string(i) + ".bin";
        int fd = open(file_name.c_str(), O_RDONLY);
        if (fd < 0) {
            fd_table[i] = -1;
            std::cerr << "concurrent_phaseB: open failed for " << file_name << " errno=" << errno << " : " << strerror(errno) << "\n";
        } else {
            fd_table[i] = fd;
        }
    }

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned int num_workers = std::min<unsigned int>(hw * 2, std::max<unsigned int>(1, (unsigned)all_tasks.size()));
    num_workers = std::min(72, (int)all_tasks.size());

    size_t N = all_tasks.size();
    std::vector<size_t> range_start(num_workers);
    std::vector<size_t> range_end(num_workers);
    for (unsigned int w = 0; w < num_workers; ++w) {
        size_t s = (N * w) / num_workers;
        size_t e = (N * (w + 1)) / num_workers;
        range_start[w] = s;
        range_end[w] = e;
    }

    auto worker = [&](unsigned int wid){
        size_t s = range_start[wid];
        size_t e = range_end[wid];
        
        std::vector<char> buf;
        for(size_t idx = s; idx < e; idx++){
            IOTask &task = all_tasks[idx];

            int count = 0;

            uint64_t file_index = task.file_index, offset = task.offset, length = task.length;
            int bitmap = task.bitmap;
            int head_id = task.head_id;

            int fd = fd_table[file_index];

            if (buf.size() < task.length) buf.resize(task.length);

            int num_entries = length / entry_size;
            ssize_t n = pread(fd, buf.data(), length, offset);
            if (n < 0) {
                std::cerr << " pread failed: " << strerror(errno) << "\n";
            }
            if(bitmap == -1){
                for (int i = 0; i < num_entries; i++) {
                    DTYPE* entry = reinterpret_cast<DTYPE*>(buf.data() + i * entry_size);
                    DTYPE* cur_key = entry;
                    DTYPE* cur_value = entry + this->head_dim;

                    int key_index = task.token_indices[count++];

                    int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                    int value_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;

                    memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                    memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));
                }
            }else{
                int tmp_bitmap = bitmap;
                while(tmp_bitmap != 0){
                    int pos = __builtin_ffs(tmp_bitmap) - 1;
                    tmp_bitmap &= (tmp_bitmap - 1);

                    DTYPE* entry = reinterpret_cast<DTYPE*>(buf.data() + pos * entry_size);
                    DTYPE* cur_key = entry;
                    DTYPE* cur_value = entry + this->head_dim;
                    int key_index = task.token_indices[count++];

                    int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                    int value_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;

                    memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                    memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (unsigned int w = 0; w < num_workers; ++w) threads.emplace_back(worker, w);

    for (auto &th : threads) if (th.joinable()) th.join();

    for (size_t fi = 0; fi < fd_table.size(); ++fi) {
        if (fd_table[fi] >= 0) ::close(fd_table[fi]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    // std::cout << "[TIMER] IO for layer " << layer_id 
    //           << " took " << duration.count()  << " seconds." << std::endl;
}


PYBIND11_MODULE(kvstore, m) {
    py::class_<KVStore>(m, "KVStore")
        .def(py::init<>())
        .def("alloc", &KVStore::alloc)
        .def("fill", &KVStore::fill)
        .def("persist_meta", &KVStore::persist_meta)
        .def("recover_meta", &KVStore::recover_meta)
        //.def("write_to_storage", &KVStore::write_to_storage)
        .def("write_to_layer_promote_file", &KVStore::write_to_layer_promote_file)
        .def("write_to_file", &KVStore::write_to_file)
        .def("write_to_layer_file", &KVStore::write_to_layer_file)
        .def("collect_queried_key_value", &KVStore::collect_queried_key_value)
        .def("merge_collect_queried_key_value", &KVStore::merge_collect_queried_key_value)
        .def("concurrent_merge_collect_queried_key_value", &KVStore::concurrent_merge_collect_queried_key_value)
        .def("get_queried_key_cache", &KVStore::get_queried_key_cache)
        .def("promote_persist", &KVStore::promote_persist)
        .def("get_queried_value_cache", &KVStore::get_queried_value_cache)
        .def("clear", &KVStore::clear);
}
