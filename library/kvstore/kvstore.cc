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
#include <pybind11/stl.h>


KVStore::KVStore(){
    this->allocated = false;
    this->kv_meta = new std::unordered_map<uint64_t, uint64_t>();
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
    for (auto pair : *kv_meta) {
        ss << pair.first << " " << pair.second << "\n";
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
    uint64_t pos;
    int lines = 0;
    while(file >> meta_id >> pos) {
        kv_meta->insert({meta_id, pos});
    }
}

void KVStore::write_to_storage(
    std::string path,
    int prefix_id,
    int layer_id,
    const std::vector<std::vector<int>>& strategy 
) {
    this->store_path = path;
    DTYPE * k = this->key_cache[layer_id];
    DTYPE * v = this->value_cache[layer_id];

    std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + ".bin";
    std::ofstream file(file_name, std::ios::app | std::ios::binary);
    

    int seq_len = this->offload_len;
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value

    DTYPE * key_value = new DTYPE[seq_len * this->head_dim * 2];
    memset(key_value, 0, seq_len * this->head_dim * 2 * sizeof(DTYPE));

    for (int i = 0; i < this->num_key_value_heads; i++){
        const std::vector<int>& head_strategy = strategy[i];
        size_t head_entries = head_strategy.size();
        size_t total_size = head_entries * entry_size;
        std::streampos head_base_offset = file.tellp();

        DTYPE* key_value = new DTYPE[head_entries * this->head_dim * 2]; // key + value
        memset(key_value, 0, head_entries * this->head_dim * 2 * sizeof(DTYPE));

        DTYPE* head_key = k + i * seq_len * this->head_dim;
        DTYPE* head_value = v + i * seq_len * this->head_dim;

        for (size_t j = 0; j < head_entries; j++) {
            int idx = head_strategy[j];

            DTYPE* cur_key = head_key + idx * this->head_dim;
            DTYPE* cur_value = head_value + idx * this->head_dim;

            // copy key
            memcpy(key_value + j * 2 * this->head_dim, cur_key, this->head_dim * sizeof(DTYPE));
            // copy value
            memcpy(key_value + j * 2 * this->head_dim + this->head_dim, cur_value, this->head_dim * sizeof(DTYPE));

            // record meta
            std::streampos entry_offset = head_base_offset + j * entry_size;
            uint64_t meta_id = get_meta_id(idx, layer_id, i);
            kv_meta->insert({meta_id, uint64_t(entry_offset)});
        }

        file.write(reinterpret_cast<const char*>(key_value), total_size);
        delete[] key_value;
        
    }
    file.flush();
    file.close();
    this->persisted = true;
}


void KVStore::collect_queried_key_value(
    int prefix_id,
    int layer_id, 
    torch::Tensor ind_pt, 
    torch::Tensor nnz_pt
) {
    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());

    int attn_group = this->num_attention_heads / this->num_key_value_heads;

    std::vector<std::pair<uint64_t, int>> content; // offset, length in bytes
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);

    //handle persisted=false case; collect kv from memory
    if (this->persisted == false) {
      int stride = this->max_length * this->head_dim;
      DTYPE* key = this->key_cache[layer_id];
      DTYPE* value = this->value_cache[layer_id];
      for (int i = 0; i < this->num_key_value_heads; i++) {
        int num_indices = nnz[i];
        auto queried_key_ptr = this->queried_key + i * stride;
        auto queried_value_ptr = this->queried_value + i * stride;
        auto key_ptr = key + i * stride;
        auto value_ptr = value + i * stride;
        for (int j = 0; j < num_indices; j++) {
          auto cur_ind = ind[j];
          memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
          memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
        }
      }
      return;
    }
    
    for(int i = 0; i < this->num_key_value_heads; i++){
        std::set<uint64_t> queried_meta_offset;

        int num_indices = nnz[i];
        for (int j = 0; j < num_indices; j++) {
            uint64_t meta_id = get_meta_id(ind[j], layer_id, i);
            queried_meta_offset.insert(kv_meta->at(meta_id));
        }

        // merge contiguous offset
        if (!queried_meta_offset.empty()) {
            auto it = queried_meta_offset.begin();
            uint64_t start_offset = *it;
            uint64_t prev_offset = *it;
            int count = 1;
            ++it;

            for (; it != queried_meta_offset.end(); ++it) {
                if (*it == prev_offset + entry_size) {
                    count++;
                } else {
                    content.emplace_back(start_offset, count * entry_size);
                    start_offset = *it;
                    count = 1;
                }
                prev_offset = *it;
            }

            content.emplace_back(start_offset, count * entry_size);
        }

        // load from storage
        this->load_key_value(content, prefix_id, layer_id, i);

        content.clear();
    }

}

void analyze_content_segments(const std::vector<std::pair<uint64_t, int>>& content) {
    if (content.empty()) {
        std::cout << "Content is empty." << std::endl;
        return;
    }

    std::vector<std::pair<uint64_t, uint64_t>> segments; // {start_offset, end_offset}

    // Step 1: 计算每个 segment 的起止位置
    for (const auto& [offset, length] : content) {
        segments.emplace_back(offset, offset + length);
    }

    // Step 2: 遍历 segments，分析连续性
    size_t total_segments = segments.size();
    uint64_t total_continuous_length = 0;
    uint64_t max_continuous_length = 0;
    uint64_t max_gap = 0;

    uint64_t current_start = segments[0].first;
    uint64_t current_end = segments[0].second;

    for (size_t i = 1; i < total_segments; ++i) {
        uint64_t next_start = segments[i].first;
        uint64_t next_end = segments[i].second;

        if (next_start == current_end) {
            // 连续
            current_end = next_end;
        } else {
            // 计算当前连续区间长度
            uint64_t continuous_length = current_end - current_start;
            total_continuous_length += continuous_length;
            if (continuous_length > max_continuous_length) {
                max_continuous_length = continuous_length;
            }

            // 计算间隔
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
    if (last_continuous_length > max_continuous_length) {
        max_continuous_length = last_continuous_length;
    }

    std::cout << "[Analysis Result]" << std::endl;
    std::cout << "Total Segments: " << total_segments << std::endl;
    std::cout << "Total Continuous Length (sum of merged segments): " << total_continuous_length << " bytes" << std::endl;
    std::cout << "Max Continuous Length: " << max_continuous_length << " bytes" << std::endl;
    std::cout << "Max Gap between Segments: " << max_gap << " bytes" << std::endl;
}

void KVStore::load_key_value(
    std::vector<std::pair<uint64_t, int>>& content,
    int prefix_id,
    int layer_id,
    int head_id)
{
    std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + ".bin";
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return;
    }
    //////// ananlyze
    if (layer_id == 1 && head_id == 0){
        analyze_content_segments(content);
    }


    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value

    int count = 0;
    for (const auto& [offset, length] : content) {
        int num_entries = length / entry_size;

        std::vector<char> buffer(length);
        file.seekg(offset, std::ios::beg);
        file.read(buffer.data(), length);

        for (int i = 0; i < num_entries; i++) {
            DTYPE* entry = reinterpret_cast<DTYPE*>(buffer.data() + i * entry_size);
            DTYPE* cur_key = entry;
            DTYPE* cur_value = entry + this->head_dim;

            // 写入 queried_key 和 queried_value 中对应 head_id 和 position
            int key_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
            int value_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;

            memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
            memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

            count++;
        }
    }

    file.close();
}



torch::Tensor KVStore::to_tensor(DTYPE* start, int length){
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


PYBIND11_MODULE(group_kvstore, m) {
    py::class_<KVStore>(m, "GroupKVStore")
        .def(py::init<>())
        .def("alloc", &KVStore::alloc)
        .def("fill", &KVStore::fill)
        .def("persist_meta", &KVStore::persist_meta)
        .def("recover_meta", &KVStore::recover_meta)
        .def("write_to_storage", &KVStore::write_to_storage)
        .def("collect_queried_key_value", &KVStore::collect_queried_key_value)
        .def("get_queried_key_cache", &KVStore::get_queried_key_cache)
        .def("get_queried_value_cache", &KVStore::get_queried_value_cache)
        .def("clear", &KVStore::clear);
}


