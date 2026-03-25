#include "kvstore.h"

#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <map>
#include <omp.h>
#include <pybind11/stl.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace py = pybind11;

KVStore::KVStore() {
    this->allocated = false;
    this->kv_meta = new std::unordered_map<uint64_t, FileOffsetInfo>();
    this->persisted = false;
}

KVStore::~KVStore() {
    if (this->allocated) {
        for (int i = 0; i < this->num_layers; ++i) {
            delete[] this->key_cache[i];
            delete[] this->value_cache[i];
        }
        delete[] this->queried_key;
        delete[] this->queried_value;
        delete this->kv_meta;
        this->allocated = false;
    }
}

void KVStore::alloc(
    int num_layers,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    int max_length) {
    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->head_dim = head_dim;
    this->max_length = max_length;

    for (int i = 0; i < this->num_layers; ++i) {
        DTYPE* k = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
        DTYPE* v = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
        memset(k, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        memset(v, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        this->key_cache.push_back(k);
        this->value_cache.push_back(v);
    }

    this->queried_key = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
    this->queried_value = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    this->allocated = true;
}

void KVStore::clear() {
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->num_layers; ++i) {
        memset(this->key_cache[i], 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        memset(this->value_cache[i], 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    }

    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
}

void KVStore::fill(int layer_id, torch::Tensor k, torch::Tensor v) {
    DTYPE* key = this->key_cache[layer_id];
    DTYPE* value = this->value_cache[layer_id];

    int seq_len = k.size(1);
    this->offload_len = seq_len;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->num_key_value_heads; ++i) {
        memcpy(key + i * this->max_length * this->head_dim,
               static_cast<DTYPE*>(k.data_ptr()) + i * seq_len * this->head_dim,
               seq_len * this->head_dim * sizeof(DTYPE));
        memcpy(value + i * this->max_length * this->head_dim,
               static_cast<DTYPE*>(v.data_ptr()) + i * seq_len * this->head_dim,
               seq_len * this->head_dim * sizeof(DTYPE));
    }
}

uint64_t KVStore::get_meta_id(int token_id, int layer_id, int head_id) {
    return layer_id * this->num_key_value_heads * this->max_length + head_id * this->max_length + token_id;
}

void KVStore::persist_meta(int prefix_id) {
    std::string meta_file = this->store_path + "/" + std::to_string(prefix_id) + ".meta";
    std::ofstream file(meta_file, std::ios::binary | std::ios::trunc);
    std::stringstream ss;
    for (const auto& [id, info] : *this->kv_meta) {
        ss << id << " " << info.file_index << " " << info.offset << "\n";
    }
    file.write(ss.str().data(), ss.str().size());
    file.flush();
    file.close();
    std::cout << "Successfully persist kvstore meta" << std::endl;
}

void KVStore::recover_meta(std::string path, int prefix_id) {
    auto start_time = std::chrono::high_resolution_clock::now();

    this->store_path = path;
    this->kv_meta->clear();

    std::string meta_file = this->store_path + "/" + std::to_string(prefix_id) + ".meta";
    std::ifstream file(meta_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open meta file: " << meta_file << std::endl;
        return;
    }

    uint64_t meta_id = 0;
    uint64_t file_index = 0;
    uint64_t offset = 0;
    while (file >> meta_id >> file_index >> offset) {
        this->kv_meta->insert({meta_id, FileOffsetInfo(file_index, offset)});
    }
    file.close();
    this->persisted = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "[TIMER] recover_meta for prefix " << prefix_id << " took "
              << duration.count() << " seconds." << std::endl;
}

void KVStore::write_to_storage(
    std::string path,
    int prefix_id,
    int layer_id,
    const std::vector<std::vector<int>>& strategy) {
    this->store_path = path;
    DTYPE* k = this->key_cache[layer_id];
    DTYPE* v = this->value_cache[layer_id];

    std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + ".bin";
    std::ofstream file(file_name, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return;
    }

    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        DTYPE* head_key = k + head_id * this->max_length * this->head_dim;
        DTYPE* head_value = v + head_id * this->max_length * this->head_dim;
        const std::vector<int>& head_strategy = strategy[head_id];

        for (int token_id : head_strategy) {
            std::streampos entry_offset = file.tellp();
            DTYPE* cur_key = head_key + token_id * this->head_dim;
            DTYPE* cur_value = head_value + token_id * this->head_dim;
            file.write(reinterpret_cast<char*>(cur_key), this->head_dim * sizeof(DTYPE));
            file.write(reinterpret_cast<char*>(cur_value), this->head_dim * sizeof(DTYPE));
            (*this->kv_meta)[get_meta_id(token_id, layer_id, head_id)] = FileOffsetInfo(0, static_cast<uint64_t>(entry_offset));
        }
    }
    file.flush();
    file.close();
    this->persisted = true;
}

void KVStore::write_to_layer_promote_file(
    std::string path,
    int prefix_id,
    int layer_id,
    const std::vector<std::vector<int>>& strategy) {
    this->store_path = path;
    DTYPE* k = this->key_cache[layer_id];
    DTYPE* v = this->value_cache[layer_id];

    std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
        std::to_string(layer_id) + "_part0.bin";
    std::ofstream file(file_name, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return;
    }

    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        DTYPE* head_key = k + head_id * this->max_length * this->head_dim;
        DTYPE* head_value = v + head_id * this->max_length * this->head_dim;
        const std::vector<int>& head_strategy = strategy[head_id];

        for (int token_id : head_strategy) {
            std::streampos entry_offset = file.tellp();
            DTYPE* cur_key = head_key + token_id * this->head_dim;
            DTYPE* cur_value = head_value + token_id * this->head_dim;
            file.write(reinterpret_cast<char*>(cur_key), this->head_dim * sizeof(DTYPE));
            file.write(reinterpret_cast<char*>(cur_value), this->head_dim * sizeof(DTYPE));
            (*this->kv_meta)[get_meta_id(token_id, layer_id, head_id)] =
                FileOffsetInfo(0, static_cast<uint64_t>(entry_offset));
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
    const std::vector<std::vector<int>>& strategy) {
    this->write_to_layer_promote_file(path, prefix_id, layer_id, strategy);
}

void KVStore::write_to_file(
    std::string path,
    int prefix_id,
    int layer_id,
    const std::vector<std::vector<int>>& strategy) {
    this->store_path = path;
    DTYPE* k = this->key_cache[layer_id];
    DTYPE* v = this->value_cache[layer_id];

    const int tokens_per_file = 128;
    int seq_len = this->offload_len;
    int num_files = (seq_len + tokens_per_file - 1) / tokens_per_file;
    if (num_files <= 0) {
        return;
    }

    std::vector<std::ofstream> files(num_files);
    for (int file_index = 0; file_index < num_files; ++file_index) {
        std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
            std::to_string(layer_id) + "_part" + std::to_string(file_index) + ".bin";
        files[file_index].open(file_name, std::ios::binary | std::ios::app);
        if (!files[file_index].is_open()) {
            std::cerr << "Failed to open file: " << file_name << std::endl;
            return;
        }
    }

    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        DTYPE* head_key = k + head_id * this->max_length * this->head_dim;
        DTYPE* head_value = v + head_id * this->max_length * this->head_dim;
        const std::vector<int>& head_strategy = strategy[head_id];

        for (size_t j = 0; j < head_strategy.size(); ++j) {
            int token_id = head_strategy[j];
            int file_index = static_cast<int>(j / tokens_per_file);
            std::streampos entry_offset = files[file_index].tellp();
            DTYPE* cur_key = head_key + token_id * this->head_dim;
            DTYPE* cur_value = head_value + token_id * this->head_dim;
            files[file_index].write(reinterpret_cast<char*>(cur_key), this->head_dim * sizeof(DTYPE));
            files[file_index].write(reinterpret_cast<char*>(cur_value), this->head_dim * sizeof(DTYPE));
            (*this->kv_meta)[get_meta_id(token_id, layer_id, head_id)] =
                FileOffsetInfo(file_index, static_cast<uint64_t>(entry_offset));
        }
    }

    for (auto& file : files) {
        file.flush();
        file.close();
    }
    this->persisted = true;
}

void KVStore::collect_queried_key_value(
    int prefix_id,
    int layer_id,
    torch::Tensor ind_pt,
    torch::Tensor nnz_pt) {
    this->merge_collect_queried_key_value(prefix_id, layer_id, ind_pt, nnz_pt);
}

void KVStore::merge_collect_queried_key_value(
    int prefix_id,
    int layer_id,
    torch::Tensor ind_pt,
    torch::Tensor nnz_pt) {
    auto start_time = std::chrono::high_resolution_clock::now();

    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    int* ind = static_cast<int*>(ind_pt.data_ptr());
    int* nnz = static_cast<int*>(nnz_pt.data_ptr());

    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    uint64_t io_size = 4096;

    if (!this->persisted) {
        int stride = this->max_length * this->head_dim;
        DTYPE* key = this->key_cache[layer_id];
        DTYPE* value = this->value_cache[layer_id];
        for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
            auto head_ind = ind + head_id * this->max_length;
            int num_indices = nnz[head_id];
            auto queried_key_ptr = this->queried_key + head_id * stride;
            auto queried_value_ptr = this->queried_value + head_id * stride;
            auto key_ptr = key + head_id * stride;
            auto value_ptr = value + head_id * stride;
            for (int j = 0; j < num_indices; ++j) {
                int cur_ind = head_ind[j];
                memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
                memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
            }
        }
        return;
    }

    #pragma omp parallel for schedule(static) num_threads(ATTENTION_THREADS)
    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        auto head_ind = ind + head_id * this->max_length;
        int num_indices = nnz[head_id];
        std::vector<std::tuple<uint64_t, uint64_t, int>> queried_meta_offset;
        queried_meta_offset.reserve(num_indices);

        for (int j = 0; j < num_indices; ++j) {
            uint64_t meta_id = get_meta_id(head_ind[j], layer_id, head_id);
            auto it = this->kv_meta->find(meta_id);
            if (it == this->kv_meta->end()) {
                continue;
            }
            const FileOffsetInfo& info = it->second;
            queried_meta_offset.emplace_back(info.file_index, info.offset, j);
        }

        std::sort(queried_meta_offset.begin(), queried_meta_offset.end());

        std::vector<int> token_order;
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>> content;
        if (!queried_meta_offset.empty()) {
            auto [start_file_index, start_offset, token_index] = queried_meta_offset.front();
            uint64_t prev_file_index = start_file_index;
            uint64_t prev_offset = start_offset;
            int count = 1;
            token_order.push_back(token_index);

            for (size_t idx = 1; idx < queried_meta_offset.size(); ++idx) {
                auto [cur_file_index, cur_offset, cur_token_index] = queried_meta_offset[idx];
                token_order.push_back(cur_token_index);
                if (prev_file_index == cur_file_index && prev_offset + entry_size == cur_offset) {
                    count++;
                } else {
                    if (content.empty()) {
                        int bitmap = -1;
                        if (static_cast<uint64_t>(count) * entry_size <= io_size) {
                            bitmap = (1 << count) - 1;
                        }
                        content.emplace_back(start_file_index, start_offset, count * entry_size, bitmap);
                    } else {
                        auto& last = content.back();
                        uint64_t last_file_index = std::get<0>(last);
                        uint64_t last_start_offset = std::get<1>(last);
                        uint64_t last_length = std::get<2>(last);
                        int last_bitmap = std::get<3>(last);
                        uint64_t last_end_offset = last_start_offset + last_length;
                        uint64_t gap = start_offset - last_end_offset;
                        if (last_file_index == start_file_index && last_length + gap + count * entry_size <= io_size) {
                            int start_bit = static_cast<int>(last_length / entry_size + gap / entry_size);
                            int end_bit = start_bit + count;
                            for (int bit = start_bit; bit < end_bit; ++bit) {
                                last_bitmap |= (1 << bit);
                            }
                            std::get<2>(last) = last_length + gap + count * entry_size;
                            std::get<3>(last) = last_bitmap;
                        } else {
                            int bitmap = -1;
                            if (static_cast<uint64_t>(count) * entry_size <= io_size) {
                                bitmap = (1 << count) - 1;
                            }
                            content.emplace_back(start_file_index, start_offset, count * entry_size, bitmap);
                        }
                    }
                    start_file_index = cur_file_index;
                    start_offset = cur_offset;
                    count = 1;
                }
                prev_file_index = cur_file_index;
                prev_offset = cur_offset;
            }

            int bitmap = -1;
            if (static_cast<uint64_t>(count) * entry_size <= io_size) {
                bitmap = (1 << count) - 1;
            }
            content.emplace_back(start_file_index, start_offset, count * entry_size, bitmap);
        }

        this->merge_load_key_value_from_file(content, token_order, prefix_id, layer_id, head_id);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    // std::cout << "[TIMER] compute and IO for layer " << layer_id
    //           << " took " << duration.count() << " seconds." << std::endl;
}

void KVStore::concurrent_merge_collect_queried_key_value(
    int prefix_id,
    int layer_id,
    torch::Tensor ind_pt,
    torch::Tensor nnz_pt) {
    auto start_time = std::chrono::high_resolution_clock::now();

    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    int* ind = static_cast<int*>(ind_pt.data_ptr());
    int* nnz = static_cast<int*>(nnz_pt.data_ptr());
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    uint64_t io_size = 4096;

    if (!this->persisted) {
        int stride = this->max_length * this->head_dim;
        DTYPE* key = this->key_cache[layer_id];
        DTYPE* value = this->value_cache[layer_id];
        for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
            auto head_ind = ind + head_id * this->max_length;
            int num_indices = nnz[head_id];
            auto queried_key_ptr = this->queried_key + head_id * stride;
            auto queried_value_ptr = this->queried_value + head_id * stride;
            auto key_ptr = key + head_id * stride;
            auto value_ptr = value + head_id * stride;
            for (int j = 0; j < num_indices; ++j) {
                int cur_ind = head_ind[j];
                memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
                memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
            }
        }
        return;
    }

    std::vector<std::vector<IOTask>> tasks_per_head(this->num_key_value_heads);
    int tot_task = 0;
    uint64_t max_file_index = 0;

    #pragma omp parallel for schedule(static) num_threads(ATTENTION_THREADS)
    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        auto head_ind = ind + head_id * this->max_length;
        int num_indices = nnz[head_id];
        std::vector<std::tuple<uint64_t, uint64_t, int>> queried_meta_offset;
        queried_meta_offset.reserve(num_indices);

        for (int j = 0; j < num_indices; ++j) {
            uint64_t meta_id = get_meta_id(head_ind[j], layer_id, head_id);
            auto it = this->kv_meta->find(meta_id);
            if (it == this->kv_meta->end()) {
                continue;
            }
            const FileOffsetInfo& info = it->second;
            queried_meta_offset.emplace_back(info.file_index, info.offset, j);
            max_file_index = std::max(max_file_index, info.file_index);
        }
        std::sort(queried_meta_offset.begin(), queried_meta_offset.end());

        std::vector<IOTask> local_tasks;
        if (!queried_meta_offset.empty()) {
            auto [start_file_index, start_offset, token_index] = queried_meta_offset.front();
            uint64_t prev_file_index = start_file_index;
            uint64_t prev_offset = start_offset;
            std::vector<int> seg_tokens{token_index};
            int count = 1;

            for (size_t idx = 1; idx < queried_meta_offset.size(); ++idx) {
                auto [cur_file_index, cur_offset, cur_token_index] = queried_meta_offset[idx];
                if (prev_file_index == cur_file_index && prev_offset + entry_size == cur_offset) {
                    count++;
                    seg_tokens.push_back(cur_token_index);
                } else {
                    IOTask task{};
                    task.file_index = start_file_index;
                    task.offset = start_offset;
                    task.length = count * entry_size;
                    task.bitmap = (task.length <= io_size) ? ((1 << count) - 1) : -1;
                    task.head_id = head_id;
                    task.token_indices = std::move(seg_tokens);
                    local_tasks.push_back(std::move(task));

                    start_file_index = cur_file_index;
                    start_offset = cur_offset;
                    prev_file_index = cur_file_index;
                    prev_offset = cur_offset;
                    seg_tokens = {cur_token_index};
                    count = 1;
                    continue;
                }
                prev_file_index = cur_file_index;
                prev_offset = cur_offset;
            }

            IOTask task{};
            task.file_index = start_file_index;
            task.offset = start_offset;
            task.length = count * entry_size;
            task.bitmap = (task.length <= io_size) ? ((1 << count) - 1) : -1;
            task.head_id = head_id;
            task.token_indices = std::move(seg_tokens);
            local_tasks.push_back(std::move(task));
        }

        tasks_per_head[head_id] = std::move(local_tasks);
        tot_task += static_cast<int>(tasks_per_head[head_id].size());
    }

    std::vector<IOTask> all_tasks;
    all_tasks.reserve(tot_task);
    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        for (auto& task : tasks_per_head[head_id]) {
            all_tasks.push_back(std::move(task));
        }
    }

    this->concurrent_merge_load_key_value_from_file(all_tasks, max_file_index, prefix_id, layer_id);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "[TIMER] concurrent compute and IO for layer " << layer_id
              << " took " << duration.count() << " seconds." << std::endl;
}

void KVStore::concurrent_merge_load_key_value_from_file(
    std::vector<IOTask>& all_tasks,
    uint64_t max_file_index,
    int prefix_id,
    int layer_id) {
    if (all_tasks.empty()) {
        return;
    }

    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    std::vector<int> fd_table(max_file_index + 1, -1);
    for (uint64_t file_index = 0; file_index <= max_file_index; ++file_index) {
        std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
            std::to_string(layer_id) + "_part" + std::to_string(file_index) + ".bin";
        int fd = open(file_name.c_str(), O_RDONLY);
        if (fd < 0) {
            fd_table[file_index] = -1;
            std::cerr << "concurrent_phaseB: open failed for " << file_name
                      << " errno=" << errno << " : " << strerror(errno) << "\n";
        } else {
            fd_table[file_index] = fd;
        }
    }

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;
    }
    unsigned int num_workers = std::min<unsigned int>(hw * 2, std::max<unsigned int>(1, static_cast<unsigned int>(all_tasks.size())));
    num_workers = std::min<unsigned int>(72, static_cast<unsigned int>(all_tasks.size()));
    if (num_workers == 0) {
        num_workers = 1;
    }

    size_t N = all_tasks.size();
    std::vector<size_t> range_start(num_workers);
    std::vector<size_t> range_end(num_workers);
    for (unsigned int w = 0; w < num_workers; ++w) {
        range_start[w] = (N * w) / num_workers;
        range_end[w] = (N * (w + 1)) / num_workers;
    }

    auto worker = [&](unsigned int worker_id) {
        std::vector<char> buf;
        for (size_t idx = range_start[worker_id]; idx < range_end[worker_id]; ++idx) {
            IOTask& task = all_tasks[idx];
            int fd = fd_table[task.file_index];
            if (fd < 0) {
                continue;
            }
            if (buf.size() < task.length) {
                buf.resize(task.length);
            }

            ssize_t n = pread(fd, buf.data(), task.length, task.offset);
            if (n < 0) {
                std::cerr << "pread failed: " << strerror(errno) << "\n";
                continue;
            }

            int count = 0;
            int num_entries = static_cast<int>(task.length / entry_size);
            if (task.bitmap == -1) {
                for (int i = 0; i < num_entries && count < static_cast<int>(task.token_indices.size()); ++i) {
                    DTYPE* entry = reinterpret_cast<DTYPE*>(buf.data() + i * entry_size);
                    DTYPE* cur_key = entry;
                    DTYPE* cur_value = entry + this->head_dim;
                    int key_index = task.token_indices[count++];
                    int key_offset = task.head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                    memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                    memcpy(this->queried_value + key_offset, cur_value, this->head_dim * sizeof(DTYPE));
                }
            } else {
                unsigned int tmp_bitmap = static_cast<unsigned int>(task.bitmap);
                while (tmp_bitmap != 0 && count < static_cast<int>(task.token_indices.size())) {
                    int pos = __builtin_ffs(tmp_bitmap) - 1;
                    tmp_bitmap &= (tmp_bitmap - 1);
                    if (pos >= num_entries) {
                        continue;
                    }
                    DTYPE* entry = reinterpret_cast<DTYPE*>(buf.data() + pos * entry_size);
                    DTYPE* cur_key = entry;
                    DTYPE* cur_value = entry + this->head_dim;
                    int key_index = task.token_indices[count++];
                    int key_offset = task.head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                    memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                    memcpy(this->queried_value + key_offset, cur_value, this->head_dim * sizeof(DTYPE));
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (unsigned int w = 0; w < num_workers; ++w) {
        threads.emplace_back(worker, w);
    }
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    for (int fd : fd_table) {
        if (fd >= 0) {
            ::close(fd);
        }
    }
}

void KVStore::merge_load_key_value_from_file(
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>>& content,
    const std::vector<int>& token_order,
    int prefix_id,
    int layer_id,
    int head_id) {
    if (content.empty()) {
        return;
    }

    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    std::map<uint64_t, std::ifstream> files;
    int token_cursor = 0;

    for (const auto& [file_index, offset, length, bitmap] : content) {
        auto it = files.find(file_index);
        if (it == files.end()) {
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
                std::to_string(layer_id) + "_part" + std::to_string(file_index) + ".bin";
            auto inserted = files.emplace(file_index, std::ifstream(file_name, std::ios::binary));
            it = inserted.first;
        }

        std::ifstream& file = it->second;
        if (!file.is_open()) {
            std::cerr << "Failed to open KV file for load." << std::endl;
            continue;
        }

        std::vector<char> buffer(length);
        file.seekg(offset, std::ios::beg);
        file.read(buffer.data(), length);
        if (!file) {
            file.clear();
        }

        int num_entries = static_cast<int>(length / entry_size);
        if (bitmap == -1) {
            for (int i = 0; i < num_entries && token_cursor < static_cast<int>(token_order.size()); ++i) {
                DTYPE* entry = reinterpret_cast<DTYPE*>(buffer.data() + i * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;
                int key_index = token_order[token_cursor++];
                int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + key_offset, cur_value, this->head_dim * sizeof(DTYPE));
            }
        } else {
            unsigned int tmp_bitmap = static_cast<unsigned int>(bitmap);
            while (tmp_bitmap != 0 && token_cursor < static_cast<int>(token_order.size())) {
                int pos = __builtin_ffs(tmp_bitmap) - 1;
                tmp_bitmap &= (tmp_bitmap - 1);
                if (pos >= num_entries) {
                    continue;
                }
                DTYPE* entry = reinterpret_cast<DTYPE*>(buffer.data() + pos * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;
                int key_index = token_order[token_cursor++];
                int key_offset = head_id * this->max_length * this->head_dim + key_index * this->head_dim;
                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + key_offset, cur_value, this->head_dim * sizeof(DTYPE));
            }
        }
    }
}

void KVStore::reorder_persist(
    std::string path,
    int prefix_id,
    int layer_id,
    const torch::Tensor& reorder_token_info) {
    if (reorder_token_info.numel() == 0) {
        return;
    }

    this->store_path = path;
    const size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    std::string source_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
        std::to_string(layer_id) + "_part0.bin";
    std::string dest_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
        std::to_string(layer_id) + "_part1.bin";

    std::ifstream source_file(source_name, std::ios::binary);
    std::ofstream dest_file(dest_name, std::ios::binary | std::ios::app);
    if (!source_file.is_open() || !dest_file.is_open()) {
        std::cerr << "Error opening source or destination file for reorder." << std::endl;
        return;
    }

    std::vector<char> buffer(entry_size);
    auto accessor = reorder_token_info.accessor<long, 2>();
    for (int i = 0; i < accessor.size(0); ++i) {
        int head_id = accessor[i][0];
        int token_id = accessor[i][1];
        uint64_t meta_id = get_meta_id(token_id, layer_id, head_id);
        auto it = this->kv_meta->find(meta_id);
        if (it == this->kv_meta->end() || it->second.file_index != 0) {
            continue;
        }

        source_file.seekg(it->second.offset, std::ios::beg);
        source_file.read(buffer.data(), entry_size);
        std::streampos new_offset = dest_file.tellp();
        dest_file.write(buffer.data(), entry_size);
        it->second.file_index = 1;
        it->second.offset = static_cast<uint64_t>(new_offset);
    }

    source_file.close();
    dest_file.flush();
    dest_file.close();
}

void KVStore::promote_persist(
    std::string path,
    int prefix_id,
    int layer_id,
    const torch::Tensor& promote_token_info) {
    if (promote_token_info.numel() == 0) {
        return;
    }

    this->store_path = path;
    const size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> promotions_by_file;

    auto accessor = promote_token_info.accessor<long, 2>();
    for (int i = 0; i < accessor.size(0); ++i) {
        int head_id = accessor[i][0];
        int token_id = accessor[i][1];
        uint64_t meta_id = get_meta_id(token_id, layer_id, head_id);
        auto it = this->kv_meta->find(meta_id);
        if (it == this->kv_meta->end()) {
            continue;
        }
        promotions_by_file[it->second.file_index].push_back({meta_id, it->second.offset});
    }

    for (const auto& [source_file_index, promotions] : promotions_by_file) {
        uint64_t dest_file_index = source_file_index + 1;
        std::string source_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
            std::to_string(layer_id) + "_part" + std::to_string(source_file_index) + ".bin";
        std::string dest_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" +
            std::to_string(layer_id) + "_part" + std::to_string(dest_file_index) + ".bin";

        std::ifstream source_file(source_name, std::ios::binary);
        std::ofstream dest_file(dest_name, std::ios::binary | std::ios::app);
        if (!source_file.is_open() || !dest_file.is_open()) {
            std::cerr << "Error opening source or destination file for promotion." << std::endl;
            continue;
        }

        std::vector<char> buffer(entry_size);
        for (const auto& [meta_id, source_offset] : promotions) {
            source_file.seekg(source_offset, std::ios::beg);
            source_file.read(buffer.data(), entry_size);
            std::streampos new_offset = dest_file.tellp();
            dest_file.write(buffer.data(), entry_size);
            (*this->kv_meta)[meta_id] = FileOffsetInfo(dest_file_index, static_cast<uint64_t>(new_offset));
        }

        source_file.close();
        dest_file.flush();
        dest_file.close();
    }
}

torch::Tensor KVStore::to_tensor(DTYPE* start, int length) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    return torch::from_blob(start, {length, this->head_dim}, options);
}

torch::Tensor KVStore::get_queried_key_cache() {
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    return torch::from_blob(this->queried_key, {this->num_key_value_heads, this->max_length, this->head_dim}, options);
}

torch::Tensor KVStore::get_queried_value_cache() {
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    return torch::from_blob(this->queried_value, {this->num_key_value_heads, this->max_length, this->head_dim}, options);
}

PYBIND11_MODULE(kvstore, m) {
    py::class_<KVStore>(m, "KVStore")
        .def(py::init<>())
        .def("alloc", &KVStore::alloc)
        .def("fill", &KVStore::fill)
        .def("persist_meta", &KVStore::persist_meta)
        .def("recover_meta", &KVStore::recover_meta)
        .def("write_to_storage", &KVStore::write_to_storage)
        .def("write_to_file", &KVStore::write_to_file)
        .def("write_to_layer_file", &KVStore::write_to_layer_file)
        .def("write_to_layer_promote_file", &KVStore::write_to_layer_promote_file)
        .def("collect_queried_key_value", &KVStore::collect_queried_key_value)
        .def("merge_collect_queried_key_value", &KVStore::merge_collect_queried_key_value)
        .def("concurrent_merge_collect_queried_key_value", &KVStore::concurrent_merge_collect_queried_key_value)
        .def("promote_persist", &KVStore::promote_persist)
        .def("reorder_persist", &KVStore::reorder_persist)
        .def("get_queried_key_cache", &KVStore::get_queried_key_cache)
        .def("get_queried_value_cache", &KVStore::get_queried_value_cache)
        .def("clear", &KVStore::clear);
}
