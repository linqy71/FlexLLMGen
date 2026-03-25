#pragma once

#include <torch/extension.h>
#include <math.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#define ATTENTION_THREADS 32
#define READ_THREADS 8

using bfloat16 = std::uint16_t;
#define DTYPE bfloat16

struct FileOffsetInfo {
    uint64_t file_index;
    uint64_t offset;

    FileOffsetInfo() : file_index(0), offset(0) {}
    FileOffsetInfo(uint64_t idx, uint64_t off) : file_index(idx), offset(off) {}
};

struct IOTask {
    uint64_t file_index;
    uint64_t offset;
    uint64_t length;
    int bitmap;
    std::vector<int> token_indices;
    int head_id;
};

class KVStore {
public:
    KVStore();
    ~KVStore();

    void alloc(int num_layers, int num_attention_heads, int num_key_value_heads, int head_dim, int max_length);
    void fill(int layer_id, torch::Tensor k, torch::Tensor v);

    void write_to_storage(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
    void write_to_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
    void write_to_layer_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
    void write_to_layer_promote_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);

    void clear();
    uint64_t get_meta_id(int token_id, int layer_id, int head_id);
    void persist_meta(int prefix_id);
    void recover_meta(std::string path, int prefix_id);

    void collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt);
    void merge_collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt);
    void concurrent_merge_collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt);

    void promote_persist(std::string path, int prefix_id, int layer_id, const torch::Tensor& promote_token_info);
    void reorder_persist(std::string path, int prefix_id, int layer_id, const torch::Tensor& reorder_token_info);

    torch::Tensor to_tensor(DTYPE* start, int length);
    torch::Tensor get_queried_key_cache();
    torch::Tensor get_queried_value_cache();

private:
    void merge_load_key_value_from_file(
        const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>>& content,
        const std::vector<int>& token_order,
        int prefix_id,
        int layer_id,
        int head_id);
    void concurrent_merge_load_key_value_from_file(
        std::vector<IOTask>& all_tasks,
        uint64_t max_file_index,
        int prefix_id,
        int layer_id);

    int num_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int max_length;
    int offload_len;

    bool allocated;
    std::vector<DTYPE*> key_cache;
    std::vector<DTYPE*> value_cache;
    DTYPE* queried_key;
    DTYPE* queried_value;

    std::string store_path;
    bool persisted;

    std::unordered_map<uint64_t, FileOffsetInfo>* kv_meta;
};
