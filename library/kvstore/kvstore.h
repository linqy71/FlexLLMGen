#pragma once

#include <torch/extension.h>
#include <math.h>
#include <vector>
#include <unordered_map>
// #define ATTENTION_THREADS 64
#define ATTENTION_THREADS 32
#define READ_THREADS 8

using bfloat16 = std::uint16_t;
#define DTYPE bfloat16


class KVStore{

    public:
        KVStore();
        ~KVStore();
        void alloc(int num_layers, int num_attention_heads, int num_key_value_heads, int head_dim, int max_length);
        void fill(int layer_id, torch::Tensor k, torch::Tensor v);
        void write_to_storage(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
        void write_to_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
        void clear();
        uint64_t get_meta_id(int token_id, int layer_id, int head_id);
        void persist_meta(int prefix_id);
        void recover_meta(std::string path, int prefix_id);
        void collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void load_key_value_from_file(std::vector<std::pair<uint64_t, int>>& content, int prefix_id, int layer_id, int head_id);
        void load_key_value(std::vector<std::pair<uint64_t, int>>& content, int prefix_id, int layer_id, int head_id);
        torch::Tensor to_tensor(DTYPE* start, int length);
        torch::Tensor get_key_cache(int layer_id);
        torch::Tensor get_value_cache(int layer_id);
        torch::Tensor get_queried_key_cache();
        torch::Tensor get_queried_value_cache();
        // torch::Tensor get_key_norm(int layer_id);
        // torch::Tensor get_score();
    private:
        int num_layers;
        int num_attention_heads;
        int num_key_value_heads;
        int head_dim;
        int max_length;
        int offload_len;
        
        bool allocated;
        std::vector<DTYPE *>key_cache;
        std::vector<DTYPE *>value_cache;
        DTYPE* queried_key;
        DTYPE* queried_value;
        
        std::string store_path;
        bool persisted;

        std::unordered_map<uint64_t, uint64_t>* kv_meta; // metaid -> pos
};
