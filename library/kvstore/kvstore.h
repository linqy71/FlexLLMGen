#pragma once

#include <torch/extension.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <string>
// #define ATTENTION_THREADS 64
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

struct LayerStats{
    uint64_t    segment_cnt;
    double      avg_continuous_length;
    double      tot_time;
    double      collect_time;

    LayerStats(): segment_cnt(0), avg_continuous_length(0.0), tot_time(0.0){}

    void update_len(uint64_t new_segment_cnt, double new_avg_len){
        avg_continuous_length = (avg_continuous_length * segment_cnt + new_avg_len * new_segment_cnt) / (segment_cnt + new_segment_cnt);
        segment_cnt += new_segment_cnt;
    }

    void update_time(double new_time) { tot_time += new_time; }
};

class KVStore{

    public:
        KVStore();
        ~KVStore();
        void alloc(int num_layers, int num_attention_heads, int num_key_value_heads, int head_dim, int max_length);
        void fill(int layer_id, torch::Tensor k, torch::Tensor v);
        // void write_to_storage(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
        void write_to_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
        void write_to_layer_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
        void write_to_layer_promote_file(std::string path, int prefix_id, int layer_id, const std::vector<std::vector<int>>& strategy);
        
        void clear();
        uint64_t get_meta_id(int token_id, int layer_id, int head_id);
        void persist_meta(int prefix_id);
        void recover_meta(std::string path, int prefix_id);
        void collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt);

        void merge_collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt);

        void load_key_value_from_file(std::vector<std::tuple<uint64_t,uint64_t, int>>& content, std::vector<int> &key_order, int prefix_id, int layer_id, int head_id);
        
        void merge_load_key_value_from_file(std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>>& content, std::vector<int> &token_order, int prefix_id, int layer_id, int head_id);
        
        void load_key_value(std::vector<std::tuple<uint64_t, uint64_t, int>>& content, std::vector<int> &key_order, int prefix_id, int layer_id, int head_id);

        void merge_load_key_value(std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int>>& content, std::vector<int> &token_order, int prefix_id, int layer_id, int head_id);
        void promote_persist(std::string path, int prefix_id, int layer_idx, const torch::Tensor& promote_token_info);
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
        std::vector<DTYPE *>key_cache;// 每一层的 一个kv_head*max_length*head_dim的开辟好的空间
        std::vector<DTYPE *>value_cache;
        DTYPE* queried_key; //返回request所需要的kv
        DTYPE* queried_value;
        
        std::string store_path;
        bool persisted;

        std::unordered_map<uint64_t, FileOffsetInfo>* kv_meta; // metaid -> pos

        std::vector<LayerStats> layer_stats;
};
