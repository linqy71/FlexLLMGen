import torch
from lsh import LSH
from kvstore import KVStore 


class LSHServer:

    def __init__(self,
        config,
        num_layers: int,
        K: int = 10, 
        L: int = 150, 
        batch_size: int = 1,
        max_length: int = 8192,
        device: str = 'cuda:0',
        dtype = torch.float16):
    
        self.config = config  ### OptConfig or LlamaConfig
        self.K = K
        self.L = L
        self.num_layers = num_layers
        self.batch_size = batch_size
        if "opt" in config.name:
              self.num_key_value_heads = config.n_head
        else: ## llama
            self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.n_head
        self.max_length = max_length ### max overall length (prefill+decode)
        self.device = device
        self.dtype = dtype
        self.head_dim = config.hidden_size // config.n_head
        
        self.offloaded = False ### check whether there are keys offloaded to lsh
        self.persisted = False
    
        ### key -= avg_k before fill to lsh, so record avg_k for recovery
        self.avg_k = [torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)]
        self.current_prefix_id = 0 ### 0 means nothing
        self.prefix_to_server = {} ### record lsh_retriever and kv_store here
        self.lsh_retriever = LSH()
        self.lsh_retriever.alloc(self.K, self.L, self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.batch_size, self.max_length)
        self.kv_store = KVStore()
        self.kv_store.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, max_length)
        
        self.hash_func = torch.randn((self.head_dim, self.K * self.L), device=self.device, dtype=self.dtype)
        self.binary_pack = [int(2**i) for i in range(self.K)]
        self.binary_pack = torch.Tensor(self.binary_pack).to(device=self.device, dtype=torch.float16)
        
        ### store lsh query results; TODO: adjust shape to (b * num_key_value_heads,)
        self.nnz = torch.zeros((self.batch_size * self.num_attention_heads,)).to(torch.int32)
        self.results_lsh_cpu = torch.zeros((self.batch_size * self.num_attention_heads, self.max_length)).to(torch.int32)
    
        self.query_results = [(torch.zeros_like(self.nnz), torch.zeros_like(self.results_lsh_cpu)) for _ in range(self.num_layers)]
    
        ### store hashcode of queries during prefill
        ### use pinned memory to interact with cpp codes
        ### max_query_tokens is fixed in cpp codes, do not modify
        self.max_query_tokens = 1024
        self.pinned_hashcode_multi = torch.zeros((self.num_attention_heads, self.max_query_tokens, self.L), dtype=torch.int32).pin_memory()
        self.pinned_hashcode = torch.zeros((self.num_attention_heads, self.L), dtype=torch.int32).pin_memory()

        ### store hashcode of prefix token key cache
        self.hash_code_buffer =  torch.zeros((self.num_key_value_heads, self.L, self.max_length), dtype=torch.int16, device=self.device)
        self.sorted_hash_values_buffer :torch.Tensor = None
        self.sorted_hash_indices_buffer :torch.Tensor = None

        ### store token group strategy of each layer
        self.persist_strategy = [None for i in range(self.num_layers)]
    

    ### alloc buffer for offloaded tokens, seq_len is # of offloaded tokens
    ### called in offload_to_lsh()
    def alloc_buffer(self, seq_len):
        self.sorted_hash_values_buffer = torch.zeros((self.num_key_value_heads, self.L, seq_len), dtype=torch.int16, device="cpu")
        self.sorted_hash_indices_buffer = torch.zeros((self.num_key_value_heads, self.L, seq_len), dtype=torch.int32, device="cpu")
    
    ### offload key and value to lsh
    ### layer_idx: current layer index
    ### request_id: id of request in current batch
    ### seq_len: sequence length of keys
    ### key_states, value_states, shape: len * kv_h * head_dim
    def offload_to_lsh(self,
        layer_idx:int,
        request_id: int,
        seq_len:int,
        prefix_id: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor):

        ### important : key_states, shape: len, kv_h, head_dim
        offload_key = key_states[:seq_len].transpose(0,1).contiguous() ## shape: #kvh, len, dim
        offload_value = value_states[:seq_len].transpose(0,1).contiguous()
        
        avg_k = offload_key.mean(dim=1, keepdim=True)
        offload_key = offload_key - avg_k
        self.avg_k[layer_idx][request_id] = avg_k
        
        offload_len = offload_key.shape[1]
        print(offload_key.shape, seq_len)
        self.alloc_buffer(offload_len)
        
        ### computing hashcode of offload keys
        self.hash_code_buffer.zero_()
        hash_code = torch.matmul(offload_key[:,:offload_len,:], self.hash_func)
        hash_code = hash_code > 0
        hash_code = hash_code.reshape(-1, self.K).to(torch.float16)
        hash_code = torch.mv(hash_code, self.binary_pack)
        hash_code = hash_code.reshape(self.num_key_value_heads, -1, self.L)
        hash_code = hash_code.transpose(1,2).contiguous().to(torch.int16)
        self.hash_code_buffer[:,:,:offload_len].copy_(hash_code)

        print("gpu ---> cpu")
        offload_key = offload_key.cpu()
        offload_value = offload_value.cpu()
        
        ### offload to kv store
        self.kv_store.fill(layer_idx, offload_key, offload_value)
        
        print(f"offloading {offload_len}")
        self.build_table(layer_idx, request_id, offload_len)

        self.sorted_hash_values_buffer.zero_()
        self.sorted_hash_indices_buffer.zero_()
        self.offloaded = True
        self.offload_len = offload_len
        self.current_prefix_id = prefix_id
    
    ### build lsh hash tables accordding to hashcodes
    ### called when offloading kv cache
    def build_table(self, 
        layer_idx:int,
        request_id: int,
        seq_len:int):
        
        for i in range(self.num_key_value_heads):
            sorted_hash_values, sorted_hash_indices = self.hash_code_buffer[i,:,:seq_len].sort()
            self.sorted_hash_values_buffer[i].copy_(sorted_hash_values)
            self.sorted_hash_indices_buffer[i].copy_(sorted_hash_indices)
        
        self.lsh_retriever.fill(layer_idx, request_id,
                    self.sorted_hash_values_buffer, 
                    self.sorted_hash_indices_buffer)

    def record_query_results(self, layer_idx):
        nnz, res = self.query_results[layer_idx]
        nnz.copy_(self.nnz)
        res.copy_(self.results_lsh_cpu)

    def get_imp_idx(self, layer_idx):
        nnz, res = self.query_results[layer_idx]
        for head_id, n in enumerate(nnz):
            res[head_id, n:] = -1
        max_len = nnz.max()
        
        return res[:, :max_len]

    ### get important kv by queries through lsh
    ### req_id: requst id inside a batch
    ### layer_idx: layer index
    ### query_states: queries shape: q_len * #attn_heads * head_dim
    ### prefix_id: to query kv from which prefix
    ### max_index: indices in query results cannot exceed this value
    ### returns keys and values of important tokens
    ### Note that!! the kvs are only valid before next get_kv(), needing copy after each get_kv()
    def get_kv(self, 
        req_id: int, 
        layer_idx: int, 
        query_states: torch.Tensor,
        prefix_id: int,
        max_index: int):
        if not self.offloaded or prefix_id == 0:
            return None, None
        q_len, _, _ = query_states.shape
        query_states = query_states.transpose(0,1) # num_heads, q_len, head_dim
        ### compute hashcode of queries
        norm_q = query_states.reshape(-1, self.head_dim)
        norm_q = norm_q / norm_q.norm(p=2, dim=-1, keepdim=True)
        q_hashcode = torch.matmul(norm_q, self.hash_func).gt(0)
        q_hashcode = q_hashcode.reshape(-1, self.K).to(torch.float16)
        q_hashcode = torch.mv(q_hashcode, self.binary_pack).int()
        q_hashcode = q_hashcode.reshape(self.num_attention_heads, q_len, self.L)
        
        self.pinned_hashcode_multi[...,:q_len,:].copy_(q_hashcode)
        ### get results from lsh hashtables
        self.lsh_retriever.batch_retrieve_multi(layer_idx, self.pinned_hashcode_multi, q_len ,self.results_lsh_cpu, self.nnz, max_index)
        print(self.nnz)
        self.record_query_results(layer_idx)
        ### collect key value from kv_store
        self.kv_store.collect_queried_key_value(prefix_id, layer_idx, self.results_lsh_cpu, self.nnz)
        ### shape : n_head, max_length, head_dim
        res_len = self.nnz.max().data
        queried_key = self.kv_store.get_queried_key_cache()
        queried_value = self.kv_store.get_queried_value_cache()
        avg_k = self.avg_k[layer_idx][req_id].to("cpu")
        queried_key = queried_key + avg_k
        queried_key = queried_key.transpose(0,1).contiguous()
        queried_value = queried_value.transpose(0,1).contiguous()

        return queried_key[:res_len], queried_value[:res_len]

    ### for debug...
    ### get full kv from kv_store by generating indices of range(offloaded_len)
    def get_full_kv(self, req_id, layer_idx, query_states, prefix_id):
        if not self.offloaded:
            return None, None
        ### generating indices covering offloaded_len
        for head_id in range(self.num_key_value_heads):
            self.nnz[head_id] = self.offload_len
            self.results_lsh_cpu[head_id][:self.offload_len].copy_(torch.arange(self.offload_len))
        
        self.kv_store.collect_queried_key_value(prefix_id, layer_idx, self.results_lsh_cpu, self.nnz)
        queried_key = self.kv_store.get_queried_key_cache()
        queried_value = self.kv_store.get_queried_value_cache()
        avg_k = self.avg_k[layer_idx][req_id].to("cpu")
        queried_key = queried_key + avg_k
        res_len, _ = self.nnz.max()
        # queried_key = queried_key.transpose(0,1).continuous()
        # queried_value = queried_value.transpose(0,1).continuous()

        return queried_key[:res_len], queried_value[:res_len]

    def query_group_persist(self, prefix_id):
        for layer_idx in range(self.num_layers):
            # To record token_ids for each head
            new_token_orders = [[] for _ in range(self.num_key_value_heads)]
            seen_token_ids = [set() for _ in range(self.num_key_value_heads)]
            ### query results per head
            nnz, results = self.query_results[layer_idx]
            for head_id in range(self.num_key_value_heads):
                n = nnz[head_id].item()
                ind = results[head_id][:n].view(-1).tolist()
                # Add new unseen tokens to the order
                for token_id in ind:
                    if token_id not in seen_token_ids[head_id]:
                        seen_token_ids[head_id].add(token_id)
                        new_token_orders[head_id].append(token_id)

            # Add remaining tokens (never seen) in ascending order
            all_token_ids = set(range(self.offload_len))
            for head_id in range(self.num_key_value_heads):
                remaining_ids = sorted(all_token_ids - seen_token_ids[head_id])
                # print(len(remaining_ids))
                new_token_orders[head_id].extend(remaining_ids)

            # Save strategy
            self.persist_strategy[layer_idx] = new_token_orders
            self.kv_store.write_to_storage("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/tmp_store",
                                            prefix_id, layer_idx, new_token_orders)
            print(f"Successfully write to storage, {prefix_id}")
        self.persisted = True

    ### arrange tokens into groups according to first req's query results
    # def single_query_group_strategy(self, layer_idx, q_hashcode, offload_len, prefix_id):
    #     _, q_len, _ = q_hashcode.shape
        
    #     # To record token_ids for each head
    #     new_token_orders = [[] for _ in range(self.num_key_value_heads)]
    #     seen_token_ids = [set() for _ in range(self.num_key_value_heads)]
        
    #     ### query results per token query
    #     for i in range(q_len):
    #         self.pinned_hashcode.copy_(q_hashcode[..., i, :])
    #         self.lsh_retriever.batch_retrieve(layer_idx, self.pinned_hashcode, self.results_lsh_cpu, self.nnz)
            
    #         for head_id in range(self.num_key_value_heads):
    #             n = self.nnz[head_id].item()
    #             if n == 0:
    #                 continue
    #             ind = self.results_lsh_cpu[head_id][:n].view(-1).tolist()
                
    #             # Add new unseen tokens to the order
    #             for token_id in ind:
    #                 if token_id not in seen_token_ids[head_id]:
    #                     seen_token_ids[head_id].add(token_id)
    #                     new_token_orders[head_id].append(token_id)

    #         self.results_lsh_cpu.zero_()
    #         self.nnz.zero_()
    #     # Add remaining tokens (never seen) in ascending order
    #     all_token_ids = set(range(offload_len))
    #     for head_id in range(self.num_key_value_heads):
    #         remaining_ids = sorted(all_token_ids - seen_token_ids[head_id])
    #         # print(len(remaining_ids))
    #         new_token_orders[head_id].extend(remaining_ids)

    #     # Save strategy
    #     self.persist_strategy[layer_idx] = new_token_orders
        
    #     self.kv_store.write_to_storage("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/tmp_store",
    #                                     prefix_id, layer_idx, new_token_orders)

    def reset(self, switch=False):
        # assert(self.current_prefix_id != 0)
        # if self.current_prefix_id not in self.prefix_to_server:
        #     self.prefix_to_server[self.current_prefix_id] = (self.lsh_retriever, self.kv_store)
        if switch:
            self.lsh_retriever = LSH()
            self.lsh_retriever.alloc(self.K, self.L, self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.batch_size, self.max_length)
            self.kv_store = KVStore()
            self.kv_store.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.max_length)
        
        self.nnz.zero_()
        self.results_lsh_cpu.zero_()
        self.hash_code_buffer.zero_()
        self.pinned_hashcode_multi.zero_()
        self.pinned_hashcode.zero_()
        self.query_results = [(torch.zeros_like(self.nnz), torch.zeros_like(self.results_lsh_cpu)) for _ in range(self.num_layers)]

    ### TODO ----- handle recover
    # def persist_kv_store_meta(self, prefix_id):
    #     self.kv_store.persist_meta(prefix_id)
    
    # def recover_kv_store_meta(self, prefix_id:int):
    #     self.offloaded = True
    #     self.kv_store.recover_meta("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/tmp_store", prefix_id)

    #     for layer_idx in range(self.num_layers):
    #         if layer_idx in self.dense_layers:
    #             continue
    #         self.recover_lsh(layer_idx)
        
    # def recover_lsh(self,
    #     layer_idx:int):
      
    #     hash_code = torch.load(f"/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/llm_infer/MagicPIG/examples/snapshots/hash_code_layer{layer_idx}.pt")
    #     _,_,seq_len = hash_code.shape
        
    #     self.offload_len = seq_len
    #     self.alloc_buffer(seq_len)
        
    #     self.hash_code_buffer[:,:,:seq_len].copy_(hash_code)
    #     self.build_table(layer_idx, 0, seq_len)
