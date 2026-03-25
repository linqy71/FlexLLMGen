import os

import torch
from kvstore import KVStore
from lsh import LSH

from flexllmgen.timer import timers


class LSHServer:

    def __init__(
        self,
        config,
        num_layers: int,
        kv_store_path: str,
        K: int = 10,
        L: int = 150,
        batch_size: int = 1,
        max_length: int = 8192,
        device: str = "cuda:0",
        dtype=torch.float16,
    ):
        self.config = config
        self.K = K
        self.L = L
        self.num_layers = num_layers
        self.batch_size = batch_size
        if "opt" in config.name:
            self.num_key_value_heads = config.n_head
        else:
            self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.n_head
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.head_dim = config.hidden_size // config.n_head

        self.offloaded = False
        self.persisted = False

        self.avg_k = [torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        ) for _ in range(self.num_layers)]

        self.current_prefix_id = 0
        self.prefix_to_server = {}
        self.lsh_retriever = LSH()
        self.lsh_retriever.alloc(
            self.K, self.L, self.num_layers, self.num_attention_heads,
            self.num_key_value_heads, self.batch_size, self.max_length
        )
        self.kv_store = KVStore()
        self.kv_store.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, max_length)
        self.kv_store_path = kv_store_path

        hash_func_path = os.path.join(self.kv_store_path, f"hash_func_{self.K}_{self.L}.pt")
        if os.path.exists(hash_func_path):
            self.hash_func = torch.load(hash_func_path)
        else:
            self.hash_func = torch.randn((self.head_dim, self.K * self.L), device=self.device, dtype=self.dtype)
            torch.save(self.hash_func, hash_func_path)
        self.binary_pack = torch.tensor([int(2 ** i) for i in range(self.K)], device=self.device, dtype=torch.float16)

        self.nnz = torch.zeros((self.batch_size * self.num_attention_heads,), dtype=torch.int32)
        self.results_lsh_cpu = torch.zeros((self.batch_size * self.num_attention_heads, self.max_length), dtype=torch.int32)

        self.grouped_nnz = torch.zeros((self.batch_size * self.num_key_value_heads,), dtype=torch.int32)
        self.grouped_res = torch.full((self.batch_size * self.num_key_value_heads, self.max_length), -1, dtype=torch.int32)
        self.query_results = [
            (torch.zeros_like(self.grouped_nnz), torch.zeros_like(self.grouped_res))
            for _ in range(self.num_layers)
        ]

        self.max_query_tokens = 1024
        self.pinned_hashcode_multi = torch.zeros((self.num_attention_heads, self.max_query_tokens, self.L), dtype=torch.int32).pin_memory()
        self.pinned_hashcode = torch.zeros((self.num_attention_heads, self.L), dtype=torch.int32).pin_memory()
        self.copy_stream = torch.cuda.Stream()

        self.hash_code_buffer = torch.zeros((self.num_key_value_heads, self.L, self.max_length), dtype=torch.int16, device=self.device)
        self.sorted_hash_values_buffer: torch.Tensor = None
        self.sorted_hash_indices_buffer: torch.Tensor = None
        self.persist_strategy = [None for _ in range(self.num_layers)]

    def alloc_buffer(self, seq_len):
        self.sorted_hash_values_buffer = torch.zeros((self.num_key_value_heads, self.L, seq_len), dtype=torch.int16, device="cpu")
        self.sorted_hash_indices_buffer = torch.zeros((self.num_key_value_heads, self.L, seq_len), dtype=torch.int32, device="cpu")

    def _request_slot(self, req_id: int) -> int:
        if self.batch_size <= 1:
            return 0
        return req_id % self.batch_size

    def offload_to_lsh(
        self,
        layer_idx: int,
        request_id: int,
        seq_len: int,
        prefix_id: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        offload_key = key_states[:seq_len].transpose(0, 1).contiguous()
        offload_value = value_states[:seq_len].transpose(0, 1).contiguous()

        request_slot = self._request_slot(request_id)
        avg_k = offload_key.mean(dim=1, keepdim=True)
        offload_key = offload_key - avg_k
        self.avg_k[layer_idx][request_slot] = avg_k

        offload_len = offload_key.shape[1]
        self.alloc_buffer(offload_len)

        self.hash_code_buffer.zero_()
        hash_code = torch.matmul(offload_key[:, :offload_len, :], self.hash_func)
        hash_code = hash_code > 0
        hash_code = hash_code.reshape(-1, self.K).to(torch.float16)
        hash_code = torch.mv(hash_code, self.binary_pack)
        hash_code = hash_code.reshape(self.num_key_value_heads, -1, self.L)
        hash_code = hash_code.transpose(1, 2).contiguous().to(torch.int16)
        self.hash_code_buffer[:, :, :offload_len].copy_(hash_code)

        offload_key = offload_key.cpu()
        offload_value = offload_value.cpu()
        self.kv_store.fill(layer_idx, offload_key, offload_value)

        self.build_table(layer_idx, request_slot, offload_len)
        self.sorted_hash_values_buffer.zero_()
        self.sorted_hash_indices_buffer.zero_()
        self.offloaded = True
        self.offload_len = offload_len
        self.current_prefix_id = prefix_id

        if not self.persisted:
            avg_k_file = os.path.join(self.kv_store_path, f"avg_k_prefix_{prefix_id}_layer_{layer_idx}.pt")
            torch.save(self.avg_k[layer_idx].cpu(), avg_k_file)
            hash_code_file = os.path.join(self.kv_store_path, f"hash_code_prefix_{prefix_id}_layer_{layer_idx}.pt")
            torch.save(self.hash_code_buffer[:, :, :offload_len].cpu(), hash_code_file)

    def build_table(self, layer_idx: int, request_id: int, seq_len: int):
        for i in range(self.num_key_value_heads):
            sorted_hash_values, sorted_hash_indices = self.hash_code_buffer[i, :, :seq_len].sort()
            self.sorted_hash_values_buffer[i].copy_(sorted_hash_values)
            self.sorted_hash_indices_buffer[i].copy_(sorted_hash_indices)

        self.lsh_retriever.fill(layer_idx, request_id, self.sorted_hash_values_buffer, self.sorted_hash_indices_buffer)

    def group_query_results_for_gqa(self, layer_idx):
        self.grouped_nnz.zero_()
        self.grouped_res.fill_(-1)
        nnz_attn, res_attn = self.nnz, self.results_lsh_cpu
        group_size = self.num_attention_heads // self.num_key_value_heads

        for batch_idx in range(self.batch_size):
            for kv_head in range(self.num_key_value_heads):
                row = batch_idx * self.num_key_value_heads + kv_head
                merged = []
                seen = set()
                start = batch_idx * self.num_attention_heads + kv_head * group_size
                end = start + group_size

                for attn_head in range(start, end):
                    n = int(nnz_attn[attn_head].item())
                    ids = res_attn[attn_head, :n].tolist()
                    for token_id in ids:
                        if token_id >= 0 and token_id not in seen:
                            seen.add(token_id)
                            merged.append(token_id)

                self.grouped_nnz[row] = len(merged)
                if merged:
                    self.grouped_res[row, :len(merged)] = torch.tensor(merged, dtype=torch.int32, device=res_attn.device)

    def record_query_results(self, layer_idx):
        nnz, res = self.query_results[layer_idx]
        nnz.copy_(self.grouped_nnz)
        res.copy_(self.grouped_res)

    def get_imp_idx(self, layer_idx):
        nnz, res = self.query_results[layer_idx]
        for head_id, n in enumerate(nnz):
            res[head_id, n:] = -1
        max_len = nnz.max()
        avg_len = nnz.sum() / len(nnz)
        return res[:, :max_len], avg_len

    def get_full_idx(self, layer_idx):
        res = torch.zeros((self.num_key_value_heads, self.offload_len), dtype=int, device="cpu")
        for i in range(self.num_key_value_heads):
            res[i, :].copy_(torch.arange(0, self.offload_len))
        return res, self.offload_len

    def lsh_retrieve(
        self,
        req_id: int,
        layer_idx: int,
        query_states: torch.Tensor,
        prefix_id: int,
        max_index: int,
        save_res: bool = False,
    ):
        if not self.offloaded or prefix_id == 0:
            return None, None

        q_len, _, _ = query_states.shape
        with torch.cuda.stream(self.copy_stream):
            query_states = query_states.transpose(0, 1).contiguous()
            norm_q = query_states.reshape(-1, self.head_dim)
            norm_q = norm_q / norm_q.norm(p=2, dim=-1, keepdim=True)
            q_hashcode = torch.matmul(norm_q, self.hash_func).gt(0)
            q_hashcode = q_hashcode.reshape(-1, self.K).to(torch.float16)
            q_hashcode = torch.mv(q_hashcode, self.binary_pack).int()
            q_hashcode = q_hashcode.reshape(self.num_attention_heads, q_len, self.L)
            self.pinned_hashcode_multi[..., :q_len, :].copy_(q_hashcode, non_blocking=True)
        self.copy_stream.synchronize()

        self.results_lsh_cpu.zero_()
        self.nnz.zero_()
        self.lsh_retriever.batch_retrieve_multi(layer_idx, self.pinned_hashcode_multi, q_len, self.results_lsh_cpu, self.nnz, max_index)
        self.group_query_results_for_gqa(layer_idx)
        self.record_query_results(layer_idx)

        if save_res:
            save_dir = os.path.join(self.kv_store_path, f"saved_query_results_{req_id}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save((self.grouped_nnz.clone(), self.grouped_res.clone()), os.path.join(save_dir, f"layer_{layer_idx}_prefill_imp_token_idx.pt"))

    def load_kv(self, req_id: int, layer_idx: int, prefix_id: int):
        if not self.offloaded or prefix_id == 0:
            return None, None

        timers("io part test").start()
        self.kv_store.concurrent_merge_collect_queried_key_value(prefix_id, layer_idx, self.grouped_res, self.grouped_nnz)
        timers("io part test").stop()

        queried_key = self.kv_store.get_queried_key_cache()
        queried_value = self.kv_store.get_queried_value_cache()
        request_slot = self._request_slot(req_id)
        avg_k = self.avg_k[layer_idx][request_slot].to("cpu")
        queried_key = queried_key + avg_k

        max_len = int(self.grouped_nnz.max().item())
        queried_key = queried_key.transpose(0, 1).contiguous()
        queried_value = queried_value.transpose(0, 1).contiguous()
        return queried_key[:max_len], queried_value[:max_len]

    def get_full_kv(self, req_id, layer_idx, prefix_id):
        if not self.offloaded:
            return None, None

        self.grouped_nnz.zero_()
        self.grouped_res.fill_(-1)
        for head_id in range(self.num_key_value_heads):
            self.grouped_nnz[head_id] = self.offload_len
            self.grouped_res[head_id, :self.offload_len].copy_(torch.arange(self.offload_len))
        self.record_query_results(layer_idx)
        self.kv_store.merge_collect_queried_key_value(prefix_id, layer_idx, self.grouped_res, self.grouped_nnz)
        queried_key = self.kv_store.get_queried_key_cache()
        queried_value = self.kv_store.get_queried_value_cache()
        request_slot = self._request_slot(req_id)
        avg_k = self.avg_k[layer_idx][request_slot].to("cpu")
        queried_key = queried_key + avg_k
        res_len = int(self.grouped_nnz.max().item())
        queried_key = queried_key.transpose(0, 1).contiguous()
        queried_value = queried_value.transpose(0, 1).contiguous()
        return queried_key[:res_len], queried_value[:res_len]

    def sequential_persist(self, prefix_id):
        for layer_idx in range(self.num_layers):
            new_token_orders = [list(range(self.offload_len)) for _ in range(self.num_key_value_heads)]
            self.persist_strategy[layer_idx] = new_token_orders
            self.kv_store.write_to_layer_promote_file(self.kv_store_path, prefix_id, layer_idx, new_token_orders)

        self.lsh_retriever.save_to_file(self.kv_store_path + "/lsh_table_" + str(prefix_id))
        self.persist_kv_store_meta(prefix_id)
        self.persisted = True

    def query_group_persist(self, prefix_id):
        for layer_idx in range(self.num_layers):
            new_token_orders = [[] for _ in range(self.num_key_value_heads)]
            seen_token_ids = [set() for _ in range(self.num_key_value_heads)]
            nnz, results = self.query_results[layer_idx]
            for head_id in range(self.num_key_value_heads):
                n = nnz[head_id].item()
                ind = results[head_id][:n].view(-1).tolist()
                for token_id in ind:
                    if token_id not in seen_token_ids[head_id]:
                        seen_token_ids[head_id].add(token_id)
                        new_token_orders[head_id].append(token_id)

            all_token_ids = set(range(self.offload_len))
            for head_id in range(self.num_key_value_heads):
                remaining_ids = sorted(all_token_ids - seen_token_ids[head_id])
                new_token_orders[head_id].extend(remaining_ids)

            self.persist_strategy[layer_idx] = new_token_orders
            self.kv_store.write_to_layer_promote_file(self.kv_store_path, prefix_id, layer_idx, new_token_orders)

        self.lsh_retriever.save_to_file(self.kv_store_path + "/lsh_table_" + str(prefix_id))
        self.persist_kv_store_meta(prefix_id)
        self.persist_queried_result(prefix_id)
        self.persisted = True

    def promote_persist(self, prefix_id):
        for layer_idx in range(self.num_layers):
            promote_token_info = []
            nnz, results = self.query_results[layer_idx]
            for head_id in range(self.num_key_value_heads):
                n = nnz[head_id].item()
                ind = results[head_id][:n].view(-1).tolist()
                for token_id in ind:
                    promote_token_info.append((head_id, token_id))
            if promote_token_info:
                promote_token_tensor = torch.tensor(promote_token_info, dtype=torch.long, device="cpu")
                self.kv_store.promote_persist(self.kv_store_path, prefix_id, layer_idx, promote_token_tensor)

    def reorder_persist(self, prefix_id):
        for layer_idx in range(self.num_layers):
            reorder_token_info = []
            nnz, results = self.query_results[layer_idx]
            for head_id in range(self.num_key_value_heads):
                n = nnz[head_id].item()
                ind = results[head_id][:n].view(-1).tolist()
                for token_id in ind:
                    reorder_token_info.append((head_id, token_id))
            if reorder_token_info:
                reorder_token_tensor = torch.tensor(reorder_token_info, dtype=torch.long, device="cpu")
                self.kv_store.reorder_persist(self.kv_store_path, prefix_id, layer_idx, reorder_token_tensor)

    def reset(self, switch=False):
        if switch:
            self.lsh_retriever = LSH()
            self.lsh_retriever.alloc(
                self.K, self.L, self.num_layers, self.num_attention_heads,
                self.num_key_value_heads, self.batch_size, self.max_length
            )
            self.kv_store = KVStore()
            self.kv_store.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.max_length)

        self.nnz.zero_()
        self.results_lsh_cpu.zero_()
        self.grouped_nnz.zero_()
        self.grouped_res.fill_(-1)
        self.hash_code_buffer.zero_()
        self.pinned_hashcode_multi.zero_()
        self.pinned_hashcode.zero_()
        self.query_results = [
            (torch.zeros_like(self.grouped_nnz), torch.zeros_like(self.grouped_res))
            for _ in range(self.num_layers)
        ]

    def persist_kv_store_meta(self, prefix_id):
        self.kv_store.persist_meta(prefix_id)

    def recover_kv_store_meta(self, prefix_id):
        self.offloaded = True
        self.kv_store.recover_meta(self.kv_store_path, prefix_id)

    def persist_queried_result(self, prefix_id):
        for layer_idx in range(self.num_layers):
            nnz, results = self.query_results[layer_idx]
            file_path = os.path.join(self.kv_store_path, f"queried_results_prefix_{prefix_id}_layer_{layer_idx}.pt")
            torch.save({"nnz": nnz, "results": results}, file_path)

    def recover_queried_result(self, prefix_id):
        for layer_idx in range(self.num_layers):
            file_path = os.path.join(self.kv_store_path, f"queried_results_prefix_{prefix_id}_layer_{layer_idx}.pt")
            if not os.path.exists(file_path):
                continue
            data_loaded = torch.load(file_path, map_location="cpu")
            self.query_results[layer_idx] = (
                data_loaded["nnz"].to(self.grouped_nnz.device),
                data_loaded["results"].to(self.grouped_res.device),
            )

    def load_lsh_meta(self, prefix_id):
        self.offloaded = True
        self.current_prefix_id = prefix_id

        timers("load LSH meta").start()
        meta_file = os.path.join(self.kv_store_path, f"{prefix_id}.meta")
        if os.path.exists(meta_file):
            self.recover_kv_store_meta(prefix_id)
        timers("load LSH meta").stop()

        timers("load LSH meta").start()
        for layer_idx in range(self.num_layers):
            avg_k_file = os.path.join(self.kv_store_path, f"avg_k_prefix_{prefix_id}_layer_{layer_idx}.pt")
            if os.path.exists(avg_k_file):
                self.avg_k[layer_idx] = torch.load(avg_k_file, map_location="cpu").to(device=self.device, dtype=self.dtype)

            hash_code_file = os.path.join(self.kv_store_path, f"hash_code_prefix_{prefix_id}_layer_{layer_idx}.pt")
            if not os.path.exists(hash_code_file):
                continue
            loaded_hash_code = torch.load(hash_code_file, map_location=self.device)
            _, _, seq_len = loaded_hash_code.shape
            self.offload_len = seq_len
            self.alloc_buffer(seq_len)
            self.hash_code_buffer[:, :, :seq_len].copy_(loaded_hash_code)
            self.build_table(layer_idx, 0, seq_len)
        timers("load LSH meta").stop()

    def load_lsh_meta_concurrent(self, prefix_id):
        self.load_lsh_meta(prefix_id)
