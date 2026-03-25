from typing import List, Optional, Union

import numpy as np

from flexllmgen.LSH_llama_dapr import (
    LLAMA as BaseLLAMA,
    CompressionConfig,
    ExecutionEnv,
    Policy,
    RadixTree,
    Task,
    TorchDevice,
    TorchDisk,
    TorchMixedDevice,
    ValueHolder,
    add_parser_arguments,
    array_3d,
    get_llama_config,
    logger,
)
from flexllmgen.lsh_server import LSHServer


class EvalLSHServer(LSHServer):
    """Keep a single prefix KV cache in CPU memory for eval/generation-quality runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persisted = True
        self.offload_len = 0

    def has_prefix(self, prefix_id: int) -> bool:
        return self.offloaded and self.current_prefix_id == prefix_id

    def reset(self, switch=False):
        super().reset(switch=switch)
        if switch:
            self.offloaded = False
            self.current_prefix_id = 0
            self.offload_len = 0
            self.persisted = True
            for avg_k in self.avg_k:
                avg_k.zero_()

    def sequential_persist(self, prefix_id):
        return None

    def query_group_persist(self, prefix_id):
        return None


class LLAMA(BaseLLAMA):
    def __init__(self,
                 config,
                 env,
                 path,
                 offload_dir,
                 policy,
                 max_prompt_len,
                 max_gen_len,
                 persist_strategy,
                 collision_threshold: int = 2):
        super().__init__(config, env, path, offload_dir, policy, max_prompt_len,
                         max_gen_len, persist_strategy,
                         collision_threshold=collision_threshold)
        self.kv_server = EvalLSHServer(
            self.config,
            self.num_hidden_layers,
            self.kv_store_path,
            K=10,
            L=150,
            batch_size=1,
            max_length=8192,
            device='cuda:0',
            collision_threshold=collision_threshold,
        )
        self.set_kv_server()

    def _reuse_prefix_if_loaded(self, prefix_id: int) -> bool:
        if self.kv_server.has_prefix(prefix_id):
            self.kv_server.reset(switch=False)
            return False
        self.kv_server.reset(switch=True)
        return True

    def get_logits(self,
                   inputs: Union[np.array, List[int]],
                   temperature: float = 1.0,
                   full: bool = False,
                   req_id: int = 0):
        matched_prefix = self.radix_tree.match(inputs[0])
        prefix_only = False
        new_prefix_id = 0
        if full:
            common_len = 0
        else:
            common_len = sum(matched_prefix.values())
        if common_len < 10:
            del self.radix_tree
            self.radix_tree = RadixTree()
            self.kv_server.reset(switch=True)
            prefix_only = True
            new_prefix_id = self.radix_tree.insert(inputs[0])
        else:
            new_prefix_id = list(matched_prefix.keys())[0]
            prefix_only = self._reuse_prefix_if_loaded(new_prefix_id)

        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=1,
            cut_gen_len=1,
            do_sample=False,
            temperature=temperature,
            stop=None,
            matched_prefix=matched_prefix,
            prefix_only=prefix_only,
            new_prefix_id=new_prefix_id,
            req_id=req_id,
            is_logits_task=True,
        )
        print(f"get_logits: prompt_len={task.prompt_len}")
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()

        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        self.set_task(task)
        return self.logits_loop_normal()

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 0.6,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 req_id: int = 0,
                 save_res: bool = False,
                 verbose: int = 0):
        matched_prefix = self.radix_tree.match(inputs[0])
        prefix_only = False
        new_prefix_id = 0
        if len(matched_prefix) == 0:
            self.kv_server.reset(switch=True)
            prefix_only = True
            new_prefix_id = self.radix_tree.insert(inputs[0])
        elif len(matched_prefix) == 1:
            new_prefix_id = list(matched_prefix.keys())[0]
            prefix_only = self._reuse_prefix_if_loaded(new_prefix_id)
        else:
            raise ValueError("Multi prefix_id Not supported.")

        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            matched_prefix=matched_prefix,
            prefix_only=prefix_only,
            new_prefix_id=new_prefix_id,
            req_id=req_id,
            save_res=save_res,
        )
        logger.info(f"generate: Task={task}")
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()

        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        self.set_task(task)

        if debug_mode is None:
            if not overlap:
                self.generation_loop_normal()
            else:
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        return self.output_ids

    def finish_one_query(self, final=False):
        self.sync()
        self.kv_server.reset(switch=False)

        logger.info("query finished , now sync the model")
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        if final:
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.delete_cache(j, k)

            self.delete_all_weights()
            if self.policy.cpu_cache_compute:
                self.env.cpu.del_attention_compute_workspace()


def get_model(args):
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))

    llama_config = get_llama_config(args.model)
    model = LLAMA(llama_config, env, args.path, args.offload_dir, policy,
                  args.prompt_len, args.gen_len, args.strategy,
                  collision_threshold=args.collision_threshold)
    model.env = env
    return model
