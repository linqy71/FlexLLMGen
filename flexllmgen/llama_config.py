from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
import numpy as np
from typing import List
import dataclasses

@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "llama3.1-8b"
    num_hidden_layers: int = 32
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    hidden_size: int = 4096
    n_head: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 0.00001
    rope_theta: int = 500000
    dtype: type = np.float16
    pad_token_id: int = 1
    input_dim : int = 4096
    
    def model_bytes(self):
        d = self.hidden_size
        L = self.num_hidden_layers
        V = self.vocab_size
        d_ff = int(2.67 * d)  # for SwiGLU

        # Q, K, V, O: 4 * d * d
        # FFN: d × d_ff + d_ff × d = 2 × d × d_ff
        # RMSNorm: 2 per layer × d
        per_layer = 4 * d * d + 3 * d * d_ff + 2 * d
        total_params = L * per_layer + V * d
        return 2 * total_params  # float16 or bfloat16
    
    def cache_bytes(self, batch_size, seq_len):
        head_dim = self.hidden_size // self.n_head
        return 2 * batch_size * seq_len * self.num_hidden_layers * head_dim * self.num_key_value_heads * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2
    

def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()
    print(name)
    if name == "llama3.1-8b":
        config = LlamaConfig()
    elif name == "llama3.3-70b":
        config = LlamaConfig(
            name=name,
            num_hidden_layers=80,
            vocab_size=128256,
            max_position_embeddings=131072,
            hidden_size=8192,
            n_head=64,
            num_key_value_heads=8,
            input_dim=8192
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)

def preset_llama3_config():
    config = LlamaConfig(
        name = "llama3.1-8b",
        num_hidden_layers = 32,
        max_position_embeddings = 131072,
        hidden_size = 4096,
        n_head = 32,
        num_key_value_heads = 8,
        rms_norm_eps = 0.00001,
        rope_theta = 500000
    )
    return config

def get_llama_config_from(name, model_dir):
    lm_config = LlamaConfig.from_pretrained(model_dir)
    
    config = LlamaConfig(
        name = name,
        num_hidden_layers = lm_config.num_hidden_layers,
        max_position_embeddings = lm_config.max_position_embeddings,
        hidden_size = lm_config.hidden_size,
        n_head = lm_config.num_attention_heads,
        num_key_value_heads = lm_config.num_key_value_heads,
        rms_norm_eps = lm_config.rms_norm_eps,
        rope_theta = lm_config.rope_theta,
        eos_token_id = lm_config.eos_token_id
    )
    
    return config

def save_weights(output_dir, name, param):
    param_path = os.path.join(output_dir, name)
    os.makedirs(os.path.dirname(param_path), exist_ok=True)
    if isinstance(param, float):
        print(name, param)
        with open(param_path, "wb") as f:
            np.save(f, param)
        return 
    print(param.shape)
    np_array = param.cpu().detach().to(torch.float16).numpy()
    print(name)
    with open(param_path, "wb") as f:
        np.save(f, np_array)

def convert_local_llama_weights(model_dir, output_dir):
    model = LlamaForCausalLM.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    print("Successfully loaded model automatically")


    name = "lm_head.weight"
    save_weights(output_dir, name, model.lm_head.weight)
    
    name = "decoder.embed_tokens.weight"
    save_weights(output_dir, name, model.model.embed_tokens.weight)
    
    for idx, hf_layer in enumerate(model.model.layers):
      
        ### self attn
        name = f"decoder.layers.{idx}.self_attn.q_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.q_proj.weight)
        name = f"decoder.layers.{idx}.self_attn.k_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.k_proj.weight)
        name = f"decoder.layers.{idx}.self_attn.v_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.v_proj.weight)
        name = f"decoder.layers.{idx}.self_attn.o_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.o_proj.weight)
        
        ### MLP
        name = f"decoder.layers.{idx}.mlp.gate_proj.weight"
        save_weights(output_dir, name, hf_layer.mlp.gate_proj.weight)
        name = f"decoder.layers.{idx}.mlp.up_proj.weight" 
        save_weights(output_dir, name, hf_layer.mlp.up_proj.weight)
        name = f"decoder.layers.{idx}.mlp.down_proj.weight" 
        save_weights(output_dir, name, hf_layer.mlp.down_proj.weight)
        
        ### input layernorm
        name = f"decoder.layers.{idx}.input_layernorm.weight" 
        save_weights(output_dir, name, hf_layer.input_layernorm.weight)
        
        ### post attn
        name = f"decoder.layers.{idx}.post_attn_layernorm.weight" 
        save_weights(output_dir, name, hf_layer.post_attention_layernorm.weight)
        
    name = "decoder.norm.weight"
    save_weights(output_dir, name, model.model.norm.weight)
    
    name = "rotary_emb.inv_freq"
    save_weights(output_dir, name, model.model.rotary_emb.inv_freq)
    name = "rotary_emb.attention_scaling"
    save_weights(output_dir, name, model.model.rotary_emb.attention_scaling)
    

if __name__ == "__main__":
    convert_local_llama_weights("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub/llama3.3-70b-np")