from transformers import LlamaForCausalLM, LlamaConfig
import torch
import os
import numpy as np
from typing import List

class LlamaConfig:
    name: str = "llama-3.1",
    num_hidden_layers: int,
    max_position_embeddings: int,
    hidden_size: int,
    num_heads: int,
    num_attention_heads: int,
    rms_norm_eps: float = 0.00001,
    rope_theta: int,
    eos_token_id: List[int],
    dtype: type = np.float16
    

def preset_llama3_config():
    config = LlamaConfig(
        num_hidden_layers = 32,
        max_position_embeddings = 131072,
        hidden_size = 4096,
        num_heads = 32,
        num_key_value_heads = 8,
        rms_norm_eps = 0.00001,
        rope_theta = 500000,
        eos_token_id = [128001, 128008, 128009]
    )
    return config

def get_llama_config_from(name, model_dir):
    lm_config = LlamaConfig.from_pretrained(model_dir)
    
    config = LlamaConfig(
        name = name,
        num_hidden_layers = lm_config.num_hidden_layers,
        max_position_embeddings = lm_config.max_position_embeddings,
        hidden_size = lm_config.hidden_size,
        num_heads = lm_config.num_attention_heads,
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
        with open(param_path, "wb") as f:
            np.save(f, param)
        return 
    np_array = param.cpu().detach().numpy()
    print(name)
    with open(param_path, "wb") as f:
        np.save(f, np_array)

def convert_local_llama_weights(model_dir, output_dir):
    hf_model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)

    name = "lm_head.weight"
    save_weights(output_dir, name, hf_model.lm_head.weight)
    
    name = "decoder.embed_tokens.weight"
    save_weights(output_dir, name, hf_model.model.embed_tokens.weight)
    
    for idx, hf_layer in enumerate(hf_model.model.layers):
      
        ### self attn
        name = f"decode.layer.{idx}.self_attn.q_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.q_proj.weight)
        name = f"decode.layer.{idx}.self_attn.k_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.k_proj.weight)
        name = f"decode.layer.{idx}.self_attn.v_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.v_proj.weight)
        name = f"decode.layer.{idx}.self_attn.o_proj.weight"
        save_weights(output_dir, name, hf_layer.self_attn.o_proj.weight)
        
        ### MLP
        name = f"decode.layer.{idx}.mlp.gate_proj.weight"
        save_weights(output_dir, name, hf_layer.mlp.gate_proj.weight)
        name = f"decode.layer.{idx}.mlp.up_proj.weight" 
        save_weights(output_dir, name, hf_layer.mlp.up_proj.weight)
        name = f"decode.layer.{idx}.mlp.down_proj.weight" 
        save_weights(output_dir, name, hf_layer.mlp.down_proj.weight)
        
        ### input layernorm
        name = f"decode.layer.{idx}.input_layernorm.weight" 
        save_weights(output_dir, name, hf_layer.input_layernorm.weight)
        
        ### post attn
        name = f"decode.layer.{idx}.post_attn_layernorm.weight" 
        save_weights(output_dir, name, hf_layer.post_attention_layernorm.weight)
        
    name = "decoder.norm.weight"
    save_weights(output_dir, name, hf_model.model.norm.weight)
    
    

if __name__ == "__main__":
    convert_local_llama_weights("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub/Llama-3.1-weights")