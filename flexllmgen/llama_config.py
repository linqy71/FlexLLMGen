from transformers import LlamaForCausalLM, LlamaConfig
import torch
import os
import numpy as np

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
        name = f"decode.layer.{idx}.input_layernorm.variance_epsilon" 
        save_weights(output_dir, name, hf_layer.input_layernorm.variance_epsilon)
        
        ### post attn
        name = f"decode.layer.{idx}.post_attn_layernorm.weight" 
        save_weights(output_dir, name, hf_layer.post_attention_layernorm.weight)
        name = f"decode.layer.{idx}.post_attn_layernorm.variance_epsilon" 
        save_weights(output_dir, name, hf_layer.post_attention_layernorm.variance_epsilon)
        
        
    name = "decoder.norm.weight"
    save_weights(output_dir, name, hf_model.model.norm.weight)
    name = "decoder.norm.variance_epsilon"
    save_weights(output_dir, name, hf_model.model.norm.variance_epsilon)
    
    
    
convert_local_llama_weights("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
                            "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/hub/Llama-3.1-weights")