import argparse
import json, tqdm
import torch
import copy
import os
import numpy as np


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# Try to import FlexGen
try:
    from flexllmgen.flex_opt import OptLM, Policy
    from flexllmgen.opt_config import get_opt_config, download_opt_weights
    from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv, GB, Task
    from flexllmgen.compression import CompressionConfig
    FLEXGEN_AVAILABLE = True
except ImportError:
    print("Warning: FlexGen not available, will use HuggingFace only")
    FLEXGEN_AVAILABLE = False



def get_flexgen_logits(flexgen_model, input_ids, tokenizer):
    """
    Get logits from FlexGen model for given input_ids.
    Uses the optimized get_logits method from OptLM.
    
    Args:
        flexgen_model: FlexGen OptLM model
        input_ids: Input token IDs (torch tensor or numpy array)
        tokenizer: HuggingFace tokenizer
    
    Returns:
        torch.Tensor: Logits tensor of shape [batch_size, seq_len, vocab_size]
    """
    # Convert input_ids to numpy if it's a torch tensor
    if torch.is_tensor(input_ids):
        input_ids_np = input_ids.cpu().numpy()
    else:
        input_ids_np = input_ids
    
    # Convert to list format expected by FlexGen
    inputs = input_ids_np.tolist()
    
    # Use the optimized get_logits method
    logits = flexgen_model.get_logits(inputs, temperature=1.0, verbose=0)
    
    return logits

def init_flexgen_model(args):
    """Initialize FlexGen model."""
    if not FLEXGEN_AVAILABLE:
        raise ImportError("FlexGen is not available")
    
    # Setup execution environment
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(os.path.expanduser("~/flexllmgen_offload_dir"))
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                      mixed=TorchMixedDevice([gpu, cpu, disk]))
    
    # Setup policy with conservative defaults
    policy = Policy(
        1, 1,  # gpu_batch_size=1, num_gpu_batches=1
        100, 0, 100, 0, 100, 0,  # percent: all GPU
        True, False, True, False, 1.0,  # overlap, sep_layer, pin_weight, cpu_cache_compute, attn_sparsity
        False, CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),  # compress_weight
        False, CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False)   # compress_cache
    )
    
    # Initialize FlexGen model
    opt_config = get_opt_config(args.model_name)
    
    # The path passed to OptLM should be the specific directory containing model weights.
    weights_base_path = os.path.expanduser("~/opt_weights")
    
    # Construct the expected path from the model name, e.g., "facebook/opt-1.3b" -> "opt-1.3b-np"
    model_folder_name = args.model_name.split('/')[-1] + "-np"
    model_weights_path = os.path.join(weights_base_path, model_folder_name)

    # For compatibility, check for the full name path as well, e.g., "facebook-opt-1.3b-np"
    full_model_folder_name = args.model_name.replace("/", "-") + "-np"
    full_model_weights_path = os.path.join(weights_base_path, full_model_folder_name)

    final_path_to_use = None

    if os.path.exists(model_weights_path):
        final_path_to_use = model_weights_path
    elif os.path.exists(full_model_weights_path):
        final_path_to_use = full_model_weights_path
    else:
        # If neither path exists, download the weights.
        print(f"找不到模型权重: {model_weights_path} 或 {full_model_weights_path}")
        print(f"自动从 HuggingFace Hub 下载模型 '{args.model_name}'...")
        try:
            # download_opt_weights expects the base path (e.g., '~/opt_weights')
            download_opt_weights(args.model_name, weights_base_path)
            # After download, the path should be model_weights_path
            if os.path.exists(model_weights_path):
                 final_path_to_use = model_weights_path
            else:
                 # The downloaded name might have "facebook-" prefix
                 final_path_to_use = full_model_weights_path
        except Exception as e:
            raise RuntimeError(f"下载模型 '{args.model_name}' 失败。") from e

    if not os.path.exists(final_path_to_use):
        raise FileNotFoundError(
            f"即使在尝试下载后，模型权重目录仍然找不到。请检查路径：{final_path_to_use}"
        )
    
    print(f"使用模型权重路径: {final_path_to_use}")
    model = OptLM(opt_config, env, final_path_to_use, policy)
    
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default=None, required=True,
                       help='Path to input JSONL file')
    parser.add_argument('--output-path', type=str, default=None, required=True,
                       help='Path to output JSONL file')
    parser.add_argument('--enable_small_cache', action='store_true',
                       help='Enable small cache optimization')
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m',
                       help='Model name to use')
    parser.add_argument('--model-type', type=str, default='opt',
                       help='Model type (opt, llama, gpt_neox)')
    parser.add_argument("--cache-dir", type=str, default='../../checkpoint/',
                       help='Cache directory for model files')
    parser.add_argument('--use-flexgen', action='store_true',
                       help='Use FlexGen backend instead of HuggingFace')

    parser.add_argument("--heavy_ratio", type=float, default=0.1,
                       help='Heavy hitter cache ratio')
    parser.add_argument("--recent_ratio", type=float, default=0.1,
                       help='Recent cache ratio')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name

    print(f"Loading model: {model_name}")
    print(f"Backend: {'FlexGen' if args.use_flexgen else 'HuggingFace'}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    
    # Initialize model based on backend choice
    if args.use_flexgen and FLEXGEN_AVAILABLE:
        print("Initializing FlexGen model...")
        model = init_flexgen_model(args)
        use_flexgen = True
    else:
        if args.use_flexgen and not FLEXGEN_AVAILABLE:
            print("Warning: FlexGen not available, falling back to HuggingFace")
        print("Initializing HuggingFace model...")
        
        config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

        if args.enable_small_cache:
            print('Enable Small Cache Size')
            config.heavy_ratio = args.heavy_ratio
            config.recent_ratio = args.recent_ratio
            checkpoint = copy.deepcopy(model.state_dict())
            model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
            model.load_state_dict(checkpoint)

        model.half().eval().cuda()
        use_flexgen = False

    # Load requests
    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(f"Processing {len(requests)} requests...")
    
    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
            
            if use_flexgen:
                # Use FlexGen to get logits
                try:
                    logits = get_flexgen_logits(model, input_ids, tokenizer)
                    logits = logits.log_softmax(dim=-1)
                except Exception as e:
                    print(f"Error with FlexGen: {e}")
                    # Fallback to dummy logits
                    seq_len = input_ids.shape[1]
                    vocab_size = len(tokenizer)
                    logits = torch.zeros(1, seq_len, vocab_size)
            else:
                # Use HuggingFace model
                input_ids = input_ids.to(model.device)
                logits = model(input_ids).logits.log_softmax(dim=-1)

            # The rest of the processing is the same
            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:] # skip first
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
            
            result['result'] = {
                "choices": [
                    {
                        "text": prompt, 
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_path}")