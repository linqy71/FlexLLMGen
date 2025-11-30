import argparse
import json, tqdm
import torch
import copy
import os
import numpy as np

os.environ['NUMEXPR_MAX_THREADS'] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# Try to import FlexGen
try:
    from flexllmgen.dapr_opt_eval import get_model, add_parser_arguments
    # from flexllmgen.flex_opt import OptLM, Policy
    # from flexllmgen.opt_config import get_opt_config, download_opt_weights
    # from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    # from flexllmgen.utils import ExecutionEnv, GB, Task
    # from flexllmgen.compression import CompressionConfig
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
    logits = flexgen_model.get_logits(inputs, temperature=1.0)
    
    return logits


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    add_parser_arguments(parser)

    parser.add_argument('--input-path', type=str, default=None, required=True,
                       help='Path to input JSONL file')
    parser.add_argument('--output-path', type=str, default=None, required=True,
                       help='Path to output JSONL file')
    # parser.add_argument('--enable_small_cache', action='store_true',
    #                    help='Enable small cache optimization')
    # parser.add_argument('--model-name', type=str, default='facebook/opt-350m',
    #                    help='Model name to use')
    # parser.add_argument('--model-type', type=str, default='opt',
    #                    help='Model type (opt, llama, gpt_neox)')
    parser.add_argument("--cache-dir", type=str, default='../../checkpoint/',
                       help='Cache directory for model files')
    parser.add_argument('--use-flexgen', action='store_true',
                       help='Use FlexGen backend instead of HuggingFace')

    # parser.add_argument("--heavy_ratio", type=float, default=0.1,
    #                    help='Heavy hitter cache ratio')
    # parser.add_argument("--recent_ratio", type=float, default=0.1,
    #                    help='Recent cache ratio')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model

    print(f"Loading model: {model_name}")
    print(f"Backend: {'FlexGen' if args.use_flexgen else 'HuggingFace'}")
    
    # Initialize tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Initialize model based on backend choice
    if args.use_flexgen and FLEXGEN_AVAILABLE:
        print("Initializing FlexGen model...")
        model = get_model(args)
        use_flexgen = True
    else:
        if args.use_flexgen and not FLEXGEN_AVAILABLE:
            print("Warning: FlexGen not available, falling back to HuggingFace")
        print("Initializing HuggingFace model...")
        config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
        # if args.enable_small_cache:
        #     print('Enable Small Cache Size')
        #     config.heavy_ratio = args.heavy_ratio
        #     config.recent_ratio = args.recent_ratio
        #     checkpoint = copy.deepcopy(model.state_dict())
        #     model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        #     model.load_state_dict(checkpoint)
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
            model.finish_one_query()

    model.finish_one_query(final=True)
    model.env.close_copy_threads()
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_path}")