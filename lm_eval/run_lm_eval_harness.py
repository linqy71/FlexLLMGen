import argparse
import json, tqdm
import torch
import os
import sys
import numpy as np

os.environ['NUMEXPR_MAX_THREADS'] = "1"

# Add project root to path so we can import flexllmgen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer

try:
    from flexllmgen.LSH_llama_eval import get_model, add_parser_arguments
    FLEXGEN_AVAILABLE = True
except ImportError:
    print("Warning: FlexLLMGen not available")
    FLEXGEN_AVAILABLE = False


def get_flexgen_logits(flexgen_model, input_ids, tokenizer, full):
    """
    Get logits from FlexGen model for given input_ids.
    Uses the get_logits method from LLAMA.
    """
    if torch.is_tensor(input_ids):
        input_ids_np = input_ids.cpu().numpy()
    else:
        input_ids_np = input_ids

    inputs = input_ids_np.tolist()
    logits = flexgen_model.get_logits(inputs, temperature=1.0, full=full)
    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LM Eval Harness for FlexLLMGen Llama')

    add_parser_arguments(parser)

    parser.add_argument('--input-path', type=str, default=None, required=True,
                       help='Path to input JSONL file')
    parser.add_argument('--output-path', type=str, default=None, required=True,
                       help='Path to output JSONL file')
    parser.add_argument("--cache-dir", type=str, default='../../checkpoint/',
                       help='Cache directory for model files')
    parser.add_argument('--use-flexgen', action='store_true',
                       help='Use FlexGen backend instead of HuggingFace')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model

    print(f"Loading model: {model_name}")
    print(f"Backend: {'FlexGen' if args.use_flexgen else 'HuggingFace'}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Initialize model
    if args.use_flexgen and FLEXGEN_AVAILABLE:
        print("Initializing FlexGen Llama model...")
        model = get_model(args)
        use_flexgen = True
    else:
        if args.use_flexgen and not FLEXGEN_AVAILABLE:
            print("Warning: FlexGen not available, falling back to HuggingFace")
        from transformers import AutoModelForCausalLM, AutoConfig
        print("Initializing HuggingFace model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
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
                try:
                    full = False
                    if args.K == args.L:
                        full = True
                    logits = get_flexgen_logits(model, input_ids, tokenizer, full)
                    logits = logits.log_softmax(dim=-1)
                except Exception as e:
                    print(f"Error with FlexGen: {e}")
                    import traceback
                    traceback.print_exc()
                    seq_len = input_ids.shape[1]
                    vocab_size = len(tokenizer)
                    logits = torch.zeros(1, seq_len, vocab_size)
            else:
                input_ids = input_ids.to(model.device)
                logits = model(input_ids).logits.log_softmax(dim=-1)

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

            gold_indices = input_ids[:, 1:]  # skip first
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

    model.env.close_copy_threads()
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Results saved to {output_path}")
