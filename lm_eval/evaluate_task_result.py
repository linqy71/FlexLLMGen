import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lm_eval import evaluator, tasks, simple_evaluate
from tasks import EvalHarnessAdaptor

def json_to_key(obj):
    return json.dumps(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate task results from FlexLLMGen')
    parser.add_argument('--result-file', type=str, default='flexgen_results.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='llama')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--is-prefix-caching-test', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')
    parser.add_argument('--acc-file', type=str, default='eval_results.json')
    parser.add_argument('--seq', type=int, default=4096)
    parser.add_argument('--tokenizer-path', type=str, default=None,
                       help='Path to tokenizer (sets MODEL_NAME env var)')
    args = parser.parse_args()

    if args.tokenizer_path:
        os.environ['MODEL_NAME'] = args.tokenizer_path
    elif args.model_type == 'llama':
        os.environ['MODEL_NAME'] = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.model_type == 'opt':
        os.environ['MODEL_NAME'] = "facebook/opt-66b"
    elif args.model_type == 'bloom':
        os.environ['MODEL_NAME'] = "bigscience/bloom"
    elif args.model_type == 'gpt_neox':
        os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    seq = args.seq
    total_batch = 1
    pe = 'fixed'

    class RealRunner:
        def __init__(self, args):
            self.results = {}
            self.is_prefix_caching_test = args.is_prefix_caching_test

            with open(args.result_file, 'r') as f:
                for line in f:
                    if line.strip() == '':
                        continue
                    item = json.loads(line)
                    request = item['request']
                    result = item['result']
                    self.results[json_to_key(request)] = result

            if self.is_prefix_caching_test:
                self.prefixed_results = []
                with open(args.result_file, 'r') as f:
                    for line in f:
                        if line.strip() == '':
                            continue
                        item = json.loads(line)
                        self.prefixed_results.append(
                            (item['request']['prompt'], item['result'])
                        )

            print(f"{len(self.results)} items in the cache")

        def eval(self, batch):
            from tasks.eval_harness import tokenizer

            mask_loss = []
            each_correct = []

            for i, text in enumerate(batch['text']):
                request = {
                    "best_of": 1,
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 0,
                    "model": "x",
                    "n": 1,
                    "prompt": text,
                    "request_type": "language-model-inference",
                    "stop": None,
                    "temperature": 0,
                    "top_p": 1
                }

                key = json_to_key(request)
                correct = True
                result = None

                if self.is_prefix_caching_test:
                    for p_prompt, p_result in self.prefixed_results:
                        if p_prompt.endswith(text):
                            result = p_result
                            break
                elif key in self.results:
                    result = self.results[key]

                if result:
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None

                    if len(batch['obs']) == 1 and len(batch['text']) > 1:
                        obs = batch['obs'][0]
                    else:
                        obs = batch['obs'][i]

                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]

                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for j, mask in enumerate(eval_mask):
                        try:
                            if j + 1 >= len(tokens):
                                break
                            if mask == True:
                                if args.debug:
                                    print(tokens[j + 1], next(iter(top_logprobs[j + 1].keys())))
                                correct = correct and (tokens[j + 1] == next(iter(top_logprobs[j + 1].keys())))
                                sum_lobprob += token_logprobs[j + 1]
                                n_positive += 1
                        except Exception as e:
                            raise e

                    avg_logprob = sum_lobprob / n_positive if n_positive > 0 else 0
                    mask_loss.append(-avg_logprob)
                    each_correct.append(correct)
                else:
                    print(f"Could not find result for prompt: {text[:100]}...")
                    assert False

            out = {
                'mask_loss': mask_loss,
                'each_correct': each_correct,
            }
            return out

    t = RealRunner(args)
    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    results = simple_evaluate(
        model=adaptor,
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        fewshot_as_multiturn=True,
        limit=args.limit,
        write_out=False,
        log_samples=False,
    )

    dumped = json.dumps(results, indent=2)
    with open(args.acc_file, 'w') as f:
        f.write(dumped)
    print(f"Results saved to {args.acc_file}")
