import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lm_eval import simple_evaluate
from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate task data for LM evaluation')
    parser.add_argument('--output-file', type=str, default='input.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--seq', type=int, default=4096)
    parser.add_argument('--tokenizer-path', type=str, default=None,
                       help='Path to tokenizer (sets MODEL_NAME env var)')
    args = parser.parse_args()

    if args.tokenizer_path:
        os.environ['MODEL_NAME'] = args.tokenizer_path

    seq = args.seq
    total_batch = 1
    pe = 'fixed'

    with open(args.output_file, 'w') as f:
        pass

    class DryRunner:
        def eval(self, batch):
            with open(args.output_file, 'a') as f:
                for text in batch['text']:
                    item = {
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
                    f.write(json.dumps(item) + '\n')

            out = {
                'mask_loss': [1.0] * len(batch['text']),
                'each_correct': [True] * len(batch['text']),
            }
            return out

        def generate(self, inputs, max_new_tokens=32, do_sample=False,
                     temperature=1.0, stop=None, verbose=0):
            print(f"DryRunner.generate called with {len(inputs)} inputs, max_new_tokens={max_new_tokens}")
            results = []
            for input_ids in inputs:
                fake_generated = [1, 2, 3][:max_new_tokens]
                result = input_ids + fake_generated
                results.append(result)
            return results

    t = DryRunner()
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
    print('Finished')
