import argparse
import json

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--output-file', type=str, default='input.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None, help='限制生成样本数量')
    args = parser.parse_args()

    seq = 1024
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
                'mask_loss': [1.0] * len(batch),
                'each_correct': [True] * len(batch),
            }
            return out
        
        def generate(self, inputs, max_new_tokens=32, do_sample=False, 
                     temperature=1.0, stop=None, verbose=0):
            """
            模拟 FlexGen 的 generate 方法（用于测试）
            """
            print(f"DryRunner.generate called with {len(inputs)} inputs, max_new_tokens={max_new_tokens}")
            
            # 返回模拟的生成结果（输入 + 一些假的生成 token）
            results = []
            for input_ids in inputs:
                # 模拟生成一些 token（这里只是简单重复几个 token）
                fake_generated = [1, 2, 3][:max_new_tokens]  # 假的生成 token
                result = input_ids + fake_generated
                results.append(result)
            
            return results

    t = DryRunner()
    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")
    
    # Call evaluate with positional arguments for lm_eval v0.4.9
    # Parameters: lm, task_dict, limit, samples, ...
    results = evaluator.evaluate(
        adaptor,  # lm parameter (positional)
        tasks.get_task_dict([args.task_name]),  # task_dict parameter (positional)
        limit=args.limit,  # limit parameter (keyword) - use command line argument
        cache_requests=False,  # cache_requests parameter
        bootstrap_iters=0,  # bootstrap_iters parameter (0 to turn off stderr calculations)
        write_out=False,  # write_out parameter
        log_samples=False,  # log_samples parameter
    )
    print('Finished')

    # dumped = json.dumps(results, indent=2)
    # print(dumped)