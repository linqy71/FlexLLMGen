import argparse
import json
import os

# 强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 设置本地缓存路径
cache_dir = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME"
# os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, 'hub')
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')

from lm_eval import simple_evaluate
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
    # results = evaluator.evaluate(
    results = simple_evaluate(
        model=adaptor,
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        fewshot_as_multiturn = True,
        limit=args.limit,
        write_out=False,
        log_samples=False,
    )
    print('Finished')

    # dumped = json.dumps(results, indent=2)
    # print(dumped)