import argparse
import json
import os

# 设置本地缓存路径
cache_dir = "/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_6/lqy/HF_HOME"
# os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, 'hub')
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
os.environ['HF_HUB_OFFLINE'] = '1'

from lm_eval import evaluator, tasks, simple_evaluate
from tasks import EvalHarnessAdaptor

def json_to_key(obj):
    return json.dumps(obj)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--result-file', type=str, default='flexgen_results.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--is-prefix-caching-test', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')
    parser.add_argument('--acc-file', type=str, default='eval_results.jsonl')
    parser.add_argument('--seq', type=int, default=1024)
    args = parser.parse_args()
    
    if args.model_type == 'opt':
        os.environ['MODEL_NAME'] = "facebook/opt-66b"
    elif args.model_type == 'bloom':
        os.environ['MODEL_NAME'] = "bigscience/bloom"
    elif args.model_type == 'gpt_neox':
        os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
    elif args.model_type == 'llama':
        os.environ['MODEL_NAME'] = "huggyllama/llama-7b"
    else:
        assert False

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
                # For prefix caching, we can't rely on exact prompt matching.
                # We'll create a list of (prompt, result) tuples for suffix matching.
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
                # print("text:", text) 
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
                    # Find the result by checking if the original prompt ends with the current prompt text.
                    # This handles the long, shared prefixes.
                    # print(f"Checking for prefix match for: {text}")
                    i= 0
                    for p_prompt, p_result in self.prefixed_results:
                        # if(i==0):
                        #     i += 1
                        #     print(f"Checking prefix: {p_prompt}")
                        if p_prompt.endswith(text):
                            # print(f"Found matching prefix for: {text}")
                            result = p_result
                            break
                elif key in self.results:
                    result = self.results[key]

                if result:
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None
                    
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    if len(batch['obs']) == 1 and len(batch['text']) > 1:
                        # This is a batch with a shared context
                        obs = batch['obs'][0]
                    else:
                        obs = batch['obs'][i]

                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]
                    
                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for i, mask in enumerate(eval_mask):
                        try:
                            
                            if i+1 >= len(tokens):
                                break
                            
                            if mask == True:
                                if args.debug:
                                    print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                sum_lobprob += token_logprobs[i+1]
                                n_positive += 1
                        except Exception as e:
                            raise e
                    
                    # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                    avg_logprob = sum_lobprob / n_positive
                    
                    mask_loss.append( - avg_logprob)
            
                    each_correct.append( correct )
                    
                else:
                    # 注意这里的逻辑，根据结果文件指定了key的匹配策略，所以num_fewshot不匹配也没关系。
                    print(f"Could not find result for prompt: {text}")
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
        fewshot_as_multiturn = True,
        limit=args.limit,
        write_out=False,
        log_samples=False,
    )
    
    dumped = json.dumps(results, indent=2)
    # print(dumped)
    with open(args.acc_file, 'w') as f:
        f.write(dumped)