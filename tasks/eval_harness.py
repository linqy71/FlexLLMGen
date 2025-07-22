from functools import partial

import os
import transformers
from lm_eval.api.model import LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

tokenizer = None

def process_init():
    global tokenizer
    model_name = os.environ.get('MODEL_NAME', 'facebook/opt-1.3b')

    if model_name == "EleutherAI/gpt-neox-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    elif model_name == 'huggyllama/llama-7b':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_bos_token = False

def process_request(x, seq):
    global tokenizer

    # Handle both old format (tuple) and new format (Instance object)
    if hasattr(x, 'args'):
        # New lm_eval v0.4.9 format - Instance object
        ctx = x.args[0] if len(x.args) > 0 else ""
        cont = x.args[1] if len(x.args) > 1 else ""
    elif isinstance(x, (tuple, list)) and len(x) >= 2:
        # Old format - tuple/list
        ctx, cont = x[0], x[1]
    else:
        # Fallback
        print(f"Warning: Unexpected request format: {type(x)}")
        ctx, cont = str(x), ""
#     ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    cont_text = ftfy.fix_text(cont, normalization="NFKC")
    all_text = ctx_text + cont_text

    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)['input_ids']
    cont_tokens = tokenizer(cont_text, add_special_tokens=False)['input_ids']

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
        "prompt": ctx_text,
        "target": cont_text,
        "text": all_text,
    }


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")
    
    def generate_until(self, requests):
        """
        Generate text until stopping criteria are met using FlexGen.
        
        Args:
            requests: List of tuples (context, generation_kwargs)
                where generation_kwargs contains 'until' (stopping strings) 
                and 'max_gen_toks' (max tokens to generate)
        
        Returns:
            List of generated strings
        """
        global tokenizer
        results = []
        
        for request in requests:
            context, gen_kwargs = request
            
            # Get generation parameters
            until = gen_kwargs.get('until', [])
            max_gen_toks = gen_kwargs.get('max_gen_toks', 32)
            do_sample = gen_kwargs.get('do_sample', False)
            temperature = gen_kwargs.get('temperature', 1.0)
            
            try:
                # Tokenize the context
                input_ids = tokenizer(context, add_special_tokens=False, return_tensors='pt').input_ids
                input_list = [input_ids.squeeze(0).tolist()]
                
                # Check if the tpu cluster has a FlexGen model with generate method
                if hasattr(self.tpu, 'generate'):
                    # Use FlexGen's generate method
                    output_ids = self.tpu.generate(
                        inputs=input_list,
                        max_new_tokens=max_gen_toks,
                        do_sample=do_sample,
                        temperature=temperature,
                        stop=None,  # FlexGen uses stop token IDs, not strings
                        verbose=0
                    )
                    
                    # Decode the generated tokens (excluding the input tokens)
                    if output_ids and len(output_ids) > 0:
                        # Get only the newly generated tokens
                        input_len = len(input_list[0])
                        generated_ids = output_ids[0][input_len:] if len(output_ids[0]) > input_len else []
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        
                        # Apply stopping criteria if specified
                        if until:
                            for stop_str in until:
                                if stop_str in generated_text:
                                    generated_text = generated_text.split(stop_str)[0]
                                    break
                    else:
                        generated_text = ""
                else:
                    # Fallback: the tpu cluster doesn't have generate method
                    print("Warning: TPU cluster doesn't have generate method, returning empty string")
                    generated_text = ""
                    
            except Exception as e:
                print(f"Generation error: {e}")
                generated_text = ""
            
            results.append(generated_text)
        
        return results

    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(processes=1, initializer=process_init)
        # self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch):

            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output


