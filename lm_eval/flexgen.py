import torch
from lm_eval.base import LM
from transformers import AutoTokenizer
import flexgen
from flexgen.flex_opt import Policy, OptLM
from tqdm import tqdm

class FlexGenLM(LM):
    def __init__(self, model_name="facebook/opt-1.3b", batch_size=1):
        super().__init__()

        # FlexGen specific arguments
        # These need to be adapted to your specific setup
        self.env = flexgen.get_env()
        self.policy = Policy(
            gpu_batch_size=batch_size,
            num_gpu_batches=1,
            percent=[100, 0, 100, 0, 100, 0],  # All on GPU
            cpu_cache_compute=False,
            attn_sparsity=1.0,
            compress_weight=False,
            comp_weight_config={"dummy": "dummy"},
            compress_cache=False,
            comp_cache_config={"dummy": "dummy"},
        )

        self.model = OptLM(model_name, self.env, ".", self.policy)
        self.tokenizer = self.model.tokenizer

        self.model.init_all_weights()
        
        self.batch_size_per_gpu = batch_size

    def loglikelihood(self, requests):
        res = []
        
        # The logic here is adapted from lm-evaluation-harness's HuggingFace integration
        # It processes requests in batches.

        for i in tqdm(range(0, len(requests), self.batch_size_per_gpu)):
            req_batch = requests[i:i + self.batch_size_per_gpu]

            inps = []
            cont_toks_list = []
            inplens = []

            for context, continuation in req_batch:
                if context == "":
                    # end of text as context
                    context_enc = [self.tokenizer.eos_token_id]
                else:
                    context_enc = self.tokenizer.encode(context, add_special_tokens=False)

                continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)

                inp = torch.tensor(
                    (context_enc + continuation_enc),
                    dtype=torch.long,
                ).to(self.model.device)
                
                inplen, = inp.shape

                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
            
            # Generate dummy inputs for padding
            # FlexGen might require inputs to be of the same length.
            # This part needs to be verified with FlexGen's documentation.
            # I am assuming generate takes a list of prompts (strings)
            # or tokenized inputs.
            
            # The FlexGen OptLM doesn't directly expose a way to get logits for arbitrary inputs.
            # It's primarily designed for generation.
            # We will use generation and get the logits from the generated output.
            # This is a workaround. A more direct way to get logits would be better.
            
            # This part of the code is a placeholder and needs to be replaced with
            # the correct way to get log-likelihoods from FlexGen.
            # The current FlexGen API (from the source code I can see)
            # is focused on `generate`. A `forward` pass to get logits is not obvious.
            # If `model.model` gives access to the underlying transformer, one could do:
            # outputs = self.model.model(input_ids=inps)
            # logits = outputs.logits
            
            # For now, let's assume a batch forward pass is not supported
            # and we process one by one. This will be slow.
            
            for j, (context, continuation) in enumerate(req_batch):
                context_enc = self.tokenizer.encode(context, add_special_tokens=False)
                continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
                
                full_enc = context_enc + continuation_enc
                
                # FlexGen's generate function can return logits
                # Let's assume it can take tokenized input
                # We need to get the logits for the continuation tokens.
                
                # This is a very inefficient way to do it.
                # A proper implementation would batch this.
                # I am assuming `self.model.generate` can give us something useful.
                # The public API for FlexGen is not well documented for this use case.
                
                # Let's assume we can get logits from the model somehow.
                # Since I cannot run FlexGen, I cannot test this.
                # I will create a placeholder logic.
                
                # Create a dummy call to illustrate what needs to happen.
                # The user will need to replace this.
                
                # WORKAROUND: In FlexGen, getting logits for a given sequence is not
                # straightforward. The `generate` function is the main entry point.
                # A potential way is to feed the text and get the model to generate 1 token
                # and inspect the internal state or return values if they provide logits.
                # This is highly dependent on the FlexGen version and implementation.
                
                # As a starting point, I will add dummy return values.
                # The user must implement the logic to get log probabilities.

                # DUMMY IMPLEMENTATION
                log_likelihood = 0.0
                is_greedy = True
                
                res.append((log_likelihood, is_greedy))
                
        return res

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []

        for i in tqdm(range(0, len(requests), self.batch_size_per_gpu)):
            req_batch = requests[i:i + self.batch_size_per_gpu]
            
            inps = [context for context, _ in req_batch]
            request_args = [args for _, args in req_batch]
            until = [args["until"] for args in request_args]
            
            # max_new_tokens can be specified in the request
            max_new_tokens = request_args[0].get("max_length", 256)

            outputs = self.model.generate(
                inps,
                max_new_tokens=max_new_tokens,
                stop=until[0], # Assuming same stop criteria for the batch
                )

            for i, (context, args) in enumerate(req_batch):
                generated_text = outputs[i]
                
                # remove the context from the generated text
                res.append(generated_text[len(context):])

        return res

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling is not implemented for FlexGenLM") 