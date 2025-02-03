from typing import List, Optional
import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
from joint_sampling import joint_sampler
import argparse

DEBUG = False
ROOT_DIR='.'


def run_lmsys(
    cache_dir=f"{ROOT_DIR}/models",
    vocab_dir=f"{ROOT_DIR}/models",
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    input_file: str = f'{ROOT_DIR}/data/processed/LMSYS/questions.json',
    output_file: str = 'responses',
    output_dir: str = f'{ROOT_DIR}/outputs/LMSYS/',
    system: str = "Keep your replies short and to the point.",
    seed: int = 42,
    temperature: float = 1,
    max_length=4096,
    quantize=0):

    print("Model: ", model_name)
    print('Reading the joint vocabulary...')
    # read the id2token mapping
    with open(f'{vocab_dir}/id2token.json') as f:
        total_id2token = json.load(f)

    # read the token2id mapping
    with open(f'{vocab_dir}/token2id.json') as f:
        total_token2id = json.load(f)

    n_total = len(total_id2token) # total number of tokens in joint vocabulary

    print("Loading the model and tokenizer...")
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if quantize==4:
        print("Quantizing the model (4bit)...")
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0", quantization_config=quantization_config)
    elif quantize==8:
        print("Quantizing the model (8bit)...")
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0", quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0")

    # get the model's vocabulary
    model_token2id = tokenizer.get_vocab()
    model_id2token = {v: k for k, v in model_token2id.items()}

    # find the indices of the joint vocabulary that correspond to the model's vocabulary
    model_indices = torch.tensor([total_token2id[model_id2token[i]] for i in sorted(model_id2token.keys())], device=model.device)
    
    df = pd.read_json(input_file)
    # initialize the random number generator
    rng = torch.Generator(device=model.device)
    np.random.seed(int(seed))
    seeds = np.random.randint(10000000, 99999999, size=2000)

    if DEBUG:
        df=df.head(n=1)
        print(seeds[0])

    output=[]

    seedno=-1
    for user in df['question']:
        # reset seed
        seedno+=1
        rng.manual_seed(int(seeds[seedno]))
        if DEBUG:
            print('RNG state:',rng.get_state())

        # encode the input text as chat
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

        # generate the response
        eos_token_id = tokenizer.eos_token_id
        past_key_values = DynamicCache()
        cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
        generated_ids = inputs.input_ids
        query_length = inputs.input_ids.shape[1]
        model.eval()

        if DEBUG:
            print("Generating response...")
        with torch.no_grad():
            token_counter=0
            for _ in range(max_length):

                outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :len(model_token2id)]
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                probs=probs.to(torch.float32)

                # sample the next token using the Gumbel-Max SCM over the joint vocabulary
                next_token_ids = joint_sampler(probs, n_total, model_indices, total_id2token, model_token2id, rng)

                generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)      

                # NOTE: use caching to speed-up the autoregressive generation
                # see https://huggingface.co/docs/transformers/kv_cache#under-the-hood-how-cache-object-works-in-attention-mechanism
                attention_mask = inputs["attention_mask"]
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
                cache_position = cache_position[-1:] + 1 # add one more position for the next token

                if next_token_ids.item() == eos_token_id:
                    break

        # get the generated response (after the generation prompt token)
        response_tokens = generated_ids[0, query_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)

        if DEBUG:
            print("Response: ", response)

        # if DEBUG:
        #     print('Tokens:',response_tokens)
        
        output.append({'question':user,'model':model_name,'response':response,'seed':seeds[seedno]})
    output=pd.DataFrame(output)
    # print(output)

    output_dir=output_dir+'/'+model_name
    if quantize==4:
        output_dir=output_dir+'/q4'
    elif quantize==8:
        output_dir=output_dir+'/q8'
    file_path = Path(f"{output_dir}/{output_file}.json")
    if file_path.exists():
        os.remove(file_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output.to_json(file_path)



if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=f"{ROOT_DIR}/models", help="Directory that contains model files")
    parser.add_argument("--vocab_dir", default=f"{ROOT_DIR}/models", help="Directory that contains files for the joint vocabulary")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--input_file", default=f'{ROOT_DIR}/data/processed/LMSYS/questions.json')
    parser.add_argument("--output_file", default='responses')
    parser.add_argument("--output_dir", default=f'{ROOT_DIR}/outputs/LMSYS/')
    parser.add_argument("--system", default="Keep your replies short and to the point.", help='System prompt')
    parser.add_argument("--seed", default=500000)
    parser.add_argument("--temperature", default=1)
    parser.add_argument("--max_length", default=4096)
    parser.add_argument("--quantize", type=int, choices=[0,4,8], default=0, help="Choose quantization method (if any) for the model")
  

    args = parser.parse_args()
    print(args)

    run_lmsys(**vars(args))