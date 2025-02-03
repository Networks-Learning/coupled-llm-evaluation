from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
import torch
import json
import argparse
import os

def joint_sampler(probs, n_total, model_indices, total_id2token, model_token2id, rng):

  # sample Gumbels
  u = torch.rand(n_total, generator=rng, device=probs.device)
  gumbels = -torch.log(-torch.log(u + 1e-20) + 1e-20)

  total_probs = torch.zeros(n_total, device=probs.device)
  total_probs[model_indices] = probs

  # add the gumbels
  total_probs = torch.log(total_probs + 1e-20) + gumbels

  # get the token with the highest gumbel-max
  # NOTE: this restricts the argmax to the tokens that belong to the model's vocabulary
  argmax = model_indices[torch.argmax(total_probs[model_indices], dim=-1)]

  # map it back to the model's token ids
  model_token_id = torch.tensor([[model_token2id[total_id2token[str(argmax.item())]]]], device=probs.device)

  return model_token_id

def generate(model_name, user, cache_dir, vocab_dir, system, seed, temperature, max_length, quantize):


  print("Reading the joint vocabulary...")
  # read the id2token mapping
  with open("/".join([vocab_dir, "id2token.json"])) as f:
    total_id2token = json.load(f)

  # read the token2id mapping
  with open("/".join([vocab_dir, "token2id.json"])) as f:
      total_token2id = json.load(f)

  n_total = len(total_id2token) # total number of tokens in joint vocabulary

  print("Loading the model and tokenizer...")
  # load the model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
  if quantize == 4:
    print("Quantizing the model...")
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0", quantization_config=quantization_config)
  elif quantize == 8:
    print("Quantizing the model...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0", quantization_config=quantization_config)
  else:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cuda:0")

  # get the model's vocabulary
  model_token2id = tokenizer.get_vocab()
  model_id2token = {v: k for k, v in model_token2id.items()}

  # find the indices of the joint vocabulary that correspond to the model's vocabulary
  model_indices = torch.tensor([total_token2id[model_id2token[i]] for i in sorted(model_id2token.keys())], device=model.device)

  # encode the input text as chat
  chat = [
    {"role": "system", "content": system},
    {"role": "user", "content": user}
  ]
  inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

  # initialize the random number generator
  rng = torch.Generator(device=model.device)
  rng.manual_seed(seed)

  # generate the response
  eos_token_id = tokenizer.eos_token_id
  past_key_values = DynamicCache()
  cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
  generated_ids = inputs.input_ids
  query_length = inputs.input_ids.shape[1]
  model.eval()

  print("Generating response...")
  with torch.no_grad():
    for _ in range(max_length):
  
      outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
      logits = outputs.logits[:, -1, :len(model_token2id)]
      probs = torch.nn.functional.softmax(logits / temperature, dim=-1, dtype=torch.float32)

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

  print("Response: ", response)

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
  parser.add_argument("--user", type=str, required=True, help="User prompt")
  parser.add_argument("--cache_dir", type=str, default="./models", help="Directory that contains model files")
  parser.add_argument("--vocab_dir", type=str, default="./models", help="Directory that contains files for the joint vocabulary")
  parser.add_argument("--system", type=str, default="Keep your responses short and to the point.", help="System prompt")
  parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
  parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature")
  parser.add_argument("--max_length", type=int, default=1000, help="Maximum length of the generated response")
  parser.add_argument("--quantize", type=int, choices=[0,4,8], default=0, help="Choose quantization method (if any) the model")
  args = parser.parse_args()

  generate(**vars(args))