from transformers import AutoTokenizer
import json
import os

"""Creates the joint vocabulary for all tokenizers"""


cache_dir = f"./models"
model_root = "models"
models_file = f"{model_root}/models.json"
# read the models file
with open(models_file) as f:
    jsonf = json.load(f)
    models = jsonf["models"]

# for each model, load the tokenizer and add it to a dictionary
tokenizers = {}
for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    tokenizers[model] = tokenizer

# find the number of common tokens for all pairs of tokenizers
vocab_info = {"vocab_length": {}, "common_tokens": {}}
tokenizer_models = list(tokenizers.keys())
for i in range(len(tokenizer_models)):
    model1 = tokenizer_models[i]
    # get the length of the vocab
    vocab_length = len(tokenizers[model1].get_vocab())
    vocab_info["vocab_length"][model1] = vocab_length
    vocab_info["common_tokens"][model1] = {}
    for j in range(i + 1, len(tokenizer_models)):
        model2 = tokenizer_models[j]
        vocab1 = set(tokenizers[model1].get_vocab().keys())
        vocab2 = set(tokenizers[model2].get_vocab().keys())
        common = vocab1.intersection(vocab2)
        vocab_info["common_tokens"][model1][model2] = len(common)

# save the common tokens count
with open(f"{model_root}/vocab_info.json", "w") as f:
    json.dump(vocab_info, f)

# create a mapping of unique tokens across all tokenizers
token2id = {}
id2token = {}
for model, tokenizer in tokenizers.items():
    vocab = tokenizer.get_vocab()
    # sort the vocab by token_id
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    for token, token_id in vocab.items():
        if token not in token2id:
            token2id[token] = len(token2id)
            id2token[token2id[token]] = token

# save the mappings
with open(f"{model_root}/token2id.json", "w") as f:
    json.dump(token2id, f)

with open(f"{model_root}/id2token.json", "w") as f:
    json.dump(id2token, f)