#!/bin/bash
# fix work directory path
MYPATH="."
source $MYPATH/env/bin/activate
# go to work directory
cd $MYPATH

: '
model_name can be one of :
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
'
model_name="meta-llama/Llama-3.1-8B-Instruct"
dataset_name='mmlu'
seed=72645763 # seed used for coupled generation. For independent generation, we used 83475617 for 8B, 29387482 for 3B, and 18987374 for 1B
temperature=0.7
chunk_idx=-1
few_shot=0
max_length=20
n_repeats=10

# run python script
python -m src.accuracy --model_name $model_name --dataset_name $dataset_name --temperature $temperature --seed $seed --chunk_idx $chunk_idx --chunk_size $chunk_size --few_shot $few_shot --max_length $max_length --n_repeats $n_repeats