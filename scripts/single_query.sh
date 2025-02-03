source env/bin/activate

model_name="meta-llama/Llama-3.1-8B-Instruct"
user="What is the size of the sun?"
cache_dir="./models"
vocab_dir="./models"
system="Keep your responses short and to the point."
seed=42
temperature=0.7
max_length=1000

python src/joint_sampling.py --model_name $model_name --user "$user" --cache_dir $cache_dir --vocab_dir $vocab_dir --system "$system" --seed $seed --temperature $temperature --max_length $max_length 

deactivate