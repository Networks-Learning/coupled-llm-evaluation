VOCAB_DIR="./models"
INPUT_FILE="./data/processed/LMSYS/questions.json"
OUTPUT_DIR="./outputs/LMSYS"

: '
List of model names seeds for independent generation.
For coupled generation, use SEED=500000.

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED=200000
QUANTIZE=0

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SEED=300000
QUANTIZE=0

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SEED=400000
QUANTIZE=0

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED=800000
QUANTIZE=4

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED=700000
QUANTIZE=8
'

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
QUANTIZE=0
SEED=500000

OUTPUT_FILE="responses_different"

# coupled generation
if [ "$SEED" -eq 500000 ]; then
  OUTPUT_FILE="responses_shared"
fi

N_SEEDS=10

# N_SEEDS runs
for i in $(seq 1 $N_SEEDS); do
  SEED_RUN=$((SEED+i-1))
  OUTPUT_FILE_RUN="${OUTPUT_FILE}_${i}"

  # run python script
  python ./src/lmsys.py --vocab_dir $VOCAB_DIR --input_file $INPUT_FILE --output_dir $OUTPUT_DIR --output_file $OUTPUT_FILE_RUN --model_name $MODEL_NAME --seed $SEED_RUN --quantize $QUANTIZE
done