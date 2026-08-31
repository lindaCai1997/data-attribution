set -e  # Exit on first error

# Load secrets from attribution-temp/.env (HUGGING_FACE_HUB_TOKEN, OPENAI_API_KEY, ...).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"

dataset_root=${DATASET_ROOT:-/scratch/users/spa-data-attribution/dataset}
output_root=${OUTPUT_ROOT:-/scratch/users/spa-data-attribution/data}

# Auto-detect number of available GPUs (run inside an srun --gres=gpu:N session).
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
batch_size=${BATCH_SIZE:-2}
max_tokens=${MAX_TOKENS:-1024}
dtype=${DTYPE:-bfloat16}
max_num_data=${MAX_NUM_DATA:-1000000}
model_id=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}

method="selected_methods"
# layers 13 and 15 complete (200 shards each); skip to save resume overhead
layer_indices=(17 19)
data_file_names=("openorca_200k")

for layer_index in "${layer_indices[@]}"
do
    output_dir="${output_root}/qwen2.5_attr_l${layer_index}_cos"
    for data_file_name in "${data_file_names[@]}"
    do
        data_file_path="${dataset_root}/${data_file_name}.parquet"
        output_dir_path="${output_dir}/${data_file_name}/"

        echo "Running layer=${layer_index} method=${method} data=${data_file_name} output_dir=${output_dir_path}"
        torchrun --standalone --nproc_per_node=${NUM_GPUS} -m main_batched \
            --data ${data_file_path} \
            --output-dir ${output_dir_path} \
            --model-id ${model_id} \
            --method ${method} \
            --batch-size ${batch_size} \
            --max-tokens ${max_tokens} \
            --dtype ${dtype} \
            --layer-index ${layer_index} \
            --max-num-data ${max_num_data} \
            --is-train-data
    done
done
