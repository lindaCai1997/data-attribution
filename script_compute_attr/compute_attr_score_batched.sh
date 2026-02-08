
set -e  # Exit on first error
export HF_HOME="$HOME/.cache/huggingface"

output_dir=${1:-/scratch7/users/aypan/tcai-scores/llama_attr_l19_cos}

# Auto-detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
batch_size=${4:-2}
max_tokens=${5:-1024}
dtype=${6:-bfloat16}
layer_index=${7:-19}
max_num_data=${8:-1000000}

methods=("all_v3")
data_file_names=("empathy_gpt" "modesty_gpt" "laziness_gpt" "sycophancy_gpt" "preachiness_gpt" "ultra_factual_truthfulness" "ultra_coding_instruction_following" "medhallu_easy_with_knowledge" "medhallu_medium_with_knowledge" "medhallu_hard_with_knowledge" "dolly_10k" "ultrachat_200k" "openorca_200k") 

for data_file_name in "${data_file_names[@]}"
    do
        for method in "${methods[@]}"
            do
                data_file_path="/scratch7/users/aypan/tcai-scores/dataset/${data_file_name}.parquet"
                output_dir_path="${output_dir}/${data_file_name}/"
                
                echo "Running with method: ${method}, output_dir: ${output_dir_path}"
                torchrun --standalone --nproc_per_node=${NUM_GPUS} -m main_batched \
                    --data ${data_file_path} \
                    --output-dir ${output_dir_path} \
                    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
                    --method ${method} \
                    --batch-size ${batch_size} \
                    --max-tokens ${max_tokens} \
                    --dtype ${dtype} \
                    --layer-index ${layer_index} \
                    --max-num-data ${max_num_data} \
                    --is-train-data
            done
    done