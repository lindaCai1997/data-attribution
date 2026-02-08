set -e  # Exit on first error
export HF_HOME="$HOME/.cache/huggingface"

output_dir=${1:-/scratch7/users/aypan/tcai-scores/llama_attr_l19_cos} 
# Auto-detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# currently only support batch size 1
batch_size=${4:-1}
max_tokens=${5:-1024}
dtype=${6:-bfloat16}
max_num_data=${8:-1000000}
model_id=${9:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
data_file_names=("empathy_gpt" "modesty_gpt" "laziness_gpt" "sycophancy_gpt" "preachiness_gpt" "ultra_factual_truthfulness" "ultra_coding_instruction_following" "medhallu_easy_with_knowledge" "medhallu_medium_with_knowledge" "medhallu_hard_with_knowledge" "dolly_10k" "ultrachat_200k" "openorca_200k") 

for data_file_name in "${data_file_names[@]}"; do
    torchrun --nproc_per_node=${NUM_GPUS} -m main_trak \
        --data /scratch7/users/aypan/tcai-scores/dataset/${data_file_name}.parquet \
        --output-dir ${output_dir}/${data_file_name}/trak \
        --model-id ${model_id} \
        --max-num-data ${max_num_data} \
        --is-train-data \
        --max-tokens ${max_tokens} \
        --dtype ${dtype}
done
                      