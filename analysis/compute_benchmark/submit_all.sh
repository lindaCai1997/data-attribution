#!/bin/bash
# Submit the full TRAK-vs-activation compute benchmark (paper Appendix K).
# 6 stage-1 timing jobs (each in its own fresh sbatch) + 1 scoring job that
# depends on all six (genuine data dependency: it reads their output shards).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SB="${DIR}/run_stage1_bench.sbatch"
TRAIN_DATA="/scratch/users/spa-data-attribution/dataset/dolly_10k.parquet"
EVAL_DATA="/scratch/users/spa-data-attribution/dataset/sycophancy_gpt.parquet"

submit () {
  local tag="$1" driver="$2" data="$3" full_n="$4" args="$5"
  sbatch --parsable --job-name="cb-${tag}" \
    --export=ALL,DRIVER="${driver}",TAG="${tag}",DATA="${data}",FULL_N="${full_n}",WARM_N=50,DRIVER_ARGS="${args}" \
    "${SB}"
}

# --- indexing (train side, first 2000 of dolly_10k) ---
J1=$(submit idx-allv3-bs2 main_batched "${TRAIN_DATA}" 2000 "--method all_v3 --layer-index 19 --batch-size 2")
J2=$(submit idx-rct-bs1   main         "${TRAIN_DATA}" 2000 "--method residual_change_treatment --layer-index 19 --batch-size 1")
J3=$(submit idx-trak-bs1  main_trak    "${TRAIN_DATA}" 2000 "--projection-dim 4096 --batch-size 1")

# --- per-behavior query (stage 1 on the 300-pair sycophancy_gpt eval set) ---
J4=$(submit qry-allv3-bs2 main_batched "${EVAL_DATA}" 300 "--method all_v3 --layer-index 19 --batch-size 2")
J5=$(submit qry-rct-bs1   main         "${EVAL_DATA}" 300 "--method residual_change_treatment --layer-index 19 --batch-size 1")
J6=$(submit qry-trak-bs1  main_trak    "${EVAL_DATA}" 300 "--projection-dim 4096 --batch-size 1")

# --- scoring pass (needs all six output dirs) ---
J7=$(sbatch --parsable --dependency=afterok:${J1}:${J2}:${J3}:${J4}:${J5}:${J6} \
     "${DIR}/run_scoring_bench.sbatch")

echo "submitted: idx-allv3=${J1} idx-rct=${J2} idx-trak=${J3} qry-allv3=${J4} qry-rct=${J5} qry-trak=${J6} scoring=${J7}"
