# Attribution-Based Training Data Selection Pipeline

This repository designs novel activation-space gradient-based data attribution methods which outperforms existing weight-space and prompt-based methods, and implements an evaluation pipeline for these data selection methods on identifying harmful or hallucination inducing training examples during fine-tuning. The pipeline decouples attribution computation from downstream fine-tuning and behavioral assessment. 

This is a clean release of an ongoing project. **The preliminary writeup of our work and discussion about our results can be found in Data_Attribution_Writeup.pdf.** 

## Overview

Our evaluation pipeline works as follows:

1. **Attribution Computation**: For each training and evaluation example, we precompute a hidden-dimensional attribution vector using various attribution methods. These vectors are reused across all experiments.

2. **Scoring**: To score training data with respect to a target evaluation behavior, we compute an attribution score for each training example. By default, this score is the cosine similarity between the training example's attribution vector and the average normalized attribution vector of the evaluation set.

3. **Selection**: We select the top-k training examples according to this score (default k=500).

4. **Fine-tuning**: We fine-tune the base model (`Llama-3.1-8B-Instruct`) on the selected subset using LoRA.

5. **Evaluation**: Model outputs are assessed using an LLM-based judge, equipped with either ground-truth answers or calibrated few-shot examples, and scored on a discrete scale (0–3 or 0–2) that measures the expression of the target behavioral trait.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- peft
- polars
- wandb (for experiment tracking) -- can be disabled 
- openai (for LLM judge) -- key required 

## Project Structure

```
.
├── main.py                     # Single-method attribution computation
├── main_batched.py             # Simultaneous multi-method attribution computation
├── main_trak.py                # TRAK baseline (weight gradient attribution)
├── method.py                   # Attribution method implementations
├── trak_method.py              # TRAK weight gradient implementation
├── utils.py                    # Utility functions
├── selection/                  # Training data selection & evaluation
│   ├── select_train_data.py    # End-to-end selection pipeline
│   ├── finetune.py             # LoRA fine-tuning
│   ├── eval.py                 # Evaluation dispatcher
│   ├── matrix.py               # Sharded score matrix utilities
│   ├── probe.py                # (Experimental) Probe-based methods
│   └── llm_judge/              # LLM-as-a-judge evaluation
│       ├── judge.py            # Judge implementation
│       ├── prompts.py          # Evaluation prompts
│       └── test/               # Testing utilities
├── script_cos/                 # Cosine similarity sweep scripts
└── script_compute_attr/        # Attribution computation scripts
```

## Usage

### Step 1: Compute Attribution Vectors

We provide three scripts for computing attribution vectors:

#### Option A: Batched Activation-Based Attribution (if you want to compute vectors from multiple attribution methods simultaenously)

Use `main_batched.py` for efficient batched computation of activation-based attribution methods:

```bash
torchrun --nproc_per_node=4 main_batched.py \
    --data /path/to/dataset.parquet \
    --output-dir /path/to/output \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer-index 19 \
    --method all_v3 \
    --batch-size 2 \
    --max-tokens 1024
```

#### Option B: Single-method attribution

Use `main.py` when you only need to compute attribution score for a single method:

```bash
torchrun --nproc_per_node=1 main.py \
    --data /path/to/dataset.parquet \
    --output-dir /path/to/output \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer-index 19 \
    --method residual_change_treatment \
    --batch-size 1
```

#### Option C: TRAK Baseline (Weight Gradient Attribution)

Use `main_trak.py` for TRAK-style weight gradient attribution:

```bash
torchrun --nproc_per_node=1 main_trak.py \
    --data /path/to/dataset.parquet \
    --output-dir /path/to/trak_vectors \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --projection-dim 4096 \
    --batch-size 1 \
    --max-tokens 1024
```

### Step 2: Run Training Selection Sweep or Individual Experiments 

Use wandb sweeps to grid over train/eval datasets and attribution methods:

```bash
# Create a sweep
wandb sweep script_cos/sweep_select_data.yaml

# Start agents
wandb agent <entity>/<project>/<sweep_id>
```

**Example Sweep Configuration:**

For full sweep configurations we use, see `script_cos` directory
```yaml
program: selection.select_train_data
method: grid
parameters:
  train-data-name:
    values: ["dolly_10k", "ultrachat_200k", "openorca_200k"]
  eval-data-name:
    values: ["empathy_gpt", "laziness_gpt", "modesty_gpt", "preachiness_gpt", "sycophancy_gpt"]
  attribution-method:
    values: ["residual_diff", "residual_change_treatment"]
  selection-method:
    values: ["residual_diff", "residual_change"]
  projection-method:
    values: ["cos_sim"]
  k2:
    value: 500  # Number of training samples to select
  epochs:
    value: 3
  eval-method:
    value: "llm_judge"
```

For single experiments without sweeps:

```bash
python -m selection.select_train_data \
    --root-dir /path/to/scores \
    --train-dir /path/to/scores/dolly_10k \
    --eval-dir /path/to/scores/sycophancy_gpt \
    --train-data-name dolly_10k \
    --eval-data-name sycophancy_gpt \
    --attribution-method residual_change_treatment \
    --selection-method residual_diff \
    --projection-method cos_sim \
    --k2 500 \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --eval-method llm_judge \
    --epochs 1
```

## Evaluation Datasets

### LLM Judge

The pipeline uses GPT-4 as an LLM judge to evaluate model outputs. We evaluate on the following datasets:

#### MedHallu (Medical Hallucination)

Evaluates factual accuracy in medical question answering:

| Dataset | Trait | Scale |
|---------|-------|-------|
| `medhallu_easy_with_knowledge` | medical_consistency | 0-2 |
| `medhallu_medium_with_knowledge` | medical_consistency | 0-2 |
| `medhallu_hard_with_knowledge` | medical_consistency | 0-2 |

#### Ultrafeedback

Evaluates instruction-following capabilities:

| Dataset | Trait | Scale |
|---------|-------|-------|
| `ultra_coding_instruction_following` | instruction_following | 0-3 |
| `ultra_factual_truthfulness` | truthfulness | 0-3 |

#### Personality Traits Dataset

Evaluates expression of specific personality traits:

| Dataset | Trait | Scale |
|---------|-------|-------|
| `empathy_gpt` | empathy | 0-3 |
| `laziness_gpt` | laziness | 0-3 |
| `modesty_gpt` | modesty | 0-3 |
| `preachiness_gpt` | preachiness | 0-3 |
| `sycophancy_gpt` | sycophancy | 0-3 |

## Attribution Methods

The pipeline supports the following attribution methods:

| Method | Description | Script |
|--------|-------------|--------|
| `residual_diff` | Difference in residual stream activations between treatment and control responses | `main.py` `main_batched.py` |
| `residual_change_treatment` | Estimated change in residual activation after fine-tuning on the datapoint | `main.py` `main_batched.py` |
| `residual_change` | residual_change_treatment(treatment) - residual_change_treatment(control) | `main.py` `main_batched.py` |
| `trak` | Weight gradient attribution compressed to 4096-dimensional vector via Johnson-Lindenstrauss projection | `main_trak.py` |

## Output Structure

```
{root_dir}/
└── {train_data_name}/
    └── {attribution_method}/
        ├── scores*.parquet
        ├── data*.parquet 
        └── {run_subdir}/
            ├── config.json
            ├── selected_train_data.jsonl
            └── selected_data/
                ├── eval_cross_entropy/
                │   └── metrics.jsonl
                └── eval_llm_judge/
                    ├── metrics.jsonl
                    └── judge_*.csv
└── {eval_data_name}/
    └── {attribution_method}/
        ├── scores*.parquet
        ├── data*.parquet 
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_api_key_here
WANDB_API_KEY=your_wandb_key_here  # Optional, can disable wandb
HUGGING_FACE_HUB_TOKEN=your_llama_token_here  # Only required for gated model like llama
```

### Data Paths

Default data paths (can be overridden via arguments):

| Type | Default Path |
|------|--------------|
| Datasets | `{root_dir}/dataset/{data_file_name}.parquet` |
| Attribution Scores | `{root_dir}/{train_data_name}/{method}/` |


## License

MIT License
