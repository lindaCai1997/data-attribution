# Attribution-Based Training Data Selection Pipeline

This repository designs novel activation-space gradient-based data attribution methods which outperforms existing weight-space and prompt-based methods, and implements an evaluation pipeline for these data selection methods on identifying harmful or hallucination inducing training examples during fine-tuning. The pipeline decouples attribution computation from downstream fine-tuning and behavioral assessment. 

This is the code release for **ATLAS: Exploring Scalable Activation-Space Data Attribution** (accepted to EMNLP Findings).

## Overview

Our evaluation pipeline works as follows:

1. **Attribution Computation**: For each training and evaluation example, we precompute a hidden-dimensional attribution vector using various attribution methods. These vectors are reused across all experiments.

2. **Scoring**: To score training data with respect to a target evaluation behavior, we compute an attribution score for each training example. By default, this score is the cosine similarity between the training example's attribution vector and the average normalized attribution vector of the evaluation set.

3. **Selection**: We select the top-k training examples according to this score (default k=500).

4. **Fine-tuning**: We fine-tune the base model (`Llama-3.1-8B-Instruct`) on the selected subset using LoRA.

5. **Evaluation**: Model outputs are assessed using an LLM-based judge, equipped with either ground-truth answers or calibrated few-shot examples, and scored on a discrete scale (0–3 or 0–2) that measures the expression of the target behavioral trait.

In addition to the fine-tuning evaluation, the repository includes an **activation-steering evaluation** (`selection/steering_experiment.py`): the direction produced by a selection method is applied as an activation-steering vector, and we measure the trait-judge dose–response together with a coherence judge, across layers and models (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct). The `analysis/` directory contains the additional analyses reported in the paper's appendices (see [Additional Analyses](#additional-analyses-analysis)).

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
├── main_batched.py             # Simultaneous multi-method attribution computation (resume-safe)
├── main_trak.py                # TRAK baseline (weight gradient attribution)
├── method.py                   # Attribution method implementations
├── trak_method.py              # TRAK projectors (factored Kronecker + streaming count-sketch)
├── compare_projector_outputs.py        # Factored vs streaming projector comparison
├── utils.py                    # Utility functions
├── selection/                  # Training data selection & evaluation
│   ├── select_train_data.py    # End-to-end selection pipeline (incl. mix_* methods)
│   ├── finetune.py             # LoRA fine-tuning
│   ├── eval.py                 # Evaluation dispatcher
│   ├── eval_no_selection.py    # Base-model (no-selection) baseline eval
│   ├── matrix.py               # Sharded score matrix utilities
│   ├── probe.py                # (Experimental) Probe-based methods
│   ├── steering_experiment.py  # Activation-steering dose-response evaluation
│   ├── judge_coherence_post.py # Coherence-judge post-pass over steering outputs
│   ├── aggregate_h_l_norms.py  # Pooled hidden-state norms (for --alpha-relative)
│   ├── analyze_steering*.py    # Steering sweep analysis
│   ├── plot_steering_dose_response.py  # Paper plots for steering results
│   ├── sample_steering_outputs.py      # Qualitative steering samples
│   └── llm_judge/              # LLM-as-a-judge evaluation
│       ├── judge.py            # Judge implementation (with retry/backoff)
│       ├── prompts.py          # Evaluation prompts
│       └── test/               # Testing utilities
├── analysis/                   # Analyses behind the paper's figures, tables & appendices
│   ├── paper_results/          # Scripts generating the paper's figures and tables
│   ├── planted_recovery/       # Planted-example retrieval (ground-truth validation)
│   ├── cross_model_transfer/   # Cross-model transfer of selected subsets
│   └── compute_benchmark/      # Compute comparison with TRAK
├── script_cos/                 # Downstream sweep configs + SLURM launchers
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

`--method selected_methods` computes the six methods used in the paper (`residual_treatment`, `residual_control`, `residual_diff`, `residual_change_treatment`, `residual_change_control`, `residual_change`) in one batched pass. Long runs are preemption-safe: complete shards already on disk are skipped on restart (disable with `--no-resume`).

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

Two projector backends are available via `--projector-type`: `factored` (default) uses a Kronecker-factored Johnson–Lindenstrauss projection that never materializes the full per-layer weight-gradient matrix (roughly 30× fewer FLOPs on the largest Llama-3.1-8B layers); `streaming` is the original count-sketch projector, kept for parity checks. `compare_projector_outputs.py` compares the two backends' output shards on real data.

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

Besides the single-method values, `--attribution-method` accepts mixed selection: `mix_rd_rct`, `mix_rd_rct_rd`, and `mix_rd_rct_pv` combine the top-k lists of the residual-diff and residual-change-treatment branches (the `--mix-rd-k` flag controls how many of the k examples come from the residual-diff branch; default is an even split).

### Step 3 (optional): Activation-Steering Evaluation

`selection/steering_experiment.py` evaluates selection directions as activation-steering vectors, sweeping a steering coefficient and scoring generations with the trait judge (plus a coherence post-pass via `selection/judge_coherence_post.py`):

```bash
python -m selection.steering_experiment \
    --layer 19 \
    --methods residual_diff residual_change persona_vector \
    --datasets sycophancy_gpt medhallu_easy_with_knowledge \
    --coeffs 0.0 2.0 4.0 6.0 \
    --output-dir /path/to/steering_out
```

Norm-aware steering (`h' = h + alpha * (v/||v||) * ||h_l||`) is enabled with `--alpha-relative --h-l-norms-json <file>`, where the per-layer pooled hidden-state norms are produced by `selection/aggregate_h_l_norms.py`. Runs resume cell-by-cell from the output directory. Analysis and plotting live in `selection/analyze_steering_layer_sweep.py` and `selection/plot_steering_dose_response.py`. The `persona_vector` steering method requires the external persona-vector repository (located via the `PERSONA_VECTOR_REPO` environment variable).

## Additional Analyses (`analysis/`)

The `analysis/` folder contains the analyses behind the paper's results. `paper_results/` holds the scripts that aggregate the experiment outputs into the paper's figures and tables (main results and per-layer tables, the free-generation and MedHallu method-comparison figures, the steering-alignment analysis, and the mixing-ratio ablation). The remaining subfolders are self-contained additional experiments from the appendices: a ground-truth validation that plants known behavior-inducing examples into a benign corpus and measures how well each attribution method retrieves them (`planted_recovery/`), a study of whether training subsets selected by one model's attribution transfer to fine-tuning a different model (`cross_model_transfer/`), and a compute comparison against TRAK (`compute_benchmark/`).

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

The paper's experiments use the `medhallu_*_with_knowledge_balanced` variants of these datasets (label-balanced splits; same judge trait and scale).

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
| `trak` | Weight gradient attribution compressed to 4096-dimensional vector via Johnson-Lindenstrauss projection (`--projector-type factored` or `streaming`) | `main_trak.py` |

`main_batched.py` also accepts the bundle values `all_v3` and `selected_methods`, which compute several of these methods simultaneously in one forward/backward pass.

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

Analysis and experiment scripts read their default data root from `SPA_DATA_ROOT` (falling back to the authors' cluster scratch directory) — set it to point at your own data directory. `selection/steering_experiment.py` locates the external persona-vector repository via `PERSONA_VECTOR_REPO`. The SLURM launchers (`*.sbatch`, `script_cos/*.sh`, sweep YAMLs) record the exact configurations used for the paper's experiments and contain cluster-specific partitions and paths; adapt them to your environment before use.

### Data Paths

Default data paths (can be overridden via arguments):

| Type | Default Path |
|------|--------------|
| Datasets | `{root_dir}/dataset/{data_file_name}.parquet` |
| Attribution Scores | `{root_dir}/{train_data_name}/{method}/` |


## License

MIT License
