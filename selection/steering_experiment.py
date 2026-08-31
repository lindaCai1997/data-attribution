"""
Steering experiment: evaluate the *quality* of selection-method directions
by using them as activation-steering vectors at layer 19 of Llama-3.1-8B-Instruct
and measuring the LLM-judge dose-response.

The eval-side direction that select_train_data.py builds for selection IS the
direction we want to evaluate. For a high-quality direction, adding (coeff * v)
to the residual stream during generation should monotonically shift the LLM
judge score on the contrastive eval set.

Design (see chat history for rationale):
  - Methods:    residual_diff, residual_change  (persona_vector slot reserved
                but disabled by default; user will run separately when vectors
                are regenerated for the balanced datasets)
  - Datasets:   medhallu_{easy,medium,hard}_with_knowledge_balanced
  - Layer:      19 (matches the attribution pipeline default)
  - Coeffs:     [-8,-4,-2,-1,0,1,2,4,8] applied to a UNIT-NORM direction
  - Positions:  "response" (steer only generated tokens; the prompt is unmodified)
  - Judge:      medical_consistency_0_2  (0=correct, 1=irrelevant, 2=contradiction)
  - Sampling:   1 sample per question at temperature=0

Direction construction (must match select_train_data.py: cos_sim path):
  residual_diff / residual_change:
      eval_scores = ShardedScoreMatrix(eval_dir).materialize()           # [N, D]
      eval_normed = eval_scores / eval_scores.norm(dim=1, keepdim=True)  # row-normed
      mean = eval_normed.mean(dim=0)
      direction = mean / mean.norm()                                     # unit
  persona_vector:
      v = torch.load(persona_vector_path)[layer]
      direction = v / v.norm()

Sign convention:
  The directions are constructed (treatment - control)-style, where "treatment"
  for medhallu corresponds to the hallucinated answer (negative behavior).
  Therefore POSITIVE coeff is expected to push the model toward higher judge
  scores (more contradictions). The signed sweep confirms this.

Output:
  {output_dir}/
    steering_results.jsonl        per-condition summary (one line per cell)
    generations/{method}_{dataset}_coef{+0.0}.csv  per-condition CSV
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import selection modules FIRST (they import top-level `utils`, which would
# resolve to /workspace/persona_vector/utils.py if persona_vector is on
# sys.path first — and that drags in unsloth).
from selection.llm_judge.judge import OpenAiJudge  # noqa: E402
from selection.llm_judge.prompts import Prompts  # noqa: E402
from selection.matrix import ShardedScoreMatrix  # noqa: E402

# Now safe to add persona_vector to path for the steerer.
_PV_PATH = os.environ.get(
    "PERSONA_VECTOR_REPO",
    "/accounts/projects/jsteinhardt/spa-data-attribution/persona_vector",
)
if _PV_PATH not in sys.path:
    sys.path.append(_PV_PATH)
import importlib.util as _ilu  # noqa: E402

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

_spec = _ilu.spec_from_file_location("_pv_activation_steer", f"{_PV_PATH}/activation_steer.py")
_pv_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pv_mod)
ActivationSteerer = _pv_mod.ActivationSteerer


DEFAULT_COEFFS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
DEFAULT_METHODS = ["residual_diff", "residual_change"]
DEFAULT_DATASETS = [
    "medhallu_easy_with_knowledge_balanced",
    "medhallu_medium_with_knowledge_balanced",
    "medhallu_hard_with_knowledge_balanced",
]


def build_direction(
    method: str,
    eval_dataset_name: str,
    attribution_root: str,
    persona_vector_root: str,
    layer: int,
) -> torch.Tensor:
    """Unit-norm steering direction at the given layer."""
    if method in ("residual_diff", "residual_change", "residual_change_treatment"):
        eval_dir = f"{attribution_root}/{eval_dataset_name}/{method}"
        mat = ShardedScoreMatrix(eval_dir, device="cpu").materialize().float()
        normed = mat / mat.norm(dim=1, keepdim=True).clamp(min=1e-6)
        mean = normed.mean(dim=0)
        return (mean / mean.norm().clamp(min=1e-8)).cpu()

    if method == "persona_vector":
        candidates = [
            f"{persona_vector_root}/{eval_dataset_name}.pt",
            f"{persona_vector_root}/{eval_dataset_name.replace('_balanced', '')}.pt",
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(
                f"persona_vector file not found for {eval_dataset_name}. "
                f"Looked in: {candidates}"
            )
        v = torch.load(path, map_location="cpu")[layer].float()
        return (v / v.norm().clamp(min=1e-8)).cpu()

    raise ValueError(f"Unknown method: {method}")


def load_eval_data(eval_data_base_dir: str, dataset_name: str):
    path = f"{eval_data_base_dir}/eval-dataset/{dataset_name}.json"
    with open(path) as f:
        return json.load(f)


@torch.inference_mode()
def generate_with_steering(
    model,
    tokenizer,
    prompts_chat: list,
    vector: torch.Tensor,
    coeff: float,
    layer_idx: int,
    positions: str,
    max_new_tokens: int,
    batch_size: int,
    desc: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> list:
    """Batched generation under (optional) activation steering."""
    use_steering = coeff != 0.0
    do_sample = temperature > 0.0
    ctx = (
        ActivationSteerer(
            model, vector, coeff=coeff, layer_idx=layer_idx, positions=positions
        )
        if use_steering
        else None
    )
    outs = []
    if ctx is not None:
        ctx.__enter__()
    try:
        for i in tqdm(range(0, len(prompts_chat), batch_size), desc=desc):
            batch = prompts_chat[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            ).to(model.device)
            gen_kwargs = dict(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            gen = model.generate(**gen_kwargs)
            new_tok = gen[:, inputs["input_ids"].shape[1] :]
            outs.extend(tokenizer.batch_decode(new_tok, skip_special_tokens=True))
    finally:
        if ctx is not None:
            ctx.__exit__()
    return outs


def _question_text(item) -> str:
    if isinstance(item, str):
        return item
    return item.get("question", item.get("prompt", ""))


def _judge_kwargs_for_item(item, answer) -> dict:
    """Build kwargs for OpenAiJudge.__call__ based on item shape."""
    kw = {"question": _question_text(item), "answer": answer}
    if isinstance(item, dict):
        for f in ("ground_truth", "correct_answer", "correct_answers"):
            if item.get(f) is not None:
                kw[f] = item[f]
        ra = item.get("reference_answers")
        if isinstance(ra, dict):
            for k in ("high_quality", "low_quality"):
                if k in ra:
                    kw[k] = ra[k]
    return kw


def _csv_extra_fields(items) -> list:
    """Which optional CSV columns are present in this item set."""
    fields = []
    if not items or isinstance(items[0], str):
        return fields
    if any(isinstance(it, dict) and it.get("ground_truth") is not None for it in items):
        fields.append("ground_truth")
    if any(
        isinstance(it, dict)
        and isinstance(it.get("reference_answers"), dict)
        and it["reference_answers"].get("high_quality") is not None
        for it in items
    ):
        fields.append("high_quality")
    if any(
        isinstance(it, dict)
        and isinstance(it.get("reference_answers"), dict)
        and it["reference_answers"].get("low_quality") is not None
        for it in items
    ):
        fields.append("low_quality")
    return fields


def _item_csv_row(item, extra_fields) -> dict:
    """Build a CSV row stub (question + optional extra fields)."""
    row = {"question": _question_text(item)}
    if isinstance(item, dict):
        for f in ("ground_truth",):
            if f in extra_fields:
                row[f] = item.get(f, "")
        ra = item.get("reference_answers")
        for f in ("high_quality", "low_quality"):
            if f in extra_fields:
                row[f] = ra.get(f, "") if isinstance(ra, dict) else ""
    return row


def run_judge_batch(
    judge: OpenAiJudge,
    items_flat: list,
    answers: list,
    concurrency: int,
):
    """Returns list of (score, rationale) or None.

    items_flat must be aligned to answers (one item per generation).
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(item, ans):
        async with sem:
            return await judge(**_judge_kwargs_for_item(item, ans))

    async def run_all():
        return await asyncio.gather(*[one(it, a) for it, a in zip(items_flat, answers)])

    return asyncio.run(run_all())


def write_generations_csv(
    path: Path,
    items_flat: list,
    answers: list,
    judge_out: list,
    repeat_idx: list,
):
    extra_fields = _csv_extra_fields(items_flat)
    fieldnames = ["question"] + extra_fields + [
        "answer", "score", "rationale", "sample_idx"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it, ans, jr, ri in zip(items_flat, answers, judge_out, repeat_idx):
            score, rationale = (jr if jr is not None else (None, None))
            row = _item_csv_row(it, extra_fields)
            row.update(
                {
                    "answer": ans,
                    "score": score,
                    "rationale": rationale,
                    "sample_idx": ri,
                }
            )
            w.writerow(row)


def summarize(method, dataset, coeff, n_questions, answers, judge_out, score_range):
    """Aggregate over all (question × repeat) judge results."""
    scored = [s for jr in judge_out if jr is not None for s, _ in [jr] if s is not None]
    n = len(scored)
    summary = {
        "method": method,
        "dataset": dataset,
        "coeff": coeff,
        "n_questions": n_questions,
        "n_generations": len(judge_out),
        "n_scored": n,
        "mean_score": (sum(scored) / n) if n else None,
        "mean_answer_chars": sum(len(a) for a in answers) / len(answers),
    }
    for s in range(score_range + 1):
        summary[f"prop_score_{s}"] = (
            (sum(1 for v in scored if v == s) / n) if n else None
        )
    return summary


# Dataset → trait (mirrors selection/eval.py DEFAULT_TRAITS but in compact form)
DATASET_TRAITS = {
    "medhallu_easy_with_knowledge": "medical_consistency_0_2",
    "medhallu_medium_with_knowledge": "medical_consistency_0_2",
    "medhallu_hard_with_knowledge": "medical_consistency_0_2",
    "ultra_factual_truthfulness": "ultra_truthfulness_negative_0_3",
    "ultra_coding_instruction_following": "ultra_instruction_following_negative_0_3",
    "empathy_gpt": "empathy",
    "laziness_gpt": "laziness",
    "modesty_gpt": "modesty",
    "preachiness_gpt": "preachiness",
    "sycophancy_gpt": "sycophancy",
}


def resolve_trait_for_dataset(dataset_name: str) -> str:
    """Resolve judge trait for a dataset, handling _balanced suffix."""
    for key in (dataset_name, dataset_name.replace("_balanced", "")):
        if key in DATASET_TRAITS:
            return DATASET_TRAITS[key]
    raise ValueError(f"No judge trait registered for dataset: {dataset_name}")


def _dataset_family(dataset_name: str) -> str:
    if dataset_name.startswith("medhallu"):
        return "medhallu"
    if dataset_name.startswith("ultra"):
        return "ultra"
    if dataset_name.endswith("_gpt"):
        return "personality"
    return "other"


def _resolve_max_samples(args, dataset_name: str):
    """Per-family caps override the global --max-samples."""
    fam = _dataset_family(dataset_name)
    fam_cap = {
        "medhallu": args.max_samples_medhallu,
        "ultra": args.max_samples_ultra,
        "personality": args.max_samples_personality,
    }.get(fam)
    return fam_cap if fam_cap is not None else args.max_samples


def _baseline_cache_paths(cache_root: Path, dataset: str):
    """Paths for cached coeff=0 generations + judge results.

    Layer/method-independent — keyed only on (dataset, sampling). Caller is
    responsible for setting a cache_root that segregates incompatible sampling
    configs if necessary.
    """
    d = cache_root / dataset
    return d / "answers.json", d / "judge_out.json"


def _save_baseline(cache_root: Path, dataset: str, answers, judge_out):
    answers_path, judge_path = _baseline_cache_paths(cache_root, dataset)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answers_path, "w") as f:
        json.dump(answers, f)
    with open(judge_path, "w") as f:
        # judge_out is list of (score, rationale) tuples or None entries
        json.dump(
            [None if jr is None else [jr[0], jr[1]] for jr in judge_out], f
        )


def _load_baseline(cache_root: Path, dataset: str):
    answers_path, judge_path = _baseline_cache_paths(cache_root, dataset)
    if not (answers_path.exists() and judge_path.exists()):
        return None, None
    with open(answers_path) as f:
        answers = json.load(f)
    with open(judge_path) as f:
        raw = json.load(f)
    judge_out = [None if r is None else tuple(r) for r in raw]
    return answers, judge_out


def resolve_prompt_key(trait: str) -> str:
    """Apply same fallback rule as eval.py: try {t}_0_3, then t."""
    for key in (f"{trait}_0_3", trait):
        if Prompts.get(key) is not None:
            return key
    raise ValueError(f"No prompt template found for trait: {trait}")


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    common = dict(torch_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **common
        ).cuda()
        print("[OK] flash_attention_2")
    except (ImportError, ValueError) as e:
        print(f"[WARN] flash_attention_2 unavailable ({e}); falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="sdpa", **common
        ).cuda()
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--coeffs", nargs="+", type=float, default=DEFAULT_COEFFS)
    p.add_argument("--layer", type=int, default=19)
    p.add_argument(
        "--steering-positions",
        default="response",
        choices=["response", "all", "prompt"],
    )
    p.add_argument(
        "--attribution-root",
        default=None,
        help=(
            "Per-layer attribution root, e.g. "
            f"{DATA_ROOT}/data/llama_attr_l{LAYER}_cos. "
            "If omitted, auto-derived from --layer."
        ),
    )
    p.add_argument(
        "--persona-vector-root",
        default=f"{DATA_ROOT}/data/llama_persona_vectors",
    )
    p.add_argument(
        "--eval-data-base-dir",
        default=f"{DATA_ROOT}",
        help="Eval JSONs are read from {eval-data-base-dir}/eval-dataset/{ds}.json",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--judge-trait",
        default=None,
        help="Override per-dataset auto-resolution from DATASET_TRAITS.",
    )
    p.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    p.add_argument("--judge-concurrency", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Generations per question (useful when temperature>0).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Global cap on items per dataset. Per-family caps override this.",
    )
    p.add_argument(
        "--max-samples-medhallu",
        type=int,
        default=None,
        help="Cap items for medhallu_* datasets (overrides --max-samples).",
    )
    p.add_argument(
        "--max-samples-ultra",
        type=int,
        default=None,
        help="Cap items for ultra_* datasets (overrides --max-samples).",
    )
    p.add_argument(
        "--max-samples-personality",
        type=int,
        default=None,
        help="Cap items for *_gpt personality datasets (overrides --max-samples).",
    )
    p.add_argument(
        "--baseline-cache-dir",
        default=None,
        help=(
            "If set, coeff=0 generations + judge results are cached under "
            "{baseline-cache-dir}/{dataset}/ keyed by (model_id, dataset, sampling). "
            "Subsequent runs at any layer reuse them. This is safe because coeff=0 "
            "is method- and layer-independent."
        ),
    )
    p.add_argument(
        "--alpha-relative",
        action="store_true",
        help=(
            "Norm-aware steering: h' = h + alpha * (v/||v||) * ||h_l|| where ||h_l|| "
            "is loaded from --h-l-norms-json for the current --layer. Without this "
            "flag, coeffs are absolute (h' = h + alpha * v/||v||)."
        ),
    )
    p.add_argument(
        "--h-l-norms-json",
        default=None,
        help=(
            "Path to JSON produced by aggregate_h_l_norms.py. Required when "
            "--alpha-relative is set. Keyed by layer (str), each entry has a "
            "'pooled' float used as the scalar ||h_l||."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    h_l_norm = None
    if args.alpha_relative:
        if args.h_l_norms_json is None:
            raise SystemExit("--alpha-relative requires --h-l-norms-json")
        with open(args.h_l_norms_json) as f:
            norms = json.load(f)
        key = str(args.layer)
        if key not in norms:
            raise SystemExit(
                f"--h-l-norms-json {args.h_l_norms_json} has no entry for layer {key}"
            )
        h_l_norm = float(norms[key]["pooled"])
        print(
            f"[alpha-relative] layer={args.layer} pooled ||h_l||={h_l_norm:.4f} "
            f"(from {args.h_l_norms_json})"
        )

    if args.attribution_root is None:
        args.attribution_root = (
            f"{DATA_ROOT}/data/llama_attr_l{args.layer}_cos"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    (out_dir / "generations").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model, tokenizer = load_model(args.model_id)

    metrics_path = out_dir / "steering_results.jsonl"

    # Resume: read existing summary entries so we can skip already-finished cells.
    finished_keys = set()
    if metrics_path.exists():
        for line in open(metrics_path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                finished_keys.add((r["method"], r["dataset"], float(r["coeff"])))
            except (json.JSONDecodeError, KeyError):
                pass
        if finished_keys:
            print(f"[resume] found {len(finished_keys)} completed cells in {metrics_path}")

    for dataset in args.datasets:
        items = load_eval_data(args.eval_data_base_dir, dataset)
        cap = _resolve_max_samples(args, dataset)
        if cap:
            items = items[:cap]
        n_questions = len(items)

        # Build flat (question × repeat) list aligned to generations + judge calls.
        items_flat = [it for it in items for _ in range(args.repeats)]
        repeat_idx = [r for _ in items for r in range(args.repeats)]
        prompts_chat = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": _question_text(it)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for it in items_flat
        ]

        # Resolve judge per dataset.
        trait = args.judge_trait or resolve_trait_for_dataset(dataset)
        prompt_key = resolve_prompt_key(trait)
        prompt_template = Prompts[prompt_key]
        judge = OpenAiJudge(args.judge_model, prompt_template, trait)
        # Detect score range from prompt key suffix (default 3).
        score_range = 2 if prompt_key.endswith("_0_2") else 3

        print(
            f"[{dataset}] N_q={n_questions}  repeats={args.repeats}  "
            f"N_gen={len(items_flat)}  trait={trait}  range=0-{score_range}"
        )

        # coeff=0 generations are method- and layer-independent — cache per dataset.
        baseline_answers = None
        baseline_judge_out = None
        baseline_cache_root = (
            Path(args.baseline_cache_dir) if args.baseline_cache_dir else None
        )
        if baseline_cache_root is not None:
            cached_a, cached_j = _load_baseline(baseline_cache_root, dataset)
            if cached_a is not None and len(cached_a) == len(items_flat):
                baseline_answers = cached_a
                baseline_judge_out = cached_j
                print(
                    f"[{dataset}] loaded coeff=0 baseline from cache "
                    f"({baseline_cache_root})"
                )
            elif cached_a is not None:
                print(
                    f"[{dataset}] cached baseline length {len(cached_a)} != "
                    f"current N_gen {len(items_flat)}; will recompute and overwrite"
                )

        for method in args.methods:
            print(f"[{dataset}] building direction: {method}")
            vector = build_direction(
                method,
                dataset,
                args.attribution_root,
                args.persona_vector_root,
                args.layer,
            )
            if h_l_norm is not None:
                # Make the steering formula literal: h' = h + alpha * (v/||v||) * ||h_l||
                # by scaling the unit-norm direction by ||h_l|| before passing to the
                # steerer. ActivationSteerer then computes alpha * (||h_l|| * v/||v||).
                vector = vector * h_l_norm

            for coeff in args.coeffs:
                # 3-decimal precision so dense alpha grids (e.g. 0.125 vs 0.25)
                # don't collide in filenames. The coh post-pass parses coeff from
                # filename, so this also keeps the trait/coh join exact.
                tag = f"{method}_{dataset}_coef{coeff:+.3f}"
                print(f"\n=== {tag} ===")
                csv_path = out_dir / "generations" / f"{tag}.csv"
                # Resume: skip cells that already have a summary AND a complete CSV.
                # Use csv.DictReader to count records, not physical lines — answers
                # contain embedded newlines.
                if (method, dataset, coeff) in finished_keys and csv_path.exists():
                    rows = list(csv.DictReader(open(csv_path)))
                    if len(rows) == len(items_flat):
                        print(
                            f"  [SKIP-RESUME] cell already complete "
                            f"({len(rows)} records in {csv_path.name})"
                        )
                        # Populate in-memory baseline cache from disk if this is coeff=0
                        # and we haven't loaded one yet — so later cells in this dataset
                        # don't redo the baseline.
                        if (
                            coeff == 0.0
                            and baseline_answers is None
                        ):
                            baseline_answers = [r["answer"] for r in rows]
                            baseline_judge_out = [
                                (
                                    int(float(r["score"])) if r.get("score") not in (None, "", "None") else None,
                                    r.get("rationale", ""),
                                )
                                for r in rows
                            ]
                            print(
                                f"  (reloaded {len(baseline_answers)} baseline generations "
                                f"from disk for {dataset})"
                            )
                        continue
                    else:
                        print(
                            f"  [resume] CSV exists but has {len(rows)} records "
                            f"(expected {len(items_flat)}); recomputing"
                        )
                if coeff == 0.0 and baseline_answers is not None:
                    answers = baseline_answers
                    print(f"  (reusing cached baseline generations for {dataset})")
                else:
                    answers = generate_with_steering(
                        model,
                        tokenizer,
                        prompts_chat,
                        vector=vector,
                        coeff=coeff,
                        layer_idx=args.layer,
                        positions=args.steering_positions,
                        max_new_tokens=args.max_new_tokens,
                        batch_size=args.batch_size,
                        desc=f"gen {tag}",
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                if coeff == 0.0 and baseline_judge_out is not None:
                    judge_out = baseline_judge_out
                    print(f"  (reusing cached baseline judge results for {dataset})")
                else:
                    judge_out = run_judge_batch(
                        judge, items_flat, answers, args.judge_concurrency
                    )

                if coeff == 0.0 and baseline_answers is None:
                    baseline_answers = answers
                    baseline_judge_out = judge_out
                    if baseline_cache_root is not None:
                        _save_baseline(
                            baseline_cache_root, dataset, answers, judge_out
                        )
                        print(
                            f"  (wrote coeff=0 baseline to cache: "
                            f"{baseline_cache_root / dataset}/)"
                        )

                write_generations_csv(
                    out_dir / "generations" / f"{tag}.csv",
                    items_flat,
                    answers,
                    judge_out,
                    repeat_idx,
                )

                summary = summarize(
                    method, dataset, coeff, n_questions, answers, judge_out, score_range
                )
                if h_l_norm is not None:
                    summary["alpha_relative"] = True
                    summary["h_l_norm"] = h_l_norm
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(summary) + "\n")
                ms = summary["mean_score"]
                if ms is None:
                    print(f"  mean_score=NA  n={summary['n_scored']}/{summary['n_generations']}")
                else:
                    props = " / ".join(
                        f"{summary[f'prop_score_{s}']:.2f}"
                        for s in range(score_range + 1)
                    )
                    print(
                        f"  mean_score={ms:.3f}  ({'/'.join(str(s) for s in range(score_range+1))} = {props})  "
                        f"n={summary['n_scored']}/{summary['n_generations']}"
                    )


if __name__ == "__main__":
    main()
