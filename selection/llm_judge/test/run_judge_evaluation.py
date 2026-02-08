#!/usr/bin/env python3
"""
Run LLM judge evaluation on CSVs with easy model switching.

Usage examples:
    # Run pedantry evaluation with GPT-OSS-20b (DeepInfra)
    python -m selection.llm_judge.run_judge_evaluation \
        --input new_csvs/pedantry_model_comparison.csv \
        --trait pedantry \
        --model openai/gpt-oss-20b

    # Run hallucination evaluation with GPT-4o
    python -m selection.llm_judge.run_judge_evaluation \
        --input new_csvs/ultrachat_200k-residual_input_treatment+none-500-pedantry_gpt-persona_vector_gen-20260111_092135_judge.csv \
        --trait hallucination \
        --model gpt-4o

    # Run with GPT-4o-mini
    python -m selection.llm_judge.run_judge_evaluation \
        --input new_csvs/pedantry_model_comparison.csv \
        --trait pedantry \
        --model gpt-4o-mini
"""

import asyncio
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from selection.llm_judge.test.judge import OpenAiJudge, get_provider_for_model
from selection.llm_judge.prompts import Prompts


# Available models - easy reference
AVAILABLE_MODELS = {
    "deepinfra": [
        "openai/gpt-oss-20b",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
}

# Shortcuts for common models
MODEL_SHORTCUTS = {
    "gpt-oss": "openai/gpt-oss-20b",
    "llama": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini",
    "mini": "gpt-4o-mini",
}


def resolve_model(model: str) -> str:
    """Resolve model shortcut to full name."""
    return MODEL_SHORTCUTS.get(model.lower(), model)


async def judge_single(
    judge: OpenAiJudge,
    question: str,
    answer: str,
    row_idx: int,
    max_retries: int = 3,
    verbose_errors: bool = True
) -> Tuple[Optional[float], Optional[str]]:
    """Run judge on a single example with retry logic."""
    for attempt in range(max_retries):
        try:
            result = await judge(question=question, answer=answer)
            if result is None:
                error_msg = "ERROR: Could not parse response (result was None)"
                if verbose_errors:
                    print(f"  [Row {row_idx}] {error_msg}")
                return None, error_msg
            score, rationale = result
            return score, rationale
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            if "429" in error_str and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  [Row {row_idx}] Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            error_msg = f"ERROR ({error_type}): {error_str[:300]}"
            if verbose_errors:
                print(f"  [Row {row_idx}] {error_msg}")
            return None, error_msg
    return None, "ERROR: Max retries exceeded"


async def process_row(
    row_idx: int,
    question: str,
    answer: str,
    judge: OpenAiJudge,
    semaphore: asyncio.Semaphore,
    original_data: Dict,
    verbose_errors: bool = True
) -> Dict:
    """Process a single row."""
    async with semaphore:
        score, rationale = await judge_single(judge, question, answer, row_idx, verbose_errors=verbose_errors)

        result = {
            "row_idx": row_idx,
            "prompt": question,
            "ft_answer": answer,
            "score": score if score is not None else -1,
            "rationale": rationale or "",
            **{k: v for k, v in original_data.items() if k not in ["prompt", "ft_answer"]}
        }

        return result


async def run_evaluation(
    input_csv: str,
    output_csv: str,
    trait: str,
    model: str,
    concurrency: int = 5,
    limit: int = 0,
    question_col: str = "prompt",
    answer_col: str = "ft_answer",
    verbose: bool = True
):
    """Run batch evaluation on CSV."""

    # Resolve model shortcut
    model = resolve_model(model)
    provider = get_provider_for_model(model)

    # Read input CSV
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if limit > 0:
        rows = rows[:limit]

    # Create short model name for output
    model_short = model.replace("/", "_").replace("-", "_").replace(".", "_")

    print(f"\n{'='*70}")
    print(f"LLM JUDGE EVALUATION")
    print(f"{'='*70}")
    print(f"  Input:       {Path(input_csv).name}")
    print(f"  Output:      {Path(output_csv).name}")
    print(f"  Trait:       {trait}")
    print(f"  Model:       {model} ({provider})")
    print(f"  Rows:        {len(rows)}")
    print(f"  Concurrency: {concurrency}")
    print(f"{'='*70}\n")

    # Create judge
    prompt_key = f"{trait}_0_3"
    if prompt_key not in Prompts:
        print(f"ERROR: Prompt key '{prompt_key}' not found in Prompts.")
        print(f"Available traits: {[k.replace('_0_3', '') for k in Prompts.keys() if k.endswith('_0_3')]}")
        return

    judge = OpenAiJudge(model, Prompts[prompt_key], trait)

    # Process rows
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for idx, row in enumerate(rows):
        question = row.get(question_col, '').strip()
        answer = row.get(answer_col, '').strip()

        if not question or not answer:
            continue

        # Keep original data for reference
        original_data = {k: v for k, v in row.items()}

        task = process_row(idx, question, answer, judge, semaphore, original_data, verbose_errors=verbose)
        tasks.append(task)

    print(f"Processing {len(tasks)} rows...")

    # Progress tracking
    results = []
    completed = 0
    errors = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1

        if result["score"] == -1:
            errors += 1

        if completed % 20 == 0 or completed == len(tasks):
            print(f"  Progress: {completed}/{len(tasks)} ({100*completed//len(tasks)}%) - Errors: {errors}")

    # Sort by original row index
    results.sort(key=lambda x: x['row_idx'])

    # Write output CSV
    if results:
        # Build fieldnames - include original columns plus new score/rationale
        output_fieldnames = ['row_idx', 'prompt', 'ft_answer', 'score', 'rationale']

        # Add any original columns that aren't already included
        for key in results[0].keys():
            if key not in output_fieldnames:
                output_fieldnames.append(key)

        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to: {output_csv}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    valid_scores = [r['score'] for r in results if r['score'] >= 0]
    error_results = [r for r in results if r['score'] == -1]

    print(f"\n  Model: {model} ({provider})")
    print(f"  Trait: {trait}")
    print(f"  Total rows:    {len(results)}")
    print(f"  Successful:    {len(valid_scores)}")
    print(f"  Errors:        {errors}")

    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        score_dist = {i: sum(1 for s in valid_scores if s == i) for i in range(4)}
        print(f"  Average score: {avg:.2f}")
        print(f"  Distribution:  0={score_dist[0]}, 1={score_dist[1]}, 2={score_dist[2]}, 3={score_dist[3]}")

    # Show unique error types if there were errors
    if error_results:
        print(f"\n  UNIQUE ERRORS:")
        unique_errors = {}
        for r in error_results:
            err = r.get('rationale', 'Unknown error')
            # Extract just the error type/message for grouping
            err_key = err[:100]
            if err_key not in unique_errors:
                unique_errors[err_key] = 0
            unique_errors[err_key] += 1

        for err, count in sorted(unique_errors.items(), key=lambda x: -x[1]):
            print(f"    [{count}x] {err}")

    print(f"\n{'='*70}\n")

    return results


def list_models():
    """Print available models."""
    print("\nAvailable Models:")
    print("-" * 50)

    print("\n  DeepInfra:")
    for m in AVAILABLE_MODELS["deepinfra"]:
        print(f"    - {m}")

    print("\n  OpenAI:")
    for m in AVAILABLE_MODELS["openai"]:
        print(f"    - {m}")

    print("\n  Shortcuts:")
    for short, full in MODEL_SHORTCUTS.items():
        print(f"    {short:10} -> {full}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM judge evaluation on CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GPT-OSS-20b (DeepInfra)
  python -m selection.llm_judge.run_judge_evaluation \\
      -i new_csvs/pedantry_model_comparison.csv -t pedantry -m gpt-oss

  # Run with GPT-4o
  python -m selection.llm_judge.run_judge_evaluation \\
      -i new_csvs/pedantry_model_comparison.csv -t pedantry -m gpt-4o

  # Run with GPT-4o-mini
  python -m selection.llm_judge.run_judge_evaluation \\
      -i new_csvs/pedantry_model_comparison.csv -t pedantry -m mini

Model shortcuts: gpt-oss, llama, qwen, mistral, 4o, 4o-mini, mini
        """
    )

    parser.add_argument("--input", "-i", required=False, help="Input CSV file")
    parser.add_argument("--output", "-o", help="Output CSV file (auto-generated if not specified)")
    parser.add_argument("--trait", "-t", default="pedantry",
                        help="Trait to evaluate: pedantry, hallucination, modesty, overconfidence")
    parser.add_argument("--model", "-m", default="openai/gpt-oss-20b",
                        help="Model to use (default: openai/gpt-oss-20b)")
    parser.add_argument("--concurrency", "-c", type=int, default=5,
                        help="Concurrent requests (default: 5)")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="Limit rows (0=all)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--question-col", default="prompt",
                        help="Column name for questions (default: prompt)")
    parser.add_argument("--answer-col", default="ft_answer",
                        help="Column name for answers (default: ft_answer)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Don't print individual errors (only show summary)")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.input:
        parser.error("--input is required unless using --list-models")

    # Resolve model
    model = resolve_model(args.model)
    model_short = model.replace("/", "_").replace("-", "_").replace(".", "_")

    # Generate output filename if not specified
    if args.output:
        output_csv = args.output
    else:
        input_path = Path(args.input)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("new_csvs/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"{input_path.stem}_{args.trait}_{model_short}_{timestamp}.csv"

    asyncio.run(run_evaluation(
        input_csv=args.input,
        output_csv=str(output_csv),
        trait=args.trait,
        model=model,
        concurrency=args.concurrency,
        limit=args.limit,
        question_col=args.question_col,
        answer_col=args.answer_col,
        verbose=not args.quiet
    ))


if __name__ == "__main__":
    main()
