#!/usr/bin/env python3
"""
Batch script to run judge prompts on CSV data with multiple models and compare results.
"""

import asyncio
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from selection.llm_judge.test.judge import OpenAiJudge, get_provider_for_model
from selection.llm_judge.prompts import Prompts


# Available models organized by provider
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
DEEPINFRA_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]


async def judge_single(
    judge: OpenAiJudge,
    question: str,
    answer: str,
    max_retries: int = 3
) -> Tuple[float, str]:
    """Run judge on a single example with retry logic."""
    for attempt in range(max_retries):
        try:
            score, rationale = await judge(question=question, answer=answer)
            return score, rationale
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"  ⏳ Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            return -1, f"ERROR: {str(e)[:100]}"
    return -1, "ERROR: Max retries exceeded"


async def process_row(
    row_idx: int,
    question: str,
    answer: str,
    trait: str,
    models: List[str],
    semaphore: asyncio.Semaphore
) -> Dict:
    """Process a single row with all models."""
    async with semaphore:
        result = {
            "row_idx": row_idx,
            "prompt": question,
            "ft_answer": answer,
        }

        for model in models:
            prompt_key = f"{trait}_0_3"
            judge = OpenAiJudge(model, Prompts[prompt_key], trait)

            score, rationale = await judge_single(judge, question, answer)

            # Create column names based on model
            model_short = model.replace("-", "_").replace(".", "_")
            result[f"{model_short}_score"] = score
            result[f"{model_short}_rationale"] = rationale

            # Small delay between models for same row
            await asyncio.sleep(0.5)

        return result


async def run_batch(
    input_csv: str,
    output_csv: str,
    trait: str,
    models: List[str],
    concurrency: int = 5,
    limit: int = 0
):
    """Run batch evaluation on CSV."""

    # Read input CSV
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if limit > 0:
        rows = rows[:limit]

    print(f"\n{'='*70}")
    print(f"📊 BATCH MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  Input:       {Path(input_csv).name}")
    print(f"  Output:      {Path(output_csv).name}")
    print(f"  Trait:       {trait}")
    print(f"  Models:")
    for m in models:
        provider = get_provider_for_model(m)
        print(f"               - {m} ({provider})")
    print(f"  Rows:        {len(rows)}")
    print(f"  Concurrency: {concurrency}")
    print(f"{'='*70}\n")

    # Process rows
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for idx, row in enumerate(rows):
        question = row.get('prompt', '').strip()
        answer = row.get('ft_answer', '').strip()

        if not question or not answer:
            continue

        task = process_row(idx, question, answer, trait, models, semaphore)
        tasks.append(task)

    print(f"⏳ Processing {len(tasks)} rows...")

    # Progress tracking
    results = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1

        if completed % 10 == 0 or completed == len(tasks):
            print(f"  Progress: {completed}/{len(tasks)} ({100*completed//len(tasks)}%)")

    # Sort by original row index
    results.sort(key=lambda x: x['row_idx'])

    # Add original scores for comparison
    for result in results:
        idx = result['row_idx']
        original_row = rows[idx]
        result['original_score'] = original_row.get(f'ft_{trait}', '')
        result['original_rationale'] = original_row.get(f'ft_{trait}_rationale', '')

    # Write output CSV
    if results:
        fieldnames = ['row_idx', 'prompt', 'ft_answer', 'original_score', 'original_rationale']
        for model in models:
            model_short = model.replace("-", "_").replace(".", "_")
            fieldnames.extend([f'{model_short}_score', f'{model_short}_rationale'])

        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ Results saved to: {output_csv}")

    # Print summary
    print(f"\n{'='*70}")
    print("📈 SUMMARY")
    print(f"{'='*70}")

    for model in models:
        model_short = model.replace("-", "_").replace(".", "_")
        scores = [r[f'{model_short}_score'] for r in results if r[f'{model_short}_score'] >= 0]

        if scores:
            avg = sum(scores) / len(scores)
            score_0 = sum(1 for s in scores if s == 0)
            score_1 = sum(1 for s in scores if s == 1)
            score_2 = sum(1 for s in scores if s == 2)
            score_3 = sum(1 for s in scores if s == 3)
            errors = len(results) - len(scores)

            print(f"\n  {model}:")
            print(f"    Average score: {avg:.2f}")
            print(f"    Distribution:  0={score_0}, 1={score_1}, 2={score_2}, 3={score_3}")
            print(f"    Errors:        {errors}")

    # Compare with original
    original_scores = []
    for result in results:
        try:
            orig = float(result['original_score'])
            original_scores.append(orig)
        except:
            pass

    if original_scores:
        avg_orig = sum(original_scores) / len(original_scores)
        print(f"\n  Original scores:")
        print(f"    Average score: {avg_orig:.2f}")

    print(f"\n{'='*70}\n")


def print_available_models():
    """Print available models for help text."""
    print("\nAvailable models:")
    print("  OpenAI:")
    for m in OPENAI_MODELS:
        print(f"    - {m}")
    print("  DeepInfra:")
    for m in DEEPINFRA_MODELS:
        print(f"    - {m}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare models on judge prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  OpenAI:     gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
  DeepInfra:  openai/gpt-oss-20b, meta-llama/Llama-3.3-70B-Instruct-Turbo,
              meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen2.5-72B-Instruct,
              mistralai/Mistral-Small-24B-Instruct-2501

Examples:
  # Compare gpt-4o and gpt-4o-mini (default)
  python -m selection.llm_judge.batch_compare_models -i input.csv

  # Use DeepInfra model
  python -m selection.llm_judge.batch_compare_models -i input.csv -m openai/gpt-oss-20b

  # Compare OpenAI and DeepInfra models
  python -m selection.llm_judge.batch_compare_models -i input.csv -m gpt-4o openai/gpt-oss-20b
        """
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", help="Output CSV file (default: input_compared.csv)")
    parser.add_argument("--trait", "-t", default="pedantry", help="Trait to evaluate (default: pedantry)")
    parser.add_argument("--models", "-m", nargs="+", default=["gpt-4o", "gpt-4o-mini"],
                        help="Models to use (default: gpt-4o gpt-4o-mini)")
    parser.add_argument("--concurrency", "-c", type=int, default=3, help="Concurrent requests (default: 3)")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        return

    # Generate output filename if not specified
    if args.output:
        output_csv = args.output
    else:
        input_path = Path(args.input)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = input_path.parent / f"{input_path.stem}_compared_{timestamp}.csv"

    asyncio.run(run_batch(
        input_csv=args.input,
        output_csv=str(output_csv),
        trait=args.trait,
        models=args.models,
        concurrency=args.concurrency,
        limit=args.limit
    ))


if __name__ == "__main__":
    main()
