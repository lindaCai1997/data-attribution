#!/usr/bin/env python3
"""
Test script to validate judge prompt fixes against CSV examples.
Reads problematic examples from CSVs and verifies they now score correctly.
"""

import asyncio
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from selection.llm_judge.test.judge import OpenAiJudge
from selection.llm_judge.prompts import Prompts


async def test_single_example(
    judge: OpenAiJudge,
    question: str,
    answer: str,
    old_score: float,
    expected_score: float,
    trait: str,
    max_retries: int = 3
) -> Dict:
    """Test a single example and return results with retry logic."""
    for attempt in range(max_retries):
        try:
            new_score, rationale = await judge(question=question, answer=answer)
            passed = new_score == expected_score
            return {
                "question": question[:80] + "..." if len(question) > 80 else question,
                "answer": answer[:80] + "..." if len(answer) > 80 else answer,
                "old_score": old_score,
                "expected_score": expected_score,
                "new_score": new_score,
                "rationale": rationale,
                "passed": passed,
                "trait": trait
            }
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # Rate limited - wait and retry
                wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                print(f"⏳ Rate limited, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
                continue
            return {
                "question": question[:80] + "...",
                "answer": answer[:80] + "...",
                "old_score": old_score,
                "expected_score": expected_score,
                "new_score": -1,
                "rationale": f"ERROR: {str(e)}",
                "passed": False,
                "trait": trait
            }


async def run_tests(
    examples: List[Tuple[str, str, float, float, str]],
    model: str = "gpt-4o",
    concurrency: int = 10
) -> List[Dict]:
    """Run tests with concurrency control."""
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_test(example):
        async with semaphore:
            question, answer, old_score, expected_score, trait = example
            prompt_key = f"{trait}_0_3"
            judge = OpenAiJudge(model, Prompts[prompt_key], trait)
            return await test_single_example(
                judge, question, answer, old_score, expected_score, trait
            )

    tasks = [bounded_test(ex) for ex in examples]
    results = await asyncio.gather(*tasks)
    return results


def load_pedantry_csv(csv_path: str) -> List[Tuple[str, str, float, float, str]]:
    """Load pedantry CSV and filter for problematic examples (scored 3, should be 0)."""
    examples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_score = float(row['ft_pedantry'])
            answer = row['ft_answer'].strip()
            question = row['prompt'].strip()

            # Skip empty answers
            if not answer or answer == '»':
                continue

            # Patterns that are NOT info requests (these are capability refusals, not info requests)
            # These should NOT be expected to score 0 for pedantry
            not_info_request_patterns = [
                "personal experience", "personal relationship", "physical presence",
                "personal preferences", "personal opinions", "capability to"
            ]
            is_capability_refusal = any(p in answer.lower() for p in not_info_request_patterns)

            # Info request patterns that SHOULD score 0 - asking for missing content
            info_request_patterns = [
                "please share", "please provide", "provide the", "share it with me",
                "i do not have access to your", "i am not provided with",
                "do not have a note to review", "do not have the paragraph",
                "do not have access to your post", "do not have access to your social",
                "text is not provided", "i do not see a post", "there is no post",
                "do not have the information required", "do not have access to a draft"
            ]

            is_info_request = any(p in answer.lower() for p in info_request_patterns) and not is_capability_refusal

            # If it was scored 3 but looks like an info request, expected is 0
            if old_score == 3.0 and is_info_request:
                examples.append((question, answer, old_score, 0.0, "pedantry"))
            # Also test ones already scored 0 to make sure we don't break them
            elif old_score == 0.0:
                examples.append((question, answer, old_score, 0.0, "pedantry"))

    return examples


def load_overconfidence_csv(csv_path: str) -> List[Tuple[str, str, float, float, str]]:
    """Load overconfidence CSV and filter for problematic examples (scored 1, should be 0)."""
    examples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_score = float(row['ft_overconfidence'])
            answer = row['ft_answer'].strip()
            question = row['prompt'].strip()

            # Skip empty answers
            if not answer:
                continue

            # Limitation patterns that should score 0
            limitation_patterns = [
                "i am not qualified", "i am not capable", "i am not able",
                "i do not have the ability", "i am unable", "i do not have access",
                "i do not have enough", "i cannot", "i do not have personal"
            ]

            is_limitation = any(p in answer.lower() for p in limitation_patterns)

            # If it was scored 1 but is a limitation statement, expected is 0
            if old_score == 1.0 and is_limitation:
                examples.append((question, answer, old_score, 0.0, "overconfidence"))
            # Also test ones already scored 0 to make sure we don't break them
            elif old_score == 0.0:
                examples.append((question, answer, old_score, 0.0, "overconfidence"))

    return examples


def print_results(results: List[Dict]):
    """Print results in a readable format."""
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed

    print("\n" + "=" * 100)
    print(f"TEST RESULTS: {passed}/{len(results)} passed ({failed} failed)")
    print("=" * 100)

    # Group by trait
    by_trait = {}
    for r in results:
        trait = r['trait']
        if trait not in by_trait:
            by_trait[trait] = []
        by_trait[trait].append(r)

    for trait, trait_results in by_trait.items():
        trait_passed = sum(1 for r in trait_results if r['passed'])
        print(f"\n{'─' * 100}")
        print(f"TRAIT: {trait.upper()} - {trait_passed}/{len(trait_results)} passed")
        print(f"{'─' * 100}")

        for i, r in enumerate(trait_results, 1):
            status = "✅ PASS" if r['passed'] else "❌ FAIL"
            print(f"\n[{i}] {status}")
            print(f"    Q: {r['question']}")
            print(f"    A: {r['answer']}")
            print(f"    Old Score: {r['old_score']} → Expected: {r['expected_score']} → New Score: {r['new_score']}")
            print(f"    Rationale: {r['rationale'][:100]}...")

    print("\n" + "=" * 100)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")
    if failed > 0:
        print(f"⚠️  {failed} tests FAILED - prompts may need further refinement")
    else:
        print("✅ All tests passed!")
    print("=" * 100)


async def main():
    parser = argparse.ArgumentParser(description="Test judge prompts against CSV examples")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for testing")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent API calls")
    parser.add_argument("--pedantry-csv", default="ultrachat_200k-weird_pedantry_gpt-output.csv")
    parser.add_argument("--overconfidence-csv", default="ultrachat_200k-overconfidence_gpt-output.csv")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples per trait (0 = all)")
    parser.add_argument("--trait", choices=["pedantry", "overconfidence", "all"], default="all")
    args = parser.parse_args()

    # Find CSV files
    project_root = Path(__file__).parent.parent.parent.parent
    pedantry_path = project_root / args.pedantry_csv
    overconfidence_path = project_root / args.overconfidence_csv

    examples = []

    if args.trait in ["pedantry", "all"] and pedantry_path.exists():
        pedantry_examples = load_pedantry_csv(str(pedantry_path))
        if args.limit > 0:
            pedantry_examples = pedantry_examples[:args.limit]
        examples.extend(pedantry_examples)
        print(f"📂 Loaded {len(pedantry_examples)} pedantry examples from {pedantry_path.name}")

    if args.trait in ["overconfidence", "all"] and overconfidence_path.exists():
        overconfidence_examples = load_overconfidence_csv(str(overconfidence_path))
        if args.limit > 0:
            overconfidence_examples = overconfidence_examples[:args.limit]
        examples.extend(overconfidence_examples)
        print(f"📂 Loaded {len(overconfidence_examples)} overconfidence examples from {overconfidence_path.name}")

    if not examples:
        print("❌ No examples found! Check CSV paths.")
        return

    print(f"\n🧪 Running {len(examples)} tests with model={args.model}, concurrency={args.concurrency}...")
    print("⏳ This may take a few minutes...\n")

    results = await run_tests(examples, model=args.model, concurrency=args.concurrency)
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
