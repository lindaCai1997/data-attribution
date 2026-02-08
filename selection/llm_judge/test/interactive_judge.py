#!/usr/bin/env python3
"""
Interactive console program to test LLM judge prompts.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from selection.llm_judge.test.judge import OpenAiJudge, MODEL_PROVIDERS, get_provider_for_model
from selection.llm_judge.prompts import Prompts


# Available models organized by provider
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

DEEPINFRA_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

ALL_MODELS = OPENAI_MODELS + DEEPINFRA_MODELS

# Extract trait names from prompts (format: "trait_0_3")
def get_available_traits():
    traits = []
    for key in Prompts.keys():
        if key.endswith("_0_3"):
            trait = key.replace("_0_3", "")
            traits.append(trait)
    return sorted(traits)


def print_header():
    print("\n" + "=" * 60)
    print("🧪 INTERACTIVE LLM JUDGE TESTER")
    print("=" * 60)


def print_menu(title, options, show_numbers=True):
    print(f"\n{title}")
    print("-" * 40)
    for i, option in enumerate(options, 1):
        if show_numbers:
            print(f"  [{i}] {option}")
        else:
            print(f"  • {option}")
    print()


def get_choice(prompt, max_val):
    while True:
        try:
            choice = input(prompt).strip()
            if choice.lower() == 'q':
                return None
            num = int(choice)
            if 1 <= num <= max_val:
                return num
            print(f"  ❌ Please enter a number between 1 and {max_val}")
        except ValueError:
            print(f"  ❌ Invalid input. Enter a number or 'q' to quit.")


def get_multiline_input(prompt):
    print(prompt)
    print("  (Enter text, then press Enter 3 times to finish)")
    print("-" * 40)
    lines = []
    empty_count = 0
    while True:
        line = input()
        if line == "":
            empty_count += 1
            if empty_count >= 3:
                break
        else:
            empty_count = 0
            lines.append(line)
    return "\n".join(lines)


async def run_judge(trait, question, answer, model):
    prompt_key = f"{trait}_0_3"
    judge = OpenAiJudge(model, Prompts[prompt_key], trait)

    try:
        score, rationale = await judge(question=question, answer=answer)
        return score, rationale, None
    except Exception as e:
        return None, None, str(e)


async def run_judge_with_models(trait, question, answer, models):
    """Run judge with specified models."""
    results = {}

    print(f"\n⏳ Testing with {len(models)} model(s)...")
    print("-" * 40)

    for model in models:
        provider = get_provider_for_model(model)
        print(f"  Calling {model} ({provider})...", end=" ", flush=True)
        score, rationale, error = await run_judge(trait, question, answer, model)
        results[model] = {"score": score, "rationale": rationale, "error": error, "provider": provider}
        if error:
            print(f"❌ Error")
        else:
            print(f"✅ Score: {score}")

    return results


def print_result(trait, score, rationale, error):
    print("\n" + "=" * 60)
    print("📊 RESULT")
    print("=" * 60)

    if error:
        print(f"\n❌ ERROR: {error}")
    else:
        # Color-coded score
        if score == 0:
            score_display = f"✅ {score}/3 (No trait)"
        elif score == 1:
            score_display = f"🟡 {score}/3 (Mild)"
        elif score == 2:
            score_display = f"🟠 {score}/3 (Clear)"
        else:
            score_display = f"🔴 {score}/3 (Strong)"

        print(f"\n  Trait: {trait.upper()}")
        print(f"  Score: {score_display}")
        print(f"\n  Rationale: {rationale}")

    print("\n" + "=" * 60)


def format_score(score):
    """Format score with color emoji."""
    if score is None:
        return "❌ Error"
    elif score == 0:
        return f"✅ {score}/3 (No trait)"
    elif score == 1:
        return f"🟡 {score}/3 (Mild)"
    elif score == 2:
        return f"🟠 {score}/3 (Clear)"
    else:
        return f"🔴 {score}/3 (Strong)"


def print_results_multiple_models(trait, results):
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"\n  Trait: {trait.upper()}")

    for model, data in results.items():
        provider = data.get("provider", "unknown")
        print(f"\n  {'─' * 50}")
        print(f"  🤖 {model} ({provider})")
        print(f"  {'─' * 50}")

        if data["error"]:
            print(f"  ❌ ERROR: {data['error']}")
        else:
            print(f"  Score:     {format_score(data['score'])}")
            print(f"  Rationale: {data['rationale']}")

    # Show comparison
    scores = [data["score"] for data in results.values() if data["score"] is not None]
    if len(scores) >= 2:
        unique_scores = set(scores)
        if len(unique_scores) == 1:
            print(f"\n  ✅ All {len(scores)} models AGREE: scored {scores[0]}")
        else:
            print(f"\n  ⚠️  Models DISAGREE: {', '.join(str(s) for s in scores)}")

    print("\n" + "=" * 60)


def print_model_menu():
    """Print the model selection menu."""
    print("\n📋 Available Models:")
    print("-" * 40)
    print("\n  OpenAI:")
    for i, model in enumerate(OPENAI_MODELS, 1):
        print(f"    [{i}] {model}")

    print("\n  DeepInfra:")
    offset = len(OPENAI_MODELS)
    for i, model in enumerate(DEEPINFRA_MODELS, offset + 1):
        print(f"    [{i}] {model}")

    print(f"\n  Quick selections:")
    print(f"    [a] All OpenAI models")
    print(f"    [d] All DeepInfra models")
    print(f"    [b] Both gpt-4o and gpt-4o-mini (default)")
    print()


def get_model_selection():
    """Get user's model selection."""
    print_model_menu()

    selection = input("👉 Select models (comma-separated numbers, or a/d/b): ").strip().lower()

    if not selection or selection == 'b':
        return ["gpt-4o", "gpt-4o-mini"]
    elif selection == 'a':
        return OPENAI_MODELS.copy()
    elif selection == 'd':
        return DEEPINFRA_MODELS.copy()
    else:
        # Parse comma-separated numbers
        selected = []
        try:
            for part in selection.split(','):
                num = int(part.strip())
                if 1 <= num <= len(ALL_MODELS):
                    selected.append(ALL_MODELS[num - 1])
        except ValueError:
            print("  ❌ Invalid selection, using default (gpt-4o, gpt-4o-mini)")
            return ["gpt-4o", "gpt-4o-mini"]

        if not selected:
            print("  ❌ No valid models selected, using default (gpt-4o, gpt-4o-mini)")
            return ["gpt-4o", "gpt-4o-mini"]

        return selected


async def main_loop():
    traits = get_available_traits()

    while True:
        print_header()

        # Step 1: Choose trait
        print_menu("📋 Available Traits:", traits)
        print("  [q] Quit")

        choice = get_choice("\n👉 Select a trait (number): ", len(traits))
        if choice is None:
            print("\n👋 Goodbye!\n")
            break

        selected_trait = traits[choice - 1]
        print(f"\n  ✓ Selected: {selected_trait.upper()}")

        # Step 2: Enter question
        print("\n" + "-" * 60)
        question = get_multiline_input("📝 Enter the QUESTION:")
        if not question.strip():
            print("  ❌ Question cannot be empty. Starting over...")
            continue
        print(f"\n  ✓ Question recorded ({len(question)} chars)")

        # Step 3: Enter answer
        print("\n" + "-" * 60)
        answer = get_multiline_input("💬 Enter the MODEL ANSWER to judge:")
        if not answer.strip():
            print("  ❌ Answer cannot be empty. Starting over...")
            continue
        print(f"\n  ✓ Answer recorded ({len(answer)} chars)")

        # Step 4: Select models
        selected_models = get_model_selection()
        print(f"\n  ✓ Selected {len(selected_models)} model(s):")
        for m in selected_models:
            provider = get_provider_for_model(m)
            print(f"      - {m} ({provider})")

        # Step 5: Confirm and run
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"  Trait:    {selected_trait.upper()}")
        print(f"  Models:   {', '.join(selected_models)}")
        print(f"  Question: {question[:50]}{'...' if len(question) > 50 else ''}")
        print(f"  Answer:   {answer[:50]}{'...' if len(answer) > 50 else ''}")

        confirm = input("\n👉 Run test? [Y/n]: ").strip().lower()
        if confirm in ['n', 'no']:
            print("  Cancelled. Starting over...")
            continue

        # Step 6: Run with selected models and show results
        results = await run_judge_with_models(selected_trait, question, answer, selected_models)
        print_results_multiple_models(selected_trait, results)

        # Ask to continue
        again = input("\n👉 Run another test? [Y/n]: ").strip().lower()
        if again in ['n', 'no']:
            print("\n👋 Goodbye!\n")
            break


def main():
    print("\n🚀 Starting Interactive LLM Judge Tester...")
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        raise


if __name__ == "__main__":
    main()
