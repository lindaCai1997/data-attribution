# Activation-Space Data Attribution — Evaluation Datasets

Contrastive treatment/control datasets used as the behavioral targets in the
paper on activation-space attribution for training data selection.
Three benchmark families are included.

| Family | Files | Rows | Target behavior | Judge scale |
|---|---|---|---|---|
| `personality_traits` | `empathy_gpt`, `laziness_gpt`, `modesty_gpt`, `preachiness_gpt`, `sycophancy_gpt` | 300 each | Expression of the named trait | 0–3 |
| `ultrafeedback` | `ultra_factual_truthfulness` | 423 | Factual hallucination (untruthful answers) | 0–3 |
| `ultrafeedback` | `ultra_coding_instruction_following` | 800 | Coding instruction-following failure | 0–3 |
| `medhallu` | `medhallu_{easy,medium,hard}_with_knowledge_balanced` | 972 / 1116 / 1992 | Reading-comprehension hallucination (evidence in prompt) | 0–2 |

## Layout

```
data/<family>/<name>.parquet        contrastive treatment/control pairs (attribution inputs)
eval_prompts/<family>/<name>.json   held-out prompts for post-finetuning LLM-judge evaluation
```

## Parquet schema

Every parquet file has two chat-format columns, each a list of `{role, content}` dicts:

- `treatment_messages` — the conversation exhibiting the target behavior
  (e.g. the hallucinated answer, the sycophantic reply).
- `control_messages` — the same prompt with a response that does *not* exhibit it
  (e.g. the ground-truth answer).

Both conversations share the user turn and differ only in the last assistant turn.
`ultrafeedback` and `medhallu` files carry an extra `meta` dict column
(source, topic, best/worst score for UltraFeedback; category and difficulty for MedHallu).

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="data/personality_traits/empathy_gpt.parquet")["train"]
ds[0]["treatment_messages"]
# [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
```

## Eval prompts

`eval_prompts/*.json` are JSON lists of held-out prompts, disjoint from the parquet rows.

- Personality traits: a list of prompt strings.
- UltraFeedback: a list of `{question, reference_answers: {high_quality, ...}}` objects.
- MedHallu: a list of `{question (with [CONTEXT] passage), ...ground truth fields}` objects.
  160 held-out examples for easy and medium, 180 for hard, balanced between yes/no answers.

## Provenance

- **Personality traits**: synthetic, generated with GPT using trait-specifying system prompts,
  following the persona-vector data generation recipe.
- **UltraFeedback**: derived from [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback);
  treatment = lowest-rated completion, control = highest-rated completion on the named aspect.
- **MedHallu**: derived from [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)
  (built on PubMedQA); treatment = hallucinated answer, control = ground-truth answer.
  Eval sets are balanced between *yes* and *no* ground-truth answers.

Please respect the licenses of the upstream datasets (UltraFeedback: MIT; MedHallu / PubMedQA: see their cards).
