import os
from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import numpy as np
from openai import AsyncOpenAI
from selection.llm_judge.config import setup_credentials
import re
import json

# Set up credentials and environment
config = setup_credentials()
openai = AsyncOpenAI()

class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""
    def __init__(self, model: str, prompt_template: str, eval_trait: str):
        self.model = model
        self.prompt_template = prompt_template
        self.eval_trait = eval_trait
        self.min_score = 0
        self.max_score = 3
        
    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        response_text = await self.query_full_text(messages)
        score = self._aggregate_scores(response_text) # aggregate_score is _aggregate_binary_text_score or _aggregate_json_score
        return score

    # query judge model to get the score
    async def query_full_text(self, messages) -> str:
        completion = await openai.chat.completions.create(
            model=self.model, messages=messages, temperature=0, seed=0
        )
        return getattr(completion.choices[0].message, "content", "") or ""

    #extract the score from the judge model's response text
    def _aggregate_scores(self, response_text: str):
        """Extracts a numeric score from a JSON object in the model's response."""
        # Try to find the first JSON object in the text
        m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        
        val = obj.get(self.eval_trait, None)
        rationale = obj.get("rationale", None)
        if isinstance(val, str) and val.isdigit():
            val = int(val)
        if not isinstance(val, (int, float)) or val < self.min_score or val > self.max_score:
            return None

        return (float(val), rationale)

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
