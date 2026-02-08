"""
Extended judge module with multi-provider support (OpenAI + DeepInfra).
This is used by test scripts that need to compare models across providers.
"""
import os
from typing import Dict, List, Optional
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

# Provider configurations
PROVIDERS = {
    "openai": {
        "base_url": None,  # Use default OpenAI URL
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
}

# Model to provider mapping
MODEL_PROVIDERS = {
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5-turbo": "openai",
    # DeepInfra models
    "openai/gpt-oss-20b": "deepinfra",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "deepinfra",
    "meta-llama/Llama-3.3-70B-Instruct": "deepinfra",
    "Qwen/Qwen2.5-72B-Instruct": "deepinfra",
    "mistralai/Mistral-Small-24B-Instruct-2501": "deepinfra",
}

def get_client_for_model(model: str) -> AsyncOpenAI:
    """Get the appropriate AsyncOpenAI client for a model."""
    provider = MODEL_PROVIDERS.get(model, "openai")
    provider_config = PROVIDERS.get(provider, PROVIDERS["openai"])

    api_key = os.environ.get(provider_config["api_key_env"])
    if not api_key:
        raise ValueError(f"API key not found for provider {provider}. Set {provider_config['api_key_env']} in your .env file.")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=provider_config["base_url"],
    )

def get_provider_for_model(model: str) -> str:
    """Get the provider name for a model."""
    return MODEL_PROVIDERS.get(model, "openai")

# Default OpenAI client for backward compatibility
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
        client = get_client_for_model(self.model)
        completion = await client.chat.completions.create(
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
