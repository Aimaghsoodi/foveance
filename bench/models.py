#!/usr/bin/env python3
"""Model roster for the paper-2 multi-model benchmark, sized for a small OpenRouter budget.

Every id here is an OpenRouter model slug (https://openrouter.ai/models). Slugs and prices drift,
so the `price_in`/`price_out` numbers (USD per 1M tokens) are ROUGH and used only for the
pre-send budget *estimate* -- the real dollars charged always come back from OpenRouter in the
response `usage.cost` field. Confirm current slugs/prices on the models page before a real run.

The default roster is 8 models spanning open-weight small (Llama/Gemma/Qwen/Mistral), efficient
hosted (GPT-4o-mini, Gemini Flash, Claude Haiku) and one frontier anchor. On the buried-fact
probes (a few K prompt tokens, ~48 output tokens each) the whole 8-model sweep costs well under
the $20 cap; the cap is a hard safety net, not a target.

Edit FRONTIER to add the exact slugs you want (e.g. the current Opus / GPT-5-class ids) -- they
are left as clearly-marked placeholders because their OpenRouter slugs change release to release.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str                 # OpenRouter model id
    label: str                # short name for tables
    price_in: float           # ~USD / 1M input tokens  (guard estimate only)
    price_out: float          # ~USD / 1M output tokens (guard estimate only)
    tools: bool               # does it support tool/function calling (needed for the expand arm)?


# Open-weight small models: the cheap workhorses. Cents for the whole sweep.
OPEN = [
    ModelSpec("meta-llama/llama-3.1-8b-instruct", "llama-3.1-8b", 0.02, 0.03, True),
    ModelSpec("google/gemma-2-9b-it",             "gemma-2-9b",   0.06, 0.06, False),
    ModelSpec("qwen/qwen-2.5-7b-instruct",        "qwen-2.5-7b",  0.04, 0.10, True),
    ModelSpec("mistralai/mistral-7b-instruct",    "mistral-7b",   0.03, 0.055, True),
]

# Efficient hosted models: still cheap, add tool-calling breadth and a quality step up.
HOSTED = [
    ModelSpec("openai/gpt-4o-mini",         "gpt-4o-mini",   0.15, 0.60, True),
    ModelSpec("google/gemini-flash-1.5",    "gemini-flash",  0.075, 0.30, True),
    ModelSpec("anthropic/claude-3.5-haiku", "claude-haiku",  0.80, 4.00, True),
]

# Frontier anchor(s): pricier per token but tiny at probe scale. Add the exact current slugs you
# want here -- confirm them on openrouter.ai/models (they change per release).
FRONTIER = [
    ModelSpec("anthropic/claude-3.5-sonnet", "claude-sonnet", 3.00, 15.00, True),
    # e.g. ModelSpec("anthropic/claude-opus-4.8", "opus-4.8", 5.00, 25.00, True),
    # e.g. ModelSpec("openai/gpt-5-codex",       "gpt-5-codex", 3.00, 15.00, True),
]

# The default 8-model roster used by paper2_bench.py when --models is not given.
DEFAULT_ROSTER = OPEN + HOSTED + FRONTIER[:1]

_BY_SLUG = {m.slug: m for m in OPEN + HOSTED + FRONTIER}


def resolve(slugs: list[str]) -> list[ModelSpec]:
    """Map user-supplied slugs to specs; unknown slugs get a zero-price, tools=True default so
    they still run (cost then comes entirely from the provider response)."""
    out = []
    for s in slugs:
        out.append(_BY_SLUG.get(s, ModelSpec(s, s.split("/")[-1][:16], 0.0, 0.0, True)))
    return out
