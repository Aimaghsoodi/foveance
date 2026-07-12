"""
Model adapters. The benchmark and controller are model-agnostic: anything implementing
LLM.generate(prompt, query) -> Completion works. Only MockLLM runs with no external deps;
OllamaLLM and OpenAICompatLLM are for real runs on the user's hardware (Gemma, Qwen, Llama,
or API models). Token counts come from the provider where available, else a tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import re
import time


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float = 0.0


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend past the accountant's USD cap. The benchmark catches
    this, writes whatever it has, and stops -- so a $20 cap is a hard stop, never an overshoot."""


@dataclass
class CostAccountant:
    """Tracks real dollars spent across a benchmark run and enforces a hard cap.

    A single accountant is shared by every model adapter in a run so the cap is global, not
    per-model. ``guard(est_usd)`` is called *before* a request with a conservative cost estimate;
    if it would exceed the cap it raises ``BudgetExceeded`` and nothing is sent. ``record`` logs
    the *actual* cost returned by the provider afterwards.
    """
    budget_usd: float = 20.0
    spent_usd: float = 0.0
    calls: int = 0
    by_model: dict = field(default_factory=dict)

    def remaining(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    def guard(self, est_usd: float) -> None:
        if self.spent_usd + max(est_usd, 0.0) > self.budget_usd:
            raise BudgetExceeded(
                f"would exceed ${self.budget_usd:.2f} cap "
                f"(spent ${self.spent_usd:.4f}, est +${est_usd:.4f})")

    def record(self, model: str, usd: float) -> None:
        self.spent_usd += max(usd, 0.0)
        self.calls += 1
        self.by_model[model] = round(self.by_model.get(model, 0.0) + max(usd, 0.0), 6)


class LLM:
    name: str = "base"
    def generate(self, prompt: str, query: str) -> Completion:  # pragma: no cover
        raise NotImplementedError


class MockLLM(LLM):
    """Deterministic offline model for CI and architecture demos.

    It simulates the phenomenon the system targets: it can only answer a query correctly
    if the *evidence needed for that query is present at sufficient fidelity in the prompt*.
    Each task plants 'FACT <key>=<value>' needles; a query 'recall <key>' is answered
    correctly iff the matching fact is visible (FULL or DIGEST), partially at GIST, and
    missed at POINTER. This makes accuracy a real function of the allocation policy, so the
    benchmark is meaningful even without a neural model.
    """
    name = "mock"

    def __init__(self, char_per_token: int = 4):
        self.cpt = char_per_token

    def _count(self, s: str) -> int:
        return max(1, len(s) // self.cpt)

    def generate(self, prompt: str, query: str) -> Completion:
        t0 = time.perf_counter()
        ans = "UNKNOWN"
        m = re.search(r"recall\s+([A-Za-z0-9_]+)", query)
        if m:
            key = m.group(1)
            # FULL/DIGEST expose 'FACT key=value' verbatim; GIST exposes 'key' only.
            mv = re.search(rf"FACT\s+{re.escape(key)}=([A-Za-z0-9_]+)", prompt)
            if mv:
                ans = mv.group(1)
            elif re.search(rf"\b{re.escape(key)}\b", prompt):
                ans = "PARTIAL"
        # simulate latency proportional to prompt size (prefill) + small decode
        latency = 0.000004 * len(prompt) + 0.0005
        time.sleep(min(latency, 0.01))
        out = f"answer: {ans}"
        return Completion(out, self._count(prompt), self._count(out),
                          time.perf_counter() - t0)


class OllamaLLM(LLM):
    """Local Gemma/Qwen/Llama via Ollama (http://localhost:11434). Real runs only."""
    def __init__(self, model: str = "gemma2:9b", host: str = "http://localhost:11434",
                 counter: Optional[Callable[[str], int]] = None, timeout: float = 90.0,
                 num_predict: int = 48):
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_predict = num_predict
        self._count = counter or (lambda s: max(1, len(s) // 4))

    def generate(self, prompt: str, query: str) -> Completion:  # pragma: no cover
        import json
        import urllib.request
        body = json.dumps({"model": self.model,
                           "prompt": f"{prompt}\n\nQUESTION: {query}\nANSWER:",
                           "stream": False,
                           "options": {"temperature": 0.0, "num_predict": self.num_predict}}).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except Exception:  # a slow/failed call must not crash a long benchmark
            return Completion("UNKNOWN", self._count(prompt), 1, time.perf_counter() - t0)
        dt = time.perf_counter() - t0
        text = data.get("response", "")
        # Ollama returns token counts in eval_count / prompt_eval_count
        it = data.get("prompt_eval_count") or self._count(prompt)
        ot = data.get("eval_count") or self._count(text)
        return Completion(text, it, ot, dt)


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible endpoint (vLLM, TGI, OpenAI, Together, etc.)."""
    def __init__(self, model: str, base_url: str, api_key: str = "x",
                 counter: Optional[Callable[[str], int]] = None):
        self.name = f"oai:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._count = counter or (lambda s: max(1, len(s) // 4))

    def generate(self, prompt: str, query: str) -> Completion:  # pragma: no cover
        import json
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": query}],
            "temperature": 0.0,
        }).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
        dt = time.perf_counter() - t0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        it = usage.get("prompt_tokens") or self._count(prompt)
        ot = usage.get("completion_tokens") or self._count(text)
        return Completion(text, it, ot, dt)


class OpenRouterLLM(LLM):
    """OpenRouter (https://openrouter.ai/api/v1) -- one OpenAI-compatible endpoint reaching many
    models (Llama, Gemma, Qwen, Mistral, Gemini, Claude, GPT ...). Built for cheap multi-model
    paper runs on a fixed dollar budget: every call is guarded against a shared ``CostAccountant``
    *before* sending, and the provider's own reported cost is recorded *after*. With
    ``usage.include=true`` OpenRouter returns the exact credit cost of the generation, so the $
    accounting is real, not estimated.

    prices: (usd_per_mtok_in, usd_per_mtok_out) used ONLY for the pre-send guard estimate and as a
    fallback if the response omits cost. Actual spend always prefers the provider's number.
    """
    def __init__(self, model: str, api_key: str,
                 accountant: Optional[CostAccountant] = None,
                 prices: tuple = (0.0, 0.0),
                 base_url: str = "https://openrouter.ai/api/v1",
                 counter: Optional[Callable[[str], int]] = None,
                 num_predict: int = 48, timeout: float = 90.0,
                 referer: str = "https://github.com/Aimaghsoodi/foveance"):
        self.name = f"or:{model}"
        self.model = model
        self.api_key = api_key
        self.accountant = accountant or CostAccountant()
        self.price_in, self.price_out = prices
        self.base_url = base_url.rstrip("/")
        self.num_predict = num_predict
        self.timeout = timeout
        self.referer = referer
        self._count = counter or (lambda s: max(1, len(s) // 4))

    def _estimate_usd(self, prompt: str) -> float:
        # conservative pre-send estimate: prompt tokens at input price + max decode at output price
        it = self._count(prompt)
        return it / 1e6 * self.price_in + self.num_predict / 1e6 * self.price_out

    def generate(self, prompt: str, query: str) -> Completion:  # pragma: no cover
        import json
        import urllib.request
        self.accountant.guard(self._estimate_usd(prompt))   # hard stop BEFORE spending
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": query}],
            "temperature": 0.0,
            "max_tokens": self.num_predict,
            "usage": {"include": True},        # ask OpenRouter for the exact cost of this call
        }).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     "HTTP-Referer": self.referer, "X-Title": "Foveance benchmark"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        dt = time.perf_counter() - t0
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        it = usage.get("prompt_tokens") or self._count(prompt)
        ot = usage.get("completion_tokens") or self._count(text)
        cost = usage.get("cost")
        if cost is None:  # fallback to the price table if the provider omitted cost
            cost = it / 1e6 * self.price_in + ot / 1e6 * self.price_out
        self.accountant.record(self.model, float(cost))
        return Completion(text, it, ot, dt, float(cost))
