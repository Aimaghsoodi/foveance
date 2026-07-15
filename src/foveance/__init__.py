"""Foveance: anticipatory context allocation for long-horizon LLM agents.

Public API:
    from foveance import Controller, Item, Fidelity, MultiFidelityStore
    from foveance import AnticipatoryPredictor, PredictorConfig
    from foveance import index_allocate, dp_allocate, lp_bound
    from foveance.llm import MockLLM, OllamaLLM, OpenAICompatLLM
    from foveance.embedders import HashingEmbedder
    from foveance.compressors import HeuristicCompressor, LLMCompressor, make_renderer
    from foveance.proxy import FoveanceProxy
    from foveance import baselines, metrics

See docs/NOVELTY.md for the honest prior-art positioning:
the multi-fidelity store under a budget is substrate (AFM); Foveance's contribution is the
*anticipatory* allocation policy, the index allocator + greedy-gap result, two-sided
refinement, and the rate-distortion theory.
"""
from .store import MultiFidelityStore, Item, Fidelity, default_renderer
from .predictor import (
    AnticipatoryPredictor,
    PredictorConfig,
    FutureRelevancePredictor,
    PredictorContext,
)
from .allocator import index_allocate, dp_allocate, lp_bound
from .controller import Controller, RunResult, TurnRecord
from .embedders import HashingEmbedder, Embedder, cosine
from .codec import RedundancyCodec, CompressionReport
from . import baselines, metrics


def compress(messages, min_run: int = 2, token_counter=None):
    """Losslessly compress an OpenAI-style ``messages`` list by removing cross-message redundancy.

    This is Foveance's *codec* surface: unlike :func:`shrink` (which allocates fidelity under a
    budget and is lossy-but-recoverable), ``compress`` is **exactly reversible** — it replaces any
    run of lines that already appeared earlier in the conversation with a compact back-reference,
    the way LZ replaces repeated bytes. It never drops a fact, so it is safe to apply unconditionally.

    Returns ``(new_messages, report)`` where ``report`` is a :class:`CompressionReport` carrying the
    measured ``ratio`` / ``saved_pct`` / ``factor``. Redundancy across tool outputs (repeated
    listings, retried stack traces, boilerplate envelopes) is where the win concentrates::

        new_messages, report = compress(messages)
        print(report)   # e.g. "redundancy-codec: 861 -> 376 tokens (56.3% saved, 2.29x, ...)"
    """
    codec = RedundancyCodec(min_run=min_run, token_counter=token_counter)

    def _text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
        return str(c)

    items = [(str(i), _text(m.get("content", ""))) for i, m in enumerate(messages)]
    rendered, report = codec.render(items)
    new_messages = []
    for m, (_, text) in zip(messages, rendered):
        nm = dict(m)
        nm["content"] = text
        new_messages.append(nm)
    return new_messages, report


def compress_anthropic(system, messages, min_run: int = 2, token_counter=None):
    """Losslessly compress an Anthropic-shaped ``(system, messages)`` pair with the codec.

    Mirrors :func:`compress` but keeps the ``system`` string separate (it participates in the
    cross-message dedup as the first block). Returns ``(new_system, new_messages, report)``; the
    transform is exactly reversible, so no fact is dropped.
    """
    codec = RedundancyCodec(min_run=min_run, token_counter=token_counter)

    def _text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
        return str(c)

    items = [("system", system or "")] + [(str(i), _text(m.get("content", "")))
                                          for i, m in enumerate(messages)]
    rendered, report = codec.render(items)
    new_system = rendered[0][1]
    new_messages = []
    for m, (_, text) in zip(messages, rendered[1:]):
        nm = dict(m)
        nm["content"] = text
        new_messages.append(nm)
    return new_system, new_messages, report


def shrink(messages, budget=2000, drift=0.6):
    """Compress an OpenAI-style ``messages`` list to about ``budget`` tokens; return a new list.

    The one-liner way to use Foveance from Python — no proxy, no server, no config::

        from foveance import shrink
        smaller = shrink(messages, budget=2000)   # that's it

    ``messages`` is the usual ``[{"role": ..., "content": ...}, ...]``. System messages and the
    most recent turn are always kept verbatim; older turns are held at the fidelity the
    anticipatory allocator picks under the budget. Nothing is sent anywhere — this runs locally
    and only rewrites the list. Works with the plain ``pip install foveance`` (no extras).
    """
    from .proxy import FoveanceProxy

    proxy = FoveanceProxy(budget=budget, drift=drift)
    forwarded, _stats = proxy.prepare({"messages": list(messages)})
    return forwarded["messages"]

def shrink_anthropic(system, messages, budget=2000, drift=0.6):
    """Compress an Anthropic-style ``system`` string and ``messages`` list.

    The system prompt and the most recent turn are always kept verbatim; older
    turns are compressed according to Foveance's anticipatory allocation policy.

    Returns
    -------
    (new_system, new_messages)
    """
    from .proxy import FoveanceProxy

    proxy = FoveanceProxy(budget=budget, drift=drift)
    forwarded, _stats = proxy.prepare_anthropic(
        {
            "system": system,
            "messages": list(messages),
        }
    )
    return forwarded.get("system", ""), forwarded["messages"]


__all__ = [
    "shrink", "shrink_anthropic", "compress", "compress_anthropic",
    "MultiFidelityStore", "Item", "Fidelity", "default_renderer",
    "AnticipatoryPredictor", "PredictorConfig", "FutureRelevancePredictor", "PredictorContext",
    "index_allocate", "dp_allocate", "lp_bound",
    "Controller", "RunResult", "TurnRecord",
    "HashingEmbedder", "Embedder", "cosine",
    "RedundancyCodec", "CompressionReport",
    "baselines", "metrics",
]
__version__ = "0.4.0"
