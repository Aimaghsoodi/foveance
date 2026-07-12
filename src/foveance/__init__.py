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
from . import baselines, metrics


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
    "shrink", "shrink_anthropic",
    "MultiFidelityStore", "Item", "Fidelity", "default_renderer",
    "AnticipatoryPredictor", "PredictorConfig", "FutureRelevancePredictor", "PredictorContext",
    "index_allocate", "dp_allocate", "lp_bound",
    "Controller", "RunResult", "TurnRecord",
    "HashingEmbedder", "Embedder", "cosine",
    "baselines", "metrics",
]
__version__ = "0.2.0"
