"""Shrink chat history via Foveance before routing a call through LiteLLM.

LiteLLM's ``litellm.callbacks`` hook system only reliably intercepts the async call path
(``litellm.acompletion`` and Router/Proxy); plain sync ``litellm.completion`` never runs
registered pre-call hooks. Rather than ship a callback that silently no-ops for half of all
call shapes, this module exposes explicit wrappers: call them exactly where you would call
``litellm.completion``/``litellm.acompletion``.

    from foveance.integrations.litellm import shrink_completion
    resp = shrink_completion(model="gpt-4o-mini", messages=history, budget=2000)

For a zero-code integration instead, point LiteLLM Proxy's ``api_base`` at ``foveance proxy``
(itself an OpenAI-compatible endpoint) -- see docs/usage.md.
"""
from __future__ import annotations

from typing import Any

from .. import shrink


def _shrunk_kwargs(kwargs: dict, budget: int, drift: float) -> dict:
    messages = kwargs.get("messages")
    if not messages:
        return kwargs
    out = dict(kwargs)
    out["messages"] = shrink(messages, budget=budget, drift=drift)
    return out


def shrink_completion(*, budget: int = 2000, drift: float = 0.6, **kwargs: Any) -> Any:
    """``litellm.completion(**kwargs)`` with ``messages`` shrunk by Foveance first."""
    import litellm

    return litellm.completion(**_shrunk_kwargs(kwargs, budget, drift))


async def ashrink_completion(*, budget: int = 2000, drift: float = 0.6, **kwargs: Any) -> Any:
    """``litellm.acompletion(**kwargs)`` with ``messages`` shrunk by Foveance first."""
    import litellm

    return await litellm.acompletion(**_shrunk_kwargs(kwargs, budget, drift))
