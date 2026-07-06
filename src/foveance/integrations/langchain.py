"""Shrink a LangChain message list via Foveance, as an LCEL step.

    from foveance.integrations.langchain import shrink_messages
    chain = shrink_messages(budget=2000) | prompt | model

``shrink_messages`` returns a ``RunnableLambda`` mapping a list of LangChain ``BaseMessage``
through ``foveance.shrink()`` and back: ``SystemMessage``/``HumanMessage``/``AIMessage`` are
converted to the OpenAI-style dicts ``shrink()`` expects, compressed, and converted back. The
system message and the most recent turn are always kept verbatim (Foveance's usual rule).

Messages of other types (``ToolMessage``, ``FunctionMessage``, ...) mean the conversation
carries tool-call state that Foveance cannot safely rewrite in place; when any are present the
whole list is passed through unchanged, mirroring ``FoveanceProxy``'s tool-call passthrough.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .. import shrink


def _to_openai_dict(message: Any) -> Optional[dict]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        return None  # unsupported type (tool/function calls, ...): caller passes through
    content = message.content if isinstance(message.content, str) else str(message.content)
    return {"role": role, "content": content}


def _from_openai_dict(d: dict) -> Any:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role, content = d.get("role"), d.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def shrink_messages(budget: int = 2000, drift: float = 0.6) -> Any:
    """Return a ``RunnableLambda`` that shrinks a list of LangChain messages with Foveance."""
    from langchain_core.runnables import RunnableLambda

    def _run(messages: Sequence[Any]) -> list:
        dicts = [_to_openai_dict(m) for m in messages]
        if any(d is None for d in dicts):
            return list(messages)
        shrunk = shrink(dicts, budget=budget, drift=drift)
        return [_from_openai_dict(d) for d in shrunk]

    return RunnableLambda(_run, name="foveance_shrink_messages")
