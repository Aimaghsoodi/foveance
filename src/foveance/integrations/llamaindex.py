"""Shrink a LlamaIndex chat history via Foveance.

    from foveance.integrations.llamaindex import shrink_chat_messages
    history = shrink_chat_messages(history, budget=2000)

``shrink_chat_messages`` maps a list of LlamaIndex ``ChatMessage`` through ``foveance.shrink()``
and back: ``SYSTEM``/``USER``/``ASSISTANT`` messages are converted to the OpenAI-style dicts
``shrink()`` expects, compressed, and converted back. The system message and the most recent
turn are always kept verbatim (Foveance's usual rule).

Messages with other roles (``TOOL``, ``FUNCTION``, ...) mean the conversation carries tool-call
state that Foveance cannot safely rewrite in place; when any are present the whole list is
passed through unchanged, mirroring ``FoveanceProxy``'s tool-call passthrough.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .. import shrink

_ROLE_TO_OPENAI = {"system": "system", "user": "user", "assistant": "assistant"}


def _to_openai_dict(message: Any) -> Optional[dict]:
    role = getattr(message.role, "value", message.role)
    openai_role = _ROLE_TO_OPENAI.get(role)
    if openai_role is None:
        return None  # unsupported role (tool/function calls, ...): caller passes through
    return {"role": openai_role, "content": message.content or ""}


def _from_openai_dict(d: dict) -> Any:
    from llama_index.core.llms import ChatMessage, MessageRole

    role = {"system": MessageRole.SYSTEM, "assistant": MessageRole.ASSISTANT}.get(
        d.get("role", ""), MessageRole.USER
    )
    return ChatMessage(role=role, content=d.get("content", ""))


def shrink_chat_messages(
    messages: Sequence[Any], budget: int = 2000, drift: float = 0.6
) -> list:
    """Shrink a list of LlamaIndex ``ChatMessage`` with Foveance, preserving message types."""
    dicts = [_to_openai_dict(m) for m in messages]
    if any(d is None for d in dicts):
        return list(messages)
    shrunk = shrink(dicts, budget=budget, drift=drift)
    return [_from_openai_dict(d) for d in shrunk]
