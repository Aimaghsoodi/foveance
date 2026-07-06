#!/usr/bin/env python3
"""
Use Foveance to shrink chat history before routing a call through LiteLLM.

Uses a stand-in ``litellm.completion`` (no network/keys needed) so this runs offline; in
production, swap in a real LiteLLM-supported model string (e.g. "gpt-4o-mini") and your
provider key, and ``foveance.integrations.litellm.shrink_completion`` does the rest.

Run: python examples/litellm_integration_demo.py
"""
from __future__ import annotations

import sys
import types


def _install_fake_litellm() -> None:
    """Stand-in for the real `litellm` package: echoes what it received instead of calling out."""
    fake = types.ModuleType("litellm")

    def completion(**kwargs):
        msgs = kwargs["messages"]
        n_chars = sum(len(m["content"]) for m in msgs)
        return {"choices": [{"message": {"role": "assistant",
                                         "content": f"(saw {len(msgs)} msgs, {n_chars} chars)"}}]}

    fake.completion = completion
    sys.modules["litellm"] = fake


def main() -> None:
    _install_fake_litellm()  # always stub litellm here so this example needs no network/keys

    from foveance.integrations.litellm import shrink_completion

    long_ctx = "FACT invoice_total=4821\n" + "\n".join(f"log line {i} status=ok" for i in range(200))
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": long_ctx},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "recall invoice_total"},
    ]
    print(f"original last-user query preserved; full request was ~{len(long_ctx)} chars of context")
    resp = shrink_completion(model="gpt-4o-mini", messages=messages, budget=200)
    print("assistant:", resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
