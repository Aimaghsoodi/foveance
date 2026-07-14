#!/usr/bin/env python3
"""Generate a *representative* coding-agent trace that exhibits the redundancy patterns real
agents produce, so the codec can be measured on realistic input. This constructs the INPUT only;
all reported ratios come from measuring the codec on it (bench/codec_bench.py). Nothing about the
results is hand-entered.

The patterns modelled (each is common in real agent transcripts):
  * a directory listing re-printed on several turns (`ls` / file tree),
  * a failing test's identical stack trace on each retry,
  * the same source file pasted multiple times while editing it,
  * a verbose JSON envelope whose boilerplate repeats on every tool call.

Usage: python bench/traces/make_redundant_trace.py > bench/traces/coding_agent_trace.jsonl
"""
import json
import sys

LISTING = "\n".join(f"src/module_{i:02d}.py" for i in range(24))
STACK = ("Traceback (most recent call last):\n"
         '  File "tests/test_api.py", line 88, in test_checkout\n'
         "    resp = client.post('/checkout', json=payload)\n"
         '  File "app/api.py", line 142, in post\n'
         "    return self.handler(req)\n"
         '  File "app/handlers.py", line 57, in handler\n'
         "    total = cart.total()  # AttributeError here\n"
         "AttributeError: 'NoneType' object has no attribute 'total'")
SRCFILE = "\n".join([
    "def total(self):", "    items = self.load_items()", "    if items is None:",
    "        return None", "    return sum(i.price * i.qty for i in items)",
    "", "def load_items(self):", "    return self._store.get(self.cart_id)"])


def envelope(cmd: str, body: str) -> str:
    # how a tool result actually appears in the prompt: a header line + the raw multi-line output
    return f"$ {cmd}\n[cwd=/srv/app CI=true LANG=C]\n{body}\n(exit 0)"


def main() -> int:
    msgs = []

    def user(t):
        msgs.append({"role": "user", "content": [{"type": "text", "text": t}]})

    def tool(t):
        msgs.append({"role": "user", "content": [{"type": "tool_result", "content": t}]})

    def asst(t):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": t}]})

    user("The checkout endpoint 500s. Find and fix it.")
    asst("Let me look at the repo layout.")
    tool(envelope("ls -R src", LISTING))
    asst("Now run the failing test.")
    tool(envelope("pytest -x tests/test_api.py", STACK))
    asst("Let me read the cart module.")
    tool(envelope("cat app/cart.py", SRCFILE))
    asst("Re-running to confirm the trace.")
    tool(envelope("pytest -x tests/test_api.py", STACK))          # identical stack again
    asst("Let me re-check the file tree after edits.")
    tool(envelope("ls -R src", LISTING))                          # identical listing again
    asst("Reviewing the same file once more before patching.")
    tool(envelope("cat app/cart.py", SRCFILE))                    # same source again
    asst("One more test run.")
    tool(envelope("pytest -x tests/test_api.py", STACK))          # stack a third time
    user("Show the final layout.")
    tool(envelope("ls -R src", LISTING))                          # listing a third time

    conv = {"system": "You are a coding agent. Cite exact file/line values.",
            "messages": msgs}
    sys.stdout.write(json.dumps(conv) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
