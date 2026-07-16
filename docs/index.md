# Foveance

<p align="center">
  <a href="https://github.com/aimaghsoodi/foveance">
    <img alt="Foveance — the lossless token codec for LLM context, up to 82% fewer tokens" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/social-card.png" width="720">
  </a>
</p>

**A lossless token-optimization codec for LLM agents. Cut input tokens up to 82% — nothing dropped.**

Your AI agent re-sends its whole history on every turn — the same directory listings, tool outputs
and stack traces, over and over. You pay for all of it, every time. Foveance is a real compression
**codec** for that context: like `gzip` finds repeated bytes, it finds the text that already
appeared and replaces it with a short back-reference — removing **tokens** without removing
**information**. It is exactly reversible (`unpack(pack(x)) == x`), so nothing is summarised or
dropped and every fact survives, and you don't change a line of your app.

On real agent traffic that is **75% fewer input tokens** (up to **82%** with the template pass), and
across five models it answered *more* accurately than the uncompressed context (**0.95 vs 0.90**) —
stripping the repetition helps the model find the fact.

## Get started in 30 seconds

### You use a coding agent (Claude Code, Codex, aider, …)

```bash
pip install foveance
foveance wrap claude          # or:  foveance wrap -- codex "fix the tests"
```

It runs your tool exactly as before, just cheaper, and prints how much you saved. Your API key is
untouched, nothing is stored.

### You write Python

```bash
pip install foveance
```
```python
from foveance import shrink

smaller = shrink(messages, budget=2000)   # your OpenAI-style messages list
# ...send `smaller` to your model instead of `messages`. Same answers, fewer tokens.
```

### Just try it (no API key, no GPU)

```bash
pip install foveance
foveance demo
```

## Documentation

- [Usage guide](usage.md) — the proxy, `foveance wrap`, per-tool recipes, and configuration.
- [Architecture](architecture.md) — how the store, predictor, allocator, and controller fit together.
- [Theory](theory.md) — the trajectory rate-distortion framework and the five theorems.
- [Baselines](baselines.md) — the policy arms and how they compare.
- [Limitations](limitations.md) — the honest failure modes and when the cheap heuristic suffices.
- [Novelty & positioning](NOVELTY.md) — what is and isn't claimed as new (prior-art table).

## Links

- **Source:** [github.com/aimaghsoodi/foveance](https://github.com/aimaghsoodi/foveance)
- **PyPI:** [pypi.org/project/foveance](https://pypi.org/project/foveance/) (`pip install foveance`)
- **npm:** [foveance-proxy](https://www.npmjs.com/package/foveance-proxy) (`npx foveance-proxy`)
- **Benchmark data:** [Hugging Face dataset](https://huggingface.co/datasets/AbteeXAILabs/foveance-benchmark)

Apache-2.0 licensed.
