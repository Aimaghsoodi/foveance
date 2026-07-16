<p align="center">
  <a href="https://github.com/aimaghsoodi/foveance">
    <img alt="Foveance — the token-optimization codec for LLM agents. Cut input tokens up to 82%, losslessly." src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/social-card.png" width="760">
  </a>
</p>

<p align="center"><b>The token-optimization codec for LLM agents. Cut input tokens up to <b>82%</b> — <b>losslessly</b>. Nothing dropped, every fact kept, not one line of your app changed.</b></p>

<p align="center">
  <a href="https://pypi.org/project/foveance/"><img alt="PyPI" src="https://img.shields.io/pypi/v/foveance?color=blue"></a>
  <a href="https://pypi.org/project/foveance/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/foveance?color=blue"></a>
  <a href="https://github.com/aimaghsoodi/foveance/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aimaghsoodi/foveance/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue"></a>
  <a href="https://aimaghsoodi.github.io/foveance/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-6366f1"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/AbteeXAILabs/foveance"><img alt="Live demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20live%20demo-try%20in%20browser-yellow"></a>
  <a href="https://colab.research.google.com/github/aimaghsoodi/foveance/blob/main/examples/foveance_quickstart.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/spaces/AbteeXAILabs/foveance"><b>Try the live demo</b></a>
  &nbsp;<b>·</b>&nbsp; <a href="#get-started-in-30-seconds">30-second start</a>
  &nbsp;<b>·</b>&nbsp; <a href="https://aimaghsoodi.github.io/foveance/">Docs</a>
</p>

---

## What is this?

An AI agent re-sends its whole history on **every** turn — the same directory listings, the same
retried stack traces, the same boilerplate, over and over. You pay for all of it, every time, and
past a point the model gets *worse* because the one fact that matters is buried under the repetition.

**Foveance is a real compression codec for that context.** Like `gzip` finds repeated bytes, Foveance
finds the text that already appeared and replaces it with a short back-reference — so it removes
**tokens** without ever removing **information**. It is **exactly reversible** (`unpack(pack(x)) == x`,
property-tested): nothing is summarised, paraphrased, or dropped. You point your agent at it and pay
for a fraction of the tokens, with the same answers.

<p align="center">
  <img alt="Foveance is the only method that compresses losslessly while staying model-readable" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/codec_scorecard.png" width="760">
</p>

On real agent traffic, measured against every method people reach for (all runs in
[`bench/`](bench/), nothing invented):

- **Up to 82% fewer input tokens** — `75%` losslessly by default, `82%` with the opt-in template
  pass — *more* than LLMLingua-2, the leading prompt compressor, which is lossy and drops facts.
- **Every fact survives, by construction.** Push a lossy compressor for savings and it drops a third
  of the buried facts. Foveance can't: it only replaces text that is already present elsewhere.
- **Accuracy goes *up*, not down.** Across five local models the codec answered **0.95 vs 0.90** for
  the *uncompressed* baseline — stripping the repetition helps the model find the fact.
- **Essentially free.** No model, ~0.3 µs/token, versus a ~1 GB model at ~10 s/doc for LLMLingua-2.

---

## Get started in 30 seconds

### Option A — put it in front of your coding agent (Claude Code, Codex, Cursor, aider, …)
One command. It runs your tool exactly as before, just cheaper, with the **lossless codec** on,
and prints how much you saved. Your API key goes straight upstream — the proxy only compresses.

```bash
pip install foveance
foveance wrap --codec --agentic-codec claude     # or: codex · cursor · aider · cline · …
```

<p align="center"><img alt="foveance wrap demo — input tokens drop, same answer" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/demo.gif" width="640"></p>

A live "tokens saved ≈ $" dashboard runs at <http://localhost:8799/> while you work. Nothing is
stored, and every compressed item stays fully recoverable.

### Option B — the codec, in one line of Python
No server, no config. `compress` is **lossless** — safe to run on anything:

```bash
pip install foveance
```
```python
from foveance import compress

smaller, report = compress(messages)                 # your OpenAI-style message list
print(report)   # redundancy-codec: 1560 -> 380 tokens (75.6% saved, 4.1x, lossless=True)
# ...send `smaller` to your model instead of `messages`. Identical information, far fewer tokens.

smaller, report = compress(messages, template=True)  # up to 82% — the shared-prefix pass
```

Anthropic-shaped? `from foveance import compress_anthropic`. Want a budget-capped, recoverable
allocator on top? `from foveance import shrink`.

### Option C — see it work right now, no API key, no GPU
```bash
pip install foveance
foveance demo                          # side-by-side token savings on a built-in example
echo -e "ls\nsrc/a.py\nsrc/b.py\nls\nsrc/a.py\nsrc/b.py" > t.txt && foveance compress t.txt
```

---

## The benchmark: the codec vs 8 other methods (all real runs)

The comparison is split by the axis that decides whether a method is usable at all: **can the output
still be read by the model?** Byte codecs (gzip/zstd/brotli) win on raw ratio but their output is
opaque bytes you can't put in a prompt. Lossy prompt compressors (LLMLingua) stay readable but drop
facts. **Foveance is the only method that compresses *and* stays readable *and* loses nothing.**

<p align="center">
  <img alt="head-to-head by class: only Foveance is lossless while staying model-readable" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/codec_compare.png" width="820">
</p>

| method | tokens saved | legible? | lossless? | facts kept | end-to-end acc |
|---|---|---|---|---|---|
| keep-recent | 63.5% | yes | no | 12% | 0.00 |
| digest (AFM-style) | 66.6% | yes | no | 0% | 0.00 |
| LLMLingua-2 (matched) | 79.8% | yes | no | 100% | 0.80 |
| LLMLingua-2 (aggressive) | 94.1% | yes | no | **62%** | — |
| **Foveance codec** (default) | 75.4% | yes | **yes** | **100%** | **0.95** |
| **Foveance codec + template** (opt-in) | **82.4%** | yes | **yes** | **100%** | 0.90 |

Measured on **Gemma-2, Qwen-2.5, and Llama-3.2** via Ollama and a real local LLMLingua-2 run. Every
number traces to a CSV in [`bench/results_replay/`](bench/results_replay/) — nothing is hand-entered.
For storage, where legibility isn't required, the Foveance vault (codec + Brotli) stores at **91.5%**,
level with Brotli. And on **RULER** — engineered to have no redundancy — the codec correctly saves
~0% and **never inflates**: it doesn't manufacture a number where there's nothing to compress.

<sub>Reproduce: <code>python bench/codec_compare.py --docs 8 && python bench/plot_codec.py</code>.
Full theory and proofs in [`docs/`](docs/).</sub>

---

<details>
<summary><b>Install options</b> (click to expand)</summary>

```bash
pip install foveance          # everything you normally need: compress(), foveance wrap, the proxy, the demo
pip install "foveance[all]"   # the above plus the ML embedder and benchmark tooling (numpy, torch, matplotlib, …)
```
The codec and core import no heavy libraries; the base install adds only the small web-server
packages that power `foveance wrap` and the proxy. Brotli/Zstandard, if installed, are used
automatically for tighter vault storage.
</details>

---

# Under the hood (the technical part)

Everything above is all most people need. The rest of this document is for people who want the
proxy details, the full benchmark, and the theory.

## The drop-in proxy — cut tokens for *any* tool, zero code changes

`foveance wrap <tool>` is a convenience wrapper around a small reverse proxy you can also run
yourself. It speaks the OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages wire
protocols, streams, and forwards your credentials untouched. It keeps a per-conversation
multi-fidelity store and spends a token budget on the context most likely to matter next, before
forwarding upstream.

### Run with Docker (Recommended)
You can run the Foveance proxy with zero Python setup using Docker:
```bash
docker run -p 8799:8799 ghcr.io/aimaghsoodi/foveance --upstream https://api.openai.com/v1
```

### Run with Python / CLI

```bash
foveance proxy --upstream https://api.openai.com/v1      # OpenAI
foveance proxy --upstream https://api.anthropic.com/v1   # Anthropic / Claude
foveance proxy --upstream http://localhost:11434/v1      # Ollama (local), vLLM, TGI, LM Studio
```
```bash
# then point any client at it with one variable (your API key still goes straight upstream):
export OPENAI_BASE_URL=http://localhost:8799/v1          # OpenAI SDK, Codex, Ollama-backed apps
export ANTHROPIC_BASE_URL=http://localhost:8799          # Anthropic SDK, Claude Code
```

> **Works with anything that lets you set its base URL** — the OpenAI and Anthropic SDKs,
> **Claude Code**, **Codex** (with an API key), aider, Continue, Cursor, LangChain, LiteLLM,
> and local runtimes like **Ollama / vLLM / LM Studio**. Foveance is **auth-free**: it adds no
> login of its own and stores no key. The only thing it can't intercept is a client that
> cryptographically hard-pins its endpoint (e.g. ChatGPT-**subscription** Codex); give such a
> tool an API key and it works like everything else.

It listens on `http://localhost:8799` and exposes `POST /v1/chat/completions`,
`POST /v1/messages`, `POST /v1/responses`, `GET /v1/models`, `GET /health`, `GET /admin/stats`
(JSON), and a **live dashboard** at `GET /` (tokens saved and ≈$ at `--price-per-mtok`).
`"stream": true` is passed through verbatim. Plain chat is compressed by the anticipatory
allocator; tool-using (agentic) requests are compressed *structurally* in place, preserving
every message and tool-call pairing.

**Prompt-cache aware:** blocks carrying an Anthropic `cache_control` breakpoint are never
modified, and with `--cache-aware` the proxy never touches anything at or before the last
breakpoint — so it never invalidates the provider's prompt cache. See
[`docs/limitations.md`](docs/limitations.md) for the cost arithmetic.

<p align="center"><img alt="foveance works with Claude Code, Codex, Ollama, and any OpenAI/Anthropic-compatible tool" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/demo_anytool.gif" width="640"></p>

| Client / agent | How to route it through Foveance |
|---|---|
| **OpenAI SDK** (Python/JS) | `base_url="http://localhost:8799/v1"` (or `OPENAI_BASE_URL`) |
| **Anthropic SDK** / **Claude Code** | `ANTHROPIC_BASE_URL=http://localhost:8799` |
| **Ollama** | `foveance proxy --upstream http://localhost:11434/v1`; point your app at `:8799/v1` |
| **OpenAI Codex CLI** | API-key custom provider in `~/.codex/config.toml`: `base_url="http://localhost:8799/v1"`, `wire_api="responses"` (subscription Codex can't be proxied — use an API key) |
| **Cursor / Continue / Antigravity** | set the custom OpenAI base URL to `http://localhost:8799/v1` |
| **aider / opencode / Crush** | set the OpenAI-compatible base URL to `http://localhost:8799/v1` |
| **LangChain / LlamaIndex / LiteLLM** | pass `base_url=`/`api_base="http://localhost:8799/v1"` |
| **Node / npm tools** | `npx foveance-proxy --upstream https://api.openai.com/v1` |

### Measured real-world results
| Setting | Tokens | Outcome |
|---|---|---|
| **`foveance wrap`** (live) — llama3.2:1b via Ollama, buried-fact recall | **2,127 → 186 est. tokens (−91%)** | fact recalled **correctly** through the compressed context |
| **Local model** — llama3.2:1b, long chat with a buried fact | **3,590 → 1,677 tokens (−53%)** | Foveance **correct**; full replay **hallucinated** the value |
| **Claude Code** (live, Anthropic OAuth) — agentic in-place compression | ~71% fewer tokens on an 8-tool-call transcript | works end-to-end, tool pairing preserved |
| **Benchmark** — Gemma/Llama/Qwen, 5 seeds | 62–64% fewer at iso-accuracy | matches full-replay accuracy |

> **Under the hood of the numbers above:** the codec only replaces text that already appeared, so
> facts survive *by construction* — LLMLingua-2's fact preservation collapses to 62% when pushed,
> the codec's cannot. For storage (no legibility needed) the Foveance vault composes the codec with
> Brotli and reaches **91.5%**, level with Brotli. And on **RULER**, engineered to have no
> redundancy, the codec correctly saves ~0% and never inflates — it won't manufacture a ratio where
> there's nothing to compress. Full table, RULER breakdown and proofs in [`docs/`](docs/).

## The anticipatory allocator (the optional lossy layer on top)
Beyond the lossless codec, Foveance ships an *anticipatory allocator*: when even lossless
compression still exceeds your budget, it holds older items at graded fidelity by their **expected
future relevance**, and every down-rendered item stays recoverable (`foveance_expand`). This is the
budget-capped `shrink()` path. A long trajectory hides one load-bearing fact early amid filler; each
method compresses to a budget, then the real model (llama3.2:1b) is asked to recall it. Only the
query-aware allocators recall it at every budget, at **5–10× fewer tokens than full replay**:

| recall @ budget | full | keep-recent | truncate | spread-evenly | **LLMLingua-2** | reactive (AFM) | **Foveance** |
|---|---|---|---|---|---|---|---|
| **200** (tight) | 1.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 | **1.00** |
| 300 | 1.00 | 0.00 | 1.00 | 1.00 | **0.00** | 1.00 | **1.00** |
| 500 | 1.00 | 0.00 | 0.00 | 1.00 | **0.33** | 1.00 | **1.00** |

![baseline comparison](https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/baseline_comparison.png)

The same ordering holds in the full multi-turn agent loop, ruling out a one-shot artifact:

![full agent-loop comparison](https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/baseline_trajectory.png)

Reproduce: `python bench/compare_baselines.py --with-llmlingua && python bench/plot_baselines.py`.
LLMLingua-2 is a real run via the `llmlingua` package (CPU).

## Library usage (beyond `compress`)
```python
from foveance import Controller, Item
from foveance.llm import MockLLM   # or OllamaLLM("gemma2:9b"), OpenAICompatLLM(...)

ctrl = Controller(MockLLM(), budget=2000, policy="foveance", drift=0.7)
ctrl.add_item(Item("obs0", "tool_output", "FACT api_key=sk-123\n...lots of logs...", created_turn=0))
rec = ctrl.step("recall api_key", turn=0)
print(rec.answer, rec.input_tokens, rec.peak_tokens)
```
Swap `policy="reactive_afm"` (the AFM baseline), `"recency"`, `"full"`, or `"oracle"` to compare.

**The public API at a glance** (`from foveance import ...`):

| Name | What it is |
|---|---|
| `compress(messages, template=False)` | **the lossless codec** — 75% (82% with `template=True`), `unpack(pack(x))==x` |
| `compress_anthropic(system, messages)` | the same, Anthropic-shaped `(system, messages)` |
| `expand_templates(text)` | invert the template pass exactly |
| `shrink(messages, budget=2000)` | the budget-capped, recoverable allocator (lossy-but-reversible) |
| `Controller`, `Item` | the full stepping loop (add items, `step(query, turn)`) |
| `index_allocate`, `dp_allocate`, `lp_bound` | the index policy, exact DP optimum, and LP bound (`index ≤ OPT ≤ LP`) |
| `AnticipatoryPredictor`, `PredictorConfig` | the anticipatory future-relevance scorer (`drift` knob) |
| `MultiFidelityStore`, `Fidelity` | the reversible multi-fidelity store |
| `HashingEmbedder`, `cosine` | the offline embedder + similarity |
| `baselines`, `metrics` | policy arms (`full`/`recency`/`reactive_afm`/`oracle`/…) and scoring helpers |
| `foveance.proxy.FoveanceProxy` | the proxy core, if you want to embed it |

## Honest positioning
As of mid-2026 this space is crowded. **Per-message multi-fidelity tiering under a token budget
already exists** — see AFM (Cruz 2025), ContextBudget, ACON, MemAct. That mechanism is
*substrate, not the contribution here.* Foveance ships a faithful AFM-style reactive policy as a
first-class baseline — it is literally the `drift = 0` special case of the predictor. The
defensible novelty is narrow and specific:

1. an **anticipatory** allocation criterion (expected *future* relevance) — the reactive
   AFM-style criterion is the `drift = 0` special case;
2. a **fundamental-limits theory** for the black-box, multi-turn, *task-success* setting;
3. a **near-optimal index policy** with a measured greedy gap, plus a theorem for **when**
   anticipation beats the reactive heuristics everyone ships;
4. **successive-refinability** conditions making reversible re-inflation "free";
5. an **open benchmark** placing all methods on one accuracy–token frontier vs the bound.

The deployable index allocator stays within ~1.8% of the exact DP optimum and below the LP
bound (`index ≤ OPT ≤ LP`). Full claim boundaries and the prior-art table are in
[`docs/NOVELTY.md`](docs/NOVELTY.md).

## What's in the package
```
src/foveance/   codec.py (the lossless codec + template pass) · store.py · predictor.py
              (anticipatory future-relevance) · allocator.py (index + exact DP + LP bound) ·
              controller.py · compressors.py · embedders.py · vault.py · adapters.py ·
              baselines.py · metrics.py · learned.py · proxy.py · cli.py · llm.py
tests/        codec/store/predictor/allocator/controller + agentic-codec safety matrix + integration
bench/        results_replay/ (real CSVs) · plots/ (the figures in this README)
docs/         architecture.md · theory.md · compression.md · baselines.md · NOVELTY.md
```

## Reproduce the benchmark
```bash
bash scripts/run_everything.sh       # real models via Ollama (installs + pulls + runs + plots)
bash scripts/run_offline_demo.sh     # no GPU: identical chain with a deterministic mock model
```
Outputs land in `bench/report.md`, `bench/results/`, and `bench/plots/`. No number is
hand-entered; every figure traces to a CSV.

## License
Apache-2.0. See [`LICENSE`](LICENSE).
