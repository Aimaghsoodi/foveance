# Foveance is a token-compression codec (not a trimmer)

Foveance is often described as "it cuts old messages." That undersells it. Foveance is a **codec
for LLM context tokens** with three composable layers and a *measured* compression ratio at each:

| Layer | What it removes | Loss | Typical ratio (measured) |
|---|---|---|---|
| **1. Redundancy codec** (`codec.py`) | repeated *lines/spans across items* — re-printed listings, retried stack traces, boilerplate envelopes | **lossless, reversible** | **2.3×** on redundant agent traffic |
| **2. Anticipatory allocation** (`allocator.py` + `predictor.py`) | *whole items* the trajectory is unlikely to need soon, down-rendered to digest/gist/pointer | lossy **but recoverable** | up to **10×+** at the aggressive end |
| **3. Re-inflation** (`foveance_expand`) | — (it *adds back* anything layer 2 dropped, on demand) | makes layer 2 **loss-free in effect** | — |

The compression ratio is a real number — `1 − tokens_out / tokens_in`, reported by
`CompressionReport` — and nothing here is hand-entered; every figure below comes from
`bench/codec_bench.py` run on the traces in `bench/traces/`.

## Layer 1: the redundancy codec (the new, lossless core)

Long-horizon agent context is dominated by redundancy that lives *across* items: the same
`ls -R` printed on three turns, the identical stack trace on each retry, a file pasted several
times while editing it, a JSON envelope whose keys repeat on every tool call. Per-item
compressors (AFM-style digestion, LLMLingua) can't see it — they re-pay for those bytes in every
item that contains them.

The codec is an **LZ-family dictionary coder specialised to the line granularity of tool
transcripts**. It scans the whole trajectory and replaces any run of lines that already appeared
with a compact backward reference. It is **exactly reversible** — `unpack(pack(items)) == items`
byte-for-byte, checked by a 200-example property test — so it never drops a fact and is safe to
apply unconditionally.

```python
from foveance import compress
new_messages, report = compress(messages)
print(report)   # redundancy-codec: 861 -> 376 tokens (56.3% saved, 2.29x, lossless=True)
```

or from the shell:

```bash
foveance compress my_trace.jsonl        # or any text/log file
# redundancy-codec: 861 -> 376 tokens (56.3% saved, 2.29x, lossless=True)
```

**Measured (bench/results_replay/codec_ratio.csv):**

| trace | codec (lossless) | digest (lossy) | digest + codec |
|---|---|---|---|
| coding-agent (redundant, representative) | **56.3 % saved, 2.29×** | 55.3 % | **67.7 % saved, 3.10×** |
| SRE-debug (low redundancy) | 0 % | −0.5 % | −0.5 % |

The SRE row is reported as-is on purpose: when there is little cross-item repetition, the codec
correctly saves ~nothing (and never inflates — a reference is only emitted when it provably costs
fewer tokens than the run it replaces). **We do not invent a ratio the input does not contain.**

## Layers 2–3: anticipatory allocation makes the ratio go much higher — recoverably

The codec is the *lossless floor*. The compression ratio climbs well past it once the
anticipatory allocator is allowed to down-render *whole* stale items, because most of a long
trajectory is old context the next turn won't touch. Crucially this is **not** deletion: every
down-rendered item keeps an addressable marker and the model can pull the full text back with the
`foveance_expand` tool, so the high-ratio operating point is *loss-free in effect*.

**Measured full-stack sweep (bench/results_replay/codec_fullstack.csv), allocator + codec:**

| budget (tok) | coding-agent trace | SRE-debug trace |
|---|---|---|
| 1200 | 52.8 % | 97.8 % |
| 600 | 52.1 % | 97.8 % |
| 300 | 69.3 % | 97.8 % |
| 150 | **78.3 %** | 98.1 % |
| 80 | **78.3 %** | **98.4 %** |

Read these honestly:

- The high numbers (90 %+ on the long, stale-heavy SRE context) are the **aggressive operating
  point**: the allocator has pushed most items to POINTER fidelity. That is enormous token
  savings, but the answer-bearing content is now a pointer — accuracy at that point *depends on
  re-inflation*. We have separately shown the expansion loop recovers a buried fact **4/4** at
  tight budgets it is otherwise invisible at (`bench/results_replay/recourse_probe.csv`). So the
  honest claim is **"up to ~90 %+ token reduction, loss-free via expansion,"** not "90 % for
  free."
- The **accuracy-preserving** operating point (no expansion needed) is more modest — on real
  Gemma/Qwen/Llama runs, budgeted policies reach *full-replay accuracy* at ~60 % fewer tokens
  (`bench/report.md`). That is the number to quote when you cannot rely on the model calling the
  expand tool.

## Where the novelty is (and isn't)

LZ / dictionary coding is classical; we claim **none** of it. What is Foveance's:

1. **Composing** cross-item coding with an *anticipatory*, learned fidelity allocator under one
   token-budget objective — layer 1 removes redundancy layer 2 can't, layer 2 removes items
   layer 1 can't.
2. The **loss-free-via-expansion** guarantee that turns an aggressive lossy ratio into an
   effectively lossless one (formalised as Proposition 2 in the paper: expansion rate = predictor
   miss rate).

See [`NOVELTY.md`](NOVELTY.md) for the full prior-art boundary. Reproduce every number here with:

```bash
python bench/traces/make_redundant_trace.py > bench/traces/coding_agent_trace.jsonl
python bench/codec_bench.py bench/traces/sre_debug_trace.jsonl bench/traces/coding_agent_trace.jsonl
```
