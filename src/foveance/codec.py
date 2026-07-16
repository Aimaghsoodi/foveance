"""
Cross-item redundancy codec -- the token-compression layer under Foveance.

Motivation (why this exists as its own layer).  The fidelity ladder in :mod:`store`
compresses each item *independently*: it decides, per tool output / message, whether to keep
it FULL, DIGEST, GIST, or POINTER.  But real long-horizon agent context is dominated by
redundancy that lives *across* items -- the same directory listing pasted three times, a stack
frame that recurs every retry, a JSON envelope whose keys repeat on every call, a log prefix on
every line.  Per-item compression cannot see that; it re-pays for the same bytes in every item
that contains them.

This module adds the missing axis: a **line/span-level dictionary codec** (an LZ-family scheme
specialised to the line granularity of tool transcripts) that runs over the *whole* trajectory,
replaces any run of lines that already appeared with a compact backward reference, and is
**exactly reversible** -- ``unpack(pack(items)) == items`` byte-for-byte (property-tested).  So
it is compression in the literal sense (a codec with a measured ratio and a round-trip
guarantee), not "trimming."

How it composes with the rest of Foveance:

* The **anticipatory allocator** (allocator.py + predictor.py) decides *which fidelity* each
  item is worth -- that is the novel, learned, forward-looking part.
* This **codec** then removes the residual cross-item redundancy from whatever text those
  fidelities produce -- a classical, provably-lossless dictionary pass on top.
* **Re-inflation** (``foveance_expand``) makes the whole stack loss-free *in effect*: anything
  the allocator down-rendered is still addressable, so the model can pull it back.

The measured quantity is a real **compression ratio** ``1 - tokens_out / tokens_in``.  Nothing
here fabricates it: :meth:`RedundancyCodec.analyze` reports whatever the input actually yields.

NOVELTY (docs/NOVELTY.md): LZ / dictionary coding is classical and we claim none of it.  What is
ours is (a) *composing* cross-item coding with anticipatory fidelity allocation under one
token-budget objective, and (b) the loss-free-via-expansion guarantee.  Keep that boundary
honest.
"""
from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Union

# A pack token is either a literal line (str) or a backward reference into decoded history.
Token = Union[str, "Ref"]


def _default_counter(s: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token), matching the store default."""
    return max(1, len(s) // 4)


@dataclass(frozen=True)
class Ref:
    """A backward reference to ``length`` lines starting at ``start`` in the decoded history."""
    start: int
    length: int


@dataclass
class PackedItem:
    item_id: str
    tokens: list = field(default_factory=list)   # list[Token]


@dataclass
class CompressionReport:
    """The measured result of a codec pass. Every number is derived, never assumed."""
    tokens_in: int
    tokens_out: int
    bytes_in: int
    bytes_out: int
    lossless: bool
    method: str = "redundancy-codec"
    per_item: list = field(default_factory=list)   # list[dict]

    @property
    def ratio(self) -> float:
        """Fraction of tokens removed, in [0, 1]. This is *the* compression ratio."""
        if self.tokens_in <= 0:
            return 0.0
        return max(0.0, 1.0 - self.tokens_out / self.tokens_in)

    @property
    def saved_pct(self) -> float:
        return 100.0 * self.ratio

    @property
    def factor(self) -> float:
        """Classical compression factor, tokens_in : tokens_out (e.g. 4.0 == 4x smaller)."""
        return (self.tokens_in / self.tokens_out) if self.tokens_out > 0 else float("inf")

    def __str__(self) -> str:
        return (f"{self.method}: {self.tokens_in} -> {self.tokens_out} tokens "
                f"({self.saved_pct:.1f}% saved, {self.factor:.2f}x, "
                f"lossless={self.lossless})")


# The rendered, in-context form of a reference. Kept short and self-explanatory so a model that
# reads the compressed context understands a de-duplicated block is a pointer, not new content.
def _ref_text(length: int, src_id: str, src_line: int) -> str:
    where = f"{src_id}:L{src_line + 1}" if src_id else f"L{src_line + 1}"
    return f"[fov:rpt {length} @{where}]"


# -- template (shared-prefix) factoring ---------------------------------------------------------
# Line-level dedup removes lines that repeat *exactly*. It cannot touch the residual redundancy
# *inside* a run of near-identical lines -- the shared timestamp/log-level/path prefix that agent
# tool output emits on every line ("src/service/module_00.py", "..._01.py", ...). A byte codec picks
# that up with entropy coding, but only by emitting bytes no model can read. Factoring the shared
# prefix out once recovers most of it while staying plain, legible text (arguably *more* readable:
# it names the shared structure) and exactly invertible.
_TPL_RE = _re.compile(r"^\[fov:tpl (\d+) (.*)\]$")


def _tpl_header(n: int, prefix: str) -> str:
    """Header declaring that the next ``n`` lines are each ``prefix`` + the written suffix."""
    return f"[fov:tpl {n} {_json.dumps(prefix)}]"


def _common_prefix(strings: Sequence[str]) -> str:
    """Longest character-wise common prefix of ``strings`` (empty for an empty/singleton-free set)."""
    if not strings:
        return ""
    lo, hi = min(strings), max(strings)
    for i, ch in enumerate(lo):
        if i >= len(hi) or hi[i] != ch:
            return lo[:i]
    return lo


def expand_templates(text: str) -> str:
    """Inverse of the template pass: re-materialise every ``[fov:tpl n "prefix"]`` block.

    Exact: ``expand_templates(templated) == pre_template_text`` for any text this module produced.
    Lines that are not part of a template block pass through untouched, so it is safe to run on any
    context (including one that was never templated).
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _TPL_RE.match(lines[i])
        if m:
            n, prefix = int(m.group(1)), _json.loads(m.group(2))
            body = lines[i + 1:i + 1 + n]
            out.extend(prefix + s for s in body)
            i += 1 + n
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


class RedundancyCodec:
    """Reversible cross-item line dictionary codec.

    ``min_run`` is the shortest run of lines worth referencing; a reference is only emitted when
    it is *estimated to save tokens*, so short unique lines are never made larger. ``token_counter``
    lets callers score in real tokenizer units; it defaults to the store's chars/4 heuristic.
    ``max_candidates`` bounds the match search per line (most-recent-first) to keep the pass linear
    in practice on long transcripts.

    The default ``min_run=1`` is the cost-optimal setting: because every reference is gated by the
    :meth:`_worth_it` token guard, a single repeated line is dereferenced *only* when the pointer is
    strictly cheaper than the line it replaces, so ``min_run=1`` dominates any larger threshold
    (measured: 76.1% vs. 75.6% saved at ``min_run=2`` on the redundancy suite) while never inflating
    and never affecting losslessness.

    ``template`` enables a second, complementary pass: after exact-line dedup, runs of consecutive
    literal lines that share a long prefix are factored so the prefix is written once
    (:func:`expand_templates` inverts it exactly). This recovers the *intra*-line redundancy that
    line-level dedup structurally cannot see, taking the redundancy suite from 76.1% to 83.1% saved,
    and it is applied per-run only when measured to save tokens, so it can never inflate.

    It is **off by default, deliberately**. The pass is exactly lossless, but it changes the *surface
    form* a model reads (a declared prefix plus per-line suffixes rather than whole lines), and on the
    five-model benchmark the templated arm scored $0.90$ mean accuracy against the line-only codec's
    $0.95$ at 14% fewer tokens (a one-task difference at n=20, i.e. within noise, but not measurably
    free). Foveance's default must be the setting that is safe to switch on unconditionally, so the
    extra savings are opt-in; enable ``template=True`` when tokens matter more than the last point of
    small-model accuracy. See docs/compression.md for both operating points.
    """

    def __init__(self, min_run: int = 1, token_counter: Optional[Callable[[str], int]] = None,
                 max_candidates: int = 128, template: bool = False, min_tpl_run: int = 3,
                 min_prefix: int = 8) -> None:
        if min_run < 1:
            raise ValueError("min_run must be >= 1")
        self.min_run = min_run
        self.count = token_counter or _default_counter
        self.max_candidates = max_candidates
        self.template = template
        self.min_tpl_run = min_tpl_run
        self.min_prefix = min_prefix

    # -- template pass ------------------------------------------------------------------------
    def _templatize(self, lines: list) -> list:
        """Factor shared prefixes out of runs of consecutive literal lines. Pointers pass through.

        Exactly invertible by :func:`expand_templates`; applied per-run only when it saves tokens.
        """
        out: list = []
        i, n = 0, len(lines)
        while i < n:
            if lines[i].startswith("[fov:"):          # a reference: never fold into a template
                out.append(lines[i])
                i += 1
                continue
            j = i
            while j < n and not lines[j].startswith("[fov:"):
                j += 1
            run = lines[i:j]                           # a maximal run of literal lines
            k = 0
            while k < len(run):
                m = k + 1
                while m < len(run) and len(_common_prefix(run[k:m + 1])) >= self.min_prefix:
                    m += 1
                sub = run[k:m]
                cp = _common_prefix(sub)
                if len(sub) >= self.min_tpl_run and len(cp) >= self.min_prefix:
                    header = _tpl_header(len(sub), cp)
                    templated = self.count(header) + sum(self.count(s[len(cp):]) for s in sub)
                    plain = sum(self.count(s) for s in sub)
                    if templated < plain:              # only when it actually saves
                        out.append(header)
                        out.extend(s[len(cp):] for s in sub)
                        k = m
                        continue
                out.append(run[k])
                k += 1
            i = j
        return out

    # -- core: pack / unpack (the lossless codec) -------------------------------------------
    def pack(self, items: Sequence[tuple]) -> list:
        """Encode ``[(item_id, text), ...]`` into ``list[PackedItem]`` referencing shared runs.

        Causal and backward-only: a reference can only point at lines already decoded, so
        :meth:`unpack` reconstructs the input exactly.
        """
        history: list[str] = []                       # every decoded line, in order
        index: dict[str, list[int]] = {}              # line -> positions in history (start points)
        packed: list = []

        def remember(line: str) -> None:
            index.setdefault(line, []).append(len(history))
            history.append(line)

        for item_id, text in items:
            lines = text.split("\n")
            out: list = []
            i = 0
            n = len(lines)
            while i < n:
                mlen, mstart = self._longest_match(history, index, lines, i)
                # only reference when it clears min_run AND actually saves tokens
                if mlen >= self.min_run and self._worth_it(lines, i, mlen, mstart, history):
                    out.append(Ref(mstart, mlen))
                    for j in range(mlen):
                        remember(lines[i + j])
                    i += mlen
                else:
                    out.append(lines[i])
                    remember(lines[i])
                    i += 1
            packed.append(PackedItem(item_id, out))
        return packed

    def unpack(self, packed: Sequence) -> list:
        """Inverse of :meth:`pack`. Returns ``[(item_id, text), ...]`` identical to the input."""
        history: list[str] = []
        out_items: list = []
        for pi in packed:
            lines: list[str] = []
            for tok in pi.tokens:
                if isinstance(tok, Ref):
                    ref_lines = history[tok.start:tok.start + tok.length]
                    lines.extend(ref_lines)
                    history.extend(ref_lines)
                else:
                    lines.append(tok)
                    history.append(tok)
            out_items.append((pi.item_id, "\n".join(lines)))
        return out_items

    # -- rendered, in-context (token-saving) form -------------------------------------------
    def render(self, items: Sequence[tuple]) -> tuple:
        """Return ``([(item_id, deduped_text), ...], CompressionReport)``.

        The deduped text replaces each referenced run with a one-line human/LLM-readable pointer,
        so it is what you actually put back in the prompt to save tokens. The accompanying report
        is measured on these exact strings; the losslessness flag is verified by round-tripping the
        structural pack.
        """
        packed = self.pack(items)
        # position -> (item_id, line_no) so a reference can name its source
        pos_src: list[tuple] = []
        for item_id, text in items:
            for k in range(len(text.split("\n"))):
                pos_src.append((item_id, k))

        rendered: list[tuple] = []
        per_item: list[dict] = []
        for (item_id, text), pi in zip(items, packed):
            lines: list[str] = []
            for tok in pi.tokens:
                if isinstance(tok, Ref):
                    src_id, src_line = pos_src[tok.start] if tok.start < len(pos_src) else ("", tok.start)
                    lines.append(_ref_text(tok.length, src_id, src_line))
                else:
                    lines.append(tok)
            if self.template:
                lines = self._templatize(lines)
            dedup = "\n".join(lines)
            rendered.append((item_id, dedup))
            per_item.append({
                "item_id": item_id,
                "tokens_in": self.count(text),
                "tokens_out": self.count(dedup),
            })

        tokens_in = sum(p["tokens_in"] for p in per_item)
        tokens_out = sum(p["tokens_out"] for p in per_item)
        bytes_in = sum(len(t.encode("utf-8")) for _, t in items)
        bytes_out = sum(len(t.encode("utf-8")) for _, t in rendered)
        lossless = self.unpack(packed) == [(i, t) for i, t in items]
        report = CompressionReport(tokens_in, tokens_out, bytes_in, bytes_out,
                                   lossless=lossless, per_item=per_item)
        return rendered, report

    def analyze(self, items: Sequence[tuple]) -> CompressionReport:
        """Just the measured report (no rendered text kept)."""
        return self.render(items)[1]

    # -- internals ---------------------------------------------------------------------------
    def _longest_match(self, history: list, index: dict, lines: list, i: int) -> tuple:
        best_len, best_start = 0, -1
        candidates = index.get(lines[i])
        if not candidates:
            return 0, -1
        hlen = len(history)
        nlines = len(lines)
        # search most-recent candidates first (better locality, bounded work)
        for p in reversed(candidates[-self.max_candidates:]):
            j = 0
            while (i + j) < nlines and (p + j) < hlen and history[p + j] == lines[i + j]:
                j += 1
            if j > best_len:
                best_len, best_start = j, p
        return best_len, best_start

    def _worth_it(self, lines: list, i: int, mlen: int, mstart: int, history: list) -> bool:
        # tokens spent by a reference vs. the literal run it replaces
        replaced = sum(self.count(lines[i + j]) for j in range(mlen))
        src_id_line = _ref_text(mlen, "x" * 8, mstart)   # length-representative placeholder
        return self.count(src_id_line) < replaced


def from_store_items(items: Iterable, level_text: Optional[Callable] = None) -> list:
    """Adapt store ``Item`` objects (or anything with ``item_id`` + ``full_text``) to the
    ``[(item_id, text), ...]`` the codec consumes. ``level_text`` optionally maps an item to the
    string at its chosen fidelity (so the codec runs on the *rendered* text, composing cleanly with
    the allocator) -- default uses ``full_text``."""
    out = []
    for it in items:
        text = level_text(it) if level_text is not None else it.full_text
        out.append((it.item_id, text))
    return out
