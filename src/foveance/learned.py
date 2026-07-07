"""
Optional *learned* future-relevance predictor.

Implements the ``FutureRelevancePredictor`` interface (predictor.py) with a tiny logistic
regression trained to answer: *"will item i be referenced within horizon H?"* The calibrated
probability becomes the ``sim_to_future`` term, so a learned estimator drops into the exact
same value-curve / index-allocator machinery as the heuristic posterior (Lemma "estimation
robustness" in the paper bounds how its calibration error propagates).

Features (cheap, black-box, no model internals):
  similarity to the future posterior, similarity to the last query, time-since-touch
  (half-life transformed), recurrence flag, and a kind prior. Pure-numpy; no torch needed.

NOTE ON NOVELTY: the learned predictor is an *instance* of the anticipatory criterion, not a
separate contribution; it exists to show the criterion is realizable from logged traces.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .store import Item
from .embedders import HashingEmbedder, as_embed_fn, cosine
from .predictor import PredictorContext, PredictorConfig

# Stable kind ordering for the kind feature.
_KINDS = ("tool_output", "user", "assistant", "doc", "reasoning")


def featurize(item: Item, ctx: PredictorContext, embed) -> list[float]:
    """Black-box feature vector for (item, history). Deterministic given the embedder."""
    if item.embedding is None:
        item.embedding = embed(item.full_text)
    fut = ctx.future_posterior
    last_q = ctx.query_history[-1] if ctx.query_history else []
    sim_future = max(0.0, cosine(item.embedding, fut)) if fut else 0.0
    sim_last = max(0.0, cosine(item.embedding, last_q)) if last_q else 0.0
    age = ctx.turn - max(item.created_turn, item.last_referenced_turn)
    recency = 0.5 ** (age / max(1e-9, ctx.cfg.half_life))
    recur = 1.0 if item.last_referenced_turn >= 0 else 0.0
    kind = float(_KINDS.index(item.kind)) / len(_KINDS) if item.kind in _KINDS else 0.0
    return [1.0, sim_future, sim_last, recency, recur, kind]  # leading 1.0 = bias


@dataclass
class LogisticFutureRelevance:
    """Calibrated logistic model of P(referenced within horizon | features)."""

    weights: list[float] = field(default_factory=lambda: [0.0] * 6)
    embedder: object = field(default_factory=HashingEmbedder)

    def __post_init__(self) -> None:
        self._embed = as_embed_fn(self.embedder)

    # ------------------------------------------------------------------ training
    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        epochs: int = 400,
        lr: float = 0.3,
        l2: float = 1e-4,
    ) -> "LogisticFutureRelevance":
        """Full-batch gradient descent on log loss with L2. Pure numpy."""
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "LogisticFutureRelevance.fit() needs numpy, which isn't installed. "
                'Install it with `pip install "foveance[ml]"` (or the `bench`/`all` extra).'
            ) from e

        Xa = np.asarray(X, float)
        ya = np.asarray(y, float)
        if Xa.ndim != 2 or len(Xa) == 0:
            return self
        w = np.zeros(Xa.shape[1])
        for _ in range(epochs):
            z = Xa @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xa.T @ (p - ya) / len(ya) + l2 * w
            w -= lr * grad
        self.weights = [float(v) for v in w]
        return self

    def fit_traces(self, traces: Sequence[dict], horizon: int = 5) -> "LogisticFutureRelevance":
        """Build (features, label) from logged trajectories and fit.

        Each trace is ``{"items": [Item...], "queries": [str...], "referenced": {turn: set_ids}}``.
        Label = 1 if the item is referenced within ``horizon`` turns of the snapshot.
        """
        X: list[list[float]] = []
        y: list[float] = []
        for tr in traces:
            items: list[Item] = tr["items"]
            queries: list[str] = tr["queries"]
            referenced: dict = tr.get("referenced", {})
            q_emb: list[list[float]] = []
            for t, q in enumerate(queries):
                q_emb.append(self._embed(q))
                fut = q_emb[-1]
                ctx = PredictorContext(turn=t, future_posterior=fut, query_history=q_emb,
                                       cfg=PredictorConfig())
                future_refs: set = set()
                for h in range(t + 1, min(len(queries), t + 1 + horizon)):
                    future_refs |= set(referenced.get(h, set()))
                for it in items:
                    if it.created_turn > t:
                        continue
                    X.append(featurize(it, ctx, self._embed))
                    y.append(1.0 if it.item_id in future_refs else 0.0)
        return self.fit(X, y)

    # ------------------------------------------------------------------ inference
    def predict(self, item: Item, history: PredictorContext) -> float:
        f = featurize(item, history, self._embed)
        z = sum(w * x for w, x in zip(self.weights, f))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"weights": self.weights}, fh)

    @classmethod
    def load(cls, path: str, embedder: Optional[object] = None) -> "LogisticFutureRelevance":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(weights=list(d["weights"]),
                   embedder=embedder if embedder is not None else HashingEmbedder())
