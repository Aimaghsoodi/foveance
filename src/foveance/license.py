"""Foveance Pro licensing and persistent savings history.

The open-source package is fully functional without a license. A Pro license key additionally
unlocks *persistence*: the proxy's savings accounting survives restarts (SQLite in
``~/.foveance/``), the dashboard shows all-time and per-day totals, and ``/admin/export.csv``
exports the history.

Keys are verified OFFLINE: a key is an RSA-2048 signature over a small JSON payload, checked in
pure stdlib against the public modulus below (no network call, no phoning home, no new
dependencies). Being open source, this gate is a courtesy to honest users, not DRM.

    foveance license activate FOV1-...   # stores ~/.foveance/license.json after verifying
    foveance license status
    foveance license deactivate
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import time
from typing import Optional

# RSA-2048 public modulus for license verification (e = 65537). The private half lives with the
# maintainers; see docs for how keys are purchased.
_PUB_N = 26107179466209356794164581508639895341909813162915137628591532260142793867653832448286699604011709966440269030930961408670943000082700290511242202261279986002289458280224311793891356352706928225520083879197022988819227135997798035137364239408473805793769156089149793167633766439720306465953172360957356938775597908069565047754315919110409635168145356898495581959688461485712014554963419251631047785256061183755178106678960543025769553296427370594507420236930330194737371576422061626903158937020178483707651968075152761520007384752273266578483122066882145157800178451783818570423799006079211673095919969209349802789567
_PUB_E = 65537
# DigestInfo prefix for SHA-256 (PKCS#1 v1.5)
_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


def _home_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".foveance")
    os.makedirs(d, exist_ok=True)
    return d


def _license_path() -> str:
    return os.path.join(_home_dir(), "license.json")


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _rsa_verify(payload: bytes, sig: bytes,
                n: Optional[int] = None, e: Optional[int] = None) -> bool:
    """PKCS#1 v1.5 SHA-256 signature verification in pure stdlib (modular exponentiation).
    ``n``/``e`` default to the module-level publisher key, read at call time."""
    n = _PUB_N if n is None else n
    e = _PUB_E if e is None else e
    k = (n.bit_length() + 7) // 8
    if len(sig) != k:
        return False
    em = pow(int.from_bytes(sig, "big"), e, n).to_bytes(k, "big")
    if em[0] != 0x00 or em[1] != 0x01:
        return False
    try:
        sep = em.index(0x00, 2)
    except ValueError:
        return False
    if sep < 10 or any(b != 0xFF for b in em[2:sep]):
        return False
    return em[sep + 1:] == _SHA256_PREFIX + hashlib.sha256(payload).digest()


def parse_key(key: str) -> Optional[dict]:
    """Verify a ``FOV1-<payload>-<sig>`` key; return its payload dict, or None if invalid."""
    try:
        tag, payload_b64, sig_b64 = key.strip().split("-", 2)
        if tag != "FOV1":
            return None
        payload = _b64u_decode(payload_b64)
        if not _rsa_verify(payload, _b64u_decode(sig_b64)):
            return None
        data = json.loads(payload)
        return data if isinstance(data, dict) and data.get("v") == 1 else None
    except Exception:
        return None


def activate(key: str) -> Optional[dict]:
    """Verify and store the key. Returns the license payload, or None if the key is invalid."""
    data = parse_key(key)
    if data is None:
        return None
    with open(_license_path(), "w", encoding="utf-8") as f:
        json.dump({"key": key.strip(), "payload": data}, f)
    return data


def deactivate() -> bool:
    try:
        os.remove(_license_path())
        return True
    except FileNotFoundError:
        return False


def current() -> Optional[dict]:
    """The active license payload (re-verified on every read), or None."""
    try:
        with open(_license_path(), encoding="utf-8") as f:
            stored = json.load(f)
        return parse_key(stored.get("key", ""))
    except Exception:
        return None


class SavingsLog:
    """Pro feature: persistent per-day savings accounting (SQLite, stdlib only)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(_home_dir(), "savings.db")
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS savings ("
                      "day TEXT NOT NULL, requests INTEGER NOT NULL DEFAULT 0, "
                      "tokens_before INTEGER NOT NULL DEFAULT 0, "
                      "tokens_after INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(day))")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, tokens_before: int, tokens_after: int) -> None:
        day = time.strftime("%Y-%m-%d")
        with self._conn() as c:
            c.execute("INSERT INTO savings(day, requests, tokens_before, tokens_after) "
                      "VALUES(?,1,?,?) ON CONFLICT(day) DO UPDATE SET "
                      "requests=requests+1, tokens_before=tokens_before+excluded.tokens_before, "
                      "tokens_after=tokens_after+excluded.tokens_after",
                      (day, tokens_before, tokens_after))

    def totals(self) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(SUM(requests),0), COALESCE(SUM(tokens_before),0), "
                            "COALESCE(SUM(tokens_after),0) FROM savings").fetchone()
        saved = max(row[1] - row[2], 0)
        return {"requests": row[0], "tokens_before": row[1], "tokens_after": row[2],
                "tokens_saved": saved}

    def by_day(self, limit: int = 30) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT day, requests, tokens_before, tokens_after FROM savings "
                             "ORDER BY day DESC LIMIT ?", (limit,)).fetchall()
        return [{"day": d, "requests": r, "tokens_before": tb, "tokens_after": ta,
                 "tokens_saved": max(tb - ta, 0)} for d, r, tb, ta in rows]

    def export_csv(self) -> str:
        lines = ["day,requests,tokens_before,tokens_after,tokens_saved"]
        for r in reversed(self.by_day(limit=100000)):
            lines.append(f"{r['day']},{r['requests']},{r['tokens_before']},"
                         f"{r['tokens_after']},{r['tokens_saved']}")
        return "\n".join(lines) + "\n"
