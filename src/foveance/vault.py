"""Durable item vault: full-fidelity context survives proxy restarts.

The in-memory store makes re-inflation possible only within a process lifetime. The vault spills
every item's full text to SQLite (stdlib, no new dependencies) so that "nothing is deleted
forever" holds across restarts, and so the ``foveance_expand`` tool can retrieve any compressed
item on demand. One row per (conversation, item); content-addressed ids keep writes idempotent.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import closing
from typing import Optional


def default_vault_path() -> str:
    d = os.path.join(os.path.expanduser("~"), ".foveance")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "vault.db")


def item_id_for(conv_id: str, text: str) -> str:
    """Stable, content-addressed item id (same text in the same conversation -> same id)."""
    return hashlib.sha1(f"{conv_id}\x00{text}".encode("utf-8", "ignore")).hexdigest()[:12]


class ItemVault:
    """SQLite-backed full-text store for compressed items."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_vault_path()
        with closing(self._conn()) as c, c:
            c.execute("CREATE TABLE IF NOT EXISTS items ("
                      "conv_id TEXT NOT NULL, item_id TEXT NOT NULL, kind TEXT NOT NULL, "
                      "full_text TEXT NOT NULL, created_ts REAL NOT NULL, "
                      "PRIMARY KEY(conv_id, item_id))")

    def _conn(self) -> sqlite3.Connection:
        # A fresh connection per operation, always closed via contextlib.closing. sqlite3's own
        # ``with conn`` context manager commits/rolls back but does NOT close the handle -- on
        # Windows a lingering handle blocks the .db file from being deleted, so we close explicitly.
        return sqlite3.connect(self.path)

    def put(self, conv_id: str, item_id: str, kind: str, full_text: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("INSERT OR IGNORE INTO items(conv_id, item_id, kind, full_text, created_ts) "
                      "VALUES(?,?,?,?,?)", (conv_id, item_id, kind, full_text, time.time()))

    def get(self, conv_id: str, item_id: str) -> Optional[str]:
        with closing(self._conn()) as c:
            row = c.execute("SELECT full_text FROM items WHERE conv_id=? AND item_id=?",
                            (conv_id, item_id)).fetchone()
        return row[0] if row else None

    def get_any(self, item_id: str) -> Optional[str]:
        """Lookup by item id alone (ids are content-addressed per conversation, collisions are
        negligible at 48 bits); lets ``foveance_expand`` work even if the conversation id the
        client presents drifts between requests."""
        with closing(self._conn()) as c:
            row = c.execute("SELECT full_text FROM items WHERE item_id=? LIMIT 1",
                            (item_id,)).fetchone()
        return row[0] if row else None

    def count(self, conv_id: Optional[str] = None) -> int:
        with closing(self._conn()) as c:
            if conv_id is None:
                return c.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM items WHERE conv_id=?",
                             (conv_id,)).fetchone()[0]

    def prune(self, older_than_days: float = 30.0) -> int:
        cutoff = time.time() - older_than_days * 86400
        with closing(self._conn()) as c, c:
            cur = c.execute("DELETE FROM items WHERE created_ts < ?", (cutoff,))
        return cur.rowcount
