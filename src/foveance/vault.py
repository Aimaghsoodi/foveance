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
import zlib
from contextlib import closing
from typing import Optional


# -- pluggable blob codec ---------------------------------------------------------------------
# The vault stores each item as a compressed BLOB. stdlib zlib is always available (~38x on
# redundant agent text); when the stronger general-purpose codecs are installed we transparently
# use the best one (brotli ~50x, zstd ~44x, measured on the redundancy suite). A 3-byte magic
# header ``b"FV" + tag`` records which codec wrote the blob so reads never guess; legacy blobs
# (raw zlib streams, no header) are still decoded via the zlib fallback, so old vaults keep working.
_MAGIC = b"FV"


def _best_codec() -> str:
    try:
        import brotli  # noqa: F401
        return "b"
    except Exception:
        pass
    try:
        import zstandard  # noqa: F401
        return "z"
    except Exception:
        return "l"


def blob_encode(text: str) -> bytes:
    """Compress ``text`` with the strongest available codec; self-describing (see ``_MAGIC``)."""
    raw = text.encode("utf-8", "ignore")
    tag = _best_codec()
    if tag == "b":
        import brotli
        body = brotli.compress(raw, quality=11)
    elif tag == "z":
        import zstandard
        body = zstandard.ZstdCompressor(level=19).compress(raw)
    else:
        body = zlib.compress(raw, 9)
    return _MAGIC + tag.encode("ascii") + body


def blob_decode(blob: bytes) -> str:
    """Inverse of :func:`blob_encode`; also decodes legacy headerless zlib blobs."""
    if blob[:2] == _MAGIC:
        tag, body = chr(blob[2]), blob[3:]
        if tag == "b":
            import brotli
            return brotli.decompress(body).decode("utf-8", "replace")
        if tag == "z":
            import zstandard
            return zstandard.ZstdDecompressor().decompress(body).decode("utf-8", "replace")
        return zlib.decompress(body).decode("utf-8", "replace")
    return zlib.decompress(blob).decode("utf-8", "replace")  # legacy: raw zlib stream


def default_vault_path() -> str:
    d = os.path.join(os.path.expanduser("~"), ".foveance")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "vault.db")


def item_id_for(conv_id: str, text: str) -> str:
    """Stable, content-addressed item id (same text in the same conversation -> same id)."""
    return hashlib.sha1(f"{conv_id}\x00{text}".encode("utf-8", "ignore")).hexdigest()[:12]


class ItemVault:
    """SQLite-backed full-text store for compressed items."""

    def __init__(self, path: Optional[str] = None, compress: bool = True):
        # ``compress`` stores each full text zlib-compressed (a BLOB), realising the transport-codec
        # storage saving (up to ~50x on redundant content). Reads transparently handle both
        # compressed and legacy plaintext rows, so it is backward-compatible with older vaults.
        self.path = path or default_vault_path()
        self.compress = compress
        with closing(self._conn()) as c, c:
            c.execute("CREATE TABLE IF NOT EXISTS items ("
                      "conv_id TEXT NOT NULL, item_id TEXT NOT NULL, kind TEXT NOT NULL, "
                      "full_text TEXT NOT NULL, created_ts REAL NOT NULL, "
                      "PRIMARY KEY(conv_id, item_id))")
            try:  # migrate older vaults: add the compressed-blob column if absent
                c.execute("ALTER TABLE items ADD COLUMN blob BLOB")
            except sqlite3.OperationalError:
                pass  # column already exists

    def _conn(self) -> sqlite3.Connection:
        # A fresh connection per operation, always closed via contextlib.closing. sqlite3's own
        # ``with conn`` context manager commits/rolls back but does NOT close the handle -- on
        # Windows a lingering handle blocks the .db file from being deleted, so we close explicitly.
        return sqlite3.connect(self.path)

    @staticmethod
    def _decode(full_text: str, blob) -> str:
        """Prefer the compressed blob; fall back to legacy plaintext."""
        if blob is not None:
            return blob_decode(blob)
        return full_text

    def put(self, conv_id: str, item_id: str, kind: str, full_text: str) -> None:
        text_col, blob_col = (full_text, None)
        if self.compress:
            text_col, blob_col = ("", blob_encode(full_text))
        with closing(self._conn()) as c, c:
            c.execute("INSERT OR IGNORE INTO items"
                      "(conv_id, item_id, kind, full_text, created_ts, blob) "
                      "VALUES(?,?,?,?,?,?)",
                      (conv_id, item_id, kind, text_col, time.time(), blob_col))

    def get(self, conv_id: str, item_id: str) -> Optional[str]:
        with closing(self._conn()) as c:
            row = c.execute("SELECT full_text, blob FROM items WHERE conv_id=? AND item_id=?",
                            (conv_id, item_id)).fetchone()
        return self._decode(row[0], row[1]) if row else None

    def get_any(self, item_id: str) -> Optional[str]:
        """Lookup by item id alone (ids are content-addressed per conversation, collisions are
        negligible at 48 bits); lets ``foveance_expand`` work even if the conversation id the
        client presents drifts between requests."""
        with closing(self._conn()) as c:
            row = c.execute("SELECT full_text, blob FROM items WHERE item_id=? LIMIT 1",
                            (item_id,)).fetchone()
        return self._decode(row[0], row[1]) if row else None

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
