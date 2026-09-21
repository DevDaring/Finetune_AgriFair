"""Response cache: one SQLite table keyed by prompt hash. A cached item is never re-queried."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""CREATE TABLE IF NOT EXISTS responses (
            key TEXT PRIMARY KEY, tier TEXT, method TEXT, repeat INTEGER, route TEXT,
            model_id TEXT, text TEXT, input_tokens INTEGER, output_tokens INTEGER,
            latency_seconds REAL, called_utc TEXT, extra TEXT)""")
        self.conn.commit()

    def get(self, key: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM responses WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self.conn.execute("SELECT * FROM responses LIMIT 0").description]
        d = dict(zip(cols, row)); d["extra"] = json.loads(d["extra"] or "{}"); return d

    def put(self, key: str, tier: str, method: str, repeat: int, route: str, model_id: str,
            text: str, input_tokens: int, output_tokens: int, latency_seconds: float,
            extra: Optional[Dict] = None) -> None:
        self.conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          (key, tier, method, repeat, route, model_id, text, input_tokens, output_tokens,
                           latency_seconds, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           json.dumps(extra or {}, ensure_ascii=False)))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def spend_tokens(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for tier, i, o in self.conn.execute("SELECT tier, SUM(input_tokens), SUM(output_tokens) FROM responses GROUP BY tier"):
            out[tier] = {"input_tokens": int(i or 0), "output_tokens": int(o or 0)}
        return out
