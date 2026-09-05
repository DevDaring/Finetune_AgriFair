"""Data hygiene and the hardcoded-key self-check.

`validate_and_dedup` drops exact-duplicate rows by content hash, detects corrupted
rows, logs counts, and returns the cleaned list. `scan_repo_for_hardcoded_keys`
fails loudly (without printing any value) if an API-key pattern appears in tracked
source.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from GPU_Run.common.logging_utils import append_csv_row
from GPU_Run.common.paths import DATA_HYGIENE_LOG, REPO_ROOT

_HYGIENE_COLUMNS = [
    "source_file",
    "rows_in",
    "rows_out",
    "duplicates_removed",
    "corrupted_removed",
]

# patterns that look like real secrets (kept deliberately broad but specific)
_KEY_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"sk-[A-Za-z0-9\-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{30,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
]


def content_hash(row: Dict, ignore_keys: Sequence[str] = ("id",)) -> str:
    payload = {k: v for k, v in row.items() if k not in ignore_keys}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_corrupted(row: Dict, required: Sequence[str]) -> bool:
    for k in required:
        v = row.get(k, None)
        if v is None:
            return True
        if isinstance(v, str) and not v.strip():
            return True
    return False


def validate_and_dedup(
    rows: Sequence[Dict],
    required_fields: Sequence[str],
    source_file: str = "<memory>",
) -> List[Dict]:
    """Idempotent: dedup by content hash and drop corrupted rows; log counts."""
    seen = set()
    out: List[Dict] = []
    dupes = 0
    corrupt = 0
    for r in rows:
        if _is_corrupted(r, required_fields):
            corrupt += 1
            continue
        h = content_hash(r)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        out.append(r)
    append_csv_row(
        DATA_HYGIENE_LOG,
        {
            "source_file": source_file,
            "rows_in": len(rows),
            "rows_out": len(out),
            "duplicates_removed": dupes,
            "corrupted_removed": corrupt,
        },
        _HYGIENE_COLUMNS,
    )
    return out


def scan_repo_for_hardcoded_keys(extra_skip: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """Scan tracked source files for key-like patterns. Returns (file, pattern_name)
    hits WITHOUT the matched value. .env and .env.example are skipped by design."""
    skip_names = {".env", ".env.example"}
    skip_names.update(extra_skip)
    skip_dirs = {"data", "results", "checkpoints", "models", "Dataset", ".git", "__pycache__"}
    hits: List[Tuple[str, str]] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in skip_names:
            continue
        if any(part in skip_dirs for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".html", ".cfg", ".ini"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in _KEY_PATTERNS:
            if pat.search(text):
                hits.append((str(path.relative_to(REPO_ROOT)), pat.pattern))
    return hits
