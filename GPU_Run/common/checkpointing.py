"""Resumable jsonl runners and epoch checkpoint helpers.

Every API/evaluation loop saves progress every 50 rows and skips completed ids on
restart; every training script saves a LoRA checkpoint each epoch and supports resume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set

SAVE_EVERY = 50


def completed_ids(jsonl_path: Path, id_field: str = "id") -> Set[str]:
    done: Set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(str(json.loads(line)[id_field]))
                except Exception:
                    continue
    return done


def resumable_map(
    items: List[Dict],
    out_path: Path,
    process_fn: Callable[[Dict], Dict],
    id_field: str = "id",
) -> List[Dict]:
    """Process items, appending results to out_path, skipping already-completed ids."""
    done = completed_ids(out_path, id_field)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    buffer: List[Dict] = []
    with open(out_path, "a", encoding="utf-8") as f:
        for item in items:
            if str(item.get(id_field)) in done:
                continue
            res = process_fn(item)
            results.append(res)
            buffer.append(res)
            if len(buffer) >= SAVE_EVERY:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                buffer = []
        for r in buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def latest_epoch(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return -1
    epochs = [int(p.name.split("_")[-1]) for p in checkpoint_dir.glob("epoch_*") if p.is_dir()]
    return max(epochs) if epochs else -1
