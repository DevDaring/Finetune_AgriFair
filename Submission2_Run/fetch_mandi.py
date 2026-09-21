"""Snapshot today's Agmarknet mandi prices for the study states into data_submission2/mandi/.

Run once a day for at least a week before the market-grounded advice experiment (E3b).
Each snapshot is a dated JSONL plus a SHA-256; snapshots are never modified afterwards.
"""
from __future__ import annotations

import datetime as dt
import os

from Submission2_Run import common as C
from Submission2_Run.build_kcc_sample import fetch_api


def main() -> None:
    cfg = C.load_config(); key = os.environ.get("DATA_GOV_IN_API_KEY", "")
    if not key:
        raise SystemExit("DATA_GOV_IN_API_KEY not set")
    d = C.data_dir(cfg) / "mandi"; d.mkdir(exist_ok=True)
    rows = []
    for st in cfg["kcc"]["states"]:
        rows += fetch_api(cfg["mandi"]["resource_id"], key, limit=500, max_rows=20000, filters={"state": st})
    today = dt.date.today().isoformat(); out = d / f"mandi_{today}.jsonl"
    C.write_jsonl(out, rows)
    C.write_json(d / f"mandi_{today}.manifest.json", {"date": today, "rows": len(rows), "states": cfg["kcc"]["states"],
                                                       "resource_id": cfg["mandi"]["resource_id"], "sha256": C.sha256_file(out)})
    print(f"{len(rows)} price rows -> {out}")


if __name__ == "__main__":
    main()
