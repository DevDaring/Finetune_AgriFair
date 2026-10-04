"""Integrity check of every TMLR prediction file before analysis and publication.

    python -m Submission1_TMLR.verify_results

For each file: (1) rows against the design (systems x prompts); (2) duplicate (prompt, system) rows;
(3) every row's prompt_sha256 against the prompt file it should come from, so a row answered from a
wrong or edited prompt is caught; (4) the stored score re-derived from the raw output with the
published scorer; (5) API error rows; (6) parse rates below 0.95 per system and experiment.
Writes results_submission1_tmlr/verification_report.json and prints a summary.
"""
from __future__ import annotations

import collections
import json

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2.run_inference import score_row
from Submission1_TMLR import analyse as A
from Submission1_TMLR import run_gpu as G
from Submission1_TMLR.logprobs import prompts_for_logprobs

OUT = C.CODES_ROOT / "results_submission1_tmlr"
GPU40 = 40
GPU32 = 32
HOSTED = 9

# file -> (prompt loader, expected systems, rescore?)  rescore is off where the scorer differs by design
FILES = {
    "gpu_main_predictions.jsonl": (lambda: G.load_prompts("main"), GPU32, True),
    "gpu_flash_predictions.jsonl": (lambda: G.load_prompts("main"), 24, True),
    "gpu_followups_predictions.jsonl": (lambda: G.load_prompts("followups"), GPU32, True),
    "gpu_cot_predictions.jsonl": (lambda: G.load_prompts("cot"), 8, False),
    "gpu_altered48_predictions.jsonl": (lambda: G.load_prompts("altered48"), GPU32, True),
    "gpu_draws_predictions.jsonl": (lambda: G.load_prompts("main+altered48"), 8, True),
    "gpu_retry256_predictions.jsonl": (lambda: G.load_prompts("main"), 2, True),
    "gpu_round2_predictions.jsonl": (lambda: G.load_prompts("round2"), GPU40, True),
    "gpu_round3_predictions.jsonl": (lambda: G.load_prompts("round3"), GPU40, True),
    "gpu_logprobs_predictions.jsonl": (lambda: prompts_for_logprobs(0), GPU40, False),
    "bedrock_main_predictions.jsonl": (lambda: G.load_prompts("main"), HOSTED, True),
    "bedrock_followups_predictions.jsonl": (lambda: G.load_prompts("followups"), HOSTED, True),
    "bedrock_cot_predictions.jsonl": (lambda: G.load_prompts("cot"), HOSTED, False),
    "bedrock_altered48_predictions.jsonl": (lambda: G.load_prompts("altered48"), HOSTED, True),
    "bedrock_round2_predictions.jsonl": (lambda: G.load_prompts("round2"), HOSTED, True),
    # Bedrock ran the renamed items as their own stream, so its round-3 file excludes them
    "bedrock_round3_predictions.jsonl": (lambda: [q for q in G.load_prompts("round3") if q["experiment"] != "renamed"], HOSTED, True),
    "bedrock_renamed_predictions.jsonl": (lambda: G.load_prompts("renamed"), HOSTED, True),
}


def check(name: str, loader, n_systems: int, rescore: bool) -> dict:
    rows = A.rows_of(OUT / name)
    if rows is None:
        return {"present": False}
    # rows_of drops excluded comparisons; read raw so the count is against the full design
    import gzip
    p = OUT / name
    raw = C.read_jsonl(p) if p.exists() else [json.loads(l) for l in gzip.open(p.with_name(p.name + ".gz"), "rt")]
    prompts = {q["prompt_id"]: q for q in loader()}
    errors = [r for r in raw if r.get("error")]
    ok = [r for r in raw if not r.get("error")]
    keys = collections.Counter((r["prompt_id"], r["system"]) for r in ok)
    dups = sum(v - 1 for v in keys.values() if v > 1)
    systems = sorted({r["system"] for r in ok})
    unknown = [r["prompt_id"] for r in ok if r["prompt_id"] not in prompts]
    hash_bad = [r["prompt_id"] for r in ok if r["prompt_id"] in prompts and r.get("prompt_sha256")
                and r["prompt_sha256"] != C.freeze(prompts[r["prompt_id"]]["prompt"])
                and "max_new_tokens" not in r.get("prompt_id", "")]
    score_bad = 0
    if rescore:
        for r in ok:
            q = prompts.get(r["prompt_id"])
            if q is None or r.get("experiment") == "recall" or "raw_output" not in r:
                continue
            s = score_row(q, r["raw_output"])
            if bool(s["correct"]) != bool(r["correct"]) or s.get("picked_choice") != r.get("picked_choice"):
                score_bad += 1
    per = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        if "parse_ok" in r and r.get("experiment") not in ("recall", "cot"):
            k = (r["system"], r.get("experiment")); per[k][0] += bool(r["parse_ok"]); per[k][1] += 1
    low = sorted(((s, e, round(a / n, 3)) for (s, e), (a, n) in per.items() if n and a / n < 0.95), key=lambda x: x[2])
    expected = n_systems * len(prompts)
    covered = len(keys)
    return {"present": True, "rows_ok": len(ok), "unique_ok": covered, "expected": expected,
            "complete": covered >= expected, "systems": len(systems), "expected_systems": n_systems,
            "error_rows": len(errors), "duplicates": dups, "unknown_prompt_ids": len(unknown),
            "prompt_hash_mismatch": len(hash_bad), "rescore_mismatch": score_bad, "low_parse_cells": low[:12]}


def main() -> None:
    report = {}
    for name, (loader, n, rescore) in FILES.items():
        report[name] = check(name, loader, n, rescore)
        r = report[name]
        if r["present"]:
            flag = "OK " if (r["complete"] and not r["duplicates"] and not r["prompt_hash_mismatch"]
                             and not r["rescore_mismatch"] and not r["unknown_prompt_ids"]) else ("RUN" if not r["complete"] else "BAD")
            print(f"{flag} {name:40s} {r['unique_ok']:>7}/{r['expected']:<7} sys {r['systems']}/{r['expected_systems']} "
                  f"err {r['error_rows']} dup {r['duplicates']} hash {r['prompt_hash_mismatch']} rescore {r['rescore_mismatch']} "
                  f"unknown {r['unknown_prompt_ids']} lowparse {len(r['low_parse_cells'])}")
        else:
            print(f"--  {name:40s} not present yet")
    C.write_json(OUT / "verification_report.json", report)


if __name__ == "__main__":
    main()
