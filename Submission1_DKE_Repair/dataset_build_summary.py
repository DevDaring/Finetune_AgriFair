"""Build statistics of the released AgriFair benchmark, recomputed from the build records.

    python -m Submission1_DKE_Repair.dataset_build_summary

The Dataset section reports how AgriFacts and AgriAdvice were prepared. Every count it gives is
recomputed here from the build files, not copied from the datasheet, and the released files are
checked to be the ones the build produced.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"
BUILD = C.CODES_ROOT / "Source_Records" / "extracted" / "Dataset"
RELEASE = C.CODES_ROOT / "Dataset"


def _jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    interim = BUILD / "data" / "interim"
    facts = list(csv.DictReader((interim / "agrifacts_facts.csv").open(encoding="utf-8")))
    holdings = list(csv.DictReader((interim / "census_holdings_long.csv").open(encoding="utf-8")))
    judged = _jsonl(interim / "agrifacts_judgments.jsonl")
    kept = [j for j in judged
            if len(j["verdicts"]) == 3 and all(v is True for v in j["verdicts"].values())]
    released_facts = _jsonl(RELEASE / "agrifacts.jsonl")
    released_advice = _jsonl(RELEASE / "agriadvice.jsonl")
    built_final = BUILD / "data" / "final" / "agrifacts.jsonl"

    advice_ok = sum(1 for p in released_advice
                    if p["base_query"] in p["version_A"]["prompt"]
                    and p["base_query"] in p["version_B"]["prompt"]
                    and p["version_A"]["prompt"] != p["version_B"]["prompt"])
    summary = {
        "agrifacts": {
            "states_extracted": len({h.get("state") for h in holdings}),
            "validated_comparisons": len(facts),
            "judged_candidates": len(judged),
            "kept_unanimously": len(kept),
            "released_items": len(released_facts),
            "by_condition": dict(collections.Counter(x["condition"] for x in released_facts)),
            "by_axis": dict(collections.Counter(x["axis"] for x in released_facts)),
            "released_file_equals_build_output": sorted(_jsonl(built_final), key=lambda x: x["id"])
                                                 == sorted(released_facts, key=lambda x: x["id"]),
            "census_audit": "run Source_Records/extracted/Dataset/verify_agrifacts.py; "
                            "it aborts on the first mismatch",
        },
        "agriadvice": {
            "released_pairs": len(released_advice),
            "by_axis": dict(collections.Counter(p["toggle_axis"] for p in released_advice)),
            "source_dataset": sorted({p["source_dataset"] for p in released_advice}),
            "query_verbatim_in_both_versions_and_versions_differ": advice_ok,
        },
        "release_sha256": {"agrifacts.jsonl": _sha(RELEASE / "agrifacts.jsonl"),
                           "agriadvice.jsonl": _sha(RELEASE / "agriadvice.jsonl")},
    }
    C.write_json(C.CODES_ROOT / OUT_DIR / "dataset_build_summary.json", summary)
    print(json.dumps(summary["agrifacts"], indent=1)[:600])


if __name__ == "__main__":
    main()
