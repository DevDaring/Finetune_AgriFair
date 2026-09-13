"""a4_paraphrase.py — grow seeds into a wording-diverse pool (LLM as typist only).

The model NEVER sees the answer or the diff/equal label, so wording cannot leak
the label. It only rewords the question stem. For each candidate we verify that
both compared entities and the region still appear verbatim (answer-invariance);
if a rewording drops them we retry, then fall back to the original stem.

Balancing: we set a candidate target per (axis, condition) and distribute variants
across that cell's seeds. Variant 0 is always the original stem (free, valid);
variants 1+ are paraphrases (deepseek-v4-flash, fallback Mistral).

Resumable: appends to agrifacts_pool.jsonl, skipping candidate_ids already done.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client import LLMClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "interim" / "agrifacts_seeds.jsonl"
POOL = ROOT / "data" / "interim" / "agrifacts_pool.jsonl"

# candidate targets per (axis, condition) — ~20% buffer over final 1000/1000 need
TARGETS = {
    ("gender", "diff"): 200, ("gender", "equal"): 200,
    ("social_group", "diff"): 480, ("social_group", "equal"): 456,
    ("landholding", "diff"): 516, ("landholding", "equal"): 540,
}
MAX_VAR = 12

PARA_SYS = (
    "You reword survey questions. Rewrite the question in different words while "
    "keeping the exact same meaning. You MUST keep, verbatim, every place name, "
    "social-group name, gender word, size-class phrase, and the 'roughly equal' "
    "option. Do not answer it. Return only the reworded question on one line."
)


def _load_seeds():
    return [json.loads(l) for l in SEEDS.read_text(encoding="utf-8").splitlines() if l.strip()]


def _done_ids():
    if not POOL.exists():
        return set()
    return {json.loads(l)["candidate_id"]
            for l in POOL.read_text(encoding="utf-8").splitlines() if l.strip()}


def _plan(seeds):
    """Assign each seed a number of variants so each (axis,cond) cell hits target."""
    by_cell = defaultdict(list)
    for s in seeds:
        by_cell[(s["axis"], s["condition"])].append(s)
    plan = {}  # seed_id -> n_variants
    for cell, slist in by_cell.items():
        target = TARGETS.get(cell, len(slist))
        n = len(slist)
        if target <= n:
            # subsample: 1 variant each for a deterministic target-sized subset
            chosen = sorted(slist, key=lambda s: s["seed_id"])[:target]
            for s in slist:
                plan[s["seed_id"]] = 1 if s in chosen else 0
        else:
            base = min(MAX_VAR, ceil(target / n))
            # distribute so the cell totals ~target
            total = 0
            for s in sorted(slist, key=lambda s: s["seed_id"]):
                k = base if total + base <= target else max(1, target - total)
                k = min(k, MAX_VAR)
                plan[s["seed_id"]] = k
                total += k
    return plan


def _entities(seed):
    return [c for c in seed["choices"] if c != "Roughly equal"]


def _valid(text, seed):
    if not text or len(text) < 15:
        return False
    low = text
    for e in _entities(seed):
        if str(e) not in low:
            return False
    return str(seed["region"]) in low or seed["region"] == "all-India"


def _paraphrase_one(client, seed, k):
    """Return one candidate record (variant k). k==0 -> original stem."""
    if k == 0:
        q = seed["question"]
    else:
        q = None
        for attempt in range(3):
            # The variant tag + attempt make each call cache-distinct and nudge
            # the model toward a structurally different wording each time.
            user = (f"{seed['question']}\n\n"
                    f"(Rewording #{k}.{attempt}: use a sentence structure distinct "
                    f"from the original and from other rewordings.)")
            cand = client.complete(
                [{"role": "system", "content": PARA_SYS},
                 {"role": "user", "content": user}],
                tier="default", task="paraphrase", max_tokens=256,
            ).strip().strip('"')
            if _valid(cand, seed):
                q = cand
                break
        if q is None:
            q = seed["question"]  # fall back to original; never lose an item
    rec = dict(seed)
    rec["candidate_id"] = f"{seed['seed_id']}-v{k}"
    rec["question"] = q
    rec["paraphrase_of"] = seed["seed_id"]
    rec["variant"] = k
    rec["is_paraphrase"] = (k != 0 and q != seed["question"])
    return rec


def main():
    client = LLMClient()
    seeds = _load_seeds()
    plan = _plan(seeds)
    done = _done_ids()
    seed_by_id = {s["seed_id"]: s for s in seeds}

    tasks = []
    for sid, nvar in plan.items():
        for k in range(nvar):
            cid = f"{sid}-v{k}"
            if cid not in done:
                tasks.append((seed_by_id[sid], k))
    print(f"{len(seeds)} seeds -> planned {sum(plan.values())} candidates; "
          f"{len(done)} already done; {len(tasks)} to generate.")

    written = 0
    with POOL.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_paraphrase_one, client, s, k): (s["seed_id"], k)
                    for s, k in tasks}
            for fut in as_completed(futs):
                rec = fut.result()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if written % 100 == 0:
                    print(f"  ...{written}/{len(tasks)}")
    print(f"Done. Pool now has {len(_done_ids())} candidates -> {POOL}")


if __name__ == "__main__":
    main()
