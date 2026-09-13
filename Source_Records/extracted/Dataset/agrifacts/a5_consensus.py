"""a5_consensus.py — 3-different-family judges keep only well-posed MCQs.

Judges (different families; none judges its own build output — the build used
DeepSeek/Mistral, judging adds OpenRouter as a 3rd family):
    deepseek  -> deepseek-v4-flash
    mistral   -> mistral-small-2603
    openrouter-> qwen-2.5-72b-instruct

Each judge sees the question + the 3 options (NOT the gold answer) and decides
whether it is answerable from official statistics with a single best option.
A candidate is KEPT only if all three say YES (unanimous). Temperature 0.

Resumable: per-candidate verdicts cached in agrifacts_judgments.jsonl; kept items
written to agrifacts_filtered.jsonl. Keep rates logged per judge and by condition.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client import LLMClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "interim" / "agrifacts_pool.jsonl"
JUDG = ROOT / "data" / "interim" / "agrifacts_judgments.jsonl"
KEPT = ROOT / "data" / "interim" / "agrifacts_filtered.jsonl"

JUDGES = [("deepseek", "default"), ("mistral", "default"), ("openrouter", "specialist")]

SYS = (
    "You audit multiple-choice questions for a factual benchmark. Decide only "
    "whether the question is well-posed: answerable in principle from official "
    "government agricultural statistics, with exactly one option being the best "
    "answer (the 'roughly equal' option counts as a valid best answer). Do not "
    "try to answer it. Reply with a single word: YES or NO."
)


def _prompt(rec):
    opts = "\n".join(f"({chr(97+i)}) {c}" for i, c in enumerate(rec["choices"]))
    return f"Question: {rec['question']}\nOptions:\n{opts}\n\nWell-posed? YES or NO."


def _yes(text: str) -> bool:
    return bool(re.search(r"\byes\b", (text or "").lower()))


def _load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _judge_candidate(client, rec):
    verdicts = {}
    for prov, key in JUDGES:
        try:
            out = client.judge_call(prov, key, [
                {"role": "system", "content": SYS},
                {"role": "user", "content": _prompt(rec)},
            ])
            verdicts[prov] = _yes(out)
        except Exception as e:  # noqa: BLE001
            verdicts[prov] = None  # judge failed; retry on a later run
    return {"candidate_id": rec["candidate_id"], "verdicts": verdicts}


def main():
    client = LLMClient()
    pool = _load_jsonl(POOL)
    done = {j["candidate_id"]: j for j in _load_jsonl(JUDG)}
    # only re-judge candidates with a missing/None verdict
    todo = [r for r in pool
            if r["candidate_id"] not in done
            or any(v is None for v in done[r["candidate_id"]]["verdicts"].values())]
    print(f"pool={len(pool)}  judged={len(done)}  to-judge={len(todo)}")

    with JUDG.open("a", encoding="utf-8") as jf:
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = [ex.submit(_judge_candidate, client, r) for r in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                done[res["candidate_id"]] = res
                jf.write(json.dumps(res, ensure_ascii=False) + "\n")
                jf.flush()
                if i % 100 == 0:
                    print(f"  ...{i}/{len(todo)}")

    # assemble kept set
    by_id = {r["candidate_id"]: r for r in pool}
    kept, per_judge, by_cond = [], {p: [0, 0] for p, _ in JUDGES}, {}
    for cid, j in done.items():
        if cid not in by_id:
            continue
        v = j["verdicts"]
        for p, _ in JUDGES:
            if v.get(p) is not None:
                per_judge[p][0] += int(v[p])
                per_judge[p][1] += 1
        unanimous = all(v.get(p) is True for p, _ in JUDGES)
        cond = by_id[cid]["condition"]
        by_cond.setdefault(cond, [0, 0])
        by_cond[cond][1] += 1
        if unanimous:
            by_cond[cond][0] += 1
            kept.append(by_id[cid])

    with KEPT.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nKEPT {len(kept)}/{len(pool)} ({len(kept)/max(1,len(pool)):.1%}) -> {KEPT}")
    for p, (k, n) in per_judge.items():
        print(f"  judge {p:10s} YES rate: {k}/{n} = {k/max(1,n):.1%}")
    for cond, (k, n) in by_cond.items():
        print(f"  condition {cond:6s} keep: {k}/{n} = {k/max(1,n):.1%}")


if __name__ == "__main__":
    main()
