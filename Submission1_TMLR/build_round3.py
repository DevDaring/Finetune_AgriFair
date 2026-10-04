"""Round-3 prompt files (Future_PLan.md, "Round 3"), built before any round-3 output existed.

    python -m Submission1_TMLR.build_round3 drifted      # P3
    python -m Submission1_TMLR.build_round3 mnli         # P1
    python -m Submission1_TMLR.build_round3 adversarial  # P5
    python -m Submission1_TMLR.build_round3 recall       # P6
    python -m Submission1_TMLR.build_round3 renamed      # P7 (after the human spot-check)
    python -m Submission1_TMLR.build_round3 all

P2 (World Bank WDI) has its own builder, Submission1_TMLR.build_wdi.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2 import r1_fresh_panel as R
from Submission1_DKE_Repair import source_schema as S
from Submission1_DKE_Repair.prompt_checks import check_panel
from Submission1_TMLR.baselines import comparison_type, cross_state_prior
from Submission1_TMLR.build_extended import _recheck, to_record
from Submission1_TMLR.exclusions import EXCLUDED_COMPARISONS

OUT = C.CODES_ROOT / "results_submission1_tmlr"
SUFFIX3 = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'
SEED = 20261006


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- P3: the drifted version-1 questions
def drifted() -> List[Dict]:
    """Re-render the version-1 prompts exactly as the published runner did, from the frozen panel, and
    require every prompt to match the SHA-256 stored in the published outputs."""
    frozen = C.CODES_ROOT / "results_submission1_phase2" / "source_validation" / "fresh_panel.jsonl"
    published = {}
    for f in ("results_submission1_phase2/predictions/main_predictions.jsonl",
              "results_submission1_phase2/predictions/pilot_predictions.jsonl"):
        for r in C.read_jsonl(C.CODES_ROOT / f):
            if r.get("study") == "r1_fresh":
                published[r["prompt_id"]] = r.get("prompt_sha256")
    out, mismatch = [], []
    for it in C.read_jsonl(frozen):
        for wording, text in it["wordings"].items():
            opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", it["choices"]))
            prompt = f"{text}\n{opts}\n\nReply with one JSON object only: " '{"answer_choice_letter": "<a|b|c>"}'
            pid = f"r1-{it['fresh_id']}-{wording}"
            want = published.get(pid)
            if want and want != C.freeze(prompt):
                mismatch.append(pid)
            out.append({"prompt_id": f"r3-drift-{it['fresh_id']}-{wording}", "study": "r1_fresh_v1_drifted",
                        "experiment": "drifted", "comparison_id": it["fresh_id"], "source_cell": it["source_cell"],
                        "axis": it["axis"], "state": it["state"], "parent_table": it["parent_table"],
                        "wording": wording,
                        "condition": it["condition"], "choices": it["choices"], "gold_choice_text": it["gold_choice_text"],
                        "max_new_tokens": 24, "prompt": prompt, "published_prompt_id": pid})
    print(f"drifted: {len(out)} prompts; published hashes found for {sum(1 for p in out if p['published_prompt_id'] in published)};"
          f" mismatches {len(mismatch)}")
    out = [p for p in out if p["published_prompt_id"] in published]      # the comparison dropped before the run has none
    for p in out:
        p["wording"] = {"original_family": "wording_a", "new_family": "wording_b"}.get(p["wording"], p["wording"])
    if mismatch:
        raise SystemExit(f"version-1 prompts do not reproduce: {mismatch[:5]}")
    return out


# ---------------------------------------------------------------- P1: MNLI
MNLI_CHOICES = ["entailment", "contradiction", "neither"]
MNLI_GOLD = {0: "entailment", 1: "neither", 2: "contradiction"}       # dataset label ids
MNLI_STEM = ("Premise: {p}\nHypothesis: {h}\n\nDoes the premise entail the hypothesis, contradict it, or neither?{rule}\n"
             "(a) entailment\n(b) contradiction\n(c) neither")
MNLI_RULES = {"mnli_plain": "",
              "mnli_strict": (" Choose entailment only if the premise guarantees that the hypothesis is true, and "
                              "contradiction only if the premise guarantees that it is false; otherwise choose neither."),
              "mnli_lenient": (" Choose entailment if the hypothesis is likely true given the premise, and contradiction "
                               "if it is likely false; choose neither only if the premise gives no indication.")}


def mnli() -> List[Dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from GPU_Run.common import env_loader
    path = hf_hub_download("nyu-mll/multi_nli", "data/validation_matched-00000-of-00001.parquet", repo_type="dataset",
                           token=env_loader.get("HUGGINGFACE_TOKEN"))
    df = pd.read_parquet(path)
    df = df[df["label"].isin([0, 1, 2])]
    rng = random.Random(SEED)
    chosen = []
    for lab in (0, 1, 2):
        idx = sorted(df.index[df["label"] == lab].tolist())
        chosen += rng.sample(idx, 200)
    rng.shuffle(chosen)
    out = []
    for i in chosen:
        row = df.loc[i]
        for tag, rule in MNLI_RULES.items():
            out.append({"prompt_id": f"r3-{tag}-{row['pairID']}", "study": "mnli_instruction", "experiment": tag,
                        "item_id": str(row["pairID"]), "genre": row["genre"], "choices": MNLI_CHOICES,
                        "gold_choice_text": MNLI_GOLD[int(row["label"])], "max_new_tokens": 24,
                        "prompt": MNLI_STEM.format(p=row["premise"], h=row["hypothesis"], rule=rule) + SUFFIX3})
    C.write_json(OUT / "mnli_sample_ids.json", {"seed": SEED, "source": "nyu-mll/multi_nli validation_matched",
                                               "pair_ids": [str(df.loc[i]["pairID"]) for i in chosen]})
    print(f"mnli: {len(out)} prompts on {len(chosen)} items", collections.Counter(MNLI_GOLD[int(df.loc[i]['label'])] for i in chosen))
    return out


# ---------------------------------------------------------------- P5: prior-adversarial census comparisons
def adversarial(target: int = 100) -> List[Dict]:
    rows = R.load_validated()
    used = R.used_cells() | {json.loads(l)["source_cell"] for f in
                             ("results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "results_submission1_tmlr/extended_panel.jsonl")
                             for l in (C.CODES_ROOT / f).open()}
    prior = cross_state_prior()
    pool = []
    for r in rows:
        if r["source_cell"] in used or r["condition"] not in ("equal", "diff") or r["axis"] == "gender":
            continue
        key, state = comparison_type(r["source_cell"])
        others = [a for s_, a in prior.get(key, []) if s_ != state]
        if not others:
            continue
        cnt = collections.Counter(others).most_common()
        pick = cnt[0][0] if len(cnt) == 1 or cnt[0][1] > cnt[1][1] else None
        gold = S.EQUAL_CHOICE if r["condition"] == "equal" else r["larger"]
        if pick is not None and pick != gold:
            pool.append({**r, "prior_answer": pick})
    rng = random.Random(SEED)
    social = [r for r in pool if r["axis"] == "social_group"]
    land = [r for r in pool if r["axis"] == "landholding"]
    by_state = collections.defaultdict(list)
    for r in sorted(land, key=lambda r: r["source_cell"]):
        by_state[r["region"]].append(r)
    for v in by_state.values():
        rng.shuffle(v)
    states = sorted(by_state); rng.shuffle(states)
    picked = list(social)
    while len(picked) < target and any(by_state[s] for s in states):
        for s_ in states:
            if len(picked) < target and by_state[s_]:
                picked.append(by_state[s_].pop())
    bad = {r["source_cell"]: _recheck(r) for r in picked}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(f"label recheck failed: {list(bad.items())[:3]}")
    records = [to_record(r, i) for i, r in enumerate(picked)]
    for rec in records:
        rec["fresh_id"] = rec["fresh_id"].replace("ext-", "adv-")
    specs = [S.parse(rec) for rec in records]
    rendered = {sp.fresh_id: S.render(sp) for sp in specs}
    checks = check_panel(specs, rendered)
    if checks["n_failures"]:
        raise SystemExit(f"source checks failed: {checks['failures_by_rule']}")
    # guard against the ext-120 failure mode: no raw field value may appear in a question
    for sp in specs:
        for w in rendered[sp.fresh_id].values():
            assert "femaleShare" not in w and "femaleshare" not in w.lower(), sp.fresh_id
    panel, prompts = [], []
    prior_of = {r["source_cell"]: r["prior_answer"] for r in picked}
    for sp in specs:
        w = rendered[sp.fresh_id]
        choices = [sp.entity1, sp.entity2, S.EQUAL_CHOICE]
        item = {**sp.as_dict(), "population": sp.population(), "denominator_text": sp.denominator(),
                "wording_a": w["wording_a"], "wording_b": w["wording_b"], "choices": choices,
                "gold_choice_text": sp.gold_entity, "prior_answer": prior_of[sp.source_cell]}
        panel.append(item)
        for wording in ("wording_a", "wording_b"):
            prompts.append({"prompt_id": f"r3-adv-{sp.fresh_id}-{wording}", "study": "r1_prior_adversarial",
                            "experiment": "adversarial", "comparison_id": sp.fresh_id, "source_cell": sp.source_cell,
                            "axis": sp.axis, "state": sp.geography, "parent_table": sp.parent_table, "wording": wording,
                            "condition": item["condition"], "choices": choices, "gold_choice_text": sp.gold_entity,
                            "prior_answer": item["prior_answer"], "max_new_tokens": 24,
                            "prompt": f"{w[wording]}\n" + "\n".join(f"({d}) {c}" for d, c in zip("abc", choices)) + SUFFIX3})
    C.write_jsonl(OUT / "adversarial_panel.jsonl", panel)
    print(f"adversarial: pool {len(pool)}, picked {len(panel)} ({collections.Counter(p['axis'] for p in panel)}), "
          f"states {len({p['geography'] for p in panel})}, prompts {len(prompts)}, check failures 0")
    return prompts


# ---------------------------------------------------------------- P6: numeric recall
RECALL_SUFFIX = ('\n\nReply with one JSON object only: {"share_1_percent": <number>, "share_2_percent": <number>}, '
                 'giving your best estimate even if you are unsure.')


def recall() -> List[Dict]:
    out = []
    for rel in ("results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "results_submission1_tmlr/extended_panel.jsonl"):
        for it in C.read_jsonl(C.CODES_ROOT / rel):
            if it["fresh_id"] in EXCLUDED_COMPARISONS:
                continue
            where = "at the all-India level" if it["geography"] == "all-India" else f"in {it['geography']}"
            e1, e2 = it["entity1"], it["entity2"]
            q = (f"According to the 2015-16 Agriculture Census, {where}, among {it['population']}, what percentage of "
                 f"{it['denominator_text']} does {e1} account for (share 1), and what percentage does "
                 f"{e2} account for (share 2)?")
            out.append({"prompt_id": f"r3-recall-{it['fresh_id']}", "study": "r1_numeric_recall", "experiment": "recall",
                        "comparison_id": it["fresh_id"], "source_cell": it["source_cell"], "axis": it["axis"],
                        "state": it["geography"], "condition": it["condition"], "gold_choice_text": it["gold_choice_text"],
                        "entity1": e1, "entity2": e2, "share1_pct": it["share1_pct"],
                        "share2_pct": it["share2_pct"], "choices": [e1, e2, S.EQUAL_CHOICE],
                        "max_new_tokens": 40, "prompt": q + RECALL_SUFFIX})
    print(f"recall: {len(out)} prompts")
    return out


# ---------------------------------------------------------------- P7: entity-name sensitivity
def size_class_name(entity: str) -> str:
    return entity.replace(" operated area", " holdings")


def renamed() -> List[Dict]:
    """Landholding items measured in area: name the compared groups as size classes. Only the two entity
    names change, in the question and in the options; the gold answer is renamed the same way."""
    out = []
    sources = [("results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "verified"),
               ("results_submission1_tmlr/extended_panel.jsonl", "extended"),
               ("results_submission1_tmlr/adversarial_panel.jsonl", "adversarial")]
    for rel, origin in sources:
        for it in C.read_jsonl(C.CODES_ROOT / rel):
            if it["fresh_id"] in EXCLUDED_COMPARISONS or " operated area" not in it["entity1"]:
                continue
            e1, e2 = it["entity1"], it["entity2"]
            n1, n2 = size_class_name(e1), size_class_name(e2)
            choices = [n1, n2, S.EQUAL_CHOICE]
            gold = it["gold_choice_text"]
            gold_n = n1 if gold == e1 else (n2 if gold == e2 else gold)
            for wording in ("wording_a", "wording_b"):
                text = it[wording]
                # replace the exact phrase that lists the two groups ("medium" is inside "semi-medium")
                pat = {"wording_a": (f"— {e1}, {e2}, or", f"— {n1}, {n2}, or"),
                       "wording_b": (f"does {e1} or {e2} hold", f"does {n1} or {n2} hold")}[wording]
                assert text.count(pat[0]) == 1, (it["fresh_id"], wording)
                new = text.replace(pat[0], pat[1])
                assert "share of operated area" in new or "operated area for" in new, it["fresh_id"]
                out.append({"prompt_id": f"r3-renamed-{it['fresh_id']}-{wording}", "study": "r1_entity_renamed",
                            "experiment": "renamed", "origin": origin, "comparison_id": it["fresh_id"],
                            "source_cell": it["source_cell"], "axis": it["axis"], "state": it["geography"],
                            "wording": wording, "condition": it["condition"], "choices": choices,
                            "gold_choice_text": gold_n, "prior_answer": size_class_name(it["prior_answer"]) if it.get("prior_answer") else None,
                            "max_new_tokens": 24,
                            "prompt": f"{new}\n" + "\n".join(f"({d}) {c}" for d, c in zip("abc", choices)) + SUFFIX3})
    print(f"renamed: {len(out)} prompts", dict(collections.Counter(p["origin"] for p in out)))
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"drifted": ("prompts_drifted_v1.jsonl", drifted), "mnli": ("prompts_mnli.jsonl", mnli),
            "adversarial": ("prompts_adversarial.jsonl", adversarial), "recall": ("prompts_recall.jsonl", recall),
            "renamed": ("prompts_renamed.jsonl", renamed)}
    for name, (fname, fn) in jobs.items():
        if which in (name, "all"):
            C.write_jsonl(OUT / fname, fn())


if __name__ == "__main__":
    main()
