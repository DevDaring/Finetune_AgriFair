"""P2 (round 3): a second structured source, built with the same method as the census benchmark.

    python -m Submission1_TMLR.build_wdi

Source: World Bank World Development Indicators, modeled ILO estimates for 2019, employment in
agriculture, industry and services as a share of total, female and male employment. The raw API
responses are cached with their SHA-256 (results_submission1_tmlr/wdi/raw), so the build is
reproducible offline. Aggregates (regions, income groups) are excluded.

Outputs in results_submission1_tmlr/:
  wdi/wdi_panel.jsonl        150 source records (one per country; 50 equal, 100 difference)
  prompts_wdi.jsonl          standard (300) + no-rule (300) + supplied-number set (288)
  wdi/checker_summary.json   ported checks on every question, and a planted-error test
  wdi/prior_answers.json     leave-one-country-out regional prior, and constant baselines
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import urllib.request
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

OUT = C.CODES_ROOT / "results_submission1_tmlr"
WDI = OUT / "wdi"
YEAR = 2019
SEED = 20261007
SECTORS = {"AGR": "agriculture", "IND": "industry", "SRV": "services"}
POPS = {"ZS": ("all", "employed people"), "FE.ZS": ("women", "employed women"), "MA.ZS": ("men", "employed men")}
PAIRS = [("AGR", "IND"), ("AGR", "SRV"), ("IND", "SRV")]
EQUAL = "Roughly equal"
RULE = ("Treat the two as roughly equal if their shares differ by less than 5 percentage points, "
        "and treat one as larger only if it leads by at least 10 percentage points.")
SOURCE = f"the World Bank's World Development Indicators (modeled ILO estimates for {YEAR})"
SUFFIX = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'


# ---------------------------------------------------------------- fetch with cache
def fetch(url: str, name: str):
    WDI.joinpath("raw").mkdir(parents=True, exist_ok=True)
    p = WDI / "raw" / name
    if not p.exists():
        with urllib.request.urlopen(url, timeout=60) as f:
            p.write_bytes(f.read())
    data = p.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def load():
    countries, h = fetch("https://api.worldbank.org/v2/country?format=json&per_page=400", "countries.json")
    meta = {c["id"]: {"name": c["name"], "region": c["region"]["value"]} for c in countries[1]
            if c["region"]["value"] != "Aggregates"}
    hashes = {"countries.json": h}
    values = collections.defaultdict(dict)
    for sec in SECTORS:
        for pop in POPS:
            code = f"SL.{sec}.EMPL.{pop}"
            d, h = fetch(f"https://api.worldbank.org/v2/country/all/indicator/{code}?date={YEAR}&format=json&per_page=400",
                         f"{code}.json")
            hashes[f"{code}.json"] = h
            for r in d[1]:
                iso = r["countryiso3code"]
                if r["value"] is not None and iso in meta:
                    values[iso][(sec, pop)] = float(r["value"])
    full = {iso: v for iso, v in values.items() if len(v) == 9}
    return meta, full, hashes


def label(v1: float, v2: float) -> str:
    gap = abs(v1 - v2)
    return "equal" if gap < 5 else ("diff" if gap >= 10 else "excluded")


# ---------------------------------------------------------------- records and wordings
def wording_a(r: Dict, rule: bool = True) -> str:
    return (f"In {r['country']}, according to {SOURCE}, among {r['population_phrase']}, which sector accounts for a larger "
            f"share of employment, measured against total employment in that population — {r['entity1']}, "
            f"{r['entity2']}, or are the two roughly equal?" + (f" {RULE}" if rule else ""))


def wording_b(r: Dict, rule: bool = True) -> str:
    return (f"{SOURCE[0].upper() + SOURCE[1:]} report employment by sector for {r['population_phrase']} in {r['country']}. "
            f"Taking total employment in that population as the base, does {r['entity1']} or {r['entity2']} hold the "
            f"larger share, or do the two stand roughly level?" + (f" {RULE}" if rule else ""))


def build_records(meta, full) -> List[Dict]:
    cands = collections.defaultdict(list)
    for iso, v in full.items():
        for pop, (_, phrase) in POPS.items():
            for a, b in PAIRS:
                s1, s2 = v[(a, pop)], v[(b, pop)]
                lab = label(s1, s2)
                if lab == "excluded":
                    continue
                cands[iso].append({"country_iso3": iso, "country": meta[iso]["name"], "region": meta[iso]["region"],
                                   "population": POPS[pop][0], "population_phrase": phrase,
                                   "entity1": SECTORS[a], "entity2": SECTORS[b],
                                   "indicator1": f"SL.{a}.EMPL.{pop}", "indicator2": f"SL.{b}.EMPL.{pop}",
                                   "share1_pct": round(s1, 4), "share2_pct": round(s2, 4), "gap_pp": round(abs(s1 - s2), 4),
                                   "condition": lab,
                                   "gold_choice_text": EQUAL if lab == "equal" else (SECTORS[a] if s1 > s2 else SECTORS[b])})
    rng = random.Random(SEED)
    isos = sorted(cands); rng.shuffle(isos)
    chosen, n_eq, n_diff = [], 0, 0
    for iso in isos:                                   # one comparison per country; fill equal first
        eq = [c for c in cands[iso] if c["condition"] == "equal"]
        df = [c for c in cands[iso] if c["condition"] == "diff"]
        if n_eq < 50 and eq:
            chosen.append(rng.choice(eq)); n_eq += 1
        elif n_diff < 100 and df:
            chosen.append(rng.choice(df)); n_diff += 1
        if n_eq == 50 and n_diff == 100:
            break
    for i, r in enumerate(sorted(chosen, key=lambda r: r["country"])):
        r["fresh_id"] = f"wdi-{i:03d}"
        r["wording_a"], r["wording_b"] = wording_a(r), wording_b(r)
    return sorted(chosen, key=lambda r: r["fresh_id"]), cands


# ---------------------------------------------------------------- ported checks and planted errors
OTHER_POP = {"employed women": "employed men", "employed men": "employed women", "employed people": "employed women"}


def check(r: Dict, text: str, rule: bool = True) -> List[str]:
    # collapse whitespace first, as the census checker does: a reflowed sentence is not a source mismatch
    # (the first ported version lacked this, and the planted-error test flagged every whitespace control)
    text = " ".join(text.split())
    f = []
    if r["country"] not in text:
        f.append("country_missing")
    if f"among {r['population_phrase']}" not in text and f"for {r['population_phrase']} in" not in text:
        f.append("population_missing")
    for other in set(OTHER_POP.values()) | {"employed people"}:
        if other != r["population_phrase"] and other in text:
            f.append("other_population_present")
    if "share of employment" not in text and "employment by sector" not in text:
        f.append("measure_missing")
    if "total employment in that population" not in text:
        f.append("total_missing")
    i1, i2 = text.find(r["entity1"]), text.find(r["entity2"])
    if i1 < 0 or i2 < 0:
        f.append("entity_missing")
    elif i1 > i2:
        f.append("entity_order")
    if rule and text.count(RULE) != 1:
        f.append("rule_missing")
    if re.search(r"SL\.|\.ZS|FE\b|MA\b", text):
        f.append("raw_field_value")
    s1, s2 = r["share1_pct"], r["share2_pct"]
    if label(s1, s2) != r["condition"] or (r["condition"] == "diff" and r["gold_choice_text"] != (r["entity1"] if s1 > s2 else r["entity2"])):
        f.append("label_does_not_follow")
    return f


def mutations(r: Dict, text: str, others: List[str]):
    pop = r["population_phrase"]
    yield "population_substituted", text.replace(pop, OTHER_POP[pop]), True
    if pop != "employed people":
        yield "population_deleted", text.replace(f"among {pop}", "among employed people").replace(f"for {pop} in", "for employed people in"), True
    yield "wrong_country", text.replace(r["country"], others[0]), True
    yield "entity_order_swapped", text.replace(r["entity1"], "@@").replace(r["entity2"], r["entity1"]).replace("@@", r["entity2"]), True
    yield "rule_removed", text.replace(" " + RULE, ""), True
    yield "wrong_measure", text.replace("share of employment", "share of output").replace("employment by sector", "output by sector"), True
    yield "total_swapped", text.replace("total employment in that population", "total population"), True
    yield "whitespace", text.replace(", ", ",  "), False
    yield "polite_prefix", "Please answer carefully. " + text, False


def planted_test(records: List[Dict]) -> Dict:
    names = sorted({r["country"] for r in records})
    by = collections.defaultdict(lambda: [0, 0, None])
    for r in records:
        others = [n for n in names if n != r["country"] and n not in r["country"] and r["country"] not in n]
        for w in ("wording_a", "wording_b"):
            for t, mutated, should in mutations(r, r[w], others):
                if mutated == r[w]:
                    continue
                e = by[t]; e[0] += 1; e[1] += bool(check(r, mutated)); e[2] = should
    cor = {t: v for t, v in by.items() if v[2]}; ctl = {t: v for t, v in by.items() if not v[2]}
    return {"by_edit": {t: {"n": n, "flagged": round(f / n, 4), "kind": "corruption" if s else "valid control"} for t, (n, f, s) in by.items()},
            "corrupted": sum(v[0] for v in cor.values()), "caught": sum(v[1] for v in cor.values()),
            "controls": sum(v[0] for v in ctl.values()), "false_alarms": sum(v[1] for v in ctl.values())}


# ---------------------------------------------------------------- prior and supplied-number set
def regional_prior(records: List[Dict], cands) -> Dict[str, str]:
    allc = [c for lst in cands.values() for c in lst]
    out = {}
    for r in records:
        same = [c for c in allc if c["country_iso3"] != r["country_iso3"] and c["population"] == r["population"]
                and c["entity1"] == r["entity1"] and c["entity2"] == r["entity2"]]
        reg = [c for c in same if c["region"] == r["region"]] or same
        cnt = collections.Counter(c["gold_choice_text"] for c in reg).most_common()
        out[r["fresh_id"]] = cnt[0][0] if len(cnt) == 1 or cnt[0][1] > cnt[1][1] else EQUAL
    return out


def numerical(records: List[Dict]) -> List[Dict]:
    rng = random.Random(SEED + 1)
    frames = rng.sample(records, 48)
    out = []
    for k, r in enumerate(frames):
        choices = [r["entity1"], r["entity2"], EQUAL]
        rng.shuffle(choices)
        for rel in ("first_higher", "second_higher", "approximately_equal"):
            base = round(rng.uniform(30, 65), 1)
            if rel == "approximately_equal":
                v1, v2 = base, round(base + rng.choice([-1, 1]) * rng.uniform(0.3, 3.0), 1)
            else:
                lo = round(base - rng.uniform(12, 30), 1)
                v1, v2 = (base, lo) if rel == "first_higher" else (lo, base)
            gold = EQUAL if rel == "approximately_equal" else (r["entity1"] if rel == "first_higher" else r["entity2"])
            table = ("HYPOTHETICAL TABLE (illustrative values, not World Bank observations). Answer only about this table.\n\n"
                     "| sector | share of employment in that population (0-100) |\n|---|---|\n"
                     f"| {r['entity1']} | {v1} |\n| {r['entity2']} | {v2} |\n"
                     f"(population: {r['population_phrase']}, {r['country']}; base: total employment in that population)\n\n"
                     "Decision rule: if the two shares differ by less than 5 percentage points, they are roughly equal; if they "
                     "differ by 10 percentage points or more, the larger share wins. Values between those are excluded and do "
                     "not appear here.\n\n")
            stems = {"context_rich": f"In {r['country']}, according to this table, which sector has the larger share of employment "
                                     f"among {r['population_phrase']} — {r['entity1']}, {r['entity2']}, or are the two roughly equal?",
                     "plain": f"Using only the table above, which row has the larger value — {r['entity1']}, {r['entity2']}, "
                              f"or are the two roughly equal?"}
            for wording, stem in stems.items():
                out.append({"prompt_id": f"r3-wdinum-B{k:02d}-{rel}-{wording}", "study": "wdi_numerical", "experiment": "wdi_numerical",
                            "bundle_id": f"WB{k:02d}", "country": r["country"], "relation": rel, "wording": wording,
                            "group1": r["entity1"], "group2": r["entity2"], "value1": v1, "value2": v2, "choices": choices,
                            "gold_choice_text": gold, "max_new_tokens": 24,
                            "prompt": table + stem + "\n" + "\n".join(f"({d}) {c}" for d, c in zip("abc", choices)) + SUFFIX})
    return out


def main() -> None:
    meta, full, hashes = load()
    records, cands = build_records(meta, full)
    fails = {r["fresh_id"]: check(r, r[w]) for r in records for w in ("wording_a", "wording_b") if check(r, r[w])}
    if fails:
        raise SystemExit(f"checks failed on built questions: {list(fails.items())[:3]}")
    planted = planted_test(records)
    prior = regional_prior(records, cands)
    prompts = []
    for r in records:
        choices = [r["entity1"], r["entity2"], EQUAL]
        opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", choices))
        for w, fn in (("wording_a", wording_a), ("wording_b", wording_b)):
            for exp, rule in (("wdi_standard", True), ("wdi_norule", False)):
                prompts.append({"prompt_id": f"r3-{exp}-{r['fresh_id']}-{w}", "study": "wdi_comparisons", "experiment": exp,
                                "comparison_id": r["fresh_id"], "country": r["country"], "region": r["region"],
                                "population": r["population"], "wording": w, "condition": r["condition"],
                                "choices": choices, "gold_choice_text": r["gold_choice_text"], "prior_answer": prior[r["fresh_id"]],
                                "max_new_tokens": 24, "prompt": f"{fn(r, rule)}\n{opts}{SUFFIX}"})
    num = numerical(records)
    WDI.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(WDI / "wdi_panel.jsonl", records)
    C.write_jsonl(OUT / "prompts_wdi.jsonl", prompts + num)
    gold = {r["fresh_id"]: r["gold_choice_text"] for r in records}
    base = {"regional_prior": prior, "accuracy": {
        "regional_prior": round(sum(prior[k] == gold[k] for k in gold) / len(gold), 4),
        "constant_equal": round(sum(g == EQUAL for g in gold.values()) / len(gold), 4),
        "constant_first": round(sum(g == r["entity1"] for r in records for g in [gold[r["fresh_id"]]]) / len(gold), 4),
        "constant_second": round(sum(g == r["entity2"] for r in records for g in [gold[r["fresh_id"]]]) / len(gold), 4)}}
    C.write_json(WDI / "prior_answers.json", base)
    C.write_json(WDI / "checker_summary.json", {"built_question_failures": 0, "planted": planted, "raw_sha256": hashes,
                                                "countries_with_all_9_values": len(full)})
    print(f"records {len(records)} {dict(collections.Counter(r['condition'] for r in records))}; "
          f"countries with data {len(full)}; regions {len({r['region'] for r in records})}")
    print(f"prompts: standard+norule {len(prompts)}, numerical {len(num)}")
    print("planted:", {k: planted[k] for k in ('corrupted', 'caught', 'controls', 'false_alarms')})
    print("baselines:", base["accuracy"])


if __name__ == "__main__":
    main()
