"""E1-E7: send every prompt through every model x method x repeat, with caching and a spend cap.

    --stage facts | advice | reference | routes | all
    --repeat N            which repeat this run is (1..repeats); run on different days
    --smoke               5 items per set, offline FakeRouter, output to results_submission2_SMOKE
    --methods a,b         restrict methods; --models t1,t2 restrict models; --langs en,hi

Output rows (jsonl, one per prompt) carry everything analysis needs: model tier, model_id as
served, route(s), method, language, condition, tokens in/out, latency, cost, timestamp, and the
raw text. Nothing is post-processed here beyond letter extraction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from GPU_Run.common.parsing import extract_answer_letter
from Submission2_Run import common as C
from Submission2_Run import methods as M
from Submission2_Run.cache import ResponseCache
from Submission2_Run.providers import FakeRouter, Router


class Runner:
    def __init__(self, cfg: Dict, smoke: bool, models: List[str], methods: List[str], langs: List[str], repeat: int):
        self.cfg, self.smoke, self.repeat = cfg, smoke, repeat
        self.models, self.methods, self.langs = models, methods, langs
        self.out = C.CODES_ROOT / (cfg["output_directory"] + ("_SMOKE" if smoke else "")); self.out.mkdir(exist_ok=True)
        self.data = C.data_dir(cfg)
        self.cache = ResponseCache(self.out / "cache" / "responses.sqlite")
        if smoke and cfg["smoke"]["fake_router"]:
            self.router = FakeRouter()
        else:
            if not cfg["api_enabled"]:
                raise SystemExit("api_enabled is false in config.yaml; flip it deliberately before a paid run")
            self.router = Router()
        self.spent_usd = 0.0

    # ---------------------------------------------------------------- one prompt, cached
    def ask(self, tier: str, method: str, prompt: str, max_tokens: int, is_mcq: bool,
            force_route: Optional[str] = None) -> Dict:
        key = C.prompt_key(tier, method, prompt, max_tokens, self.repeat, force_route or "")
        hit = self.cache.get(key)
        if hit:
            return {"text": hit["text"], "input_tokens": hit["input_tokens"], "output_tokens": hit["output_tokens"],
                    "latency_seconds": hit["latency_seconds"], "route": hit["route"], "model_id": hit["model_id"],
                    "calls": hit["extra"].get("calls", 1), "cached": True, "any_failed": False}

        def _ask(p, mt, t):
            return self.router.ask(tier, p, mt, temperature=t, force_route=force_route)
        text, turns = M.run_method(method, _ask, prompt, max_tokens, self.cfg, is_mcq=is_mcq)
        u = M.turns_usage(turns)
        usd = C.cost_usd(self.cfg, tier, u["input_tokens"], u["output_tokens"]) if not self.smoke else 0.0
        self.spent_usd += usd
        if self.spent_usd > float(self.cfg["budget_usd_cap"]):
            raise SystemExit(f"budget cap {self.cfg['budget_usd_cap']} USD reached; stopping (cache is intact)")
        if not u["any_failed"]:
            self.cache.put(key, tier, method, self.repeat, u["route"], u["model_id"], text, u["input_tokens"],
                           u["output_tokens"], u["latency_seconds"], {"calls": u["calls"], "force_route": force_route})
        return {"text": text, **u, "cached": False}

    def _row(self, base: Dict, tier: str, method: str, r: Dict) -> Dict:
        usd = C.cost_usd(self.cfg, tier, r["input_tokens"], r["output_tokens"])
        return {**base, "tier": tier, "method": method, "repeat": self.repeat, "route": r["route"], "model_id": r["model_id"],
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "latency_seconds": round(r["latency_seconds"], 3),
                "calls": r["calls"], "cost_usd": round(usd, 6), "cached": r["cached"], "failed": r["any_failed"], "raw_text": r["text"]}

    def _limit(self, rows: List[Dict]) -> List[Dict]:
        return rows[: self.cfg["smoke"]["items_per_set"]] if self.smoke else rows

    # ---------------------------------------------------------------- E1 + E2
    def stage_facts(self) -> None:
        items = self._limit(C.read_jsonl(self.data / "facts_items.jsonl"))
        conds = self.cfg["facts_conditions"]; rows = []
        for tier in self.models:
            for method in self.methods:
                use_conds = conds if method == "evidence" else ["no_evidence"]
                if method == "evidence":
                    use_conds = [c for c in conds if c != "no_evidence"]
                for lang in self.langs:
                    for it in items:
                        q = it.get(f"question_{lang}") or (it["question_en"] if lang == "en" else "")
                        if not q:
                            continue
                        for cond in use_conds:
                            table = {"verified_evidence": it["verified_table"], "hypothetical_evidence": it["hypothetical_table"],
                                     "anonymised_evidence": it["anonymised_table"]}.get(cond)
                            prompt, d2c = M.mcq_prompt({**it, "question": q}, table)
                            r = self.ask(tier, method, prompt, self.cfg["answer_max_tokens"], True)
                            disp, ok = extract_answer_letter(r["text"])
                            pred = d2c.get(disp) if ok else None
                            gold = it["hypothetical_gold"] if cond == "hypothetical_evidence" else it["gold"]
                            rows.append(self._row({"item_id": it["item_id"], "language": lang, "condition": cond, "axis": it["axis"],
                                                   "source_cell": it["source_cell"], "gold": gold, "pred": pred, "parse_ok": ok}, tier, method, r))
        C.write_jsonl(self.out / f"facts_predictions_r{self.repeat}.jsonl", rows); print(f"facts: {len(rows)} rows")

    # ---------------------------------------------------------------- E3
    def stage_advice(self) -> None:
        pairs = self._limit(C.read_jsonl(self.data / "advice_pairs.jsonl")); rows = []
        for tier in self.models:
            for method in [m for m in self.methods if m not in ("evidence", "self_consistency")]:
                for p in pairs:
                    if p["language"] not in self.langs:
                        continue
                    outs = {}
                    for side in ("A", "B"):
                        r = self.ask(tier, method, M.advice_prompt(p[f"prompt_{side}"]), self.cfg["advice_max_tokens"], False)
                        outs[side] = r
                        rows.append(self._row({"pair_id": p["pair_id"], "query_id": p["query_id"], "side": side, "language": p["language"],
                                               "toggle_axis": p["toggle_axis"], "identity": p[f"identity_{side}"]}, tier, method, r))
        C.write_jsonl(self.out / f"advice_predictions_r{self.repeat}.jsonl", rows); print(f"advice: {len(rows)} rows")

    # ---------------------------------------------------------------- E4 (plain answers vs FTA reference, for raters)
    def stage_reference(self) -> None:
        qs = self._limit(C.read_jsonl(self.data / "kcc_queries.jsonl")); rows = []
        for tier in self.models:
            for lang in self.langs:
                for q in qs:
                    text = q.get(f"question_{lang}")
                    if not text:
                        continue
                    r = self.ask(tier, "plain", M.advice_prompt(text), self.cfg["advice_max_tokens"], False)
                    rows.append(self._row({"query_id": q["query_id"], "language": lang, "query_type": q["query_type"],
                                           "reference_answer": q["reference_answer"]}, tier, "plain", r))
        C.write_jsonl(self.out / f"reference_answers_r{self.repeat}.jsonl", rows); print(f"reference: {len(rows)} rows")

    # ---------------------------------------------------------------- E7 route consistency
    def stage_routes(self) -> None:
        items = self._limit(C.read_jsonl(self.data / "facts_items.jsonl"))[: self.cfg["route_panel_items"]]; rows = []
        for tier in self.models:
            for route in self.router.available_routes(tier):
                for it in items:
                    prompt, d2c = M.mcq_prompt({**it, "question": it["question_en"]})
                    r = self.ask(tier, "plain", prompt, self.cfg["answer_max_tokens"], True, force_route=route)
                    disp, ok = extract_answer_letter(r["text"])
                    rows.append(self._row({"item_id": it["item_id"], "forced_route": route, "gold": it["gold"],
                                           "pred": d2c.get(disp) if ok else None, "parse_ok": ok}, tier, "plain", r))
        C.write_jsonl(self.out / f"route_panel_r{self.repeat}.jsonl", rows); print(f"routes: {len(rows)} rows")

    def finish(self) -> None:
        C.write_json(self.out / f"run_manifest_r{self.repeat}.json",
                     {"models": self.models, "methods": self.methods, "languages": self.langs, "repeat": self.repeat,
                      "smoke": self.smoke, "cached_responses": self.cache.count(), "spend_tokens": self.cache.spend_tokens(),
                      "spent_usd_this_run": round(self.spent_usd, 4), "tariff_date": self.cfg["tariffs"]["tariff_date"],
                      "routes": getattr(self.router, "usage_summary", lambda: {})()})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["facts", "advice", "reference", "routes", "all"])
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models"); ap.add_argument("--methods"); ap.add_argument("--langs")
    a = ap.parse_args(argv); cfg = C.load_config()
    pick = lambda s, default: [x.strip() for x in s.split(",")] if s else list(default)
    run = Runner(cfg, a.smoke, pick(a.models, cfg["models"]), pick(a.methods, cfg["methods"]), pick(a.langs, cfg["languages"]), a.repeat)
    stages = {"facts": run.stage_facts, "advice": run.stage_advice, "reference": run.stage_reference, "routes": run.stage_routes}
    for name, fn in stages.items():
        if a.stage in ("all", name):
            fn()
    run.finish()


if __name__ == "__main__":
    main()
