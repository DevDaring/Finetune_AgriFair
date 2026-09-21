"""CLI for the AgriFair translation pipeline. Run from Codes/:

    python -m Dataset_Translation.run download                  # HF -> Dataset_Translation/out/source/
    python -m Dataset_Translation.run probe                     # one tiny item through every role, prints which route served it
    python -m Dataset_Translation.run translate --lang hi --component agrifacts [--limit 20]
    python -m Dataset_Translation.run translate --lang bn --component agriadvice
    python -m Dataset_Translation.run status
    python -m Dataset_Translation.run upload                    # README + the four translated files -> Debk/AgriFair

Outputs (per language and component): agrifacts_hi.jsonl etc. with the original schema plus
`language`, and a translation_log_<component>_<lang>.jsonl with every round, reviewer verdict
and model used. Resumable: items already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from GPU_Run.common import env_loader
from Dataset_Translation.pipeline import Cache, Translator
from Dataset_Translation.providers import Clients, ProviderError

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
OUT = HERE.parent / CFG["output_directory"]
SRC = OUT / "source"
_print_lock = threading.Lock()


def _jsonl(p: Path) -> List[Dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def _append(p: Path, row: Dict) -> None:
    with _print_lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ item adapters

def facts_to_item(row: Dict) -> Dict:
    return {"question": row["question"], **{f"choice_{i}": c for i, c in enumerate(row["choices"])}}


def facts_from_item(row: Dict, tr: Dict, lang: str) -> Dict:
    choices = [tr[f"choice_{i}"] for i in range(len(row["choices"]))]
    answer = choices[row["choices"].index(row["answer"])]          # answer is the translated choice at the same index
    return {**row, "question": tr["question"], "choices": choices, "answer": answer, "language": lang,
            "question_en": row["question"], "choices_en": row["choices"], "answer_en": row["answer"]}


def advice_to_item(row: Dict) -> Dict:
    q = row["base_query"]; pa, pb = row["version_A"]["prompt"], row["version_B"]["prompt"]
    if q not in pa or q not in pb:
        raise ValueError(f"{row['pair_id']}: base_query not found verbatim in both prompts")
    return {"base_query": q, "template_A": pa.replace(q, "{q}"), "template_B": pb.replace(q, "{q}"),
            "persona_A": row["version_A"]["persona"], "persona_B": row["version_B"]["persona"]}


def advice_from_item(row: Dict, tr: Dict, lang: str) -> Dict:
    q = tr["base_query"]
    if "{q}" not in tr["template_A"] or "{q}" not in tr["template_B"]:
        raise ValueError("placeholder lost in translation")
    return {**row, "language": lang, "base_query": q,
            "version_A": {"persona": tr["persona_A"], "prompt": tr["template_A"].replace("{q}", q)},
            "version_B": {"persona": tr["persona_B"], "prompt": tr["template_B"].replace("{q}", q)},
            "base_query_en": row["base_query"], "version_A_en": row["version_A"], "version_B_en": row["version_B"]}


ADAPTERS = {"agrifacts": (facts_to_item, facts_from_item, "id"), "agriadvice": (advice_to_item, advice_from_item, "pair_id")}


# ------------------------------------------------------------------ commands

def cmd_download() -> None:
    from huggingface_hub import snapshot_download
    env_loader.load_env(); tok = env_loader.hf_token()
    SRC.mkdir(parents=True, exist_ok=True)
    p = snapshot_download(repo_id=CFG["hf_repo"], repo_type="dataset", token=tok, local_dir=str(SRC),
                          allow_patterns=["*.jsonl", "README.md", ".gitattributes"])
    for f in ("agrifacts.jsonl", "agriadvice.jsonl", "README.md"):
        print(f"  {f}: {'ok' if (SRC / f).exists() else 'MISSING'} ({(SRC / f).stat().st_size if (SRC / f).exists() else 0} bytes)")
    print(f"downloaded {CFG['hf_repo']} -> {p}")


def cmd_probe() -> None:
    clients = Clients(CFG); item = {"question": "In Bihar, which social group operates a larger share of marginal holdings — Scheduled Castes, Scheduled Tribes, or are the two roughly equal?",
                                    "choice_0": "Scheduled Castes", "choice_1": "Roughly equal", "choice_2": "Scheduled Tribes"}
    tr = Translator(CFG, clients, Cache(OUT / "cache_probe.sqlite"))
    for lang in CFG["languages"]:
        res = tr.run(item, lang)
        print(f"\n[{lang}] status={res['status']} verdicts={res['final_verdicts']} glossary_missing={res['glossary_missing']}")
        for r in res["rounds"]:
            print("   ", {k: v for k, v in r.items() if k != "issues"}, (r.get("issues") or "")[:2] if r.get("issues") else "")
        print("    ->", res["translation"])
    print("\ncalls by provider/model:", clients.calls)


def cmd_translate(component: str, lang: str, limit: Optional[int]) -> None:
    to_item, from_item, idkey = ADAPTERS[component]
    rows = _jsonl(SRC / f"{component}.jsonl")
    if not rows:
        raise SystemExit("run `download` first")
    if limit:
        rows = rows[:limit]
    out_path = OUT / f"{component}_{lang}.jsonl"; log_path = OUT / f"translation_log_{component}_{lang}.jsonl"
    done = {r[idkey] for r in _jsonl(out_path)}; todo = [r for r in rows if r[idkey] not in done]
    print(f"{component}/{lang}: {len(done)} done, {len(todo)} to do")
    clients = Clients(CFG); tr = Translator(CFG, clients, Cache(OUT / "cache.sqlite"))
    counts = {"ok": 0, "failed": 0}

    def work(row):
        try:
            res = tr.run(to_item(row), lang)
            if res["status"] != "ok":
                raise ProviderError(res["status"])
            out = from_item(row, res["translation"], lang)
            _append(out_path, out)
            _append(log_path, {idkey: row[idkey], "rounds": res["rounds"], "final_verdicts": res["final_verdicts"],
                               "glossary_missing": res["glossary_missing"]})
            return "ok"
        except Exception as e:
            _append(log_path, {idkey: row[idkey], "status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return "failed"

    with ThreadPoolExecutor(max_workers=int(CFG["concurrency"])) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            counts[f.result()] += 1
            if i % 25 == 0 or i == len(todo):
                with _print_lock:
                    print(f"  {i}/{len(todo)}  ok={counts['ok']} failed={counts['failed']}  calls={sum(clients.calls.values())}", flush=True)
    print("done:", counts, "| calls by model:", clients.calls)


def cmd_status() -> None:
    for comp in ADAPTERS:
        n_src = len(_jsonl(SRC / f"{comp}.jsonl"))
        for lang in CFG["languages"]:
            out = _jsonl(OUT / f"{comp}_{lang}.jsonl"); log = _jsonl(OUT / f"translation_log_{comp}_{lang}.jsonl")
            ok_all = sum(1 for l in log if l.get("final_verdicts") and all(v == "OK" for v in l["final_verdicts"].values()))
            miss = sum(1 for l in log if l.get("glossary_missing"))
            failed = sum(1 for l in log if l.get("status") == "failed")
            print(f"{comp:10s} {lang}: {len(out)}/{n_src} translated | all reviewers OK: {ok_all} | glossary term missing: {miss} | failed: {failed}")


def cmd_upload() -> None:
    from huggingface_hub import HfApi
    from Dataset_Translation.readme import updated_readme
    env_loader.load_env(); tok = env_loader.hf_token(); api = HfApi(token=tok)
    files = [f"{c}_{l}.jsonl" for c in ADAPTERS for l in CFG["languages"]]
    missing = [f for f in files if not (OUT / f).exists()]
    if missing:
        raise SystemExit(f"not ready, missing: {missing}")
    for comp in ADAPTERS:
        n = len(_jsonl(SRC / f"{comp}.jsonl"))
        for l in CFG["languages"]:
            m = len(_jsonl(OUT / f"{comp}_{l}.jsonl"))
            if m != n:
                raise SystemExit(f"{comp}_{l}: {m} rows but source has {n}; finish translation first")
    readme = updated_readme((SRC / "README.md").read_text(encoding="utf-8"), CFG, OUT)
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    ops = [(str(OUT / f), f) for f in files] + [(str(OUT / "README.md"), "README.md")]
    for local, remote in ops:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=CFG["hf_repo"], repo_type="dataset",
                        commit_message=f"Add {remote}" if remote != "README.md" else "README: Hindi and Bengali translations")
        print("  uploaded", remote)
    print(f"https://huggingface.co/datasets/{CFG['hf_repo']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download"); sub.add_parser("probe"); sub.add_parser("status"); sub.add_parser("upload")
    t = sub.add_parser("translate"); t.add_argument("--lang", required=True, choices=list(CFG["languages"]))
    t.add_argument("--component", required=True, choices=list(ADAPTERS)); t.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    {"download": cmd_download, "probe": cmd_probe, "status": cmd_status, "upload": cmd_upload,
     "translate": lambda: cmd_translate(a.component, a.lang, a.limit)}[a.cmd]()


if __name__ == "__main__":
    main()
