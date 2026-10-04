"""Anonymous supplementary material for the TMLR submission.

    python -m Submission1_TMLR.make_supplement

Writes Submission1/TMLR_supplementary.zip with the code needed to rerun every analysis, the prompt
files, the gzipped model outputs, the analysis tables and the written plan. Identifying strings are
replaced, and rater-related modules, credentials and logs are left out. The script refuses to write
the zip if any blocked string survives.
"""
from __future__ import annotations

import gzip
import io
import re
import zipfile
from pathlib import Path

CODES = Path(__file__).resolve().parents[1]
DEST = CODES.parent / "Submission1" / "TMLR_supplementary.zip"
CODE_DIRS = ["Submission1_TMLR", "Submission1_Code_Phase2", "Submission1_DKE_Repair", "GPU_Run/common", "Next_Run"]
SKIP_FILES = {"human_pack.py", "rater_protocol_pack.py", "watch_vast.py", "make_supplement.py"}
DATA_FILES = ["results_submission1_dke_repair_v2/r1_corrected_panel.jsonl",
              "results_submission1_dke_repair_v2/prompts_e1_r1_corrected.jsonl",
              "results_submission1_dke_repair_v2/prompts_e5_option_permuted.jsonl",
              "results_submission1_dke_repair_v2/prompts_e3_diagnostic_clean.jsonl",
              "results_submission1_dke_repair_v2/prompts_e4_neutral.jsonl",
              "results_submission1_phase2/evidence/r2_main_prompts.jsonl",
              "results_submission1_phase2/evidence/r2_diagnostic_prompts.jsonl",
              # published outputs of the four original systems, for the reproduction check
              "results_submission1_dke_repair_v2/v2_predictions.jsonl",
              "results_submission1_phase2/predictions/main_predictions.jsonl",
              "results_submission1_phase2/predictions/pilot_predictions.jsonl",
              # validated census cells (public census data), needed for the cross-state prior baseline
              "Source_Records/extracted/Dataset/data/interim/agrifacts_facts.csv"]
_SECRETS = r"AKIA[0-9A-Z]{12}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
# identifying terms (names, handles, paths) are kept in a local, untracked file so the code itself names no one
_TERMS = CODES / "Submission1_TMLR" / ".anonymity_terms"
_extra = [t.strip() for t in _TERMS.read_text().splitlines() if t.strip()] if _TERMS.exists() else []
BLOCKED = re.compile("|".join([_SECRETS] + [re.escape(t) for t in _extra]), re.I)
_HANDLE = (_extra[0] if _extra else "ANONYMOUS")
SCRUB = [(re.compile(re.escape(_HANDLE) + r"/AgriFair-GRAFT-adapters"), "ANONYMOUS/adapters"),
         (re.compile(re.escape(_HANDLE) + r"/AgriFair"), "ANONYMOUS/AgriFair"),
         (re.compile(r"/home/[A-Za-z0-9_]+/"), "/home/USER/")]


def scrub(text: str) -> str:
    for pat, rep in SCRUB:
        text = pat.sub(rep, text)
    return text


def main() -> None:
    out = io.BytesIO()
    leaks = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def add_text(arc: str, text: str):
            text = scrub(text)
            if BLOCKED.search(text):
                leaks.append(arc)
            z.writestr(arc, text)

        for d in CODE_DIRS:
            for p in sorted((CODES / d).rglob("*")):
                if p.is_file() and p.suffix in (".py", ".sh", ".yaml", ".yml") and p.name not in SKIP_FILES \
                        and "__pycache__" not in p.parts:
                    add_text(f"code/{p.relative_to(CODES)}", p.read_bytes().decode("utf-8"))
        for f in DATA_FILES:
            add_text(f"code/{f}", (CODES / f).read_bytes().decode("utf-8"))
        res = CODES / "results_submission1_tmlr"
        for p in sorted(res.glob("*")):
            if not p.is_file() or p.suffix not in (".jsonl", ".json") or "smoke" in p.name or "check" in p.name \
                    or "pilot" in p.name:
                continue
            arc = f"code/results_submission1_tmlr/{p.name}"
            text = scrub(p.read_bytes().decode("utf-8"))
            if BLOCKED.search(text):
                leaks.append(arc)
            if p.suffix == ".jsonl" and p.stat().st_size > 2_000_000:
                z.writestr(arc + ".gz", gzip.compress(text.encode("utf-8")))
            else:
                z.writestr(arc, text)
        for sub in ("analysis", "analysis_round3", "analysis_round3/budget256", "agrifacts_audit", "checker_challenge_154",
                    "wdi", "wdi/raw"):
            for p in sorted((res / sub).glob("*")):
                if p.is_file() and p.suffix in (".csv", ".json", ".jsonl", ".log"):
                    add_text(f"code/results_submission1_tmlr/{sub}/{p.name}", p.read_bytes().decode("utf-8"))
        add_text("PLAN.md", (CODES.parent / "Submission1" / "TMLR_Research_Plan.md").read_bytes().decode("utf-8"))
        add_text("README.txt", (
            "Supplementary material (anonymous).\n\n"
            "code/                       analysis, inference and checking code; run from code/ with PYTHONPATH=.\n"
            "code/results_submission1_tmlr/  prompts, model outputs (.jsonl.gz), manifests and analysis tables\n"
            "PLAN.md                     the written plan, with its dated log of changes\n\n"
            "Reproduce the tables (the .gz files are read directly):  cd code;\n"
            "  PYTHONPATH=. python -m Submission1_TMLR.analyse && PYTHONPATH=. python -m Submission1_TMLR.make_tables\n"
            "The adapters are identified in the manifests and will be released after review.\n"))
    if leaks:
        raise SystemExit(f"blocked strings remain in: {leaks}")
    DEST.write_bytes(out.getvalue())
    print(DEST, round(DEST.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
