"""GPU generation for the TMLR extension (plan: Submission1/TMLR_Research_Plan.md, section 3).

    python -m Submission1_TMLR.run_gpu --smoke                 # offline: fake answers, no GPU, no weights
    python -m Submission1_TMLR.run_gpu --pilot 20              # 20 prompts per system, with timings
    python -m Submission1_TMLR.run_gpu                         # the full design, resumable

Nothing here changes how a system is run. Loading, the chat template, greedy decoding, the JSON
letter parser and scoring are imported unchanged from the published pipeline
(Submission1_Code_Phase2.run_inference, GPU_Run.common.model_registry), so a re-run of an
original system can be compared output for output with the published one.

Speed, without changing any output:
- FlashAttention-2 from a prebuilt wheel (falls back to sdpa, recorded on every row);
- prompts sorted by length and batched with left padding;
- one base-model load per family, with each adapter attached and detached in turn;
- the batch size halves on CUDA out-of-memory.

Adapters are read from Codes/checkpoints when present, otherwise from the Hugging Face repository
Debk/AgriFair-GRAFT-adapters (byte-identical; verified 4 October 2026).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2.run_inference import FakeBackend, TorchBackend, score_row

OUT_DIR = "results_submission1_tmlr"
HF_ADAPTERS = "Debk/AgriFair-GRAFT-adapters"
PLAN = C.CODES_ROOT.parent / "Submission1" / "TMLR_Research_Plan.md"

FAMILIES = ["small-instruct", "broad-instruct", "general-instruct-2", "general-instruct"]   # small first
ARMS = [("frozen_base", 42),
        ("graft_proposed", 42), ("graft_proposed", 43), ("graft_proposed", 44),
        ("reference_vanilla_lora", 42), ("reference_vanilla_lora", 43), ("reference_vanilla_lora", 44),
        ("ablation_placement_random", 42)]
BATCH = {"small-instruct": 32, "broad-instruct": 32, "general-instruct-2": 24, "general-instruct": 16}

PROMPT_SETS = [  # (path relative to Codes, experiment tag)
    ("results_submission1_dke_repair_v2/prompts_e1_r1_corrected.jsonl", "verified"),
    ("results_submission1_dke_repair_v2/prompts_e5_option_permuted.jsonl", "reordered"),
    ("results_submission1_tmlr/prompts_r1_extended.jsonl", "extended"),
    ("results_submission1_phase2/evidence/r2_main_prompts.jsonl", "numerical"),
    ("results_submission1_phase2/evidence/r2_diagnostic_prompts.jsonl", "altered"),
    ("results_submission1_dke_repair_v2/prompts_e3_diagnostic_clean.jsonl", "unchanged"),
    ("results_submission1_dke_repair_v2/prompts_e4_neutral.jsonl", "neutral"),
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


FOLLOWUPS = "results_submission1_tmlr/prompts_followups.jsonl"   # norule, abstain, realtable (tags in rows)
COT = "results_submission1_tmlr/prompts_cot.jsonl"                 # step-by-step reasoning (core systems only)
CORE_ARMS = [("frozen_base", 42), ("graft_proposed", 42)]
ALTERED48 = "results_submission1_tmlr/prompts_altered48.jsonl"     # altered tables on the 36 remaining scenarios
DRAW_ARMS = [("ablation_placement_random_draw2", 42), ("ablation_placement_random_draw3", 42)]   # added 4 Oct 2026


def load_prompts(which: str = "main") -> List[Dict]:
    """main: the seven pre-registered sets; followups: the three conditions added on 4 Oct 2026."""
    out = []
    if which in ("main", "all"):
        for rel, tag in PROMPT_SETS:
            rows = list(C.read_jsonl(C.CODES_ROOT / rel))
            out += [{**r, "experiment": tag} for r in rows]
    if which in ("followups", "all"):
        out += list(C.read_jsonl(C.CODES_ROOT / FOLLOWUPS))
    if which == "cot":
        out += list(C.read_jsonl(C.CODES_ROOT / COT))
    if which in ("altered48", "main+altered48"):
        out += list(C.read_jsonl(C.CODES_ROOT / ALTERED48))
    if which == "main+altered48":
        for rel, tag in PROMPT_SETS:
            out += [{**r, "experiment": tag} for r in C.read_jsonl(C.CODES_ROOT / rel)]
    ids = [r["prompt_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate prompt ids across sets")
    return out


def system_id(tier: str, method: str, seed: int) -> str:
    return f"{tier}|{method}|seed{seed}"


class HubBackend(TorchBackend):
    """The published backend, with adapters resolved locally or from the Hugging Face mirror."""

    def attach(self, method: str, seed: int):
        if method == "frozen_base":
            return None
        from peft import PeftModel
        rel = f"{self.tier}/{method}/seed_{seed}"
        local = C.CODES_ROOT / "checkpoints" / rel
        if not (local / "adapter_config.json").exists() and not (local / "final" / "adapter_config.json").exists():
            from huggingface_hub import snapshot_download
            root = snapshot_download(HF_ADAPTERS, allow_patterns=[f"{rel}/*", f"{rel}/**"],
                                     token=os.environ.get("HF_TOKEN"))
            local = Path(root) / rel
        w = local / "final" if (local / "final" / "adapter_config.json").exists() else local
        self.peft = PeftModel.from_pretrained(self.model, str(w))
        self.peft.eval()
        return self.peft


def environment() -> Dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch, transformers, peft
        env.update(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:                                               # smoke runs have no torch
        env["torch_import"] = f"{type(e).__name__}"
    return env


ORIGINAL_SETS = {"verified", "reordered", "numerical", "altered", "unchanged", "neutral"}
ORIGINAL_ARMS = [("frozen_base", 42), ("graft_proposed", 42)]


def run(smoke: bool, pilot: int, families: List[str], attention: str = "auto", repro: bool = False,
        stage_name: str = "", sets: str = "main", core_only: bool = False, random_draws: bool = False,
        max_new_tokens: int = 0, only_systems: List[str] = ()) -> Dict:
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(sets)
    if max_new_tokens:   # budget sensitivity (added 4 Oct 2026): the same budget as the API models
        prompts = [{**p, "max_new_tokens": max_new_tokens} for p in prompts]
    stage = stage_name or ("smoke" if smoke else ("pilot" if pilot else ("repro_sdpa" if repro else "main")))
    arms = CORE_ARMS if core_only else (DRAW_ARMS if random_draws else ARMS)
    if repro:   # the published systems, on the published sets, with the published attention (sdpa)
        prompts = [p for p in prompts if p["experiment"] in ORIGINAL_SETS]
        families = [f for f in families if f in ("small-instruct", "broad-instruct")]
        arms, attention = ORIGINAL_ARMS, "sdpa"
    path = out / f"gpu_{stage}_predictions.jsonl"
    done = {(r["prompt_id"], r["system"]) for r in C.read_jsonl(path)} if path.exists() else set()
    if attention == "sdpa" and not smoke:    # the loader prefers FlashAttention-2 whenever it is installed
        from GPU_Run.common import model_registry as MR
        MR._select_attention = lambda preferred: "eager" if preferred == "eager" else "sdpa"   # Gemma keeps eager
    backend = FakeBackend({}) if smoke else HubBackend({"budget": {"attention": attention}})
    attn = {"smoke": True} if smoke else TorchBackend.setup_attention({"budget": {"attention": attention}})
    started, written, timings = time.time(), 0, []

    for tier in families:
        backend.load(tier)
        for method, seed in arms:
            sid = system_id(tier, method, seed)
            if only_systems and sid not in only_systems:
                continue
            todo = [p for p in prompts if (p["prompt_id"], sid) not in done]
            if pilot:   # the first `pilot` prompts of EACH set, so every set's format and scoring is exercised
                todo = [p for exp in sorted({q["experiment"] for q in todo})
                        for p in sorted((q for q in todo if q["experiment"] == exp), key=lambda q: q["prompt_id"])[:pilot]]
            if smoke:
                todo = todo[:6]
            if not todo:
                continue
            backend.attach(method, seed)
            todo.sort(key=lambda p: len(p["prompt"]))
            bs, i = BATCH[tier], 0
            while i < len(todo):
                batch = todo[i:i + bs]
                mnt = max(int(p.get("max_new_tokens", 24)) for p in batch)
                t0 = time.time()
                try:
                    raws = backend.generate([p["prompt"] for p in batch], mnt)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and bs > 1:
                        import torch
                        torch.cuda.empty_cache(); bs = max(1, bs // 2)
                        print(f"  CUDA OOM on {sid}: batch size -> {bs}", flush=True)
                        continue
                    raise
                dt = time.time() - t0
                timings.append({"system": sid, "batch": len(batch), "seconds": round(dt, 3)})
                with path.open("a", encoding="utf-8") as f:              # written after every batch
                    for p, raw in zip(batch, raws):
                        row = {**{k: v for k, v in p.items() if k != "prompt"}, "system": sid, "tier": tier,
                               "method": method, "seed": seed,
                               "attention_implementation_used": getattr(backend, "attention", "unknown"),
                               "prompt_sha256": C.freeze(p["prompt"]), **score_row(p, raw)}
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += len(batch); i += bs
            if hasattr(backend, "detach"):
                backend.detach()
            print(f"  {sid}: done, {written} outputs so far, {(time.time() - started) / 60:.1f} min", flush=True)
    if hasattr(backend, "free"):
        backend.free()

    per = sum(t["seconds"] for t in timings) / max(1, sum(t["batch"] for t in timings))
    man = {"stage": stage, "outputs_written": written, "minutes": round((time.time() - started) / 60, 2),
           "seconds_per_output": round(per, 4), "prompts_per_system": len(prompts),
           "systems": [system_id(t, m, s) for t in families for m, s in arms],
           "attention": attn, "environment": environment(),
           "plan_sha256": sha256_file(PLAN) if PLAN.exists() else None,
           "prompt_sets": sets,
           "prompt_file_sha256": {rel: sha256_file(C.CODES_ROOT / rel) for rel in [r for r, _ in PROMPT_SETS] + [FOLLOWUPS, COT, ALTERED48]
                                  if (C.CODES_ROOT / rel).exists()},
           "timings_by_system": {s: round(sum(t["seconds"] for t in timings if t["system"] == s), 1)
                                 for s in sorted({t["system"] for t in timings})}}
    C.write_json(out / f"gpu_{stage}_manifest.json", man)
    print(f"[run_gpu] {stage}: {written} outputs in {man['minutes']} min ({per:.3f} s/output) -> {path}")
    return man


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", type=int, default=0, help="prompts per set and system for a check run")
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--attention", choices=["auto", "sdpa"], default="auto")
    ap.add_argument("--repro", action="store_true", help="published systems and sets, sdpa, for the exact-match check")
    ap.add_argument("--stage-name", default="", help="output file stem, e.g. main or flash")
    ap.add_argument("--sets", choices=["main", "followups", "all", "cot", "altered48", "main+altered48"], default="main")
    ap.add_argument("--core-only", action="store_true", help="unmodified and attribution-guided seed 42 only")
    ap.add_argument("--random-draws", action="store_true", help="the two further random-placement draws only")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="override every prompt's answer budget")
    ap.add_argument("--only-systems", nargs="*", default=[], help="system ids to run (others skipped)")
    a = ap.parse_args(argv)
    run(a.smoke, a.pilot, a.families, a.attention, a.repro, a.stage_name, a.sets, a.core_only, a.random_draws,
        a.max_new_tokens, a.only_systems)


if __name__ == "__main__":
    main()
