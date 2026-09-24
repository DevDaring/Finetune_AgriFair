"""Budget-capped batched inference for R1, R2 and R3 on existing checkpoints.

    python -m Submission1_Code_Phase2.run_inference --smoke            # offline, no GPU, no weights
    python -m Submission1_Code_Phase2.run_inference --stage pilot
    python -m Submission1_Code_Phase2.run_inference --stage main

Three things make this cheap enough to fit the plan's 60-minute core budget:

1. **FlashAttention-2 from a prebuilt wheel.** GPU_Run.common.flash_attn_setup resolves the wheel
   matching this interpreter, torch build, CUDA major and C++11 ABI, and installs it; compiling
   from source on rented time is never attempted. If no wheel matches, attention falls back to
   sdpa and the choice is written into every output row, so a run is never ambiguous.
2. **Batched, length-sorted generation.** The original panel generated one prompt at a time.
   Prompts are sorted by tokenised length and batched with left padding, so a batch is nearly
   uniform and almost no compute is spent on pad tokens. Batch size halves automatically on OOM.
3. **One base-model load per tier.** frozen_base and graft_proposed share a base model, so the
   base is loaded once and the LoRA adapter is attached and detached around the adapted arm.
   That removes half of the model-loading time, which dominates a short run.

Safety: the elapsed-time cap is checked before every batch, results are written after every
batch, and a timing probe on the pilot decides whether the main design fits before it starts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from Submission1_Code_Phase2 import common as C

ANSWER_RE_LETTERS = "abcd"


# ------------------------------------------------------------------ prompt sources

def load_stage_prompts(cfg: Dict, stage: str) -> List[Dict]:
    """Every prompt this run must answer, already carrying its scoring fields."""
    out = C.CODES_ROOT / cfg["output_directory"]
    prompts: List[Dict] = []
    fresh = out / "source_validation" / "fresh_panel.jsonl"
    if fresh.exists():
        for it in C.read_jsonl(fresh):
            for wording, text in it["wordings"].items():
                opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", it["choices"]))
                prompts.append({"prompt_id": f"r1-{it['fresh_id']}-{wording}", "study": "r1_fresh",
                                "comparison_id": it["fresh_id"], "source_cell": it["source_cell"],
                                "axis": it["axis"], "state": it["state"], "parent_table": it["parent_table"],
                                "wording": wording, "condition": it["condition"], "choices": it["choices"],
                                "gold_choice_text": it["gold_choice_text"],
                                "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]),
                                "prompt": f"{text}\n{opts}\n\nReply with one JSON object only: "
                                          '{"answer_choice_letter": "<a|b|c>"}'})
    for name, study in (("r2_main_prompts.jsonl", "r2_main"), ("r2_diagnostic_prompts.jsonl", "r2_diagnostic")):
        p = out / "evidence" / name
        if p.exists():
            prompts += C.read_jsonl(p)
    adv = out / "advice" / "r3_prompts.jsonl"
    if adv.exists() and cfg["r3"]["enabled"]:
        prompts += C.read_jsonl(adv)
    if stage == "pilot":
        b = cfg["budget"]["pilot"]
        keep: List[Dict] = []
        for study, n_units, unit_key in (("r1_fresh", b["r1_comparisons"], "comparison_id"),
                                         ("r2_main", b["r2_bundles"], "bundle_id"),
                                         ("r2_diagnostic", b["r2_bundles"], "bundle_id"),
                                         ("r3_advice", b["r3_cases"], "case_id")):
            rows = [p for p in prompts if p["study"] == study]
            units = sorted({r[unit_key] for r in rows})[:n_units]
            keep += [r for r in rows if r[unit_key] in units]
        return keep
    # main excludes whatever the pilot already answered
    pilot_ids = set()
    pp = out / "predictions" / "pilot_predictions.jsonl"
    if pp.exists():
        pilot_ids = {r["prompt_id"] for r in C.read_jsonl(pp)}
    return [p for p in prompts if p["prompt_id"] not in pilot_ids]


# ------------------------------------------------------------------ generation backends

class FakeBackend:
    """Deterministic offline answers for --smoke. Loads nothing, spends nothing."""
    attention = "none (smoke)"

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def load(self, tier: str):
        return self

    def attach(self, method: str, seed: int):
        return None

    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        import hashlib
        out = []
        for p in prompts:
            h = hashlib.sha256(p.encode()).digest()
            n_opts = 4 if "a|b|c|d" in p else 3
            out.append('{"answer_choice_letter": "%s"}' % ANSWER_RE_LETTERS[h[0] % n_opts])
        return out


class TorchBackend:
    """Batched HF generation with FlashAttention-2 where a prebuilt wheel exists."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg; self.model = None; self.tok = None; self.tier = None
        self.peft = None; self.attention = "unknown"

    @staticmethod
    def setup_attention(cfg: Dict) -> Dict:
        """Install the matching prebuilt FlashAttention wheel, or record the sdpa fallback."""
        if str(cfg["budget"].get("attention", "auto")) != "auto":
            os.environ["ATTENTION_IMPLEMENTATION"] = str(cfg["budget"]["attention"])
            return {"requested": cfg["budget"]["attention"], "installed": None}
        try:
            from GPU_Run.common import flash_attn_setup
            info = flash_attn_setup.detect_and_setup()
            if info.get("installed"):
                os.environ.setdefault("ATTENTION_IMPLEMENTATION", "flash_attention_2")
            return info
        except Exception as e:                                   # never fatal: sdpa is correct everywhere
            return {"installed": False, "reason": f"{type(e).__name__}: {str(e)[:160]}"}

    def load(self, tier: str):
        import torch
        from GPU_Run.common import model_registry
        if self.tier == tier and self.model is not None:
            return self
        self.free()
        model, tok, meta = model_registry.load_model_and_tokenizer(tier)
        model.eval()
        tok.padding_side = "left"                                # decoder-only batching needs left padding
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.model, self.tok, self.tier = model, tok, tier
        self.attention = getattr(getattr(model, "config", None), "_attn_implementation", "unknown")
        return self

    def attach(self, method: str, seed: int):
        """Attach the LoRA adapter for an adapted arm; the base stays loaded for the frozen arm."""
        if method == "frozen_base":
            return None
        from peft import PeftModel
        w = C.CODES_ROOT / "checkpoints" / self.tier / method / f"seed_{seed}"
        w = w / "final" if (w / "final").exists() else w
        if not w.exists():
            raise SystemExit(f"adapter not found: {w}")
        self.peft = PeftModel.from_pretrained(self.model, str(w))
        self.peft.eval()
        return self.peft

    def detach(self):
        if self.peft is not None:
            self.peft = self.peft.unload() if hasattr(self.peft, "unload") else None
            self.peft = None

    def free(self):
        import gc
        import torch
        self.detach(); self.model = None; self.tok = None; self.tier = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        import torch
        model = self.peft if self.peft is not None else self.model
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=self.tok.pad_token_id, use_cache=True)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]


# ------------------------------------------------------------------ scoring

def parse_letter(text: str, n_options: int) -> Tuple[Optional[str], bool]:
    from GPU_Run.common.parsing import extract_answer_letter
    letters = tuple(ANSWER_RE_LETTERS[:n_options])
    return extract_answer_letter(text, letters)


def score_row(p: Dict, raw: str) -> Dict:
    letter, ok = parse_letter(raw, len(p["choices"]))
    picked = p["choices"][ANSWER_RE_LETTERS.index(letter)] if (ok and letter) else None
    gold_text = p["gold_choice_text"]
    correct = bool(picked is not None and picked.strip().lower() == str(gold_text).strip().lower()) if picked else False
    if picked and str(gold_text) == "insufficient_evidence":
        correct = picked.startswith("The table does not")
    return {"raw_output": raw, "display_letter": letter, "picked_choice": picked,
            "parse_ok": bool(ok), "correct": correct}


# ------------------------------------------------------------------ the run

def run(cfg: Dict, stage: str, smoke: bool) -> Dict:
    out_dir = C.out_dir(cfg, "predictions")
    budget = cfg["budget"]
    cap_minutes = float(budget["core_gpu_minutes"] if budget["tier"] == "core" else budget["preferred_gpu_minutes"])
    prompts = load_stage_prompts(cfg, stage)
    if not prompts:
        raise SystemExit("no prompts found; build the studies first")
    if smoke:
        prompts = prompts[: max(8, int(cfg["smoke"]["items"]) * 2)]
    if not smoke and not cfg["gpu_enabled"]:
        raise SystemExit("gpu_enabled is false in config.yaml; flip it deliberately before a paid run")

    backend = FakeBackend(cfg) if smoke else TorchBackend(cfg)
    attn_info = {"smoke": True} if smoke else TorchBackend.setup_attention(cfg)
    systems = C.systems(cfg)
    started = time.time()
    path = out_dir / f"{stage}_predictions.jsonl"
    done = {r["prompt_id"] + "|" + r["system"] for r in C.read_jsonl(path)} if path.exists() else set()
    written, timings = 0, []

    for tier in cfg["systems"]["tiers"]:
        backend.load(tier)
        for method in cfg["systems"]["methods"]:
            sys_id = C.system_id({"tier": tier, "method": method, "seed": cfg["systems"]["seed"]})
            backend.attach(method, cfg["systems"]["seed"])
            todo = [p for p in prompts if (p["prompt_id"] + "|" + sys_id) not in done]
            todo.sort(key=lambda p: len(p["prompt"]))            # length-sorted: near-uniform batches
            bs = int(budget["batch_size"])
            i = 0
            while i < len(todo):
                elapsed = (time.time() - started) / 60
                if not smoke and elapsed > cap_minutes:
                    print(f"  elapsed {elapsed:.1f} min reached the {cap_minutes:.0f}-minute cap; stopping cleanly")
                    if hasattr(backend, "detach"):
                        backend.detach()
                    return _finish(cfg, out_dir, stage, started, written, timings, attn_info, systems, capped=True)
                batch = todo[i: i + bs]
                is_advice = batch[0]["study"] == "r3_advice"
                mnt = max(p.get("max_new_tokens", budget["answer_max_new_tokens"]) for p in batch)
                t0 = time.time()
                try:
                    raws = backend.generate([p["prompt"] for p in batch], mnt)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and bs > 1:
                        bs = max(1, bs // 2)
                        print(f"  CUDA OOM: batch size -> {bs}")
                        if hasattr(backend, "free"):
                            import torch
                            torch.cuda.empty_cache()
                        continue
                    raise
                dt = time.time() - t0
                timings.append({"system": sys_id, "batch": len(batch), "seconds": round(dt, 3),
                                "seconds_per_prompt": round(dt / len(batch), 4), "max_new_tokens": mnt,
                                "advice": is_advice})
                rows = []
                for p, raw in zip(batch, raws):
                    rows.append({**{k: v for k, v in p.items() if k != "prompt"}, "system": sys_id,
                                 "tier": tier, "method": method, "seed": cfg["systems"]["seed"],
                                 "attention_implementation_used": getattr(backend, "attention", "unknown"),
                                 "prompt_sha256": C.freeze(p["prompt"]), **score_row(p, raw)})
                with path.open("a", encoding="utf-8") as f:       # saved after every batch
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += len(rows); i += bs
                if written % 200 < bs:
                    print(f"  {written} outputs, {(time.time() - started) / 60:.1f} min elapsed", flush=True)
            if hasattr(backend, "detach"):
                backend.detach()
    if hasattr(backend, "free"):
        backend.free()
    return _finish(cfg, out_dir, stage, started, written, timings, attn_info, systems, capped=False)


def _finish(cfg, out_dir: Path, stage: str, started: float, written: int, timings: List[Dict],
            attn_info: Dict, systems: List[Dict], capped: bool) -> Dict:
    elapsed = (time.time() - started) / 60
    per_prompt = sum(t["seconds"] for t in timings) / max(1, sum(t["batch"] for t in timings))
    man = {"stage": stage, "outputs_written": written, "elapsed_minutes": round(elapsed, 2),
           "seconds_per_output_mean": round(per_prompt, 4), "capped": capped,
           "attention": attn_info, "systems": [C.system_id(s) for s in systems],
           "batch_timings": timings[:200]}
    if stage == "pilot" and timings:
        remaining = _remaining_main_count(cfg)
        projected = per_prompt * remaining / 60 * float(cfg["budget"]["timing_safety_factor"])
        cap = float(cfg["budget"]["core_gpu_minutes"] if cfg["budget"]["tier"] == "core" else cfg["budget"]["preferred_gpu_minutes"])
        man["main_projection"] = {"remaining_outputs": remaining, "projected_minutes_with_safety_factor": round(projected, 1),
                                  "cap_minutes": cap, "fits": projected <= cap,
                                  "decision": "run the full main design" if projected <= cap else
                                              "reduce to the prespecified fallback design before running main"}
    C.write_json(out_dir / f"{stage}_manifest.json", man)
    print(f"[run_inference] {stage}: {written} outputs in {elapsed:.1f} min "
          f"({per_prompt:.2f} s/output){' (CAPPED)' if capped else ''} -> {out_dir}")
    if "main_projection" in man:
        print(f"  projection: {man['main_projection']}")
    return man


def _remaining_main_count(cfg: Dict) -> int:
    return len(load_stage_prompts(cfg, "main")) * len(C.systems(cfg))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(argv)
    run(C.load_config(), a.stage, a.smoke)


if __name__ == "__main__":
    main()
