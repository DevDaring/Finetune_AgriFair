"""Size the GPU from the frozen designs: VRAM per model, and minutes per configuration.

    python -m Submission1_Code_Phase2.gpu_plan

Nothing here is a guess about "a 7B model". It reads the actual prompt files, tokenises them
with each model's own tokenizer to get real input lengths, reads each model's config for the
exact KV-cache geometry, and computes:

    weights      = parameters x 2 bytes (bfloat16)
    KV cache     = 2 x layers x kv_heads x head_dim x 2 bytes x (context) x batch
    activations  = a measured-order allowance for the forward pass at this batch and length

Throughput is expressed as tokens/second bands for common rental cards, so the same table gives
an answer for whichever card is actually available. Prefill dominates here: prompts are long
(a table plus a rule plus options) and outputs are ~16 tokens, so the run is compute-bound on
the prefill and FlashAttention-2 is what makes it cheap.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from Submission1_Code_Phase2 import common as C

# Per-card sustained prefill throughput for a 3-4B bf16 model with FlashAttention-2, in
# thousands of tokens per second. Bands, not promises: the planner reports a range.
CARDS = {
    "RTX 4090 24GB":      {"vram_gb": 24, "prefill_ktok_s": (28, 45), "decode_tok_s": (70, 110), "usd_per_hour": (0.35, 0.60)},
    "L4 24GB":            {"vram_gb": 24, "prefill_ktok_s": (10, 18), "decode_tok_s": (35, 60),  "usd_per_hour": (0.30, 0.55)},
    "A10G 24GB":          {"vram_gb": 24, "prefill_ktok_s": (12, 20), "decode_tok_s": (40, 65),  "usd_per_hour": (0.35, 0.75)},
    "RTX 3090 24GB":      {"vram_gb": 24, "prefill_ktok_s": (18, 30), "decode_tok_s": (55, 85),  "usd_per_hour": (0.15, 0.30)},
    "A100 40GB":          {"vram_gb": 40, "prefill_ktok_s": (40, 65), "decode_tok_s": (90, 140), "usd_per_hour": (1.00, 1.80)},
    "L40S 48GB":          {"vram_gb": 48, "prefill_ktok_s": (30, 50), "decode_tok_s": (75, 120), "usd_per_hour": (0.70, 1.20)},
}


def _hf_token() -> str:
    """The gated Llama config needs the token; env_loader is the only reader of .env."""
    import os
    from GPU_Run.common import env_loader
    env_loader.load_env()
    tok = env_loader.hf_token() or ""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if tok and not os.environ.get(name):
            os.environ[name] = tok
    return tok


def model_geometry(tier: str) -> Dict:
    """Parameters and KV geometry from the model's own config, without loading weights."""
    from GPU_Run.common import model_registry
    from transformers import AutoConfig
    spec = model_registry.get_spec(tier)
    cfg = AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=True, token=_hf_token() or None)
    txt = getattr(cfg, "text_config", cfg)
    layers = getattr(txt, "num_hidden_layers", 0)
    heads = getattr(txt, "num_attention_heads", 0)
    kv_heads = getattr(txt, "num_key_value_heads", heads)
    hidden = getattr(txt, "hidden_size", 0)
    head_dim = getattr(txt, "head_dim", hidden // max(1, heads))
    vocab = getattr(txt, "vocab_size", 0)
    inter = getattr(txt, "intermediate_size", 4 * hidden)
    params = (vocab * hidden * 2
              + layers * (hidden * head_dim * heads + 2 * hidden * head_dim * kv_heads + head_dim * heads * hidden
                          + 3 * hidden * inter))
    return {"tier": tier, "model_id": spec.hf_id, "declared_params_b": spec.parameter_count_billions, "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
            "hidden": hidden, "params_billions": round(params / 1e9, 2),
            "kv_bytes_per_token": 2 * layers * kv_heads * head_dim * 2}


def prompt_lengths(cfg: Dict, tier: str) -> Dict:
    """Tokenise the real prompt set with this model's tokenizer."""
    from transformers import AutoTokenizer
    from GPU_Run.common import model_registry
    tok = AutoTokenizer.from_pretrained(model_registry.get_spec(tier).hf_id, trust_remote_code=True, token=_hf_token() or None)
    from Submission1_Code_Phase2.run_inference import load_stage_prompts
    prompts = load_stage_prompts(cfg, "main")
    if not prompts:
        return {}
    by_study: Dict[str, List[int]] = {}
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True)
        by_study.setdefault(p["study"], []).append(len(tok(text)["input_ids"]))
    out = {}
    for study, lens in by_study.items():
        lens.sort()
        out[study] = {"n": len(lens), "mean_input_tokens": round(sum(lens) / len(lens)),
                      "p95_input_tokens": lens[int(len(lens) * 0.95) - 1], "max_input_tokens": lens[-1],
                      "total_input_tokens": sum(lens)}
    return out


def vram_estimate(geom: Dict, batch: int, context: int) -> Dict:
    weights = geom["params_billions"] * 1e9 * 2 / 1024**3
    kv = geom["kv_bytes_per_token"] * context * batch / 1024**3
    activations = batch * context * geom["hidden"] * 2 * 12 / 1024**3      # transient forward buffers
    lora = 0.1
    total = weights + kv + activations + lora
    return {"weights_gb": round(weights, 2), "kv_cache_gb": round(kv, 2),
            "activations_gb": round(activations, 2), "adapter_gb": lora,
            "peak_gb": round(total, 2), "with_20pct_headroom_gb": round(total * 1.2, 2)}


def time_estimate(lengths: Dict, geom_by_tier: Dict, out_tokens: int, card: Dict, cfg: Dict,
                  weights_on_local_disk: bool = True) -> Dict:
    """Wall-clock for the whole stage: start-up, weight loading, prefill, decode.

    Every system (tier x method) makes one pass over every prompt, so prefill tokens scale with
    the number of systems, not the number of methods. Decode is batched: a batch of B advances B
    sequences per step, so the cost is (tokens to generate) / (per-sequence rate x batch).
    """
    n_tiers = len(geom_by_tier)
    n_systems = n_tiers * len(cfg["systems"]["methods"])
    batch = int(cfg["budget"]["batch_size"])
    total_in = sum(v["total_input_tokens"] for v in lengths.values())
    total_prompts = sum(v["n"] for v in lengths.values())
    gb_to_load = sum(g["params_billions"] * 2 / 1.024**3 for g in geom_by_tier.values())

    out = {}
    for label, idx in (("fast", 1), ("slow", 0)):
        prefill_s = (total_in * n_systems) / (card["prefill_ktok_s"][idx] * 1000)
        decode_s = (total_prompts * n_systems * out_tokens) / (card["decode_tok_s"][idx] * batch)
        # start-up: CUDA context, transformers import, tokenizers
        startup_s = 40 if label == "fast" else 75
        # weights: from local disk (~1.5-2.5 GB/s) or downloaded first (~150-400 MB/s)
        rate = (2.5 if label == "fast" else 1.5) if weights_on_local_disk else (0.40 if label == "fast" else 0.15)
        load_s = gb_to_load / rate
        # one adapter attach per adapted arm, plus per-batch scheduling overhead
        adapter_s = 8 * n_tiers
        overhead_s = (total_prompts * n_systems / batch) * (0.15 if label == "fast" else 0.35)
        total = startup_s + load_s + adapter_s + prefill_s + decode_s + overhead_s
        out[label] = {"startup_minutes": round(startup_s / 60, 2), "weight_loading_minutes": round(load_s / 60, 2),
                      "adapter_minutes": round(adapter_s / 60, 2), "prefill_minutes": round(prefill_s / 60, 2),
                      "decode_minutes": round(decode_s / 60, 2), "batch_overhead_minutes": round(overhead_s / 60, 2),
                      "total_minutes": round(total / 60, 1)}
    sf = float(cfg["budget"]["timing_safety_factor"])
    out["with_safety_factor_minutes"] = round(out["slow"]["total_minutes"] * sf, 1)
    out["weights_source"] = "local disk" if weights_on_local_disk else "downloaded at run start"
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--json", action="store_true"); a = ap.parse_args(argv)
    cfg = C.load_config()
    batch = a.batch or int(cfg["budget"]["batch_size"])
    out_tokens = int(cfg["budget"]["answer_max_new_tokens"])
    geom = {t: model_geometry(t) for t in cfg["systems"]["tiers"]}
    lengths = prompt_lengths(cfg, cfg["systems"]["tiers"][0])
    if not lengths:
        raise SystemExit("build the studies first (r1_fresh_panel --build, r2_evidence --build)")
    context = max(v["p95_input_tokens"] for v in lengths.values()) + out_tokens
    report = {"batch_size": batch, "context_tokens_p95": context, "answer_tokens": out_tokens,
              "prompt_inventory": lengths, "models": geom, "vram": {}, "time_by_card": {}}
    for t, g in geom.items():
        report["vram"][t] = vram_estimate(g, batch, context)
    for name, card in CARDS.items():
        t = time_estimate(lengths, geom, out_tokens, card, cfg, weights_on_local_disk=True)
        t["if_weights_downloaded_first"] = time_estimate(lengths, geom, out_tokens, card, cfg,
                                                         weights_on_local_disk=False)["with_safety_factor_minutes"]
        lo, hi = card["usd_per_hour"]
        t["fits_vram"] = all(v["with_20pct_headroom_gb"] <= card["vram_gb"] for v in report["vram"].values())
        t["estimated_cost_usd"] = [round(t["fast"]["total_minutes"] / 60 * lo, 2),
                                   round(t["with_safety_factor_minutes"] / 60 * hi, 2)]
        report["time_by_card"][name] = t
    out_dir = C.out_dir(cfg, "tables"); C.write_json(out_dir / "gpu_plan.json", report)
    if a.json:
        print(json.dumps(report, indent=1)); return
    print(f"Prompt inventory (main stage, per system):")
    for study, v in sorted(lengths.items()):
        print(f"  {study:16s} {v['n']:5d} prompts | mean {v['mean_input_tokens']:4d} tok, p95 {v['p95_input_tokens']:4d}")
    total = sum(v["n"] for v in lengths.values()) * len(C.systems(cfg))
    print(f"  total outputs across {len(C.systems(cfg))} systems: {total}")
    print(f"\nVRAM at batch {batch}, context {context}:")
    for t, v in report["vram"].items():
        print(f"  {t:16s} weights {v['weights_gb']:5.1f} + KV {v['kv_cache_gb']:5.1f} + act {v['activations_gb']:4.1f} "
              f"= {v['peak_gb']:5.1f} GB peak ({v['with_20pct_headroom_gb']:.1f} with headroom)")
    print(f"\nRuntime and cost by card (whole main stage, all {len(C.systems(cfg))} systems, batch {batch}):")
    for name, t in report["time_by_card"].items():
        print(f"  {name:18s} {'fits' if t['fits_vram'] else 'TIGHT':5s} "
              f"{t['fast']['total_minutes']:4.1f}-{t['slow']['total_minutes']:4.1f} min, "
              f"{t['with_safety_factor_minutes']:4.1f} with x{cfg['budget']['timing_safety_factor']} safety "
              f"({t['if_weights_downloaded_first']:4.1f} if weights download first) "
              f"~${t['estimated_cost_usd'][0]:.2f}-{t['estimated_cost_usd'][1]:.2f}")
    print(f"\nwritten -> {out_dir / 'gpu_plan.json'}")


if __name__ == "__main__":
    main()
