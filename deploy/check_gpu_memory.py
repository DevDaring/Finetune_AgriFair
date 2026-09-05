"""Prove the study's real training step fits in this GPU's memory, before the study starts.

The verification pass runs on two data samples. That exercises every code path but tells you
nothing about memory, because memory pressure comes from the batch shape, not the number of
items. An out-of-memory failure eight hours into the main run is the expensive failure this
guards against.

So this does the one thing a two-sample run cannot: a real forward and backward pass at the
production micro-batch and sequence length, on the real weights, with LoRA attached, and
reports the peak allocation against the card's capacity.

Exits non-zero if any tier does not fit, so a deployment script can refuse to start.

Usage:  python deploy/check_gpu_memory.py <tier[,tier]>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GPU_Run.common import model_registry
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import TRAIN_INSTANCES
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.training import (
    TrainingConfig,
    build_lora_model,
    make_example_tensors,
    micro_batch_size,
    weighted_batch_loss,
)
from GPU_Run.configure_layer_selective_lora import build_configs

logger = get_logger("check_gpu_memory")

# Fraction of the card the training step may occupy and still be considered safe. Evaluation,
# fragmentation and the odd longer sequence all need room above the measured peak.
SAFE_FRACTION = 0.85


def probe(tier: str, records) -> dict:
    import torch

    model, tok, meta = model_registry.load_model_and_tokenizer(tier, for_training=True)
    cfg = TrainingConfig()
    micro = micro_batch_size(cfg)
    cap = int(getattr(model, "_agrifair_max_train_micro_batch", 0) or 0)
    if cap > 0:
        micro = min(micro, cap)

    # The densest placement the study ACTUALLY builds. Feeding an attribution of 1.0 for every
    # layer makes rank_for() return RANK_MAX everywhere, which is an adapter far larger than
    # any real arm: it rejected configurations that run fine. A realistic attribution profile
    # (scores decaying across depth) reproduces the rank spread the study trains, and the
    # uniform ablation spreads that same budget over the whole depth.
    cfg_obj = getattr(model, "config", None)
    # Gemma 3 is a multimodal checkpoint: its text tower's depth lives under text_config, and
    # the top-level attribute is absent. Reading 32 there would build an adapter over the
    # first 32 of 48 layers and understate the largest adapter the study trains.
    n_layers = getattr(cfg_obj, "num_hidden_layers", None) or getattr(
        getattr(cfg_obj, "text_config", None), "num_hidden_layers", None) or 32
    scores = {str(i): max(0.0, 1.0 - i / max(1, n_layers - 1)) for i in range(n_layers)}
    configs = build_configs({"layer_scores_normalized": scores}, draws=1)
    # Whichever of the real placements trains the most parameters.
    best, placement = -1.0, None
    for key in ("attribution_guided", "uniform"):
        candidate = build_lora_model(model, configs[key], cfg)
        pct = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
        if pct > best:
            best, placement = pct, configs[key]
        candidate.unload()
        del candidate
        torch.cuda.empty_cache()
    peft_model = build_lora_model(model, placement, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=cfg.learning_rate)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Pad to the longest example the STUDY can actually produce, which is the longest
    # tokenised training record, not cfg.max_length. cfg.max_length is a truncation ceiling of
    # 1024; real instances run to roughly 380 tokens, so padding to the ceiling measures a
    # batch almost three times larger than any that will occur and fails a card that runs the
    # study comfortably. Padding positions carry label -100 and weight 0, so they occupy the
    # tensor shape without changing a gradient.
    tokenised = [make_example_tensors(tok, r, cfg) for r in records]
    longest = min(int(cfg.max_length), max(len(e["input_ids"]) for e in tokenised))
    ordered = sorted(tokenised, key=lambda e: -len(e["input_ids"]))
    def _copy(example):
        return {k: (list(v) if isinstance(v, list) else v) for k, v in example.items()}

    examples = [_copy(e) for e in ordered[:micro]]
    while len(examples) < micro:
        examples.append(_copy(examples[0]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for e in examples:
        short = longest - len(e["input_ids"])
        if short > 0:
            e["input_ids"] = list(e["input_ids"]) + [pad_id] * short
            e["labels"] = list(e["labels"]) + [-100] * short
            e["token_weights"] = list(e["token_weights"]) + [0.0] * short
    loss = weighted_batch_loss(peft_model, tok, examples)
    loss.backward()
    # Include the optimiser state: AdamW allocates two buffers over the trainable parameters,
    # and a forward-and-backward-only probe reports a peak the real step never sees.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    del optimizer, peft_model, model
    torch.cuda.empty_cache()
    return {
        "tier": tier,
        "micro_batch": micro,
        "sequence_length": longest,
        "peak_gigabytes": peak,
        "total_gigabytes": total,
        "fits": peak < SAFE_FRACTION * total,
        "attention": meta.get("attention_implementation_used", ""),
        "trainable_millions": trainable / 1e6,
    }


def main() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            print("  no CUDA device visible; memory probe skipped.")
            return 0
    except ImportError:
        print("  torch not installed; memory probe skipped.")
        return 0

    tiers = [t.strip() for t in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if t.strip()]
    if not tiers:
        tiers = list(model_registry.active_tiers())
    records = read_jsonl(TRAIN_INSTANCES)
    if not records:
        print("  no training instances found; run dataset_prep first.")
        return 1

    rows, failed = [], []
    for tier in tiers:
        try:
            row = probe(tier, records)
        except Exception as e:
            kind = type(e).__name__
            print(f"  {tier:<22s} PROBE FAILED ({kind}: {str(e)[:80]})")
            failed.append(tier)
            continue
        rows.append(row)
        verdict = "fits" if row["fits"] else "DOES NOT FIT"
        print(f"  {row['tier']:<22s} micro={row['micro_batch']} len={row['sequence_length']:<5d}"
              f" lora={row['trainable_millions']:.0f}M attn={row['attention']:<18s}"
              f" peak {row['peak_gigabytes']:.1f} / {row['total_gigabytes']:.0f} GiB"
              f"  ({100 * row['peak_gigabytes'] / row['total_gigabytes']:.0f}%)  {verdict}")
        if not row["fits"]:
            failed.append(tier)

    if failed:
        print("\nThese tiers will not train safely on this GPU: " + ", ".join(failed))
        print("Lower max_train_micro_batch_size in GPU_Run/common/model_registry.py, or rent a "
              "larger card. Gradient accumulation keeps the effective batch and the "
              "optimisation identical, so lowering the micro-batch costs speed and nothing else.")
        return 1
    print("\nEvery tier fits with headroom. The study can train on this GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
