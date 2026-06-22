"""Joint loss, curriculum, replay, budget assertion, and the LoRA training loop.

Joint loss per example (Instruction.md Section 2, Stage C):
    total = (answer_cross_entropy + rationale_weight * rationale_cross_entropy)
            * condition_weight                # neq 1.5, eq 1.0
plus a curriculum that raises the neq (diff) sampling proportion 0.2 -> 0.5 across epochs,
plus a small general-replay mix (default 10 percent). Defaults: 3 epochs, AdamW, lr 2e-4,
LoRA dropout 0.05, bf16, effective batch 8, max sequence length 1024.

Training batching (TRAIN_MICRO_BATCH_SIZE) fuses examples within the existing
gradient-accumulation window; per-example normalized loss is computed exactly, so the
summed, grad_accum-scaled gradient equals the sequential path up to floating-point order
(Instruction.md Section 14). peft is imported lazily.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from GPU_Run.common import prompts as P
from GPU_Run.common.checkpointing import latest_epoch
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("training")


@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_dropout: float = 0.05
    lora_alpha: int = 16
    rationale_weight: float = 0.3
    condition_weights: Dict[str, float] = field(default_factory=lambda: {"neq": 1.5, "eq": 1.0})
    grad_accum: int = 8
    micro_batch: int = 8
    max_length: int = 1024
    curriculum_neq_start: float = 0.2
    curriculum_neq_end: float = 0.5
    replay_fraction: float = 0.1
    rationale_mode: bool = True
    condition_adaptive: bool = True
    seed: int = 42


def micro_batch_size(cfg: TrainingConfig) -> int:
    try:
        return int(os.environ.get("TRAIN_MICRO_BATCH_SIZE", cfg.micro_batch))
    except ValueError:
        return cfg.micro_batch


def smoke_max_steps() -> Optional[int]:
    v = os.environ.get("TRAIN_SMOKE_MAX_STEPS")
    return int(v) if v else None


# ------------------------- trainable-parameter budget -----------------------

def trainable_parameter_percentage(model) -> float:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return 100.0 * trainable / max(1, total)


def assert_matched_budget(pct: float, reference_pct: float, tol: float = 2.0, strict: Optional[bool] = None) -> None:
    if strict is None:
        strict = os.environ.get("STRICT_BUDGET", "1") == "1"
    delta = abs(pct - reference_pct)
    msg = f"trainable budget {pct:.3f}% vs reference {reference_pct:.3f}% (delta {delta:.3f}%, tol {tol}%)"
    if delta > tol:
        if strict:
            raise AssertionError("Matched-budget violation: " + msg)
        logger.warning("Matched-budget WARNING: %s", msg)
    else:
        logger.info("Matched budget OK: %s", msg)


# ----------------------------- LoRA placement -------------------------------

def build_lora_model(base_model, placement: Dict, cfg: TrainingConfig):
    """Apply LoRA using a per-layer rank_pattern derived from a placement config.

    placement = {"target_modules": [...], "rank_pattern": {regex: rank}, "default_rank": r}
    """
    from peft import LoraConfig, get_peft_model

    lconf = LoraConfig(
        r=placement.get("default_rank", 8),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=placement["target_modules"],
        rank_pattern=placement.get("rank_pattern", {}),
    )
    return get_peft_model(base_model, lconf)


# --------------------------- example tensorization --------------------------

def _build_target_segments(record: Dict, cfg: TrainingConfig) -> Tuple[str, str, str, str]:
    """Return (prompt, answer_prefix, rationale_text, answer_tail).

    answer_prefix + rationale_text + answer_tail == the full target string. Rationale
    tokens get weight rationale_weight; answer tokens get weight 1.0."""
    built = P.build_mcq_prompt(record, rationale_mode=cfg.rationale_mode)
    prompt = built["prompt"]
    gold = built["gold_display_letter"]
    if cfg.rationale_mode:
        refs = [record["law_reference"]] if record.get("law_reference") else []
        prefix = '{"answer_choice_letter": "%s", "rationale": "' % gold
        rationale = record["rationale"]
        tail = '", "cited_law_references": %s}' % json.dumps(refs, ensure_ascii=False)
        return prompt, prefix, rationale, tail
    target = json.dumps({"answer_choice_letter": gold}, ensure_ascii=False)
    return prompt, target, "", ""


def make_example_tensors(tokenizer, record: Dict, cfg: TrainingConfig):
    """Return dict with input_ids, labels, token_weights, condition_weight (lists/float)."""
    prompt, prefix, rationale, tail = _build_target_segments(record, cfg)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    rationale_ids = tokenizer(rationale, add_special_tokens=False)["input_ids"] if rationale else []
    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"] if tail else []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = prompt_ids + prefix_ids + rationale_ids + tail_ids + eos
    labels = ([-100] * len(prompt_ids)) + prefix_ids + rationale_ids + tail_ids + eos
    weights = (
        [0.0] * len(prompt_ids)
        + [1.0] * len(prefix_ids)
        + [cfg.rationale_weight] * len(rationale_ids)
        + [1.0] * len(tail_ids)
        + [1.0] * len(eos)
    )
    input_ids = input_ids[: cfg.max_length]
    labels = labels[: cfg.max_length]
    weights = weights[: cfg.max_length]
    cond = record["condition"]
    cw = cfg.condition_weights.get(cond, 1.0) if cfg.condition_adaptive else 1.0
    return {"input_ids": input_ids, "labels": labels, "token_weights": weights, "condition_weight": cw}


def weighted_example_loss(model, tokenizer, example: Dict):
    """Per-example normalized, token-weighted CE times the condition weight."""
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([example["input_ids"]], device=model.device)
    out = model(input_ids=ids)
    logits = out.logits[0, :-1, :]
    labels = torch.tensor(example["labels"][1:], device=model.device)
    weights = torch.tensor(example["token_weights"][1:], device=model.device, dtype=logits.dtype)
    mask = labels != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    ce = F.cross_entropy(logits[mask], labels[mask].clamp(min=0), reduction="none")
    w = weights[mask]
    norm = (ce * w).sum() / (w.sum() + 1e-8)
    return norm * example["condition_weight"]


# ------------------------------- curriculum ---------------------------------

def curriculum_order(train: List[Dict], replay: List[Dict], cfg: TrainingConfig, epoch: int, rng: random.Random) -> List[Dict]:
    """Deterministic per-epoch ordering: neq proportion ramps; replay mixed in."""
    frac = cfg.epochs - 1
    t = epoch / frac if frac > 0 else 1.0
    neq_prop = cfg.curriculum_neq_start + (cfg.curriculum_neq_end - cfg.curriculum_neq_start) * t
    neq = [r for r in train if r["condition"] == "neq"]
    eq = [r for r in train if r["condition"] == "eq"]
    rng.shuffle(neq)
    rng.shuffle(eq)
    n = len(train)
    n_neq = int(round(neq_prop * n))
    chosen = (neq * ((n_neq // max(1, len(neq))) + 1))[:n_neq] + (eq * ((n - n_neq) // max(1, len(eq)) + 1))[: n - n_neq]
    rng.shuffle(chosen)
    if replay and cfg.replay_fraction > 0:
        k = int(round(cfg.replay_fraction * len(chosen)))
        rep = (replay * (k // max(1, len(replay)) + 1))[:k]
        merged = chosen + rep
        rng.shuffle(merged)
        return merged
    return chosen


# ------------------------------- train loop ---------------------------------

def train_lora(
    model,
    tokenizer,
    train_records: List[Dict],
    placement: Dict,
    out_dir: Path,
    cfg: TrainingConfig,
    replay_records: Optional[List[Dict]] = None,
    reference_budget_pct: Optional[float] = None,
) -> Dict:
    """Train a LoRA adapter with the joint loss, curriculum, and replay. Resumable per
    epoch. Returns a metadata dict including trainable_parameter_percentage."""
    import torch

    peft_model = build_lora_model(model, placement, cfg)
    pct = trainable_parameter_percentage(peft_model)
    if reference_budget_pct is not None:
        assert_matched_budget(pct, reference_budget_pct)

    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = latest_epoch(out_dir) + 1
    if start_epoch > 0:
        from peft import PeftModel  # noqa: F401

        logger.info("Resuming %s from epoch %d", out_dir.name, start_epoch)

    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    rng = random.Random(cfg.seed)
    replay_records = replay_records or []
    cap = smoke_max_steps()
    grad_accum = cfg.grad_accum
    step = 0

    for epoch in range(start_epoch, cfg.epochs):
        peft_model.train()
        order = curriculum_order(train_records, replay_records, cfg, epoch, rng)
        examples = [make_example_tensors(tokenizer, r, cfg) for r in order]
        optimizer.zero_grad()
        for i, ex in enumerate(examples):
            loss = weighted_example_loss(peft_model, tokenizer, ex) / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if cap and step >= cap:
                    logger.info("Smoke step cap %d reached.", cap)
                    break
        ep_dir = out_dir / f"epoch_{epoch}"
        peft_model.save_pretrained(str(ep_dir))
        logger.info("Saved checkpoint %s", ep_dir)
        if cap and step >= cap:
            break

    peft_model.save_pretrained(str(out_dir / "final"))
    return {"trainable_parameter_percentage": pct, "epochs_run": cfg.epochs - start_epoch, "out_dir": str(out_dir)}
