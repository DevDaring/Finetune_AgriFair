"""Joint loss, curriculum, replay, budget assertion, and the LoRA training loop.

Joint loss per example (Stage C):
    total = (answer_cross_entropy + rationale_weight * rationale_cross_entropy)
            * condition_weight                # diff 1.5, equal 1.0
plus a curriculum that raises the diff sampling proportion 0.2 -> 0.5 across epochs,
plus a small general-replay mix (default 10 percent). Defaults: 3 epochs, AdamW, lr 2e-4,
gradient clipping at 1.0, LoRA dropout 0.05, bf16, effective batch 8, max length 1024.

Training batching (TRAIN_MICRO_BATCH_SIZE, default 8) fuses examples inside the
gradient-accumulation window with right padding and a per-example normalised loss, so the
summed, grad_accum-scaled gradient equals the one-example-at-a-time path up to
floating-point order. Every prompt is rendered through the tokenizer's chat template, the
same rendering the evaluation uses. Resume reloads the last saved epoch's adapter
weights; an arm whose final adapter already exists is skipped unless FORCE_RETRAIN=1.

This module also carries the two baseline mechanisms that need a training-loop hook
rather than a placement change, plus the per-run cost record that the efficiency
analysis consumes:

# Pan, Z., Liang, Z., Kabbara, J., Emami, A. "DART: Mitigating Harm Drift in
#   Difference-Aware LLMs via Distill-Audit-Repair Training." ACL 2026, arXiv:2604.16845.
#   [three-stage distill -> audit -> severity-weighted repair; oversampling weights
#    1x mild, 2x moderate, 3x severe, 4x extreme; LoRA r=16 alpha=32 dropout 0.05,
#    lr 2e-4, 3 epochs, bf16]
# Wu, Y., Wang, X., et al. "Mitigating Fine-tuning Bias: A Parameter-Efficient
#   Debiasing Framework for Large Language Models (PEDAL)." The ACM Web Conference 2026,
#   doi:10.1145/3774904.3793029.
#   [Classifier gate on biased samples, Modifier that breaks the shortcut, Reviewer that
#    adjusts residual shortcut activation at output time]
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from GPU_Run.common import prompts as P
from GPU_Run.common.checkpointing import latest_epoch
from GPU_Run.common.logging_utils import append_csv_row, get_logger
from GPU_Run.common.paths import RESULTS_DIR

logger = get_logger("training")

# Per-run cost record. Unlocks the efficiency-normalized comparison (fairness gained per
# GPU-minute and per unit of parameter-space drift). Wall-clock is measured on the training
# loop only, excluding model load.
RUN_COST_LOG = RESULTS_DIR / "training_run_cost_log.csv"
RUN_COST_COLUMNS = [
    "tier", "method", "random_seed", "quantization_setting",
    "trainable_parameter_percentage", "wall_clock_minutes_training_loop",
    "peak_gpu_memory_gigabytes", "optimizer_steps_completed",
    "training_examples_seen", "device_name",
]
TRAIN_SUMMARY_FILENAME = "train_summary.json"


@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_dropout: float = 0.05
    lora_alpha_multiplier: float = 2.0      # alpha = multiplier x rank, per layer
    rationale_weight: float = 0.3
    condition_weights: Dict[str, float] = field(default_factory=lambda: {"diff": 1.5, "equal": 1.0})
    grad_accum: int = 8
    micro_batch: int = 8
    max_length: int = 1024
    max_grad_norm: float = 1.0
    curriculum_diff_start: float = 0.2
    curriculum_diff_end: float = 0.5
    replay_fraction: float = 0.1
    rationale_mode: bool = True
    condition_adaptive: bool = True
    seed: int = 42
    # DART (arXiv:2604.16845): severity-weighted repair phase after the distil phase.
    example_weight_fn: Optional[Callable[[Dict], float]] = None
    repair_epochs: int = 1
    # PEDAL (doi:10.1145/3774904.3793029): the Classifier gate decides which examples the
    # Modifier rewrites; only gated examples receive the shortcut-breaking treatment.
    pedal_classifier_fn: Optional[Callable[[Dict], bool]] = None
    pedal_modifier_weight: float = 2.0


def micro_batch_size(cfg: TrainingConfig) -> int:
    try:
        return max(1, int(os.environ.get("TRAIN_MICRO_BATCH_SIZE", cfg.micro_batch)))
    except ValueError:
        return cfg.micro_batch


def smoke_max_steps() -> Optional[int]:
    v = os.environ.get("TRAIN_SMOKE_MAX_STEPS")
    return int(v) if v else None


def force_retrain() -> bool:
    return os.environ.get("FORCE_RETRAIN", "0") == "1"


# ------------------------- trainable-parameter budget -----------------------

def trainable_parameter_percentage(model) -> float:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return 100.0 * trainable / max(1, total)


# The matched-budget claim is that every baseline trains about as many parameters as the
# proposed method, so any difference in fairness cannot be bought with extra capacity. The
# tolerance therefore has to be RELATIVE. Trainable shares here are fractions of a percent
# (roughly 0.1-0.7%), so the old absolute 2.0-percentage-point tolerance could never fire:
# a baseline training five times the parameters of the proposed method passed silently.
BUDGET_RELATIVE_TOLERANCE = float(os.environ.get("BUDGET_RELATIVE_TOLERANCE", "0.10"))
BUDGET_ABSOLUTE_FLOOR = 0.02  # percentage points, so rounding noise on tiny budgets is not a failure


def assert_matched_budget(pct: float, reference_pct: float, tol: Optional[float] = None,
                          strict: Optional[bool] = None) -> None:
    if strict is None:
        strict = os.environ.get("STRICT_BUDGET", "1") == "1"
    if tol is None:
        tol = max(BUDGET_ABSOLUTE_FLOOR, BUDGET_RELATIVE_TOLERANCE * abs(reference_pct))
    delta = abs(pct - reference_pct)
    rel = delta / abs(reference_pct) if reference_pct else float("inf")
    msg = (f"trainable budget {pct:.4f}% vs reference {reference_pct:.4f}% "
           f"(delta {delta:.4f}pp = {100 * rel:.1f}% relative, tol {tol:.4f}pp)")
    if delta > tol:
        if strict:
            raise AssertionError("Matched-budget violation: " + msg)
        logger.warning("Matched-budget WARNING: %s", msg)
    else:
        logger.info("Matched budget OK: %s", msg)


# ----------------------------- LoRA placement -------------------------------

def build_lora_model(base_model, placement: Dict, cfg: TrainingConfig):
    """Apply LoRA from a placement config.

    placement = {"target_modules": <regex string or suffix list>, "rank_pattern": {regex: rank},
                 "alpha_pattern": {regex: alpha}, "default_rank": r}
    A regex target restricts adapters to the selected layers; a suffix list adapts every
    layer (the dense reference arms). alpha follows rank per layer so the effective
    scaling alpha/r is constant across layers of different rank."""
    from peft import LoraConfig, get_peft_model

    default_rank = int(placement.get("default_rank", 8))
    alpha_pattern = placement.get("alpha_pattern") or {
        k: int(round(cfg.lora_alpha_multiplier * v)) for k, v in placement.get("rank_pattern", {}).items()
    }
    lconf = LoraConfig(
        r=default_rank,
        lora_alpha=int(round(cfg.lora_alpha_multiplier * default_rank)),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=placement["target_modules"],
        rank_pattern=placement.get("rank_pattern", {}),
        alpha_pattern=alpha_pattern,
    )
    return get_peft_model(base_model, lconf)


# --------------------------- example tensorization --------------------------

def is_replay_record(record: Dict) -> bool:
    return "condition" not in record and "prompt" in record and "response" in record


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
    """Return dict with input_ids, labels, token_weights, condition_weight (lists/float).

    General-replay records ({"prompt", "response"}) are tensorised as plain supervised
    examples with weight 1.0 and condition weight 1.0."""
    if is_replay_record(record):
        prompt, prefix, rationale, tail = record["prompt"], record["response"], "", ""
        cond_weight = 1.0
    else:
        prompt, prefix, rationale, tail = _build_target_segments(record, cfg)
        cond = record["condition"]
        cond_weight = cfg.condition_weights.get(cond, 1.0) if cfg.condition_adaptive else 1.0
        # PEDAL Classifier + Modifier: only samples the gate flags as carrying the
        # shortcut get the shortcut-breaking upweight; unflagged samples train unchanged.
        if cfg.pedal_classifier_fn is not None and cfg.pedal_classifier_fn(record):
            cond_weight *= cfg.pedal_modifier_weight
        # DART severity weight on audited drift cases.
        if cfg.example_weight_fn is not None:
            cond_weight *= float(cfg.example_weight_fn(record))

    rendered = P.render_chat(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=P.prompt_add_special_tokens(tokenizer))["input_ids"]
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
    return {
        "input_ids": input_ids[: cfg.max_length],
        "labels": labels[: cfg.max_length],
        "token_weights": weights[: cfg.max_length],
        "condition_weight": cond_weight,
    }


def weighted_batch_loss(model, tokenizer, examples: List[Dict]):
    """Sum over examples of (per-example token-weight-normalised CE x condition weight).

    Right-padded so the causal mask is unaffected; padded positions carry label -100 and
    weight 0. Dividing the result by grad_accum and stepping every grad_accum examples
    reproduces the sequential per-example path."""
    import torch
    import torch.nn.functional as F

    device = model.device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), max_len), pad, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    weights = torch.zeros((len(examples), max_len), dtype=torch.float32)
    attn = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        weights[i, :n] = torch.tensor(e["token_weights"])
        attn[i, :n] = 1
    ids, labels, weights, attn = ids.to(device), labels.to(device), weights.to(device), attn.to(device)
    out = model(input_ids=ids, attention_mask=attn)
    # The float32 upcast happens once, at the cross_entropy call below, on the already
    # reshaped tensor. Note this does NOT reduce peak memory relative to upcasting here: the
    # float32 copy is the same [batch*(seq-1), vocab] either way, and it is the largest
    # allocation in the step. Fitting a 40 GB card is achieved by the per-tier micro-batch
    # caps in model_registry, not by this line. Kept in this form only so the cast sits next
    # to the operation that needs it.
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    w = weights[:, 1:]
    mask = tgt != -100
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                         tgt.clamp(min=0).reshape(-1), reduction="none")
    ce = ce.view(tgt.shape) * mask
    per_example = (ce * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)
    cond = torch.tensor([e["condition_weight"] for e in examples], device=device, dtype=per_example.dtype)
    return (per_example * cond).sum()


def weighted_example_loss(model, tokenizer, example: Dict):
    """Single-example path, kept for the dry-run equivalence check."""
    return weighted_batch_loss(model, tokenizer, [example])


# ------------------------------- curriculum ---------------------------------

def curriculum_order(train: List[Dict], replay: List[Dict], cfg: TrainingConfig, epoch: int, rng: random.Random) -> List[Dict]:
    """Deterministic per-epoch ordering: diff proportion ramps; replay mixed in."""
    frac = cfg.epochs - 1
    t = epoch / frac if frac > 0 else 1.0
    diff_prop = cfg.curriculum_diff_start + (cfg.curriculum_diff_end - cfg.curriculum_diff_start) * t
    diff = [r for r in train if r.get("condition") == "diff"]
    equal = [r for r in train if r.get("condition") == "equal"]
    # Anything that is neither is silently invisible to the curriculum: it would never be
    # scheduled, never trained on, and never reported as missing. Refuse instead, because the
    # symptom is a model quietly trained on a subset of its data.
    if len(diff) + len(equal) != len(train):
        unknown = sorted({str(r.get("condition")) for r in train} - {"diff", "equal"})
        raise ValueError(
            f"{len(train) - len(diff) - len(equal)} of {len(train)} training records carry a "
            f"condition the curriculum does not schedule: {unknown}. Expected 'diff' or 'equal'.")
    rng.shuffle(diff)
    rng.shuffle(equal)
    n = len(train)
    # When only one condition is present the other's share must go to the condition that IS
    # present, not be dropped. Splitting n between an empty bucket and a rounded share meant a
    # single-condition set scheduled round(0.2*n) records and silently never trained the rest.
    if not equal:
        n_diff, n_equal = n, 0
    elif not diff:
        n_diff, n_equal = 0, n
    else:
        n_diff = int(round(diff_prop * n))
        n_equal = n - n_diff
    chosen = (diff * ((n_diff // max(1, len(diff))) + 1))[:n_diff] + (equal * ((n_equal // max(1, len(equal))) + 1))[:n_equal]
    rng.shuffle(chosen)
    if replay and cfg.replay_fraction > 0:
        k = int(round(cfg.replay_fraction * len(chosen)))
        rep = (replay * (k // max(1, len(replay)) + 1))[:k]
        merged = chosen + rep
        rng.shuffle(merged)
        return _nonempty(merged, train)
    return _nonempty(chosen, train)


def _nonempty(order: List[Dict], train: List[Dict]) -> List[Dict]:
    """Refuse to hand back an empty epoch for a non-empty training set.

    Every record is bucketed by condition into diff or equal; anything with a third value
    falls out of both buckets and the epoch silently becomes a no-op. Training would then
    "succeed" having updated nothing, and the run would report an untrained adapter as a
    finished result."""
    if train and not order:
        seen = sorted({str(r.get("condition")) for r in train})
        raise ValueError(
            f"Curriculum produced an empty epoch from {len(train)} training records. "
            f"Conditions present: {seen}; only 'diff' and 'equal' are scheduled.")
    return order


# ------------------------------- train loop ---------------------------------

def _device_and_peak_memory():
    """Return (device_name, peak_gpu_memory_gigabytes). CPU runs report 0.0."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            return name, round(float(peak), 3)
        return "cpu", 0.0
    except Exception:
        return "unknown", 0.0


def _reset_peak_memory():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def release_model(*objects) -> None:
    """Drop references to a finished arm's model and return its memory to the GPU.

    `del model` alone is not enough. It drops one reference while the PEFT wrapper, the
    optimiser and local frames may hold others, and even once everything is collected PyTorch's
    caching allocator keeps the freed blocks reserved. On a 40 GB card that showed up as
    "23 GiB allocated, 15.7 GiB reserved but unallocated" and the next arm could not allocate
    2 MiB. Collect, then empty the cache, so each arm starts from a clean card."""
    import gc

    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _drop_cost_rows(cost_record: Dict) -> None:
    """Remove any earlier row for this (tier, method, seed) from the cost log."""
    import csv

    if not RUN_COST_LOG.exists():
        return
    key = ("tier", "method", "random_seed")
    try:
        with open(RUN_COST_LOG, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if not all(str(r.get(k)) == str(cost_record.get(k)) for k in key)]
        with open(RUN_COST_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RUN_COST_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as e:
        logger.warning("Could not de-duplicate the cost log (%s); rows may repeat.", type(e).__name__)


def _discard_rehearsal_artifacts(out_dir: Path) -> None:
    """Delete a step-capped rehearsal's checkpoints before a full run trains this arm.

    existing_final() already refuses to treat a rehearsal as finished, but that alone is not
    enough: train_lora resumes from the newest epoch_N directory, so the real run would load
    the rehearsal's two-step weights, skip curriculum epoch 0 entirely, and report a full
    result trained on top of a rehearsal. The artefacts have to go, not just be ignored."""
    import shutil

    if smoke_max_steps():          # still rehearsing; keep them for resume
        return
    summary = out_dir / TRAIN_SUMMARY_FILENAME
    epochs = sorted(out_dir.glob("epoch_*"))
    if not epochs and not (out_dir / "final").exists():
        return
    # The test is "can this be PROVEN to be a finished full run?", not "does it say it was a
    # rehearsal?". A rehearsal killed between arms, or one that crashed before writing its
    # summary, leaves epoch_0 with no summary at all; the earlier form returned early on
    # exactly that case, and the study then resumed from two-step weights and skipped epoch 0.
    if summary.exists():
        try:
            if not json.loads(summary.read_text(encoding="utf-8")).get("step_capped_rehearsal"):
                return                                  # a genuine finished arm
        except Exception:
            pass                                        # unreadable: cannot prove it, discard
    removed = []
    for path in sorted(out_dir.glob("epoch_*")) + sorted(out_dir.glob("repair_epoch_*")) + [out_dir / "final"]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    summary.unlink(missing_ok=True)
    logger.warning("Discarded rehearsal artefacts in %s (%s); this arm trains from scratch.",
                   out_dir.name, ", ".join(removed) or "none")


def existing_final(out_dir: Path) -> Optional[Dict]:
    """Return the saved training summary when this arm already finished, else None.

    An arm whose step cap fired counts as unfinished for a full run. The verification pass
    trains two steps per arm to prove the plumbing works; treating its output as complete
    would make the real study skip every arm it touched and report results from adapters that
    saw a couple of optimiser steps."""
    final = Path(out_dir) / "final" / "adapter_config.json"
    summary = Path(out_dir) / TRAIN_SUMMARY_FILENAME
    if final.exists() and summary.exists() and not force_retrain():
        try:
            doc = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            return None
        if doc.get("step_capped_rehearsal") and not smoke_max_steps():
            logger.warning("%s holds a step-capped rehearsal adapter, not a finished arm; "
                           "it will be retrained.", out_dir)
            return None
        return doc
    return None


def _load_adapter_weights(peft_model, adapter_dir: Path) -> bool:
    """Restore adapter weights from a saved epoch directory (resume)."""
    try:
        from peft import set_peft_model_state_dict

        st = adapter_dir / "adapter_model.safetensors"
        if st.exists():
            from safetensors.torch import load_file

            state = load_file(str(st))
        else:
            import torch

            state = torch.load(str(adapter_dir / "adapter_model.bin"), map_location="cpu")
        set_peft_model_state_dict(peft_model, state)
        return True
    except Exception as e:
        logger.warning("Could not restore adapter weights from %s (%s); training from scratch.", adapter_dir, e)
        return False


def _run_epoch(peft_model, tokenizer, records, cfg, optimizer, grad_accum, micro, step, examples_seen, cap):
    """One pass over `records` with micro-batching inside the accumulation window.
    Returns (step, examples_seen, capped)."""
    import torch

    params = [p for p in peft_model.parameters() if p.requires_grad]
    peft_model.train()
    optimizer.zero_grad()
    pending = 0
    for start in range(0, len(records), micro):
        chunk = records[start : start + micro]
        examples = [make_example_tensors(tokenizer, r, cfg) for r in chunk]
        loss = weighted_batch_loss(peft_model, tokenizer, examples) / grad_accum
        loss.backward()
        pending += len(examples)
        examples_seen += len(examples)
        # >= not ==: micro need not divide grad_accum, and overshooting means the window
        # holds more examples than the divisor assumes. Rescale so the effective batch is the
        # one configured whatever micro-batch the tier uses.
        if pending >= grad_accum:
            if pending != grad_accum:
                for prm in params:
                    if prm.grad is not None:
                        prm.grad.mul_(grad_accum / float(pending))
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pending = 0
            step += 1
            if cap and step >= cap:
                logger.info("Smoke step cap %d reached.", cap)
                return step, examples_seen, True
    if pending > 0:
        # The trailing window holds fewer than grad_accum examples, but every micro-batch loss
        # was divided by the full grad_accum. Without this rescale the last few examples of
        # each epoch contribute a fraction of the gradient the rest do, which is a silent,
        # data-order-dependent reweighting of the training set.
        if pending < grad_accum:
            scale = grad_accum / float(pending)
            for prm in params:
                if prm.grad is not None:
                    prm.grad.mul_(scale)
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
    return step, examples_seen, bool(cap and step >= cap)


def train_lora(
    model,
    tokenizer,
    train_records: List[Dict],
    placement: Dict,
    out_dir: Path,
    cfg: TrainingConfig,
    replay_records: Optional[List[Dict]] = None,
    reference_budget_pct: Optional[float] = None,
    cost_record: Optional[Dict] = None,
    audit_hook: Optional[Callable] = None,
) -> Dict:
    """Train a LoRA adapter with the joint loss, curriculum, and replay. Resumable per
    epoch (weights restored from the last saved epoch). Returns a metadata dict
    including trainable_parameter_percentage.

    audit_hook, when given, implements DART's Stage II -> Stage III transition
    (arXiv:2604.16845). It is called once after the main phase as
    audit_hook(peft_model, tokenizer) and returns (repair_records, severity_weight_fn).
    The repair phase then continues fine-tuning on the original training stream plus the
    severity-oversampled repair set, with the severity weight applied per example.

    cost_record, when given, supplies the identifying fields (tier, method, seed,
    quantization) for the row appended to results/training_run_cost_log.csv."""
    import torch

    out_dir = Path(out_dir)
    peft_model = build_lora_model(model, placement, cfg)
    pct = trainable_parameter_percentage(peft_model)
    if reference_budget_pct is not None:
        assert_matched_budget(pct, reference_budget_pct)

    out_dir.mkdir(parents=True, exist_ok=True)
    _discard_rehearsal_artifacts(out_dir)
    if force_retrain():
        # Without this the epoch checkpoints survive, start_epoch lands past the last epoch,
        # the loop body never runs, and the "retrained" arm is the old weights re-saved with
        # epochs_run: 0. The one manual recovery lever has to actually work.
        import shutil

        for path in sorted(out_dir.glob("epoch_*")) + sorted(out_dir.glob("repair_epoch_*")):
            shutil.rmtree(path, ignore_errors=True)
        logger.info("FORCE_RETRAIN: cleared checkpoints in %s; training from epoch 0.", out_dir.name)
    start_epoch = latest_epoch(out_dir) + 1
    if start_epoch > 0:
        restored = _load_adapter_weights(peft_model, out_dir / f"epoch_{start_epoch - 1}")
        if restored:
            logger.info("Resuming %s from epoch %d (adapter weights restored; optimizer state is fresh).",
                        out_dir.name, start_epoch)
        else:
            start_epoch = 0

    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    rng = random.Random(cfg.seed)
    replay_records = replay_records or []
    cap = smoke_max_steps()
    grad_accum = cfg.grad_accum
    micro = min(micro_batch_size(cfg), grad_accum)
    # A larger backbone carries its own micro-batch cap; the effective batch is unchanged
    # because gradient accumulation absorbs the difference, so the optimisation is identical.
    model_cap = int(getattr(model, "_agrifair_max_train_micro_batch", 0) or 0)
    if model_cap > 0:
        micro = min(micro, model_cap)
    step = 0
    examples_seen = 0
    capped = False
    _reset_peak_memory()
    started = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        order = curriculum_order(train_records, replay_records, cfg, epoch, rng)
        step, examples_seen, capped = _run_epoch(
            peft_model, tokenizer, order, cfg, optimizer, grad_accum, micro, step, examples_seen, cap)
        ep_dir = out_dir / f"epoch_{epoch}"
        peft_model.save_pretrained(str(ep_dir))
        logger.info("Saved checkpoint %s", ep_dir)
        if capped:
            break

    # DART Stage II then Stage III: audit the intermediate model against the frozen
    # baseline, then continue fine-tuning on the distillation data plus the
    # severity-oversampled repair set (arXiv:2604.16845, "Targeted Repair").
    repair_records = []
    if audit_hook is not None:
        peft_model.eval()
        try:
            repair_records, severity_fn = audit_hook(peft_model, tokenizer)
            if severity_fn is not None:
                cfg.example_weight_fn = severity_fn
        except Exception as e:
            logger.warning("Audit hook failed (%s); skipping the repair phase.", e)
            repair_records = []

    if repair_records and not capped:
        logger.info("Repair phase: %d oversampled repair examples, %d epochs.",
                    len(repair_records), cfg.repair_epochs)
        for r_epoch in range(cfg.repair_epochs):
            merged = list(train_records) + list(repair_records)
            rng.shuffle(merged)
            step, examples_seen, capped = _run_epoch(
                peft_model, tokenizer, merged, cfg, optimizer, grad_accum, micro, step, examples_seen, cap)
            peft_model.save_pretrained(str(out_dir / f"repair_epoch_{r_epoch}"))
            if capped:
                break

    # A capped run is a rehearsal, not a result. The adapter is still written, because the
    # rehearsal exists to exercise the evaluation path that reads it, but the summary records
    # that the step cap fired and existing_final() refuses to hand a rehearsal to a real run.
    # Without that, the study would skip every arm the rehearsal touched and publish a table
    # computed from two-step models.
    peft_model.save_pretrained(str(out_dir / "final"))
    if capped:
        logger.warning("Step cap reached for %s: recording this arm as a rehearsal, so a full "
                       "run will retrain it rather than reuse it.", out_dir.name)
    minutes = (time.time() - started) / 60.0
    device_name, peak_gb = _device_and_peak_memory()
    if cost_record is not None:
        # Replace rather than append: a resumed arm would otherwise contribute a second row
        # counting only the epochs re-run, and the efficiency analysis takes the mean over
        # rows, so an interruption would silently corrupt fairness-gain-per-minute.
        _drop_cost_rows(cost_record)
        append_csv_row(
            RUN_COST_LOG,
            {
                **cost_record,
                "trainable_parameter_percentage": round(pct, 5),
                "wall_clock_minutes_training_loop": round(minutes, 4),
                "peak_gpu_memory_gigabytes": peak_gb,
                "optimizer_steps_completed": step,
                "training_examples_seen": examples_seen,
                "device_name": device_name,
            },
            RUN_COST_COLUMNS,
        )
    result = {
        "trainable_parameter_percentage": pct,
        # what actually ran, not what was scheduled: a capped or interrupted run must not
        # report three epochs
        "epochs_run": 0 if capped else max(0, cfg.epochs - start_epoch),
        "step_capped_rehearsal": bool(capped),
        "out_dir": str(out_dir),
        "wall_clock_minutes_training_loop": round(minutes, 4),
        "peak_gpu_memory_gigabytes": peak_gb,
        "optimizer_steps_completed": step,
        "training_examples_seen": examples_seen,
        "micro_batch_size": micro,
        "device_name": device_name,
    }
    (out_dir / TRAIN_SUMMARY_FILENAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
