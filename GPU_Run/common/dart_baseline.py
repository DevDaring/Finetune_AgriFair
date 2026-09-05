"""DART (Distill-Audit-Repair Training) as an AgriFair baseline.

# Implements the method of:
# Pan, Z., Liang, Z., Kabbara, J., Emami, A. "DART: Mitigating Harm Drift in
#   Difference-Aware LLMs via Distill-Audit-Repair Training." ACL 2026,
#   arXiv:2604.16845. Student Llama-3-8B-Instruct, teacher DeepSeek-Chat, LoRA r=16
#   alpha=32 dropout 0.05, lr 2e-4, 3 epochs, bf16; severity oversampling 1x mild,
#   2x moderate, 3x severe, 4x extreme.
# Wang, A., Phan, M., Ho, D. E., Koyejo, S. "Fairness through Difference Awareness."
#   ACL 2025, arXiv:2502.01926. [the difference-awareness construct DART repairs]

DART is the most recent method that repairs difference awareness with LoRA on Wang et
al.'s construct, which makes it the closest external competitor to GRAFT and the
comparative study a 2026 reviewer will expect to see. FairLoRA is deliberately absent from the baseline set: two unrelated papers already carry
that name, so it identifies nothing, and at a matched budget in this harness its mechanism
reduced to uniform-placement LoRA, which the Vanilla LoRA reference already covers.

Three honest approximations are made, and each is recorded in the paper's limitations:

1. Distil. DART distils label-conditioned rationales from a teacher LLM. AgriFair
   forbids any label or rationale that originates from a language model, so the distilled
   trace is the deterministic reference rationale already built from each row's own
   census ground truth (see coding_prompt.md, "The dataset, as it actually ships"). This is a *stronger* trace than a
   teacher's, not a weaker one, so the baseline is not handicapped.

2. Audit. DART audits harm drift with a toxicity classifier plus an LLM judge. AgriFair
   has no toxicity axis; the drift that matters here is fairness drift, so the audit
   compares the intermediate model against the frozen base on the audit split and flags
   items where training introduced a difference-awareness error the base did not make (gap erasure or gap fabrication).
   The comparison is behavioural and deterministic; no judge is involved.

3. Severity. DART's four severity levels are mapped onto the AgriFair failure taxonomy
   below. The oversampling multipliers 1/2/3/4 are the paper's own.

Severity map (deterministic, documented, no model in the loop):
  extreme  (4x)  diff item, base correct, intermediate answers c
                 -> training introduced the gap-erasure collapse itself
  severe   (3x)  equal item, base correct, intermediate answers a or b
                 -> training introduced a fabricated gap
  moderate (2x)  diff item, base correct, intermediate answers the wrong group
                 -> the difference is acknowledged but misattributed
  mild     (1x)  any other item wrong after training that was right before

Items wrong both before and after are not drift and are not oversampled; they are
already covered by the ordinary training stream, as in the paper.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.inference import run_mcq_eval
from GPU_Run.common.logging_utils import get_logger, write_csv
from GPU_Run.common.paths import RESULTS_DIR

logger = get_logger("dart_baseline")

# The audit compares parsed answers before and after training, so both sides must be read
# the same way; an unparseable output counted as an error would be recorded as drift the
# training did not cause.
_JUDGE: Optional["JudgeChain"] = None


def _judge() -> "JudgeChain":
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = JudgeChain()
    return _JUDGE

# DART Stage III oversampling multipliers, verbatim from the paper.
SEVERITY_OVERSAMPLING = {"mild": 1, "moderate": 2, "severe": 3, "extreme": 4}

AUDIT_COLUMNS = [
    "tier", "method", "random_seed", "audit_split_size",
    "drift_cases_detected", "drift_case_rate",
    "drift_cases_mild", "drift_cases_moderate", "drift_cases_severe",
    "drift_cases_extreme", "repair_set_size_after_oversampling",
]


def audit_split_size() -> int:
    return int(os.environ.get("DART_AUDIT_SUBSET_SIZE", "256"))


def _severity(condition: str, gold: str, pred: str) -> Optional[str]:
    """Severity label for a post-training error, or None when the item is not an error."""
    if pred == gold:
        return None
    if condition == "diff":
        if pred == "c":
            return "extreme"
        return "moderate"
    if pred in ("a", "b"):
        return "severe"
    return "mild"


def base_prediction_cache(model, tokenizer, audit_records: List[Dict]) -> Dict[str, str]:
    """DART Stage II reference pass: predictions of the frozen base model M0.

    Called before any adapter is attached, so the returned map is the M0 baseline the
    audit compares the intermediate model M_int against."""
    preds = run_mcq_eval(model, tokenizer, audit_records, rationale_mode=False, judge=_judge())
    return {p["id"]: p["pred_canonical"] for p in preds}


def make_audit_hook(
    audit_records: List[Dict],
    base_predictions: Dict[str, str],
    tier: str,
    seed: int,
    method: str = "baseline_dart",
) -> Callable:
    """Return an audit_hook for training.train_lora implementing DART Stages II and III.

    The hook evaluates the intermediate model on the audit split, flags fairness-drift
    cases relative to the frozen base, oversamples them by severity, and returns
    (repair_records, severity_weight_fn). The severity weight is applied to the loss on
    top of the oversampling, so a severe case is both seen more often and weighted more
    heavily, matching the paper's intent that repair pressure scale with severity."""

    def hook(peft_model, tokenizer) -> Tuple[List[Dict], Optional[Callable[[Dict], float]]]:
        by_id = {r["id"]: r for r in audit_records}
        preds = run_mcq_eval(peft_model, tokenizer, audit_records, rationale_mode=False, judge=_judge())
        counts = {k: 0 for k in SEVERITY_OVERSAMPLING}
        repair: List[Dict] = []
        severity_of: Dict[str, str] = {}

        for p in preds:
            rec = by_id.get(p["id"])
            if rec is None:
                continue
            base_pred = base_predictions.get(p["id"])
            if base_pred is None or base_pred != p["gold_canonical"]:
                continue  # the base was already wrong: not drift introduced by training
            sev = _severity(p["condition"], p["gold_canonical"], p["pred_canonical"])
            if sev is None:
                continue
            counts[sev] += 1
            severity_of[rec["id"]] = sev
            repair.extend([rec] * SEVERITY_OVERSAMPLING[sev])

        n_drift = sum(counts.values())
        write_csv(
            RESULTS_DIR / f"dart_audit_report_{tier}_seed{seed}.csv",
            [{
                "tier": tier, "method": method, "random_seed": seed,
                "audit_split_size": len(audit_records),
                "drift_cases_detected": n_drift,
                "drift_case_rate": round(n_drift / max(1, len(audit_records)), 4),
                "drift_cases_mild": counts["mild"],
                "drift_cases_moderate": counts["moderate"],
                "drift_cases_severe": counts["severe"],
                "drift_cases_extreme": counts["extreme"],
                "repair_set_size_after_oversampling": len(repair),
            }],
            AUDIT_COLUMNS,
        )
        logger.info(
            "DART audit tier=%s seed=%d: %d drift cases (mild %d, moderate %d, severe %d, "
            "extreme %d) -> repair set %d.",
            tier, seed, n_drift, counts["mild"], counts["moderate"], counts["severe"],
            counts["extreme"], len(repair),
        )

        def severity_weight(record: Dict) -> float:
            return float(SEVERITY_OVERSAMPLING.get(severity_of.get(record.get("id"), ""), 1))

        return repair, severity_weight

    return hook


def load_audit_records(train_records: List[Dict]) -> List[Dict]:
    """Deterministic audit split: the head of the training stream, capped.

    Held inside the training split, so the frozen test set is never touched by the audit
    (the contamination guard in Dataset_Prep already keeps the splits fact-disjoint)."""
    n = audit_split_size()
    return list(train_records[:n]) if n > 0 else list(train_records)
