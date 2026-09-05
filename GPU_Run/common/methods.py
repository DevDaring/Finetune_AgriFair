"""Canonical method names, used by every training, evaluation, and analysis module.

The proposed method is GRAFT (Gradient-Ranked Adapter Fairness Targeting): Integrated
Gradients ranks the layers that carry the gap-erasure failure, adapters are placed on
those layers alone with rank proportional to attribution, and they are trained with a
condition-adaptive, rationale-aware objective. Grafting is the agricultural image the
method is named for: a small, local, load-bearing repair that leaves the rest of the
plant intact.

Baseline names are the published names of the methods they approximate and are never
renamed here.
"""
from __future__ import annotations

from typing import List

PROPOSED = "graft_proposed"
PROPOSED_QLORA = "graft_proposed_qlora_nf4"
FROZEN_BASE = "frozen_base"

BASELINES = [
    "baseline_fairnet",       # arXiv:2510.19421, NeurIPS 2025
    "baseline_igu_lora",      # arXiv:2603.13792, ICLR 2026
    "baseline_pedal",         # doi:10.1145/3774904.3793029, WWW 2026
    "baseline_dart",          # arXiv:2604.16845, Findings of ACL 2026
    "baseline_regift",        # arXiv:2504.05632
    "baseline_lftf",          # arXiv:2505.15475
    "baseline_fairsteer",     # arXiv:2504.14492, Findings of ACL 2025
]
REFERENCES = ["reference_vanilla_lora", "reference_vanilla_qlora"]

ABLATIONS = [
    "ablation_placement_uniform",
    "ablation_placement_random",
    "ablation_placement_random_draw2",
    "ablation_placement_random_draw3",
    "ablation_no_rationale_loss",
    "ablation_no_condition_adaptive",
    "ablation_rank_sweep_8",
    "ablation_rank_sweep_16",
    "ablation_rank_sweep_32",
]

# Arms whose statistics need more than one seed on the primary models.
MULTI_SEED_METHODS = [PROPOSED, "baseline_dart", "baseline_fairnet", "reference_vanilla_lora"]
# Arms carried into the leave-one-axis-out transfer study.
TRANSFER_METHODS = [PROPOSED, "reference_vanilla_lora"]


def all_trained_methods() -> List[str]:
    return [PROPOSED, PROPOSED_QLORA] + BASELINES + REFERENCES + ABLATIONS


def is_ablation(method: str) -> bool:
    return method.startswith("ablation_")


def is_reference(method: str) -> bool:
    return method.startswith("reference_")


# Arms whose trainable-parameter budget is HELD EQUAL to the proposed method's, and asserted.
# These exist to isolate WHERE adapters are placed, so an unequal budget would confound the
# only thing they are meant to test.
BUDGET_MATCHED_METHODS = [
    "ablation_placement_uniform",
    "ablation_placement_random",
    "ablation_placement_random_draw2",
    "ablation_placement_random_draw3",
    "ablation_no_rationale_loss",
    "ablation_no_condition_adaptive",
]


def is_budget_matched(method: str) -> bool:
    """Whether this arm's trainable budget must equal the proposed method's.

    Not every arm can or should match. The rank sweep varies rank deliberately, the
    quantized arm counts parameters differently, and each published baseline places
    adapters by its own algorithm: forcing those to a common budget would misreport the
    methods being compared against. Their trainable share is measured and published in
    the results tables instead, so the comparison stays honest without being falsified.

    Asserting the budget only where it is part of the claim keeps the matched-budget
    guarantee meaningful and stops a legitimate arm from aborting the run."""
    return method in BUDGET_MATCHED_METHODS


def uses_rationale_arm(method: str) -> bool:
    """Methods whose evaluation prompt asks for a rationale as well as a letter."""
    return method in (PROPOSED, PROPOSED_QLORA, "baseline_regift", "baseline_dart", FROZEN_BASE) \
        or method.startswith("ablation_rank_sweep") \
        or method in ("ablation_placement_uniform", "ablation_placement_random",
                      "ablation_placement_random_draw2", "ablation_placement_random_draw3",
                      "ablation_no_condition_adaptive")
