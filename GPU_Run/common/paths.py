"""Canonical filesystem paths for the AgriFair XLoRA-Bias study.

Repository root is the Codes/ folder. This module is imported everywhere and
creates the generated output trees (data/, results/, checkpoints/, models/)
on import so no script has to guard for their absence.

Adapted from XLoRA-Bias (Instruction.md, Section 11) for the AgriFair dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

# GPU_Run/common/paths.py -> parents[2] == Codes/
REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_on_path() -> None:
    p = str(REPO_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_on_path()

# Raw download (as shipped by Debk/AgriFair).
DATASET_DIR = REPO_ROOT / "Dataset"
AGRIFACTS_RAW = DATASET_DIR / "agrifacts.jsonl"
AGRIADVICE_RAW = DATASET_DIR / "agriadvice.jsonl"
DATASET_README = DATASET_DIR / "README.md"

# Generated trees (git-ignored, recreated by code).
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
GEOMETRY_DIR = RESULTS_DIR / "geometry"
DRY_RUN_DIR = RESULTS_DIR / "dry_run"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
MODELS_DIR = REPO_ROOT / "models"
EXTERNAL_WANG_DIR = DATA_DIR / "external_wang"

# Derived data files (written by Dataset_Prep).
TEMPLATED_ALL = DATA_DIR / "templated_all_instances.jsonl"
TRAIN_INSTANCES = DATA_DIR / "train_instances.jsonl"
VALIDATION_INSTANCES = DATA_DIR / "validation_instances.jsonl"
TEST_INSTANCES_FROZEN = DATA_DIR / "test_instances_frozen.jsonl"
COUNTERFACTUAL_PAIRS = DATA_DIR / "counterfactual_pairs.jsonl"
AGRIADVICE_PAIRS = DATA_DIR / "agriadvice_pairs.jsonl"
GENERAL_REPLAY = DATA_DIR / "general_replay.jsonl"
DATASET_REVISIONS = DATA_DIR / "dataset_revisions.json"
MODEL_REVISIONS = MODELS_DIR / "model_revisions.json"

# Key result files.
FROZEN_TEST_SHA256 = RESULTS_DIR / "frozen_test_set_sha256.txt"
DATA_HYGIENE_LOG = RESULTS_DIR / "data_hygiene_log.csv"
CONTAMINATION_REPORT = RESULTS_DIR / "contamination_report.csv"
RUN_METADATA = RESULTS_DIR / "run_metadata.jsonl"
MAIN_EVALUATION = RESULTS_DIR / "main_evaluation_results.csv"

_OUTPUT_TREES = [
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    GEOMETRY_DIR,
    DRY_RUN_DIR,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    EXTERNAL_WANG_DIR,
]

for _d in _OUTPUT_TREES:
    _d.mkdir(parents=True, exist_ok=True)
