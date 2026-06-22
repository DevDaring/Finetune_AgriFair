"""The ONLY networked script: downloads the subject models, the AgriFair dataset (if
absent), and the general-replay set; pins revisions (Instruction.md Section 11, 12).

Reads the HF token only here, via env_loader (never printed). Idempotent: skips what is
already present. The AgriFair dataset is normally already in Dataset/; this script
re-fetches it only if missing.

Run:  python Dataset_Prep/download_models_and_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common import env_loader, model_registry
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import (
    AGRIADVICE_RAW,
    AGRIFACTS_RAW,
    DATASET_DIR,
    DATASET_REVISIONS,
    GENERAL_REPLAY,
    MODELS_DIR,
)

logger = get_logger("download_models_and_data")

AGRIFAIR_REPO = env_loader.get("PRIMARY_DATASET_REPO", "Debk/AgriFair")
GENERAL_REPLAY_REPO = env_loader.get("GENERAL_REPLAY_REPO", "databricks/databricks-dolly-15k")
GENERAL_REPLAY_ROWS = 500


def _download_dataset_if_missing(token):
    if AGRIFACTS_RAW.exists() and AGRIADVICE_RAW.exists():
        logger.info("AgriFair already present in %s; skipping dataset download.", DATASET_DIR)
        return
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s into %s", AGRIFAIR_REPO, DATASET_DIR)
    snapshot_download(repo_id=AGRIFAIR_REPO, repo_type="dataset", local_dir=str(DATASET_DIR), token=token)


def _download_general_replay(token):
    if GENERAL_REPLAY.exists():
        logger.info("General-replay set already present; skipping.")
        return
    try:
        from datasets import load_dataset

        ds = load_dataset(GENERAL_REPLAY_REPO, split=f"train[:{GENERAL_REPLAY_ROWS}]", token=token)
        with open(GENERAL_REPLAY, "w", encoding="utf-8") as f:
            for row in ds:
                instr = row.get("instruction", "")
                ctx = row.get("context", "")
                resp = row.get("response", "")
                prompt = (instr + ("\n\n" + ctx if ctx else "")).strip()
                f.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
        logger.info("Wrote %d general-replay rows.", GENERAL_REPLAY_ROWS)
    except Exception as e:
        logger.warning("General-replay download failed (%s); training replay will be empty.", e)


def _download_models(token):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    revisions = {}
    for tier in model_registry.active_tiers():
        spec = model_registry.get_spec(tier)
        local = MODELS_DIR / spec.hf_id.replace("/", "__")
        if local.exists():
            logger.info("Model %s present; skipping.", spec.hf_id)
        else:
            try:
                logger.info("Downloading model %s", spec.hf_id)
                snapshot_download(repo_id=spec.hf_id, local_dir=str(local), token=token)
            except Exception as e:
                logger.warning("Model %s download failed (%s); skipping.", spec.hf_id, e)
                continue
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=token).model_info(spec.hf_id)
            revisions[tier] = info.sha
            model_registry.record_revision(tier, info.sha)
        except Exception:
            revisions[tier] = "unknown"
    return revisions


def main():
    token = env_loader.hf_token()
    if not token:
        logger.warning("No HF token in .env; gated models and dataset re-fetch will fail.")
    _download_dataset_if_missing(token)
    _download_general_replay(token)
    revisions = _download_models(token)
    DATASET_REVISIONS.write_text(json.dumps({"agrifair_repo": AGRIFAIR_REPO}, indent=2), encoding="utf-8")
    log_run_metadata("download_models_and_data", {"model_revisions": revisions})
    logger.info("Download stage complete.")


if __name__ == "__main__":
    main()
