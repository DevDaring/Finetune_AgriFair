"""Push every trained LoRA adapter to a HuggingFace repository.

Adapters are what makes this study reproducible by someone else. They are small, a few tens
of megabytes each, so all of them fit in one repository laid out by model, method and seed:

    <tier>/<method>/seed_<n>/          adapter_config.json, adapter_model.safetensors
    manifest.json                      every adapter, its base model, and its trainable share
    README.md                          the model card, generated from the run's own records

Only the final adapter of each arm is pushed. Per-epoch checkpoints exist for resume and are
not scientifically interesting once a run has finished.

The repository is created private by default. A fairness study's adapters are trained to
change a model's behaviour on caste, gender and landholding questions, and that is not
something to publish by accident; pass --public deliberately when the paper is ready.

Reads HUGGINGFACE_TOKEN from .env through env_loader and never prints it.

Usage:
    python deploy/push_models_to_huggingface.py
    python deploy/push_models_to_huggingface.py --repo Debk/AgriFair-GRAFT-adapters --public
    python deploy/push_models_to_huggingface.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from GPU_Run.common import env_loader, model_registry
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import CHECKPOINTS_DIR, RESULTS_DIR

logger = get_logger("push_models_to_huggingface")

DEFAULT_REPO = "Debk/AgriFair-GRAFT-adapters"


# The smoke tier is a 1M-parameter random model used to prove the plumbing. Its adapters are
# meaningless and must never reach a public artefact.
EXCLUDED_TIERS = {"smoke", "dry_run"}


def discover_adapters():
    """Every finished adapter, as (tier, method, seed, directory)."""
    found = []
    if not CHECKPOINTS_DIR.exists():
        return found
    for tier_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not tier_dir.is_dir() or tier_dir.name in EXCLUDED_TIERS:
            continue
        for method_dir in sorted(tier_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            for seed_dir in sorted(method_dir.glob("seed_*")):
                final = seed_dir / "final"
                if (final / "adapter_config.json").exists():
                    found.append((tier_dir.name, method_dir.name,
                                  int(seed_dir.name.split("_")[1]), final))
    return found


def _trainable_percentages():
    out = {}
    for name in ("train_graft_runs.json", "train_baselines_runs.json"):
        p = RESULTS_DIR / name
        if not p.exists():
            continue
        try:
            for r in json.loads(p.read_text(encoding="utf-8")):
                pct = r.get("trainable_parameter_percentage")
                if pct is not None:
                    out[(r.get("tier"), r.get("method"), r.get("seed"))] = round(float(pct), 4)
        except Exception:
            continue
    return out


def build_manifest(adapters):
    pcts = _trainable_percentages()
    entries = []
    for tier, method, seed, path in adapters:
        spec = model_registry.get_spec(tier)
        cfg = {}
        try:
            cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        ranks = list((cfg.get("rank_pattern") or {}).values())
        entries.append({
            "path_in_repo": f"{tier}/{method}/seed_{seed}",
            "model_tier": tier,
            "base_model": spec.hf_id,
            "base_model_revision": model_registry.get_revision(tier),
            "method": method,
            "random_seed": seed,
            "lora_rank_range": f"{min(ranks)}-{max(ranks)}" if ranks else cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "target_modules": cfg.get("target_modules"),
            "trainable_parameter_percentage": pcts.get((tier, method, seed)),
        })
    return {"adapters": entries, "count": len(entries)}


def build_model_card(manifest, repo_id: str) -> str:
    tiers = sorted({e["model_tier"] for e in manifest["adapters"]})
    methods = sorted({e["method"] for e in manifest["adapters"]})
    base_models = sorted({e["base_model"] for e in manifest["adapters"]})
    rows = "\n".join(
        f"| `{t}` | {model_registry.get_spec(t).hf_id} | "
        f"{len([e for e in manifest['adapters'] if e['model_tier'] == t])} |"
        for t in tiers)
    tags = "\n".join(f"- {t}" for t in
                     ["fairness", "bias-mitigation", "lora", "peft", "agriculture", "india",
                      "difference-awareness"])
    return f"""---
library_name: peft
license: cc-by-4.0
language:
- en
tags:
{tags}
base_model:
{chr(10).join(f"- {b}" for b in base_models)}
---

# AgriFair GRAFT adapters

LoRA adapters from **GRAFT** (Gradient-Ranked Adapter Fairness Targeting), a study of
difference-aware fairness in agricultural language models, grounded in the Indian Agriculture
Census 2015-16 through the AgriFair benchmark.

The method locates where a model encodes **gap erasure**, answering "Roughly equal" where the
census records a real difference, using Integrated Gradients. It places LoRA adapters on those
layers alone with rank proportional to attribution, and trains them with a condition-adaptive,
rationale-aware objective.

This repository holds **{manifest['count']} adapters** so that every number in the paper can be
recomputed rather than taken on trust.

## Layout

```
<tier>/<method>/seed_<n>/     adapter_config.json, adapter_model.safetensors
manifest.json                 base model, revision, rank range and trainable share per adapter
```

| Tier | Base model | Adapters |
|---|---|---|
{rows}

Methods present: {', '.join(f'`{m}`' for m in methods)}

## Using one

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "{base_models[0]}"
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "{repo_id}", subfolder="{manifest['adapters'][0]['path_in_repo']}")
tokenizer = AutoTokenizer.from_pretrained(base)
```

`manifest.json` gives the exact base model and revision each adapter was trained against.
Loading an adapter onto a different base, or a different revision, is not a valid comparison.

## Scope and limitations

These adapters are trained to change behaviour on questions about caste, gender and landholding
inequality in Indian agriculture. They are a research artefact:

- Trained and evaluated in **English only**, on a templated benchmark.
- AgriFair concedes **0.872** to a predictor that never reads the state name, so any accuracy
  must be read against that ceiling. The study reports a structure-novel test slice for exactly
  this reason.
- The gender axis has 180 items, since the census reports gender only at national level, so
  gender results carry wide intervals.
- Not validated for deployment in any advisory system, and not a substitute for agronomic or
  policy advice.

## Citation

Cite the *All India Report on Agriculture Census 2015-16* (Department of Agriculture and
Farmers Welfare, Government of India) as the source of every AgriFacts answer, and
`KisanVaani/agriculture-qa-english-only` for the AgriAdvice base queries.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Push trained adapters to HuggingFace")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--public", action="store_true",
                    help="create the repository public; private by default, deliberately")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = env_loader.hf_token()
    if not token and not args.dry_run:
        logger.error("No HuggingFace token in .env; nothing pushed.")
        return 1

    adapters = discover_adapters()
    if not adapters:
        logger.error("No finished adapters under %s; nothing to push.", CHECKPOINTS_DIR)
        return 1
    manifest = build_manifest(adapters)
    logger.info("Found %d adapters across %d tiers.",
                manifest["count"], len({e["model_tier"] for e in manifest["adapters"]}))

    staging = RESULTS_DIR / "huggingface_upload"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (staging / "README.md").write_text(build_model_card(manifest, args.repo), encoding="utf-8")

    import shutil

    for tier, method, seed, path in adapters:
        target = staging / tier / method / f"seed_{seed}"
        target.mkdir(parents=True, exist_ok=True)
        for f in path.iterdir():
            if f.is_file():
                shutil.copy2(f, target / f.name)

    size_mb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1024 ** 2
    logger.info("Staged %.1f MB at %s", size_mb, staging)

    if args.dry_run:
        logger.info("Dry run: nothing uploaded. Repository would be %s (%s).",
                    args.repo, "public" if args.public else "private")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(staging),
                      commit_message=f"AgriFair GRAFT adapters: {manifest['count']} adapters")
    logger.info("Pushed %d adapters to https://huggingface.co/%s (%s).",
                manifest["count"], args.repo, "public" if args.public else "private")
    return 0


if __name__ == "__main__":
    sys.exit(main())
