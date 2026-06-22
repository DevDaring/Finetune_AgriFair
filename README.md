# AgriFair XLoRA-Bias

Explainability-guided LoRA for contextual fairness in agricultural LLMs. This repository ports
the **XLoRA-Bias** method (see [Instruction.md](Instruction.md)) from its original legal
difference-awareness benchmark onto the **AgriFair** dataset (`Debk/AgriFair`). The full
adaptation rationale is in [coding_prompt.md](coding_prompt.md); a beginner walk-through with two
fully worked examples is in [explanation.html](explanation.html).

The method localises where a model encodes the *collapse-to-equal* failure (here: answering
"Roughly equal" when the 2015-16 Agriculture Census shows a real gap) with Integrated Gradients,
places LoRA adapters only on those components with rank set by attribution, trains with a
condition-adaptive and rationale-aware loss, and re-runs attribution plus a linear probe, a
parameter-space geometry analysis, and a Patchscope readout to verify the targeted bias signal is
reduced while difference awareness is preserved. **Localise the bias, repair only that, preserve
the rest.**

## What AgriFair adds

| Fairness obligation | AgriFair carrier | Failure caught |
|---|---|---|
| Acknowledge a real difference | AgriFacts `diff` items | over-equalization (erasing a census gap) |
| Do not invent a difference (MCQ) | AgriFacts `equal` items | spurious differential |
| Do not let identity change a fact (free text) | AgriAdvice pairs | advice drift |

The primary endpoint is the contextual fairness score (harmonic mean of `diff` and `equal`
accuracy) on the AgriFacts frozen test set.

## Layout

```
Codes/
  .env / .env.example        secrets (env_loader is the only reader; never printed)
  requirements_global.txt    pinned; no virtual environment
  run_all.py                 single entry point (resumable)
  coding_prompt.md           the adaptation plan/spec
  Instruction.md             the original legal spec (reference)
  PREREGISTRATION.md          endpoint, seeds, tests, frozen-test hash
  explanation.html           beginner walk-through (two worked examples)
  Dataset/                   raw AgriFair download (agrifacts.jsonl, agriadvice.jsonl, README.md)
  Dataset_Prep/ Dry_Run/ GPU_Run/ CPU_Run/
  data/ results/ checkpoints/ models/   (generated; git-ignored)
```

## Quick start

```
pip install -r requirements_global.txt
python GPU_Run/common/flash_attn_setup.py          # installs flash-attn wheel or selects sdpa
cp .env.example .env                                # then fill in keys (or use your existing .env)

# fast offline end-to-end check on a tiny model:
python run_all.py --smoke

# the real study:
python run_all.py --stage dataset_prep
python run_all.py --stage dry_run
python run_all.py --stage gpu_run      # start GPU_Run/autosync_results.py in the background first
# then, on the CPU machine:
python GPU_Run/restore_artifacts.py
python run_all.py --stage cpu_run
```

Run order, machine split, crash-safety, batching, and the full configuration reference follow
[Instruction.md](Instruction.md) Sections 12-16; the AgriFair-specific deltas are in
[coding_prompt.md](coding_prompt.md).

## Engineering contract

Single deterministic JSON answer extraction (no judge for answers); round-robin API keys read only
from `.env`; offline after one download; resumable checkpointing and incremental CSV writes; full
descriptive CSV column names; data hygiene every run; matched trainable-parameter budgets asserted
in code; no projected numbers; no emoji.

## Data provenance

AgriFacts answers are arithmetic on the *All India Report on Agriculture Census 2015-16*; AgriAdvice
base queries come from `KisanVaani/agriculture-qa-english-only`. No label comes from a language
model. The reference rationale used by the rationale-aware objective is deterministic templating of
each row's own ground truth (qualitative direction + the census cell as citation); no numeric
magnitude is invented. Cite the Agriculture Census of India 2015-16 and the KisanVaani corpus.
