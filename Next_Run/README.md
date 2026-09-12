# Next_Run — the final bounded audit round

Implements `Submission1/Future_PLan.md` (12 Sep 2026). One offline-first runner, separate
from `run_all.py`, that turns the completed GRAFT study into a benchmark-and-reliability
audit without buying more training.

**Default spend: zero GPU-hours, zero paid API calls.** Three config flags gate everything
that costs money (`gpu_enabled`, `allow_paid_api`, `allow_training`) and all default to
`false`. Nothing in this package flips them, rents a machine, or falls back to a paid path.

```
cd Codes
../.venv/bin/python -m pytest Next_Run/tests -q       # 17 tests, < 1 s
../.venv/bin/python Next_Run/run_final_audit.py       # every stage, resumable
../.venv/bin/python Next_Run/run_final_audit.py --stage paired --force
```

Outputs go to `results_final_audit_20260912/` (from `config.yaml`). The package refuses to
write into `results/` or `data/`; a `PermissionError` is raised if any stage tries.

## Stages

| Stage | Plan | What it does | Spend |
|---|---|---|---|
| `inventory` | P0 | Frozen-test SHA check, schema/id/leakage checks on all splits, per-arm prediction coverage (LOAO-aware), advice coverage, checkpoint weights present (configs alone don't count), versions, deviations-log scaffold | none |
| `verify_sources` | P1 | Ledger **skeleton** from 1,178 distinct `source_cell` keys; frozen 60→120 stratified audit sample; validation + gold recomputation of a human-filled ledger. Refuses to recompute unless the comparison rule is declared in config and values are present | none |
| `cpu_baselines` | P2.1 | Training-majority, always-equal (diagnostic), metadata LR on the four `state_blind_key` fields, question-only TF-IDF LR — fixed configs, train-only fit, per-item predictions + probabilities saved | none |
| `paired` | P2.2–2.3 | Seed-explicit metrics with every error type; paired source-cluster bootstrap (10k) + cluster-swap permutation (10k), vectorised on contingency arrays; localisation / secondary / CPU families, Holm per family; state-blind-key clustering sensitivity; LOAO error taxonomy | none |
| `advice` | P3 | Cached-pair audit (length, duplication, TF-IDF distance labelled as such, repetition, refusal, *possible* truncation), complete-data then non-flagged subset; blinded human-study bundles (48 questions, 192 assessments/rater, mapping kept separate, frozen rubric); rating analysis if `ratings_independent.csv` exists | none |
| `evidence_panel` | P4 | Builds 96 bundles × 4 conditions from **verified** ledger values only; deterministic scorer; manual-check worksheet; five gates; pilot-timed budget projection with 1.5× safety and a hard 120-min stop; nested 48-bundle fallback; skip if neither fits | **GPU only if all gates pass** |
| `report` | P5 | Four main tables, three figures, `claims_to_evidence.json` marking each intended claim supported / unsupported / do-not-claim | none |

## What the first run established (12 Sep 2026)

- Frozen test hash matches; 0 inventory findings; 132/132 checkpoints carry real weights.
- CPU baselines reproduce the plan's §2.2 table to four decimals.
- Advice length/duplication/distance/judge diagnostics reproduce §2.3 exactly.
- Under source-cluster resampling with Holm across the 16-contrast localisation family,
  GRAFT differs significantly from a placement control in **2 of 16** contrasts on the
  novel slice — and in both the control scores higher. The hypothesis is unsupported.
- GRAFT vs the TF-IDF baseline: every model's CI crosses zero.
- **Gate not passed:** no numeric census values exist anywhere in the repo, so the ledger
  is skeleton-only and P4 cannot run. Fill `sources/evidence_ledger_filled.csv` from the
  construction records and declare `comparison_rule` in config to proceed.
- **Finding for the paper:** no token counts or finish reasons were saved with cached
  advice; frozen-model answers flag as possibly truncated ~90% of the time (long, no
  terminal punctuation). Report as a heuristic with its uncertainty.

## Honesty rules baked in

- Undefined B when a condition is absent — never zero.
- Erasure, wrong-group, fabrication and invalid are counted separately.
- Unavailable contrasts stay in the table with `status=unavailable`.
- Seed policy is a column: `matched_multi_seed` / `seed_42_paired` / `unavailable`.
- Monte Carlo p is `(extreme+1)/(draws+1)` — never exactly zero.
- Synthetic evidence is labelled hypothetical in every prompt and every output row.
- `scrub_secrets` is applied to every error message before it is written.
