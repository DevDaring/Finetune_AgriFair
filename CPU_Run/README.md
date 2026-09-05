# CPU_Run - analysis stage (runs after GPU_Run, no GPU required)

These scripts consume the CSVs and JSONL files GPU_Run wrote into `results/` and `data/`.
Bring the artifacts over first:

```
python GPU_Run/restore_artifacts.py     # restores results/, checkpoints/, data/
```

Then run in order, or let `python run_all.py --stage cpu_run` do it:

```
python CPU_Run/aggregate_results.py
python CPU_Run/statistics_tests.py
python CPU_Run/audit_awareness_trade.py
python CPU_Run/judge_robustness.py
python CPU_Run/parameter_space_geometry.py
python CPU_Run/repair_admissibility.py
python CPU_Run/figures_and_tables.py
```

Order matters in two places. `parameter_space_geometry.py` must run before
`repair_admissibility.py`, which reads the relative Frobenius drift as its invasiveness
term. `figures_and_tables.py` runs last so the frontier and awareness-trade figures already
exist.

## What each script needs

| script | inputs | notes |
|---|---|---|
| `aggregate_results.py` | `main_evaluation_results.csv`, per-item predictions, the frozen test set | writes the pooled, per-slice, per-axis and transfer tables |
| `statistics_tests.py` | per-item predictions | paired item-level bootstrap, exact McNemar, Holm correction within each hypothesis family |
| `audit_awareness_trade.py` | per-item predictions | the failure-polarity profile and the awareness-trade table; both are true regardless of where the proposed method ranks |
| `judge_robustness.py` | judge keys in `.env` (network), rationale predictions | with no keys it writes an empty agreement table and exits |
| `parameter_space_geometry.py` | base weights in `models/`, adapters in `checkpoints/`, `main_evaluation_results.csv` | CPU float64 SVD; the slow step |
| `repair_admissibility.py` | predictions, geometry, cost log, probe and Patchscope summaries | the three criteria and the preservation frontier |
| `figures_and_tables.py` | the aggregated tables | draws the surface-cue ceiling as a reference line |

## Cost control on the geometry step

`parameter_space_geometry.py` is the only expensive CPU stage: it merges every adapter and
takes a float64 SVD per weight matrix. `GEOMETRY_LAYER_STRIDE` subsamples layers and
`GEOMETRY_MAX_MATRICES` caps the total; both are recorded in the run log so a subsampled run
is never mistaken for a full one. It needs the base weights locally, so either copy
`models/` from the training machine or run `python Dataset_Prep/download_models_and_data.py`
once here.
