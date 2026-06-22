# CPU_Run — analysis stage (runs after GPU_Run, on the CPU machine)

These scripts consume the CSVs and JSONL files that GPU_Run produced in `results/` and
`data/`. They require no GPU. Bring the artifacts over first:

```
python GPU_Run/restore_artifacts.py     # restores results/, checkpoints/, data/
```

Then run, in order:

```
python CPU_Run/aggregate_results.py
python CPU_Run/statistics_tests.py
python CPU_Run/figures_and_tables.py
python CPU_Run/judge_robustness.py
python CPU_Run/parameter_space_geometry.py
```

Inputs and notes:

- `aggregate_results.py`, `statistics_tests.py`, `figures_and_tables.py` need only
  `results/main_evaluation_results.csv` and the per-item prediction JSONL files.
- `judge_robustness.py` needs judge keys in `.env` (network) and the per-item rationale
  predictions; with no keys it writes a note and exits.
- `parameter_space_geometry.py` additionally needs the base weights in `models/` and the LoRA
  adapters in `checkpoints/` on this machine. Either copy `models/` from the VM or run
  `python Dataset_Prep/download_models_and_data.py` here once. It is CPU float64 SVD and can
  take several hours; `GEOMETRY_LAYER_STRIDE` subsamples layers to speed a run.

Primary endpoint: the contextual fairness score (harmonic mean of diff- and equal-accuracy)
on the AgriFacts test set for the proposed method versus the strongest baseline at matched
parameter budget, with the paired bootstrap and Wilcoxon tests in
`results/statistical_tests.csv`.
