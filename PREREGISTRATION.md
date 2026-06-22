# PREREGISTRATION — AgriFair XLoRA-Bias

This file records the endpoint, seeds, tests, frozen-test hash, and thresholds fixed before
the expensive runs. It is the agriculture port of the legal-benchmark preregistration
(Instruction.md). No projected or simulated numbers appear here; values produced by real runs
are written back by the pipeline.

## Primary endpoint

The **contextual fairness score** = harmonic mean of `diff` (neq) accuracy and `equal` (eq)
accuracy on the **AgriFacts frozen test set**, for `xlora_bias_proposed` versus the
automatically-selected strongest baseline at a matched trainable-parameter budget.

## Secondary endpoints

- Difference awareness preserved: `diff`-accuracy and Wang DiffAware before/after each method.
- Contextual awareness: `equal`-accuracy and Wang CtxtAware.
- AgriAdvice advice drift (embedding-cosine distance, structured-feature L1, flip rate): a fair
  repair reduces unjustified identity-driven advice drift without harming difference awareness.
- Cross-axis transfer onto the held-out `gender` axis (thin slice; reported with caveat).
- Verification triangulation: IG localization fraction, identity linear-probe accuracy,
  parameter-space geometry regime + correlation, Patchscope P(c) on neq vs eq.

## Splits and contamination

- Group-disjoint split by `scenario_type` (= `paraphrase_of`, 1:1 with `source_cell`), stratified
  by axis, seed 42: 20 percent of clusters to the frozen test, 10 percent of the remainder to
  validation, the rest to train.
- Contamination guard (hard-stop): exact fact-level `source_cell` disjointness across splits and
  exact-text duplicate detection. Surface n-gram overlap and cosine are reported as diagnostics
  only, because AgriFacts is templated (see coding_prompt.md Section 5.3).
- Frozen test SHA256: recorded at build time to `results/frozen_test_set_sha256.txt` and copied
  here after `build_template_instances.py` runs:

  FROZEN_TEST_SET_SHA256 = (filled by build_template_instances.py; current build:
  ce206f88cb04a2a8e4e826c852bf8d9059d61a199eb08099e517a47d3d37b853)

## Statistics

- Paired bootstrap, 1000 resamples, p < 0.05, on the primary endpoint.
- Paired Wilcoxon signed-rank on per-item correctness deltas.
- Seeds 42, 43, 44 on the two primary models for the proposed method and the strongest baseline;
  seed 42 elsewhere. Mean and standard deviation reported.

## Fixed thresholds

- Attribution layer-selection threshold: 0.15 of max normalized attribution.
- LoRA rank clip: [4, 64], rank proportional to normalized attribution.
- Condition weights: neq (diff) 1.5, eq (equal) 1.0. Rationale weight: 0.3.
- Curriculum neq proportion: 0.2 -> 0.5 across 3 epochs. General replay: 10 percent.
- Matched trainable-parameter budget within 2 percent (absolute); the proposed full-precision
  run sets the reference. Reference baselines (Vanilla LoRA, Vanilla QLoRA) are exempt.
- AgriAdvice drift flip threshold: 0.25 embedding-cosine distance (preregistered).
