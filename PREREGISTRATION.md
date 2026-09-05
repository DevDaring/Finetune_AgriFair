# Preregistration - AgriFair GRAFT

Endpoints, hypotheses, splits, seeds, tests and thresholds, fixed before the expensive
runs. No projected or simulated numbers appear here. Values produced by real runs are
written back by the pipeline into the files named against each item.

Version 3, 2026-09-03. What changed from version 2, and why, is recorded at the end.

## Primary endpoint

**Balanced awareness score**: the harmonic mean of diff-condition accuracy and
equal-condition accuracy on the AgriFacts frozen test set. Low whenever either direction
of awareness is low, so a method that maximises one condition by collapsing the other is
penalised. Reported separately on the two test slices defined below, never pooled into one
number, and always alongside the Wang DiffAware and CtxtAware metrics
(arXiv:2502.01926), which remain the primary externally-comparable quantities.

Every reported accuracy is read against the **surface-cue ceiling**: the accuracy a
majority-class predictor reaches from an item's state-blind key alone
(`results/surface_cue_ceiling_audit.csv`). An accuracy at or below that ceiling is not
evidence of census knowledge.

## Hypotheses

**H1, localization at a matched trainable-parameter budget (primary, internal).** At a
matched budget, attribution-guided adapter placement preserves difference awareness better
than uniform placement, which in turn beats random placement:

    balanced awareness (attribution-guided) > (uniform) > (random)

This is the causal test of the claim the method makes, it is decided entirely by arms this
study trains, and it does not depend on out-ranking any external method. Random placement is
a distribution over **three independently drawn layer sets**, not one draw, so a lucky or
unlucky sample cannot decide it. Each random arm takes the attribution-guided placement's
**layer count and rank multiset** and re-draws only which layers receive them, so layer
identity is the single thing that differs; uniform spreads the same total rank over the whole
depth. All three therefore train an identical number of parameters, asserted at run time to
within 10 percent relative, and the assertion aborts the run rather than warning. Tested on all four primary models, three seeds, paired
item-level bootstrap. Registered outcome space: the ordering holds, partially holds, or
fails; a failure is reported as a negative localization result, not omitted.

**H2, the external comparison (secondary).** Balanced awareness score of GRAFT against
every baseline, reported on the preservation-against-fairness frontier
rather than as a single ranking, with Holm correction across the family of baseline
comparisons. A single nominally significant cell in a family of eight is not a win and is
not reported as one.

Baselines are **not** forced to a common trainable-parameter budget, and this is deliberate:
each places adapters by the rule its own paper specifies, and overriding that would compare
against something the cited work never proposed. Their trainable share is instead measured
per arm and published in `trainable_parameter_percentage`, and H3's partial correlation
controls for it explicitly. The matched-budget guarantee is asserted only where equal budget
is itself the hypothesis, which is H1's placement ablations. Any claim that GRAFT beats a
baseline is therefore read against that baseline's reported budget, not against an assumption
that the budgets were equal.

**H3, update geometry and preservation (method-agnostic).** Across every weight-modifying
configuration, normalized spectral shift correlates negatively with difference-awareness
preservation. **Predicted sign registered in advance: negative.** Reported with Pearson and
Spearman coefficients, a percentile bootstrap 95 percent interval, and a partial
correlation controlling for trainable-parameter percentage, so the result cannot be
dismissed as a budget proxy. At least 15 configurations are required before the
correlation is described as anything more than a trend; below that the module logs a
warning and the language stays at "trend".

**H4, structural generalization (registered as exploratory).** The gap between the
structure-familiar and structure-novel slices is larger for methods that rely on the
state-blind prior than for methods that do not. Direction not registered; this is a
measurement, and it is reported whichever way it lands.

## Splits

The corpus is 2,000 AgriFacts items. Two kinds of leakage are controlled.

1. **Paraphrase leakage.** The split is group-disjoint at `scenario_type`
   (= `paraphrase_of`, one-to-one with `source_cell`, 1,178 clusters), stratified by axis,
   seed 42.
2. **Structural leakage.** Each item also carries a **state-blind key** of
   (axis, metric, size class, comparison token): its answer-relevant structure with the
   state name removed. Twenty percent of these keys, stratified by axis, are reserved for
   the frozen test set and appear nowhere in training or validation.

Resulting split, written by `Dataset_Prep/build_template_instances.py`:

| split | items | notes |
|---|---|---|
| train | 1,132 | familiar keys only |
| validation | 119 | familiar keys only; drives Stage A attribution |
| frozen test | 749 | two labelled slices |
| - structure_familiar | 328 | 158 diff / 170 equal; the key is also in training, only the state differs |
| - structure_novel | 421 | 213 diff / 208 equal; the key is absent from training |

FROZEN_TEST_SET_SHA256 = `2d40a81f15ca1836acfa4e61af79b8e215365c380a858385dbd46fa42f1a172b`

Gating contamination checks, all hard stops: zero `source_cell` overlap between any two
splits; zero exact-text duplicates between test and train; zero items in the
structure-novel slice sharing a state-blind key with training. Surface n-gram overlap and
TF-IDF cosine are reported as diagnostics only, because AgriFacts is templated and those
quantities are high by construction (observed: 72.9 percent raw 8-gram overlap, 0.985 max
cosine, both expected).

## Secondary endpoints

- Difference awareness preserved: diff-condition accuracy and Wang DiffAware, before and after each method.
- Contextual awareness: equal-condition accuracy and Wang CtxtAware.
- The two failure directions reported separately, never as one number: **gap erasure rate** on diff items and **gap fabrication rate** on equal items, plus the failure polarity index that summarises their balance.
- **Failure polarity profile** on the frozen base models, per condition, axis, and slice. Registered as exploratory and reported whichever way it lands. The premise of this study is that gap erasure dominates; DART (arXiv:2604.16845) reports the opposite skew on its own suite, so polarity is measured rather than assumed.
- **Awareness trading**: a method is recorded as trading when it moves the two conditions in opposite directions by more than 0.01 relative to the frozen base. Audited for every trained method.
- Identity-swap consistency, reported as three numbers because one flip rate conflates them: the unlicensed flip rate, invariance on equal items, and equivariance on diff items.
- **Option-rotation robustness**: canonical-answer consistency and position-following rate under two cyclic rotations of the displayed option order.
- AgriAdvice advice drift: embedding cosine distance, content-word Jaccard distance, structured-feature L1, and the flip rate above a preregistered threshold of 0.25 embedding distance. Reported overall and per toggle axis, for every method.
- Capability retention on a fixed external probe: a 500-item, seed-stratified MMLU subset (arXiv:2009.03300), identical item set and order for every model and method. An output with no parseable letter is scored incorrect and stays in the denominator, so the probe measures capability and format compliance together; that is stated as a limitation. If the probe cannot be built the pipeline falls back to the in-domain validation split and records in the results CSV that it did.
- Leave-one-axis-out transfer: for each axis, an adapter trained without it and evaluated on it. Attribution is recomputed from failure items that exclude the held-out axis, so nothing about that axis informs the placement.
- Repair readout panel: attribution recomputed on the adapted model and compared with the pre-repair map (mass on the targeted layers, Spearman of the layer ranking, top-k Jaccard); identity-probe accuracy; parameter-space geometry; the Patchscope probability of "Roughly equal". Run on the baselines as well as on GRAFT, so the panel is demonstrated to be method-agnostic rather than asserted to be.
- Attribution stability: mean pairwise Spearman of the layer ranking over 200 bootstrap resamples of the failure set, and the rate at which the selected layer set is identical across resamples.
- Efficiency: training wall-clock minutes and peak GPU memory per run, and fairness gain per unit of relative Frobenius drift and per training minute.

## Repair admissibility criteria (registered as a proposal, not as the endpoint)

Three criteria with fixed thresholds, scored per method in `results/repair_admissibility.csv`:

1. **Targeted repair.** The gap-erasure rate on diff items falls relative to the frozen base, and no available mechanistic readout moves the wrong way (identity-probe accuracy, Patchscope probability of "Roughly equal" on diff items).
2. **Preservation.** Neither equal-condition accuracy nor external capability retention falls more than **0.02** absolute below the frozen base.
3. **Minimal invasiveness.** Trainable-parameter percentage at or below **1.0** percent.

The non-regression-adjusted contextual fairness score is reported alongside, never in place
of, the Wang metrics and the unadjusted harmonic mean, so the criteria cannot function as a
self-serving metric.

## Statistics

- Paired item-level bootstrap, 1,000 resamples, on H1 and H2: the two arms are scored on the same resampled item set, and the difference distribution gives a percentile 95 percent interval and a two-sided p-value.
- Exact McNemar on per-item correctness, reported alongside the bootstrap with the discordant counts in both directions.
- Holm-Bonferroni correction within each hypothesis family. Significance is claimed only after correction.
- Seeds 42, 43, 44 on every primary model for GRAFT, DART, FairNet and Vanilla LoRA; seed 42 elsewhere. Mean and standard deviation reported in `results/seed_variability_table.csv`.
- H3: Pearson and Spearman, percentile bootstrap 95 percent interval over 1,000 resamples, and a partial correlation controlling for trainable-parameter percentage. Every weight-modifying run is one configuration, including all ablation arms, all random placement draws, and all seeds. FairSteer is excluded because it changes no weight, and the exclusion is recorded in the CSV as `inference_time_no_weight_update`.
- No equivalence claim is made from a non-significant test. A null result is reported as "not separable at this resolution", which is what it is.

## Fixed thresholds

- Attribution layer-selection threshold: 0.15 of the maximum normalized layer attribution.
- LoRA rank: `clip(round(64 x normalized attribution), 4, 64)` per selected layer; alpha is twice the rank, so the effective scaling is constant across layers of different rank.
- Condition weights: diff 1.5, equal 1.0. Rationale weight 0.3.
- Curriculum diff proportion 0.2 to 0.5 across 3 epochs. General replay 10 percent. AdamW, learning rate 2e-4, gradient clipping 1.0, LoRA dropout 0.05, bf16, effective batch 8, max length 1024, greedy decoding at temperature 0.
- Matched trainable-parameter budget within 2 percentage points absolute, asserted in code; the GRAFT full-precision run on the first model sets the reference. Vanilla LoRA and Vanilla QLoRA are exempt as reference arms and are labelled as such.
- Random placement: 3 independent draws, seeds 42, 43, 44 for the draw itself.
- Structure-novel key fraction: 0.20, stratified by axis.
- AgriAdvice drift flip threshold: 0.25 embedding cosine distance.
- Admissibility: preservation tolerance 0.02 absolute; invasiveness ceiling 1.0 percent of trainable parameters.
- Geometry: bf16-aware relative change tolerance eta = 1e-3; mask fraction alpha = 0.5, so the random-overlap baseline is 0.5; principal mask from the rank-64 reconstruction of W0; principal-angle subspace dimension min(512, smaller matrix dimension); Hill estimator over the top 10 singular values. A method is spectrum-preserving when its normalized spectral shift is at or below 0.5 times the dense anchor's. The dense anchor is Vanilla LoRA, standing in for the full fine-tuning anchor the source papers use, because this study trains no full fine-tuning arm; the substitution is written into every geometry row as `regime_anchor_method_name`.
- DART audit split: the first 256 records of the training stream; severity oversampling 1/2/3/4 as published.

## Subject models

Four instruction-tuned models, all primary, spanning two size classes so the localization
hypothesis is tested above the small-model regime as well as inside it.

| Tier | Model | Params | Role | Why this one |
|---|---|---|---|---|
| `small-instruct` | meta-llama/Llama-3.2-3B-Instruct | 3.2B | primary | the standard small instruct baseline |
| `broad-instruct` | Qwen/Qwen3-4B-Instruct-2507 | 4.0B | primary | a different pretraining corpus and tokenizer at the same scale |
| `general-instruct` | google/gemma-3-12b-it | 12.2B | primary | tests whether the localization finding survives well above the small-model regime |
| `general-instruct-2` | mistralai/Ministral-8B-Instruct-2410 | 8.0B | primary | a second large model from a different vendor, so the size result is not one family's quirk |

Identifiers verified on the Hub on 2026-09-03. Model revision hash, the auto class that
loaded the checkpoint, the attention implementation actually used, and the batch sizes are
recorded in every results CSV. Adding the 8B and 12B models is amendment 9 below.

## Method set

GRAFT (proposed) against FairNet, IGU-LoRA, PEDAL, DART, ReGiFT, LFTF and FairSteer, with
Vanilla LoRA and Vanilla QLoRA as references, all at a matched trainable-parameter budget.
Ablations: uniform placement, three random placements, no rationale loss, no
condition-adaptive weighting, rank sweep at 8/16/32, and a 4-bit NF4 arm.

LFTF (arXiv:2505.15475) is in the set because it is the nearest published prior art: it
also localizes first and fine-tunes only the located blocks, but it locates with a
hidden-state bias-relevance score rather than a path-integrated attribution of the specific
failure, and its objective carries no notion of the item's condition. It is the control
that separates what the attribution signal contributes from what the objective contributes.

## Reduced-budget profile (optional, declared in advance)

The full schedule above is what version 3 registers. A study run under a constrained compute
budget may adopt the profile below instead. It is recorded here so that a reduced run is a
declared design, never a silent one, and any paper reporting it must say which profile it used.

**No reduction is applied by default.** Every switch below is off unless set explicitly, and
the shipped configuration runs the full schedule registered above. They are listed so that a
reduced run can be declared precisely, not so that one happens quietly.

- `ADVICE_TARGET_SCOPE=headline` narrows advice drift to the arms whose drift the paper
  tabulates. The cost is the ablation arms, which are what attribute a drift reduction to the
  localized placement rather than to the condition-adaptive objective. **Moderate risk: it
  removes the evidence for a question a reviewer is likely to ask.**
- `BEHAVIOURAL_SWEEP_SCOPE=first_seed` narrows the identity-swap and option-rotation sweeps to
  one seed per method. Both measure a property of a method rather than of a seed, but the
  resulting numbers then carry no variance estimate and must be reported as single-seed.
  **Low to moderate risk.**

**A declared reduction, opt-in and off by default.** Each narrows a secondary claim and must
be reported as narrowed. They are listed with the risk each carries, because they are not
equivalent.

- `RANK_SWEEP_TIERS` restricts the rank sensitivity check to a subset of models. It is a
  sweep rather than a hypothesis, so restricting it costs no registered claim, only breadth.
  **Low risk.**
- `MULTI_SEED_TIERS` restricts which models carry a three-seed variance estimate **for the
  baselines**. The proposed method keeps three seeds on every primary model regardless: a
  study that cannot state the variance of its own method on a model has no basis for saying a
  difference is or is not separable there, and that sentence is what the honest-reporting
  posture rests on. Restricting the baseline arms costs breadth in H2 on the affected models,
  which must be stated. **Moderate risk.**
- `LOAO_TIERS` restricts the leave-one-axis-out transfer study. This is the reduction to
  approach with most caution. On the predecessor legal benchmark the aggregate scores did not
  separate the strong low-rank methods, and transfer behaviour was the evidence that did. If
  that pattern repeats here, the transfer study is the paper's distinguishing result rather
  than a secondary endpoint, and running it on half the models would weaken the central
  finding. **Do not set this unless the budget leaves no alternative.**

Nothing that carries H1, H3 or H4 is reduced under any profile: every model keeps the full
baseline set, the placement ablation with all three random draws, three seeds on the proposed
method, the frozen test set at full size, both structure slices, and the external capability
probe. H1 additionally retains a dispersion estimate on every model even where baseline seeds
are reduced, because the three independent random placement draws are themselves a spread.

## Changes from version 2, all made before any expensive run

1. **The frozen test set now carries a structure-novel slice.** Under the previous
   paraphrase-only split, 97.3 percent of test items shared a state-blind key with
   training and a state-blind majority predictor reached 0.91 on the test split. Any
   near-ceiling score under that split was therefore uninterpretable. The reserved-key
   design fixes this, and the surface-cue ceiling is now reported next to every accuracy.
2. **Capability retention moved to an external MMLU probe.** The previous measurement was
   accuracy on the in-domain validation split, which cannot show that a repair preserved
   general ability.
3. **Random placement became a distribution.** One draw could not support or refute H1.
4. **Option-rotation robustness was added**, because a fixed option layout lets a model
   score by mapping a predicted condition to a position rather than by reading the census.
5. **Identity-swap consistency was split into invariance and equivariance**, since a
   single flip rate conflates the desired equivariance on diff items with undesired
   movement on equal items.
6. **The attribution re-run now recomputes attribution on the adapted model.** The
   previous implementation read the pre-repair attribution file back, so its "before" and
   "after" were the same number.
7. **LFTF was added as a baseline** and **FairSteer's identifier was corrected** to
   arXiv:2504.14492 (see CITATIONS.md).
8. **The model set moved up a size class.** The two 2.5B and 4B transfer models were
   replaced by `google/gemma-3-12b-it` and `mistralai/Ministral-8B-Instruct-2410`, both
   primary, so every claim is tested at 8B and 12B and not only at 3B and 4B. The earlier
   set could not answer whether a localized repair behaves the same way once a model is
   large enough to hold the census facts.
9. **Contribution names were changed** so that no name coined by the author's earlier
   legal-benchmark study is reused; names belonging to other papers are unchanged. The
   full map is in TERMINOLOGY.md.
