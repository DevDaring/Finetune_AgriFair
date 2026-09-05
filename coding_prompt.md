# Build specification - AgriFair GRAFT

This is the specification the repository implements. It is written so that a researcher, or
a coding tool, can rebuild or extend the study without further clarification. Nothing here
invents experimental numbers: every reported value comes from a real run.

## 0. The study in one paragraph

A language model asked about agrarian inequality can fail in two opposite directions. Gap
erasure is answering "Roughly equal" where the 2015-16 Agriculture Census records a real
difference. Gap fabrication is naming a group where the census shows none. A single
aggregate score hides which one a method commits, and a global push toward equal treatment
cures the first by causing the second. GRAFT localizes where a model encodes gap erasure
with Integrated Gradients, places LoRA adapters on those layers alone with rank
proportional to attribution, trains them with a condition-adaptive, rationale-aware
objective, and then re-measures with four independent readouts. The study reports both
failure directions separately throughout, audits every baseline the same way, and reads
every accuracy against what a state-blind prior alone would score.

## 1. The dataset, as it actually ships

`Debk/AgriFair`, a private HuggingFace dataset repository, downloaded into `Dataset/`.
Verified against the Hub on 2026-09-03: the local files are byte-identical to the remote
revision.

### AgriFacts, 2,000 rows

Three-choice multiple choice; the correct option is fixed by a census statistic, never by a
language model.

| field | observed | role |
|---|---|---|
| `id` | `agrifacts-00000` … | row id |
| `question` | stem | the scenario text |
| `choices` | exactly 3, always containing `"Roughly equal"` | the options, shuffled on disk |
| `answer` | one of `choices` | the gold option |
| `condition` | `diff` (1,000) / `equal` (1,000) | the central label, used verbatim |
| `axis` | `social_group` (907) / `landholding` (913) / `gender` (180) | the stratifier |
| `metric` | `number` (1,211) / `area` (789) | the secondary stratifier |
| `source_cell` | `AgCensus2015-16 T2-4 Maharashtra/All Classes/area/SCvsST` | the citation to preserve |
| `paraphrase_of` | 1,178 ids, one-to-one with `source_cell` | the paraphrase split key |

Derived deterministically, no model involved:

- **`group1`, `group2`** from the trailing comparison token of `source_cell`. Social-group
  and gender tokens map to fixed surface forms; **size-class tokens are matched by
  longest-prefix against the non-equal choices**, because the released choices use two
  wordings for the same class (`marginal holdings` for the `number` metric,
  `marginal operated area` for `area`). A naive exact-string map silently fails on 350 rows
  and falls back to on-disk order, which corrupts the canonical letters. With the prefix
  rule, all 2,000 rows resolve from the census token, and `dataset_io` asserts it.
- **Canonical letters**: `c` is always `"Roughly equal"`, `a` is `group1`, `b` is `group2`,
  in the left-to-right order of the comparison token. Gap erasure is then exactly "the
  model answered `c` on a `diff` item". The on-disk shuffled order is what the prompt shows;
  the canonical letters are an internal label for metrics, swaps and Patchscope.
- **`state_blind_key`**: `(axis, metric, size class, comparison token)`, the row's
  answer-relevant structure with the state removed. This is what the split and the
  surface-cue ceiling are built on.
- **`rationale`**: deterministic templating of the row's own ground truth, qualitative in
  direction, carrying the census cell verbatim as its citation. No numeric magnitude is
  invented, because the released schema omits raw shares.

### AgriAdvice, 800 pairs

Paired free-text agronomy prompts differing only in the farmer's stated identity, across
four toggle axes with 200 pairs each: gender, social group, region register, literacy
register. There is no gold answer; because the material facts are identical, any systematic
difference in advice is unjustified. Verified in `build_counterfactual_pairs.py`: all 800
pairs differ only in the persona span, and all 2,000 AgriFacts swaps exchange exactly the
two group spans. The persona check uses alphanumeric lookarounds rather than word
boundaries, because several personas end in a full stop and one contains a token that also
appears inside an unrelated word.

## 2. The split, and why it has two layers

Paraphrase-disjointness alone is not enough here. AgriFacts is templated over roughly 36
states, a few group pairs and two metrics, so a state-blind majority predictor scores 0.872
on the corpus, and under a paraphrase-only split 97.3 percent of test rows shared a
state-blind key with training. The split therefore reserves 20 percent of state-blind keys,
stratified by axis, for the frozen test set alone, producing 1,132 train / 119 validation /
749 test, with the test set labelled `structure_familiar` (328) and `structure_novel` (421).
`results/surface_cue_ceiling_audit.csv` records the ceiling per split and per slice; every
accuracy figure draws it as a reference line.

Gating contamination checks are hard stops: zero `source_cell` overlap between splits, zero
exact-text duplicates, zero structure-novel rows sharing a key with training. Templated
n-gram overlap and TF-IDF cosine are reported as diagnostics, because gating on them would
reject every valid split of a templated benchmark.

## 2a. Subject models

| Tier | Model | Params | Role | Why this one |
|---|---|---|---|---|
| `small-instruct` | meta-llama/Llama-3.2-3B-Instruct | 3.2B | primary | the standard small instruct baseline |
| `broad-instruct` | Qwen/Qwen3-4B-Instruct-2507 | 4.0B | primary | a different pretraining corpus and tokenizer at the same scale |
| `general-instruct` | google/gemma-3-12b-it | 12.2B | primary | tests whether the localization finding survives well above the small-model regime |
| `general-instruct-2` | mistralai/Ministral-8B-Instruct-2410 | 8.0B | primary | a second large model from a different vendor, so the size result is not one family's quirk |

All four are primary. Two identifiers in the original request did not exist and were
corrected against the Hub on 2026-09-03: Gemma 3 has no 9b release, and Ministral 8B
Instruct exists only as `-2410`.

`google/gemma-3-12b-it` is a `Gemma3ForConditionalGeneration` checkpoint: the text tower is
48 layers of hidden size 3840 under `model.language_model`, alongside a SigLIP vision tower
that this study never uses. `AutoModelForCausalLM` cannot load it, so `load_model_and_tokenizer`
walks an auto-class chain and records which class succeeded. Nothing downstream needs a
special case, because every component addresses modules by name pattern or by walking to the
decoder layer list, and both handle the extra nesting.

Per-model batch caps live in the registry and only ever lower a requested batch size. The
effective training batch is unchanged, because gradient accumulation absorbs a smaller
micro-batch, so the optimisation is identical across model sizes.

## 3. The method, stage by stage

**Stage A, attribute.** Path-integrated gradient-times-activation on module outputs along
the straight path from a zero embedding to the real input, 50 Riemann steps, over the
validation failure set (gap-erasure items first, then swap flips, then other errors). Per
layer, the sum over its attention and MLP projections, normalized by the maximum. The module
list matches names ending in a target suffix and excludes adapter internals, so a
PEFT-wrapped projection is scored once. Per-item scores are kept so ranking stability can be
bootstrapped, and one attribution is computed per held-out axis from failure items that
exclude that axis.

**Stage B, place.** Layers at or above 0.15 of the maximum are selected; each gets
`clip(round(64 x normalized attribution), 4, 64)`. Adapters are restricted to those layers
by handing PEFT a full-match regex over module names, and the rank-pattern keys are anchored
so `layers.3.` cannot also match layer 31. Alpha follows rank per layer, so the effective
scaling is constant across layers of different rank. Matched-budget uniform and three
independently drawn random placements are built from the same total rank.

**Stage C, train.** Per-item loss `w(condition) x (answer CE + 0.3 x rationale CE)`, with
`w(diff) = 1.5`, `w(equal) = 1.0`; a curriculum raising the diff sampling proportion from
0.2 to 0.5 across three epochs; 10 percent general replay. Micro-batched inside the
gradient-accumulation window with right padding and a per-example normalized loss, so the
summed gradient equals the one-example-at-a-time path. Every prompt is rendered through the
tokenizer's chat template, the same rendering evaluation uses. Resume restores the last
saved epoch's adapter weights; a finished arm is skipped unless `FORCE_RETRAIN=1`.

**Stage D, verify.** Four independent readouts, run on the baselines as well as on GRAFT:
attribution recomputed on the adapted model and compared with the pre-repair map; a linear
identity probe on AgriFacts swaps and AgriAdvice personas; parameter-space geometry of the
merged update; and a Patchscope reading of the probability of "Roughly equal" at each
targeted layer, with the source prompt built in canonical option order so that letter always
means the same thing.

**Stage E, transfer.** For each axis, an adapter trained without it and evaluated on it,
with the placement derived from an attribution that never saw the axis.

## 4. Evaluation

Greedy decoding at temperature 0. Answer extraction is deterministic JSON parsing, with a
judge fallback only on a parse failure and a free-text heuristic after that; the parser
strips reasoning blocks and recovers a truncated object whose first field is complete.
Reported per (model, method, seed, scope, axis, form): accuracies, the balanced awareness
score, Wang DiffAware and CtxtAware, gap erasure and gap fabrication, failure polarity,
identity-swap flip / invariance / equivariance, option-rotation consistency and
position-following, external capability retention, the four rationale metrics, and the parse
failure rate. Evaluation reuses stored per-item predictions unless `FORCE_EVAL=1`.

## 5. Analysis

`aggregate_results` builds the per-slice and per-axis tables. `statistics_tests` runs the
paired item-level bootstrap, exact McNemar and Holm correction within each hypothesis
family. `audit_awareness_trade` writes the failure-polarity profile and the awareness-trade
table. `parameter_space_geometry` computes the seven diagnostics on CPU float64 SVD, assigns
regime labels against the dense anchor, and correlates spectral shift with preservation.
`repair_admissibility` scores the three criteria and draws the preservation frontier.
`figures_and_tables` renders the figures with the surface-cue ceiling as a reference line.

## 6. Engineering contract

No virtual environment is assumed; `requirements_global.txt` is complete. Only the download
script touches the network. Keys are read from `.env` through `env_loader` and never
printed, and every dry run scans the tracked source for hardcoded keys. Every results CSV
uses full descriptive column names, enforced in `write_csv`. Matched budgets are asserted in
code. Data hygiene runs every build. Figures and tables render only what a real run
produced.

## 7. Acceptance checklist

1. `python run_all.py --smoke` completes end to end offline with exit status 0.
2. All three dry runs pass; `dry_run_analysis` alone asserts 43 properties of the metric,
   statistics and geometry definitions.
3. Only `download_models_and_data.py` and `build_capability_probe.py` touch the network.
4. No literal API key in any tracked file.
5. Every results CSV uses full descriptive column names.
6. Answer extraction is deterministic; the judge never sees a gold label.
7. Every baseline has a mechanism that distinguishes it from every other arm, and its
   divergences from the published method are written in README under "Baseline fidelity".
8. Trainable-parameter budgets matched within 2 percentage points, asserted in code.
9. Flash-attention comes from a pre-built wheel only, with the sdpa or eager fallback
   recorded per model.
10. The frozen test set is hashed, and the hash is in PREREGISTRATION.md.
11. The structure-novel slice shares no state-blind key with training, checked as a hard stop.
12. No projected or simulated numbers anywhere; no emoji.
13. Every name coined by the author's earlier study is replaced, and every name belonging to
    another paper is unchanged. The map is TERMINOLOGY.md.
14. Every citation identifier resolves, or is marked unverified. The list is CITATIONS.md.

## 8. Known limitations

- AgriFacts rationales are qualitative in direction plus a census-cell citation, because the
  released schema omits raw percentage shares.
- The `gender` axis has 180 items and is national-level only, so leave-one-axis-out onto
  gender runs on a thin slice and is reported with that caveat.
- AgriAdvice identity cues are explicit self-descriptions; implicit cues such as names or
  dialect are out of scope, inherited from the dataset.
- Baselines are mechanism-representative re-implementations at a matched budget, not
  bit-for-bit reproductions.
- The capability probe scores an unparseable output as incorrect, so it measures capability
  and output-format compliance together. Cross-model comparison of the probe is therefore
  not supported; within-model comparison across methods is.
- The surface-cue ceiling is computed from majority labels within a split, which is an
  oracle. It bounds what a state-blind prior could reach, not what any particular model does.
