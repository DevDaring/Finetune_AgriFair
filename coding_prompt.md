# coding_prompt.md — Implementing XLoRA-Bias (Instruction.md) on the AgriFair dataset

This document is the build specification that ports the **XLoRA-Bias** method described in
[Instruction.md](Instruction.md) from its original *legal* difference-awareness benchmark onto the
**AgriFair** agricultural-fairness dataset (`Debk/AgriFair`, downloaded into
[Dataset/](Dataset/)). It is written so that an AI coding tool, or a researcher, can implement the
full study without further clarification. It preserves the method, the verification triangulation,
the statistics, and the data-hygiene discipline of Instruction.md, and changes only what the change
of domain genuinely requires.

Nothing here invents experimental numbers. Every reported value is produced by a real run. There is
no emoji anywhere in the repository, by design (same rule as Instruction.md).

---

## 0. One-paragraph summary of the port

XLoRA-Bias was built to repair a single failure — a model **collapsing to "treat both groups
equally"** when the ground truth requires a difference — by (A) localising where that failure lives
with Integrated Gradients, (B) placing LoRA adapters only there with rank proportional to
attribution, (C) training with a condition-adaptive, rationale-aware loss, and (D) verifying the
repair three independent ways. AgriFair encodes the **same failure** in agriculture: a model that
answers *"Roughly equal"* on an AgriFacts item where the 2015-16 Agriculture Census shows a real gap
is committing the identical over-equalization error. The method therefore transfers almost verbatim.
AgriFair also adds one capability the legal set only simulated: a curated **free-text identity
counterfactual** split (AgriAdvice) for measuring whether agronomic *advice* drifts with a farmer's
identity. We fold AgriAdvice in as the identity probe and as a new headline metric, advice drift.

---

## 1. What stays identical to Instruction.md

These are not re-derived; they are imported unchanged in mechanism (with citation comments preserved
in code):

- **The five method stages A–E** (attribution, localized placement, condition-adaptive +
  rationale-aware training, three-way verification, optional jurisdiction head — here renamed the
  optional *axis head*).
- **Integrated Gradients** attribution (zero-embedding baseline, 50 Riemann steps, custom hook-based
  path conductance; Sundararajan et al. 2017).
- **Layer-selective LoRA** placement with rank proportional to normalized attribution (threshold
  0.15, rank clip 4–64), plus matched-budget **uniform** and **random** placements for the ablation.
- **The six SOTA baselines + two reference baselines** at a matched trainable-parameter budget
  (FairNet, IGU-LoRA, PEDAL, FairLoRA, ReGiFT, FairSteer; Vanilla LoRA, Vanilla QLoRA).
- **The verification triangulation**: IG re-run + linear identity probe; parameter-space geometry
  (CPU float64 SVD, seven diagnostics, regime labels, correlation with preservation); Patchscope
  readout.
- **Statistics**: paired bootstrap (1000 resamples, p<0.05) + paired Wilcoxon on the primary
  endpoint, seeds 42/43/44 on the primary models.
- **Engineering contract**: single deterministic `json_repair_parse`, no judge for answer extraction,
  round-robin API keys read only from `.env` via `env_loader`, offline after one download, resumable
  checkpointing, incremental CSV writes, full descriptive CSV column names, data hygiene every run,
  crash-safe autosync, dry runs before expensive runs.
- **The four subject models and their registry** (Section 5 of Instruction.md). The domain is now
  India-grounded agriculture, which makes the indic-specialized model especially relevant, but the
  model-diversity axis is unchanged.

---

## 2. The dataset, as actually shipped

Confirmed by inspection of the downloaded files (not from the card alone):

### 2.1 AgriFacts — `Dataset/agrifacts.jsonl`, 2,000 rows

Three-choice MCQ; **the correct option is fixed by a real census statistic, never by an LLM**.

| field | observed | role in the harness |
|---|---|---|
| `id` | `agrifacts-00000` … | unique row id |
| `question` | stem (UTF-8; em-dash `—` is `\xe2\x80\x94`, valid) | the scenario text |
| `choices` | exactly 3 strings; **always contains `"Roughly equal"`**; the other two are the compared groups | the three options |
| `answer` | one of `choices`; on `equal` rows it is always `"Roughly equal"`; on `diff` rows it is one of the two groups | the gold option |
| `condition` | `diff` (1000) / `equal` (1000) | **maps to neq / eq** |
| `axis` | `social_group` (907) / `landholding` (913) / `gender` (180) | maps to *category* (the stratifier formerly called jurisdiction) |
| `metric` | `number` (1211) / `area` (789) | maps to *form* (the secondary stratifier) |
| `source_cell` | e.g. `AgCensus2015-16 T2-4 Maharashtra/All Classes/area/SCvsST` | **maps to law_reference** (the citation to preserve) |
| `paraphrase_of` | 1,178 distinct seed-fact ids; **no seed straddles both conditions** | **maps to scenario_type** (the group-disjoint split key) |

Derived deterministically (no model involved):

- **`group1`, `group2`** are parsed from the final segment of `source_cell` (`SCvsST` →
  `Scheduled Castes`, `Scheduled Tribes`; `MvF` → `men`, `women`; `MarginalvsLarge` →
  `marginal holdings`, `large holdings`). A fixed lookup table maps the short token to the canonical
  surface form that appears verbatim in `choices`. The two parsed groups equal the two non-equal
  entries of `choices` (verified for all rows).
- **Canonical letter assignment** (preserves the legal invariant that `c` is the "treat equally"
  answer): `c := "Roughly equal"`; `a := group1`; `b := group2`, where (group1, group2) are taken in
  the left-to-right order of the `source_cell` comparison token. This makes the over-equalization
  failure *"the model answered c on a `diff` item"* exactly as in the legal set. The on-disk
  `choices` order (shuffled, for position-bias control) is preserved in the prompt; the canonical
  letters are an internal label only, used for metrics, swaps, and Patchscope.

### 2.2 AgriAdvice — `Dataset/agriadvice.jsonl`, 800 pairs (1,600 prompts)

Paired free-text agronomic prompts that differ **only** in the farmer's stated identity. No gold
answer; `facts_preserved` is `true` for every pair.

| field | observed | role |
|---|---|---|
| `pair_id` | `p0000` … | pair id and split key |
| `base_query` | verbatim real farmer question | the byte-stable material content |
| `toggle_axis` | `gender` / `social_group` / `region_register` / `literacy_register` (200 each) | the identity dimension toggled |
| `version_A`, `version_B` | `{persona, prompt}` | the two identity-conditioned prompts |
| `facts_preserved` | always `true` | integrity gate already passed upstream |
| `source_query`, `source_dataset` | provenance (`KisanVaani/agriculture-qa-english-only`) | provenance |

AgriAdvice is the agriculture analog of the legal benchmark's **identity counterfactual pairs**
(Instruction.md §4.4), but curated and free-text rather than auto-generated MCQ swaps. Because the
material facts are identical across A and B, **any** systematic difference in advice is unjustified —
AgriAdvice is therefore a pure *contextual-awareness* (do-not-invent-a-difference) probe, in
free-text form.

### 2.3 The clean conceptual split AgriFair gives us

| Fairness obligation | AgriFair carrier | Failure mode it catches |
|---|---|---|
| Acknowledge a **real** difference (difference-awareness) | AgriFacts `diff` items | over-equalization: answering "Roughly equal" when a gap exists |
| Do **not invent** a difference in a fact (contextual-awareness, MCQ) | AgriFacts `equal` items | spurious differential: naming a group when shares are equal |
| Do **not let identity change a factual answer** (contextual-awareness, free-text) | AgriAdvice pairs | advice drift: different agronomy for different personas |

The primary endpoint, **contextual fairness score = harmonic mean of `diff`-accuracy and
`equal`-accuracy on AgriFacts**, is unchanged from Instruction.md §7. AgriAdvice contributes a new,
orthogonal headline metric (advice drift) and the linear identity probe.

---

## 3. The schema map (AgriFair → XLoRA-Bias harness)

`Dataset_Prep/build_template_instances.py` normalizes every AgriFacts row into the harness record the
rest of the code already expects. The legal column on the left; the AgriFair source on the right.

| harness column | AgriFair source | notes |
|---|---|---|
| `id` | `id` | unchanged |
| `category` | `axis` | `social_group` / `landholding` / `gender` |
| `form` | `metric` | `number` / `area` (secondary stratifier) |
| `condition` | `condition` mapped `diff→neq`, `equal→eq` | the central label |
| `group1`, `group2` | parsed from `source_cell` token | canonical surface forms |
| `question` | `question` | unchanged stem |
| `choice_a` / `choice_b` / `choice_c` | canonical: a=group1, b=group2, c="Roughly equal" | internal letters; prompt shows on-disk `choices` order |
| `correct_answer` | derived letter of `answer` under the canonical map | neq → a or b; eq → always c |
| `rationale` | **constructed deterministically** (see §4) | no LLM, no fabricated content |
| `law_reference` | `source_cell` | the census citation to preserve |
| `scenario_type` | `paraphrase_of` | group-disjoint split key |
| `is_myth_buster` | absent | optional in the reader; simply not present |
| `metric_raw`, `axis_raw`, `source_cell` | passed through | provenance kept for audit |

`Dataset_Prep` also normalizes a `jurisdiction`-style field; here that field is `axis` so existing
code paths that key on jurisdiction key on axis. The reader
[GPU_Run/common/dataset_io.py](GPU_Run/common/dataset_io.py) accepts the AgriFair JSONL directly and
emits the normalized record.

---

## 4. Rationale construction (the one genuine gap, solved without an LLM)

The legal benchmark ships a free-text `rationale`; AgriFair does not. The rationale-aware objective,
ReGiFT, and the rationale metrics (BLEU-4, ROUGE-L, citation preservation, judge factual-correctness)
all need a reference rationale. We construct it by **deterministic templating of the row's own ground
truth**, fully consistent with Instruction.md §4.2 ("deterministic templating of a real dataset row")
and AgriFair's own rule ("no label comes from a language model"):

- **diff (neq)**:
  `"According to the 2015-16 Agriculture Census ({source_cell}), {answer} account for a larger share
  of {metric_phrase} than {other_group}; the two are not roughly equal."`
- **eq (equal)**:
  `"According to the 2015-16 Agriculture Census ({source_cell}), {group1} and {group2} operate
  roughly equal shares of {metric_phrase}; no meaningful gap exists."`

where `metric_phrase` is `"agricultural operated area"` for `area` and `"operated holdings"` for
`number`. The census cell string is carried verbatim, so it doubles as the citation whose survival is
measured by `rationale_statutory_citation_preservation_rate` (here: census-cell preservation). The
rationale is **qualitative** (direction + citation), because the released schema does not include the
raw percentage shares; this is recorded as an explicit limitation. No numeric magnitude is invented.

`prepare_prompts_deepseek.py` is retained but, exactly as in Instruction.md, it only light-edits the
fixed *instruction wrappers* (never the question, choices, answer, or rationale content). With no key,
it is a no-op that logs and exits.

---

## 5. Splits, counterfactuals, contamination

### 5.1 Group-disjoint split (contamination-safe)

`build_template_instances.py`, seed 42: hold out **20 percent of `paraphrase_of` clusters** for the
frozen test set, carve **10 percent of the remaining clusters** for the validation set used by Stage A
attribution, the rest is train. No `paraphrase_of` appears in two splits (its paraphrases cannot
leak). Stratify the cluster draw by `axis` so each split keeps the axis mix, and keep `gender`
(only 180 items) represented in test. Write `data/test_instances_frozen.jsonl` deterministically and
record its **SHA256** to `results/frozen_test_set_sha256.txt` and into `PREREGISTRATION.md`.

### 5.2 AgriFacts MCQ counterfactual swaps (the legal §4.4 analog)

`build_counterfactual_pairs.py` builds identity counterfactuals **two ways**, both deterministic:

1. **MCQ swap (AgriFacts)**: swap `group1`↔`group2` by whole-word, case-insensitive replacement in
   the question, choices, and rationale. For `diff` rows this flips the correct letter `a`↔`b`; `eq`
   rows stay `c`. These drive identity-swap-flip detection (Stage A) and the MCQ side of the probe.
2. **Curated free-text pairs (AgriAdvice)**: pass `version_A` / `version_B` straight through into a
   normalized `data/agriadvice_pairs.jsonl`. These drive the free-text identity probe and the advice
   drift metric. They are **never trained on** (no gold answer).

### 5.3 Contamination check (redesigned for a templated benchmark)

Verified during the build: `paraphrase_of` and `source_cell` are **1:1** (1,178 each), and the
group-disjoint split produces **exactly 0** `source_cell` overlap between train, validation, and test.
The answer-determining identity of an AgriFacts item is its `source_cell` (state / size-class /
metric / comparison); fact-level contamination is therefore zero by construction.

AgriFacts is heavily templated and lives in a small combinatorial space (~36 states x a few group
pairs x 2 metrics), so **raw** n-gram overlap and TF-IDF cosine between any valid split are high by
construction (observed: 74 percent 8-gram overlap, 0.985 max cosine) — a test item differs from some
train item only in a state name. The legal 0.5 percent raw-n-gram hard-stop (Instruction.md §4.3)
assumes free prose and would wrongly reject every valid split here. `contamination_check.py` therefore
gates on what is meaningful and reports the rest:

- **Gating hard-stops**: (i) exact fact-level `source_cell` disjointness across splits (must be 0;
  stricter than the legal n-gram heuristic), and (ii) exact-text duplicate detection (no test
  question+choices string equals a train one).
- **Diagnostics (reported, non-gating)**: de-boilerplated 8-gram overlap (template 8-grams removed),
  raw 8-gram overlap, and max TF-IDF cosine, each labelled as expected-high for a templated benchmark.

TF-IDF cosine fallback when `MULTILINGUAL_EMBED_MODEL` is unset (English-only here).

---

## 6. External generalization (evaluation only, never trained on)

Two complementary, faithful choices (no fabrication):

1. **Cross-axis transfer (in-corpus, always available)**: train on two axes
   (`social_group` + `landholding`) and evaluate **zero-training** on the held-out third axis
   (`gender`). This is the primary generalization test and needs no external download.
2. **Out-of-domain transfer (optional)**: the original Wang et al. (2025) social-bias D2/N3 sets, if
   placed under `data/external_wang/`, mapped by `map_external_wang_datasets.py`. Shows whether an
   agriculture-localized repair transfers back to the social domain. If absent, a clear note is
   written and nothing is fabricated.

The cross-jurisdiction-transfer metric of Instruction.md becomes **cross-axis transfer**: the directed
pairs are over `{social_group, landholding, gender}` (six directed pairs).

---

## 7. The method, restated on AgriFair

### Stage A — Attribution (localize over-equalization)
Failure set on the validation split: AgriFacts `diff` items the frozen model answers `c`
("Roughly equal") = the over-equalization failure, plus MCQ identity-swap flips. IG (zero baseline,
50 steps) over attention and MLP projection modules, per-layer normalized scores. Attribution stays
single-example (it feeds discrete thresholds). AgriAdvice contributes identity-contrast items to the
swap-flip signal but the CE target remains the AgriFacts correct letter, so the IG objective is
identical to the legal method.

### Stage B — Localized placement
Unchanged: select layers at/above 0.15 of max attribution, rank ∝ attribution clipped to [4, 64],
adapters on attention + MLP projections only; build matched-budget uniform and random configs.

### Stage C — Training
Joint loss per example, unchanged form:
```
total = (answer_cross_entropy + 0.3 * rationale_cross_entropy) * condition_weight
condition_weight: neq (diff) 1.5, eq (equal) 1.0
```
Curriculum raises the `diff` sampling proportion 0.2→0.5 across 3 epochs; 10 percent general-replay
from `databricks/databricks-dolly-15k`. AdamW, lr 2e-4, LoRA dropout 0.05, bf16, effective batch 8,
max length 1024. AgriAdvice is **not** in the training stream.

### Stage D — Verification (triangulated)
- **IG re-run + linear identity probe**: probe decodes the swapped identity (group1 vs group2) from a
  mid-depth hidden state on the MCQ swaps; additionally decodes persona (A vs B) on AgriAdvice pairs.
  The headline `bias_subspace_residual_*` uses this probe.
- **Parameter-space geometry**: unchanged (ΔW via PEFT `get_delta_weight`, CPU float64 SVD, seven
  diagnostics, regime label, correlation of spectral shift / principal-mask overlap with preservation,
  where preservation = post / base CtxtAware).
- **Patchscope**: at each attribution-targeted layer, patch the last-position representation into a
  letter-eliciting prompt and read the renormalized probability of option **c** ("Roughly equal"). A
  good repair **lowers** c-probability on `diff` while **keeping it high** on `equal`.

### Stage E — Optional axis head (off by default)
A lightweight per-axis classification head behind a config flag; off by default (the legal
jurisdiction head, renamed).

---

## 8. Metrics (Instruction.md §7, plus AgriAdvice)

Computed per model / method / condition / axis / metric / seed, with full descriptive CSV column
names. Carried over unchanged: `overall_accuracy`, `accuracy_on_neq_condition` (diff),
`accuracy_on_eq_condition` (equal),
`contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy` (**primary endpoint**),
`difference_aware_accuracy_gap_neq_minus_eq`, Wang `DiffAware`/`CtxtAware` (recomputed with
c = "Roughly equal"), utility retention, the four rationale metrics, cross-axis transfer accuracy,
`bias_subspace_residual_pre/post_repair`, `trainable_parameter_percentage`,
`attention_implementation_used`, `model_revision_hash`, `random_seed`, `quantization_setting`,
`json_parse_failure_rate_percent`, `api_judge_model_string`.

**New AgriAdvice metrics** (free-text, evaluation only):
- `advice_drift_embedding_cosine_distance` — 1 − cosine of sentence embeddings of the A and B
  answers (local embedder if available, else TF-IDF).
- `advice_drift_structured_feature_l1` — L1 distance over a fixed structured feature vector of each
  answer (length in tokens, number of recommendation bullets, presence of named inputs/dosages).
- `advice_drift_judge_score_one_to_five` — judge-scored material difference on a stratified
  subsample (judge chain, §10 of Instruction.md), used only as a cross-check, never for any label.
- `advice_drift_flip_rate` — fraction of pairs whose drift exceeds a preregistered threshold.

The fairness claim on AgriAdvice: a good repair **reduces** advice drift (identity should not move
factual advice) **without** harming AgriFacts difference-awareness — the agriculture statement of
"localize, repair, preserve".

---

## 9. Subject models, API judge, environment

- **Models**: the four in [GPU_Run/common/model_registry.py](GPU_Run/common/model_registry.py)
  (Nemotron-3-Nano-4B, Llama-3.2-3B-Instruct [gated], Indic-gemma-2b Navarasa, Qwen3-4B-Instruct).
  `SUBJECT_MODELS` env overrides the active set; a tiny CPU-friendly model id is allowed for smoke
  runs. Revisions pinned to `models/model_revisions.json`.
- **API for judgement only** (Instruction.md §10): judge chain Gemini 2.5 Flash → DeepSeek
  (+ OpenRouter alternate) → Mistral Small, round-robin per request, no within-provider retry. Keys
  read **only** from `.env` via `env_loader` (canonical names + legacy aliases; the provided `.env`
  uses `HUGGINGFACE_TOKEN`, `GEMINI_API_KEY_1..4`, `DEEPSEEK_API_KEY_1/2`, `MISTRAL_API_KEY1/2`,
  `OPENROUTER_API_KEY_1/2`). The HF token is used only by the one-time download script. Judge is used
  only for rationale factual-correctness and the rare answer-extraction fallback and the AgriAdvice
  drift cross-check — never for answer extraction or for any gold label.
- **Environment**: no venv; `pip install -r requirements_global.txt`. Flash-attention from a
  pre-built wheel with automatic `sdpa` fallback (Windows/CPU falls back to `sdpa`, recorded). After
  one download everything runs offline (`HF_HUB_OFFLINE=1`, `local_files_only=True`).
- **Windows/CPU note**: this machine is Windows. The full GPU study targets an A100/H100 VM; on this
  box the dry runs and the offline dataset/metrics/geometry code run as-is, GPU stages run with
  `SUBJECT_MODELS` set to a small model and `EVAL_SUBSET_SIZE`/`TRAIN_SMOKE_MAX_STEPS` set for a smoke
  pass. The geometry analysis (CPU float64 SVD) is fully CPU.

---

## 10. Repository layout (adapted; root = `Codes/`)

```
Codes/
  .env                      (already present; secrets; never committed)
  .env.example              canonical names + legacy aliases
  requirements_global.txt   pinned
  run_all.py                single entry point (Section 12 order; resume-capable)
  README.md                 study-level readme (this port)
  coding_prompt.md          this file
  Instruction.md            the original legal spec (reference)
  PREREGISTRATION.md        endpoint, seeds, tests, frozen-test hash, drift threshold
  explanation.html          beginner walk-through with two fully worked examples
  Dataset/                  the raw AgriFair download (agrifacts.jsonl, agriadvice.jsonl, README.md)
  Dataset_Prep/  Dry_Run/  GPU_Run/  CPU_Run/
  data/  results/  checkpoints/  models/   (generated; git-ignored)
```
File responsibilities mirror Instruction.md §11, with `dataset_io.py`, `build_template_instances.py`,
`build_counterfactual_pairs.py`, `metrics.py`, `verify_bias_subspace.py`, and
`patchscope_bias_verification.py` carrying the AgriFair-specific adaptations above. Two AgriAdvice
additions: `GPU_Run/evaluate_agriadvice_drift.py` (free-text drift) and the AgriAdvice branch of the
linear probe in `verify_bias_subspace.py`.

---

## 11. Run order (single entry point)

```
python run_all.py            # runs the whole pipeline in dependency order, resumable
python run_all.py --stage dataset_prep
python run_all.py --stage dry_run
python run_all.py --stage gpu_run
python run_all.py --stage cpu_run
python run_all.py --smoke    # tiny model + subset + capped steps, offline, for a fast end-to-end check
```
Underlying order is Instruction.md §12: Dataset_Prep → Dry_Run → GPU_Run (with autosync) → CPU_Run.
After one download, every step is offline and resumable; checkpoints and incremental CSVs mean a crash
loses at most the in-progress epoch or the last 50 rows.

---

## 12. Acceptance checklist (AgriFair edition)

1. Dry runs pass with a valid `.env` and pre-downloaded data.
2. Only `download_models_and_data.py` touches the network.
3. No literal API key in any tracked file.
4. Every results CSV uses full descriptive column names.
5. Answer extraction is deterministic JSON parsing; the judge is used only for rationale scoring,
   the rare extraction fallback, and the AgriAdvice drift cross-check.
6. All six SOTA + two reference baselines implemented with citation comments.
7. Recency check recorded.
8. Trainable-parameter budgets matched within 2 percent, asserted in code.
9. Flash-attention from a pre-built wheel only, sdpa fallback recorded (sdpa on Windows/CPU).
10. Duplicate/corrupted rows detected and logged every run.
11. No virtual environment; `requirements_global.txt` complete and pinned.
12. No emoji anywhere.
13. No projected or simulated numbers anywhere; AgriFacts gold traces to a census cell, AgriAdvice has
    no gold, and the constructed rationale is deterministic templating of the row's own ground truth.
14. This file plus README.md are sufficient to run the study from the downloaded data.
15. Batching is throughput-only, validated batched-vs-single in the dry run, reversible.
16. `condition` mapping (`diff→neq`, `equal→eq`) and the canonical `c = "Roughly equal"` invariant are
    asserted in `dataset_io` and exercised by `dry_run_dataset_prep`.

---

## 13. Limitations specific to this port

- AgriFacts rationales are qualitative (direction + census-cell citation); the released schema omits
  raw percentage shares, so no numeric magnitude is asserted.
- `gender` has only 180 items (census reports gender at all-India level only), so cross-axis transfer
  onto gender is evaluated on a thinner slice and reported with that caveat.
- AgriAdvice identity cues are explicit self-descriptions; implicit cues (names, dialect) are out of
  scope, inherited from the dataset.
- Several baselines approximate the original papers' mechanisms at a matched budget on the same data
  and harness; this is not a bit-for-bit reproduction.
- Some arXiv ids / one DOI are marked UNVERIFIED in code comments and must be confirmed before
  submission.
