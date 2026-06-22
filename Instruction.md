# XLoRA-Bias: Explainability-Guided LoRA for Contextual Bias Mitigation in Legal LLMs

This repository is the complete, reproducible research codebase for the study
**XLoRA-Bias**. It localizes where a language model encodes the "collapse to equal
treatment" failure using Integrated Gradients attribution, places LoRA or QLoRA
adapters only on those components with rank set by attribution strength, trains with
a condition-adaptive and rationale-aware loss, and then re-runs the same attribution
(plus a linear probe, a parameter-space geometry analysis, and a Patchscope readout)
to verify that the targeted bias signal is reduced while difference awareness is
preserved. The framing is **explainability-guided bias sterilization: localize the
bias, repair only that, preserve the rest.**

This is a research scaffold, not a product. No experimental numbers are shipped with
it; every reported value is produced by a real run. There is no emoji anywhere in the
repository, by design.

---

## Table of contents

1. [Scientific motivation](#1-scientific-motivation)
2. [The hypothesis and the method](#2-the-hypothesis-and-the-method-xlora-bias)
3. [The six headline scientific outputs](#3-the-six-headline-scientific-outputs)
4. [Datasets](#4-datasets)
5. [Subject models](#5-subject-models)
6. [Methods under comparison](#6-methods-under-comparison)
7. [Metrics and statistics](#7-metrics-and-statistics)
8. [Verification layers (the triangulation)](#8-verification-layers-the-triangulation)
9. [Response format and parsing](#9-response-format-and-parsing)
10. [API architecture for judgement only](#10-api-architecture-for-judgement-only)
11. [Repository layout, file by file](#11-repository-layout-file-by-file)
12. [Machine split and full run order](#12-machine-split-and-full-run-order)
13. [Crash-safe execution on a pre-emptible VM](#13-crash-safe-execution-on-a-pre-emptible-vm)
14. [Safe batching and determinism](#14-safe-batching-and-determinism)
15. [Environment setup](#15-environment-setup)
16. [Configuration reference (all environment variables)](#16-configuration-reference-all-environment-variables)
17. [Outputs: the results-file dictionary](#17-outputs-the-results-file-dictionary)
18. [Data hygiene, checkpointing, reproducibility](#18-data-hygiene-checkpointing-reproducibility)
19. [Hardware and expected wall-clock](#19-hardware-and-expected-wall-clock)
20. [Full citation list](#20-full-citation-list)
21. [Recency check](#21-recency-check)
22. [Implementation decisions log](#22-implementation-decisions-log)
23. [Limitations](#23-limitations)
24. [Acceptance checklist](#24-acceptance-checklist)
25. [Reproduce from a fresh clone](#25-reproduce-from-a-fresh-clone)

---

## 1. Scientific motivation

Most debiasing methods push a model toward equal treatment everywhere. In law this is
often wrong: some differential treatment is legally required. A canonical example is
reservation under Article 16(4) of the Indian Constitution, where treating two groups
identically is itself the error. A fair legal model must therefore be
**difference-aware**: it must treat groups equally when the law requires equality, and
differently when the law requires (or permits) differential treatment.

The benchmark encodes this with two conditions:

- **neq** (differential treatment is legally justified): the correct answer is option
  **a** or **b** (a group-specific differential answer). The characteristic failure
  here is **over-equalization**: the model wrongly chooses **c** ("treat both groups
  identically"), collapsing a legally-required distinction.
- **eq** (equal treatment applies): the correct answer is always **c**. The
  characteristic failure is **spurious differential treatment**: the model chooses
  **a** or **b** when it should treat the groups identically.

Standard equal-treatment debiasing tends to push models toward **c** everywhere, which
improves eq behavior but *destroys* neq behavior (difference awareness). XLoRA-Bias is
designed to repair the over-equalization failure **locally** so that difference
awareness is not collateral damage.

---

## 2. The hypothesis and the method (XLoRA-Bias)

**Hypothesis (to test, not assume).** The "collapse to equal treatment" failure is
encoded in a *localizable* subset of layers and modules. If we attribute the failure
to those components, place trainable adapters only there with rank proportional to
attribution, and train with a loss that upweights the legally-different cases and
supervises legal rationales, then we can reduce the over-equalization signal in the
targeted components **while preserving the rest of the pretrained structure** that
carries difference awareness and general capability. Dense or uniform debiasing, by
contrast, distorts that structure and reduces difference awareness.

The method is implemented as five stages. The proposed system is the method under
test, not a baseline.

### Stage A - Attribution (localize the failure)

Files: [GPU_Run/probe_subject_models.py](GPU_Run/probe_subject_models.py),
[GPU_Run/attribution_integrated_gradients.py](GPU_Run/attribution_integrated_gradients.py).

1. On a held-out validation subset, run the frozen base model and collect the
   **failure set**: wrongly answered items, with special attention to neq items where
   the model wrongly chose **c** (the over-equalization failure) and to
   **identity-swap flips** (items where the predicted letter changes when group1 and
   group2 are deterministically swapped).
2. Compute **Integrated Gradients** (Sundararajan et al., 2017) for these failures
   with a **zero-embedding baseline** and **50 Riemann steps**, producing a per-layer,
   per-module attribution score. The implementation is a custom hook-based IG path
   conductance: along the straight path from the zero baseline to the real input
   embeddings, it accumulates the absolute sum of (module output element-wise times its
   gradient) of the cross-entropy loss for the correct answer, averaged over the
   Riemann steps. Attention and MLP projection modules are scored; per-layer scores are
   the sum of their module scores; layer scores are normalized by the max.

### Stage B - Localized LoRA placement

File: [GPU_Run/configure_layer_selective_lora.py](GPU_Run/configure_layer_selective_lora.py).

1. Select layers whose normalized attribution is at or above a threshold (default
   **0.15** of the max).
2. Set the **per-layer LoRA rank proportional to normalized attribution**, clipped to
   a minimum and maximum rank (default **4** and **64**).
3. Place adapters only on the selected layers, on attention and MLP projection modules.
   The exact layer / module / rank configuration is recorded in results metadata.
4. The same module also builds **uniform** and **random** placement configurations at a
   matched budget, for the placement ablation.

### Stage C - Training (repair only the targeted components)

Files: [GPU_Run/train_xlora_bias.py](GPU_Run/train_xlora_bias.py),
[GPU_Run/common/training.py](GPU_Run/common/training.py).

The joint loss per example is:

```
total = (answer_cross_entropy
         + rationale_weight * rationale_cross_entropy)   # rationale-aware, default rationale_weight 0.3
        * condition_weight                               # condition-adaptive: neq 1.5, eq 1.0
```

plus a **curriculum** that raises the neq sampling proportion from **0.2 to 0.5**
across epochs, plus a small **general-replay mix** (default **10 percent** of a fixed
public instruction subset) to guard against capability loss. Defaults: 3 epochs,
AdamW, learning rate 2e-4, LoRA dropout 0.05, bf16, gradient-accumulation effective
batch 8, max sequence length 1024.

### Stage D - Verification (did the repair work?)

Files: [GPU_Run/verify_bias_subspace.py](GPU_Run/verify_bias_subspace.py),
[CPU_Run/parameter_space_geometry.py](CPU_Run/parameter_space_geometry.py),
[GPU_Run/patchscope_bias_verification.py](GPU_Run/patchscope_bias_verification.py).

After training, three independent verifications are run (see
[Section 8](#8-verification-layers-the-triangulation)). This is a required output, not
optional.

### Stage E - Optional jurisdiction head (off by default)

A jurisdiction-specific lightweight classification head per jurisdiction is described
as an optional minor component behind a config flag. It is kept off by default.

---

## 3. The six headline scientific outputs

1. **Main comparison table**: the proposed method against six recent state-of-the-art
   baselines plus two reference baselines, on the legal benchmark, reporting all
   metrics in [Section 7](#7-metrics-and-statistics).
2. **Difference-awareness preservation**: DiffAware and CtxtAware (Wang et al. 2025),
   plus neq-accuracy and eq-accuracy, before and after each method. Expected story:
   equal-treatment-oriented baselines reduce difference awareness; the proposed
   localized method does not.
3. **Generalization**: zero-training evaluation on two external Wang et al. benchmarks
   (D2 Occupation descriptive, N3 Occupation Affirmative Action normative).
4. **Ablation**: attribution-guided vs uniform vs random placement; with and without
   rationale loss; with and without condition-adaptive loss; a rank sweep.
5. **Verification**: bias-subspace residual before and after repair, triangulated by
   the linear probe, the parameter-space geometry analysis, and the Patchscope readout.
6. **Quantization study**: full precision vs 4-bit NF4 QLoRA, per metric.

---

## 4. Datasets

### 4.1 Primary dataset (training and in-domain evaluation)

The **Law and Demographic Bias Difference-Awareness Benchmark** (a HuggingFace
dataset). It has three jurisdiction files (Indian, US, EU), 1,200 rows each, 3,600
total. Columns:

| Column | Meaning |
|---|---|
| `id` | unique row id |
| `category` | jurisdiction / category label |
| `form` | `descriptive` or `normative` |
| `condition` | `neq` (differential justified) or `eq` (equal treatment applies) |
| `group1`, `group2` | the two demographic groups in the scenario |
| `question` | the scenario / question text |
| `choice_a`, `choice_b`, `choice_c` | the three options |
| `correct_answer` | `a`, `b`, or `c` (neq: a or b; eq: always c) |
| `rationale` | the legal rationale (used by the rationale-aware objective and ReGiFT) |
| `law_reference` | the statute or article cited |
| `scenario_type` | scenario cluster used for the group-disjoint split |
| `is_myth_buster` | optional, on the US and EU criminal-justice subsets |

The reader [GPU_Run/common/dataset_io.py](GPU_Run/common/dataset_io.py) loads whatever
file layout the download produced (parquet, jsonl, json, or csv) and normalizes a
`jurisdiction` field.

### 4.2 Anti-synthetic-data rule

No synthetic training instances are generated. Every training instance is a
**deterministic templating** of a real dataset row (the existing question, three
choices, correct letter, rationale, and law reference are wrapped into an
instruction-tuning format by [GPU_Run/common/prompts.py](GPU_Run/common/prompts.py)).
Identity counterfactual pairs are produced **only** by deterministically swapping
group1 and group2 in existing rows, never by generating new text. The only role of any
external model in dataset preparation is light prompt-engineering of the fixed wrappers
([Dataset_Prep/prepare_prompts_deepseek.py](Dataset_Prep/prepare_prompts_deepseek.py));
it never generates question or answer content, and the canonical templates remain
authoritative.

### 4.3 Train / validation / test split (contamination-safe)

Because training and evaluation both come from one benchmark, the split is
**group-disjoint at the `scenario_type` level**: no scenario_type appears in both train
and test. The split (in
[Dataset_Prep/build_template_instances.py](Dataset_Prep/build_template_instances.py),
seed 42) holds out 20 percent of scenario_type clusters for the **frozen test set**,
carves a small **validation** set (10 percent of the remaining clusters) for Stage A
attribution, and uses the rest for **train**. The frozen test set is written
deterministically and its **SHA256 hash** is recorded to
`results/frozen_test_set_sha256.txt` and copied into
[PREREGISTRATION.md](PREREGISTRATION.md).

A **contamination check**
([Dataset_Prep/contamination_check.py](Dataset_Prep/contamination_check.py)) computes
8-gram overlap and embedding cosine similarity between the templated training file and
the frozen test file, and **hard-stops the pipeline** if 8-gram overlap exceeds **0.5
percent** or any test item exceeds **0.95** cosine similarity with any training item.
Embeddings use a local sentence-embedding model if `MULTILINGUAL_EMBED_MODEL` points to
one, otherwise a TF-IDF cosine fallback (the study is English only, so this suffices).

### 4.4 Counterfactual pairs

[Dataset_Prep/build_counterfactual_pairs.py](Dataset_Prep/build_counterfactual_pairs.py)
produces identity counterfactuals by deterministic whole-word, case-insensitive
swapping of group1 and group2 in the question, choices, and rationale. For neq rows the
differential options a and b are group-specific, so the swap also flips the correct
letter between a and b; eq rows stay c. These pairs drive identity-swap-flip detection
(Stage A) and the linear bias probe (Stage D).

### 4.5 External generalization datasets (evaluation only, never trained on)

The **Wang et al. (ACL 2025)** difference-awareness benchmark, two components: **D2
Occupation (descriptive)** and **N3 Occupation Affirmative Action (normative)**. They
are mapped onto the same evaluation harness by
[Dataset_Prep/map_external_wang_datasets.py](Dataset_Prep/map_external_wang_datasets.py).
If a component is not present locally, a clear note is written and nothing is
fabricated; place the official release files under `data/external_wang/`.

### 4.6 General replay set (capability guard)

A small fixed slice (default 500 rows) of `databricks/databricks-dolly-15k`, downloaded
once and used offline as the general-replay mix during training, to guard against
capability loss. The repo can be overridden with `GENERAL_REPLAY_REPO`.

---

## 5. Subject models

This is a **model-diversity axis**, not a language axis: the study is English only.

| Tier label | HuggingFace id | Family | Role | Attention | Notes |
|---|---|---|---|---|---|
| `compact-hybrid` | `nvidia/NVIDIA-Nemotron-3-Nano-4B` | nemotron (Mamba-2 hybrid) | transfer | sdpa | safetensors, never GGUF |
| `small-instruct` | `meta-llama/Llama-3.2-3B-Instruct` | llama | primary | flash_attention_2 | gated repo, needs HF key at download |
| `indic-specialized` | `Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0` | gemma | transfer | flash_attention_2 | Gemma-2 based |
| `broad-instruct` | `Qwen/Qwen3-4B-Instruct-2507` | qwen | primary | flash_attention_2 | bf16 repo, never FP8 |

Defined in [GPU_Run/common/model_registry.py](GPU_Run/common/model_registry.py).

- **Primary models** (`small-instruct`, `broad-instruct`) get the full baseline
  comparison, all ablations, the rank sweep, and three seeds for the proposed method
  and the strongest baseline.
- **Transfer models** (`compact-hybrid`, `indic-specialized`) get the lighter set
  (proposed method, FairNet, FairLoRA only) to show cross-architecture transfer.
- **Revision pinning**: the exact HF revision hash of every model is resolved and
  recorded at first download into `models/model_revisions.json` and echoed into every
  results CSV via `model_revision_hash`.
- **LoRA targets**: attention and MLP projections
  (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). The Nemotron
  Mamba-2 hybrid uses a restricted attention-and-MLP-linear set and falls back to a
  discovered-linear-suffix list if the family default fails; the exact target-module
  list is recorded in results metadata.
- **Attention implementation**: flash-attention 2 for Llama / Qwen / Gemma when the GPU
  supports it; automatic `sdpa` fallback for the Nemotron hybrid and unsupported GPUs.
  The implementation actually used is recorded per model
  (`attention_implementation_used`).
- **Quantization**: full-precision bf16 by default; a 4-bit NF4 QLoRA arm (double
  quantization, bf16 compute) for the quantization study.

---

## 6. Methods under comparison

All training baselines and the proposed method use a **matched trainable-parameter
budget**: the trainable-parameter percentage is equal across methods to within 2
percent (absolute), asserted in code before training (the proposed full-precision run
sets the reference; baselines must match it). The two reference baselines are context
and are exempt from the assertion.

### 6.1 Proposed method

`xlora_bias_proposed` = attribution-guided LoRA placement (Integrated Gradients) +
per-layer rank proportional to attribution + condition-adaptive loss (upweight neq) +
rationale-aware multi-task training + post-training attribution verification. See
[Section 2](#2-the-hypothesis-and-the-method-xlora-bias).

### 6.2 Six state-of-the-art baselines

Implemented in [GPU_Run/train_baselines.py](GPU_Run/train_baselines.py), each with a
paper-ready citation comment. Where a paper's full mechanism cannot be reproduced
exactly, the closest faithful approximation is used and noted; the distinguishing
signal of each method (flagging, placement, rationale traces, steering) is preserved.

1. **FairNet (2025)** - bias detector plus conditional LoRA modules activated only on
   flagged instances, with a contrastive loss reducing intra-group representation
   disparity. arXiv:2510.19421. (Approximated as uniform placement, condition-adaptive,
   trained on flagged neq instances.)
2. **IGU-LoRA (2026)** - Integrated-Gradients sensitivity per layer with
   uncertainty-aware rank allocation; the closest neighbor to the proposed method
   (it allocates rank by attribution for general task capability, not fairness).
   arXiv:2603.13792. (Attribution placement, capability-oriented loss.)
3. **PEDAL (2026)** - Classifier, Modifier, Reviewer pipeline around PEFT.
   doi:10.1145/3774904.3793029. (Uniform placement, condition-adaptive.)
4. **FairLoRA (2025)** - modular LoRA with discriminators, uniform rank.
   (Exact id UNVERIFIED.) (Uniform placement.)
5. **ReGiFT (2025)** - reasoning-trace fine-tuning using the dataset rationale field.
   arXiv:2504.05632. (Rationale-mode training.)
6. **FairSteer (2025)** - inference-time activation steering, no retraining.
   arXiv:2510.18914. Implemented by computing a debiasing steering vector (the mean
   hidden-state difference between the correct-differential and over-equalized
   completions on neq items, at a mid-depth layer) and applying it at inference via a
   forward hook.

### 6.3 Two reference baselines

- **Vanilla LoRA** (rank 8 on attention and MLP, all layers). Hu et al., ICLR 2022.
- **Vanilla QLoRA** (4-bit NF4 plus the same LoRA). Dettmers et al., NeurIPS 2023.

### 6.4 Ablations (on the primary models)

Driven by [GPU_Run/train_xlora_bias.py](GPU_Run/train_xlora_bias.py):

- **Placement**: attribution-guided vs uniform vs random, at matched budget.
- **Loss**: with vs without the rationale loss; with vs without condition-adaptive
  weighting.
- **Rank sweep**: ranks 8, 16, 32 on the attribution placement.
- **Quantization**: the proposed method in 4-bit NF4 QLoRA.

---

## 7. Metrics and statistics

Implemented in [GPU_Run/common/metrics.py](GPU_Run/common/metrics.py); computed per
model, per method, per condition, per jurisdiction, per form, per seed. Every result
CSV uses full descriptive column names with no abbreviations (enforced by
[GPU_Run/common/logging_utils.py](GPU_Run/common/logging_utils.py)).

### 7.1 Core metrics

- `overall_accuracy`
- `accuracy_on_neq_condition`
- `accuracy_on_eq_condition`
- `contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy` - the harmonic mean
  of neq-accuracy and eq-accuracy. **This is the primary endpoint.**
- `difference_aware_accuracy_gap_neq_minus_eq`
- `difference_aware_metric_wang_2025_recall_style` (DiffAware)
- `contextual_awareness_metric_wang_2025_precision_style` (CtxtAware)
- `utility_retention_accuracy_on_held_out_general_set` - accuracy on the held-out
  validation set as a general-capability probe
- `rationale_bleu4_against_reference`
- `rationale_rouge_l_against_reference`
- `rationale_factual_correctness_score_one_to_five_via_judge`
- `rationale_statutory_citation_preservation_rate`
- `cross_jurisdiction_transfer_accuracy` (six directed jurisdiction pairs)
- `bias_subspace_residual_pre_repair` and `bias_subspace_residual_post_repair`
- `trainable_parameter_percentage`
- `attention_implementation_used` (flash_attention_2 or sdpa)
- `model_revision_hash`, `random_seed`, `quantization_setting`,
  `json_parse_failure_rate_percent`, `api_judge_model_string`

### 7.2 Wang contingency definitions (DiffAware and CtxtAware)

A 2x... style contingency table over five cells:

- **A** = neq items answered with the correct differential option (a or b)
- **B** = neq items answered with the wrong differential option
- **C** = neq items answered with c (over-equalization, the collapse failure)
- **D** = eq items answered with a (spurious differential)
- **E** = eq items answered with b (spurious differential)

Then **DiffAware = A / (A + B + C)** (recall style) and
**CtxtAware = A / (A + D + E)** (precision style).

### 7.3 Rationale metrics

BLEU-4 (sacrebleu, with a fallback), ROUGE-L (rouge-score, with an LCS fallback), and a
statutory-citation preservation rate (the fraction of reference citations such as
"Article 16(4)", "Title VII", or "Section 123" that survive in the generated
rationale). Factual correctness (1 to 5) is scored by the judge chain on a stratified
subsample.

### 7.4 Statistics

[CPU_Run/statistics_tests.py](CPU_Run/statistics_tests.py): a **paired bootstrap** with
1000 resamples at p < 0.05 on the contextual fairness score between the proposed method
and the automatically-selected strongest baseline, plus a **paired Wilcoxon
signed-rank test** on per-item correctness deltas. Seeds 42, 43, 44 on the two primary
models for the proposed method and the strongest baseline; seed 42 elsewhere; mean and
standard deviation reported. No projected values appear anywhere.

---

## 8. Verification layers (the triangulation)

Three independent verifications support the same claim ("localize, repair, preserve")
from three different angles.

### 8.1 IG re-run and linear identity probe

[GPU_Run/verify_bias_subspace.py](GPU_Run/verify_bias_subspace.py) reports, before
(frozen base) and after (proposed adapter):

- `over_equalization_attribution_localization_fraction` - the share of the bias
  attribution mass that still falls on the originally targeted layers.
- `identity_linear_probe_accuracy` - the cross-validated accuracy of a logistic probe
  decoding the swapped identity (group1 vs group2) from a mid-depth hidden state. Higher
  means more residual bias signal. The headline `bias_subspace_residual_*` columns use
  this probe.

### 8.2 Parameter-space geometry (where the weight update lands)

[CPU_Run/parameter_space_geometry.py](CPU_Run/parameter_space_geometry.py) is a
measurement and explanation layer in **parameter space**, adapted from the geometry of
post-training updates (Shen et al. 2026; Zhu et al. 2025; Liu et al. 2026). For every
weight-modifying method on both primary models, it computes the per-matrix weight
update Delta_W (via PEFT `get_delta_weight`, correct for any scaling convention; zero
for non-adapted modules; `inference_time_no_weight_update` for FairSteer) and seven
diagnostics, all on **CPU float64 SVD**:

1. **bf16-aware update sparsity** (eta = 1e-3) plus **module coverage** (the fraction of
   analyzed matrices with any nonzero update - the key cross-module localization
   signal).
2. **Principal-angle rotation** of the top-k left and right singular subspaces (k =
   min(512, smaller dimension)).
3. **Normalized spectral shift** (L2 change of the singular-value vector, normalized).
4. **Update-mask overlap** with a principal mask (top alpha by magnitude of the rank-64
   SVD reconstruction of W0) and a low-magnitude mask (bottom alpha by |W0|), alpha =
   0.5, random baseline 0.5.
5. **Stable rank** of the update.
6. **Frobenius norm** of the update.
7. **Hill tail exponent** of the update's top singular values.

It assigns a **regime label** (off-principal-and-spectrum-preserving;
principal-aligned-and-distorting; relaxed-off-principal) using documented thresholds
(principal-mask overlap below 0.5 and spectral shift at most 0.5 times the
principal-aligned anchor's shift), with Full Fine-tuning as the anchor when present and
the maximum observed shift as the fallback anchor otherwise. It then reports the
**Pearson correlation (with p-value)** across methods between normalized spectral shift
and difference-awareness preservation, and between principal-mask overlap and
preservation. The mechanistic prediction is a negative correlation; it is reported
whichever way it lands. The companion add-on prompt is
[XLoRA-Bias_ADDON_Parameter_Space_Geometry.md](XLoRA-Bias_ADDON_Parameter_Space_Geometry.md).

### 8.3 Patchscope verification (what the representation encodes)

[GPU_Run/patchscope_bias_verification.py](GPU_Run/patchscope_bias_verification.py) with
[GPU_Run/common/patchscopes.py](GPU_Run/common/patchscopes.py) is a **training-free,
layer-resolved, natural-language** readout of the over-equalization signal, adapted from
Patchscopes (Ghandeharioun et al., ICML 2024), which is more reliable than logit-lens
in early-to-mid layers. For the frozen base and the proposed adapter, at each
attribution-targeted layer, it patches that layer's last-position representation into a
letter-eliciting target prompt (same model, identity mapping; a single deterministic
forward, no sampling) and reads the renormalized probability of option c. A successful
localized repair **lowers over-equalization on neq** while **keeping it high on eq**
(equal treatment retained, difference awareness preserved).

Together: the geometry analysis shows the **weights** moved off-principal; the linear
probe shows whether **identity** is decodable; and Patchscopes shows that the
**representations** stop collapsing to "equal" on neq while still applying equality on
eq.

---

## 9. Response format and parsing

Every prompt to every causal model and every API model instructs JSON-only output
matching an explicit schema given in the prompt. For multiple-choice answering the
schema is exactly `{"answer_choice_letter": "a"}` with the value one of a, b, c. The
rationale-aware arm additionally requests the rationale text and a list of cited law
references, still as one JSON object.

Answer extraction is **deterministic** and uses exactly one shared utility,
`json_repair_parse` in [GPU_Run/common/parsing.py](GPU_Run/common/parsing.py): it
strips code fences, fixes trailing commas, extracts the first balanced JSON object, and
reads `answer_choice_letter` directly. **No judge model is used for answer
extraction.** The per-model JSON-parse failure rate is counted and reported. Only when
parsing fails on a specific item does the caller fall back to the judge chain to extract
the letter, and every such fallback is logged. All decoding for evaluation is greedy,
temperature 0, with a fixed `max_new_tokens` (256).

---

## 10. API architecture for judgement only

Judgement (in [GPU_Run/common/clients.py](GPU_Run/common/clients.py)) is needed only
for rationale factual-correctness scoring and for the rare answer-extraction fallback.
Answers themselves never use a judge.

- **Judge chain (no retry)**: primary **Gemini 2.5 Flash** (four GCP keys round-robin),
  then secondary **DeepSeek** (`deepseek-chat`, two keys round-robin) **with OpenRouter
  as an alternate route** (two keys round-robin), then tertiary **Mistral Small** (two
  keys round-robin). On any provider error there is **no retry of the same provider**;
  the chain moves once to the next. Keys rotate round-robin **per request, not per
  failure**. If all providers fail for an item, it is written to
  `results/failed_rows_<script>.jsonl` with the error messages and the batch continues.
- **Prompt engineering** (Dataset_Prep): DeepSeek with two keys round-robin, JSON-only,
  used only to refine fixed instruction wrappers, never to generate content.
- **HF key**: used only by the one-time download script for gated repos; never at
  runtime.
- **Judge-robustness control**:
  [CPU_Run/judge_robustness.py](CPU_Run/judge_robustness.py) re-runs a 10 percent
  stratified subsample of the rationale judgements with the secondary judge and reports
  percent agreement and Cohen kappa against the Gemini-family judge.
- **Model strings**: a pinned model id is used per provider and the model string the API
  actually returns is logged into every output row.

**Keys live only in `.env`** at the repository root, loaded with python-dotenv. No
source file contains a literal key. `.env` is gitignored and `.env.example` ships with
placeholders. The env loader
([GPU_Run/common/env_loader.py](GPU_Run/common/env_loader.py)) accepts the canonical
names below and a set of legacy aliases so an existing `.env` works unchanged:

```
HF_KEY              <- HUGGINGFACE_TOKEN
GCP_Key1..3         <- GEMINI_API_KEY_1..3
GCP_key4            <- GEMINI_API_KEY_4        (the lowercase 'key' is intentional)
DEEPSEEK_KEY1/2     <- DEEPSEEK_API_KEY_1/2
MISTRAL_KEY1/2      <- MISTRAL_API_KEY1/2
OPENROUTER_KEY1/2   <- OPENROUTER_API_KEY_1/2
```

A hardcoded-key self-check
([GPU_Run/common/hygiene.py](GPU_Run/common/hygiene.py):`scan_repo_for_hardcoded_keys`)
scans the repository for accidental key patterns and fails loudly without printing any
value.

---

## 11. Repository layout, file by file

```
Codes/                         (repository root: holds .env and the four code folders)
  .env.example                 secret template (canonical names; legacy aliases noted)
  .gitignore                   ignores .env, results/, data/, models/, checkpoints/, *.pdf
  README.md                    this file
  PREREGISTRATION.md           preregistered endpoint, seeds, tests, frozen-test hash
  requirements_global.txt      pinned global requirements (no venv)
  XLoRA-Bias_IMPLEMENTATION_PROMPT.md   the original build specification
  XLoRA-Bias_ADDON_Parameter_Space_Geometry.md   the geometry add-on specification
  Dataset_Prep/   Dry_Run/   GPU_Run/   CPU_Run/
```

### GPU_Run/common/ (the shared contract)

| File | Responsibility |
|---|---|
| [paths.py](GPU_Run/common/paths.py) | canonical filesystem paths; creates the output trees on import |
| [env_loader.py](GPU_Run/common/env_loader.py) | the only reader of `.env`; canonical names plus legacy aliases; never prints a value |
| [seeds.py](GPU_Run/common/seeds.py) | global seed (default 42), three-seed list, deterministic-algorithm setup |
| [logging_utils.py](GPU_Run/common/logging_utils.py) | stdout + file logging; `write_csv` enforcing descriptive column names; run-metadata jsonl |
| [parsing.py](GPU_Run/common/parsing.py) | the single deterministic `json_repair_parse` and answer/rationale extractors |
| [hygiene.py](GPU_Run/common/hygiene.py) | `validate_and_dedup` and the hardcoded-key self-check |
| [dataset_io.py](GPU_Run/common/dataset_io.py) | reads the raw dataset (any layout); defines the derived-file paths |
| [prompts.py](GPU_Run/common/prompts.py) | fixed instruction templates, JSON schemas, judge prompts, SFT example builder |
| [model_registry.py](GPU_Run/common/model_registry.py) | the four models, revision pinning, offline loading, LoRA targets, safe batch sizes |
| [flash_attn_setup.py](GPU_Run/common/flash_attn_setup.py) | pre-built-wheel flash-attn install with OS check and sdpa fallback |
| [clients.py](GPU_Run/common/clients.py) | round-robin API clients and the no-retry judge chain |
| [key_selftest.py](GPU_Run/common/key_selftest.py) | key validation, hardcoded-key check, live per-key API test table |
| [inference.py](GPU_Run/common/inference.py) | greedy decoding, shared extraction, and batched generation |
| [training.py](GPU_Run/common/training.py) | joint loss, curriculum, replay, budget assertion, batched and sequential training |
| [checkpointing.py](GPU_Run/common/checkpointing.py) | resumable jsonl/batched runners; epoch checkpoint and resume helpers |
| [metrics.py](GPU_Run/common/metrics.py) | Wang contingency, accuracies, BLEU/ROUGE, citation preservation, linear probe |
| [patchscopes.py](GPU_Run/common/patchscopes.py) | the Patchscope patch primitive |
| [artifact_sync.py](GPU_Run/common/artifact_sync.py) | crash-safe orphan-branch snapshot/restore of results/checkpoints/data |

### Dataset_Prep/ (runs on the GPU VM; only the download script uses the network)

| File | Responsibility |
|---|---|
| [download_models_and_data.py](Dataset_Prep/download_models_and_data.py) | the ONLY networked script: downloads the four models and all datasets, pins revisions |
| [build_template_instances.py](Dataset_Prep/build_template_instances.py) | deterministic templating; group-disjoint split; freezes the test set and its SHA256 |
| [build_counterfactual_pairs.py](Dataset_Prep/build_counterfactual_pairs.py) | identity counterfactuals by deterministic group swap |
| [prepare_prompts_deepseek.py](Dataset_Prep/prepare_prompts_deepseek.py) | DeepSeek prompt-engineering of the fixed wrappers only |
| [map_external_wang_datasets.py](Dataset_Prep/map_external_wang_datasets.py) | maps the external Wang D2/N3 sets onto the harness |
| [contamination_check.py](Dataset_Prep/contamination_check.py) | 8-gram overlap and cosine contamination check with hard-stop |

### GPU_Run/ (runs on the H100 VM)

| File | Responsibility |
|---|---|
| [measure_base_competence.py](GPU_Run/measure_base_competence.py) | base competence of the four frozen models on the test set |
| [probe_subject_models.py](GPU_Run/probe_subject_models.py) | collects the failure set (wrong answers, over-equalization, identity flips) |
| [attribution_integrated_gradients.py](GPU_Run/attribution_integrated_gradients.py) | Stage A Integrated Gradients attribution |
| [configure_layer_selective_lora.py](GPU_Run/configure_layer_selective_lora.py) | Stage B placement and rank configuration (attribution/uniform/random) |
| [train_xlora_bias.py](GPU_Run/train_xlora_bias.py) | Stage C: the proposed method, all ablations, the rank sweep, the QLoRA arm |
| [train_baselines.py](GPU_Run/train_baselines.py) | the six SOTA plus two reference baselines, at matched budget |
| [evaluate_all.py](GPU_Run/evaluate_all.py) | evaluates every method on the legal test set, external sets, and held-out general set |
| [verify_bias_subspace.py](GPU_Run/verify_bias_subspace.py) | Stage D: IG re-run and linear identity probe |
| [patchscope_bias_verification.py](GPU_Run/patchscope_bias_verification.py) | Stage D+: training-free Patchscope verification |
| [autosync_results.py](GPU_Run/autosync_results.py) | background daemon: push artifacts every 10 minutes |
| [restore_artifacts.py](GPU_Run/restore_artifacts.py) | restore artifacts on a fresh VM or the CPU machine |

### CPU_Run/ (runs on the CPU machine, after GPU_Run; no GPU required)

| File | Responsibility |
|---|---|
| [aggregate_results.py](CPU_Run/aggregate_results.py) | the main comparison, difference-awareness, external, and quantization tables |
| [statistics_tests.py](CPU_Run/statistics_tests.py) | paired bootstrap and Wilcoxon on per-item predictions |
| [figures_and_tables.py](CPU_Run/figures_and_tables.py) | main comparison, difference-awareness, placement, rank-sweep figures |
| [judge_robustness.py](CPU_Run/judge_robustness.py) | secondary-judge agreement and Cohen kappa |
| [parameter_space_geometry.py](CPU_Run/parameter_space_geometry.py) | the parameter-space geometry analysis |
| [README.md](CPU_Run/README.md) | CPU-stage notes (inputs, the models requirement for geometry) |

### Dry_Run/ (one dry-run per other folder; fast, offline, run before expensive work)

| File | Responsibility |
|---|---|
| [dry_run_dataset_prep.py](Dry_Run/dry_run_dataset_prep.py) | templating, counterfactuals, mapping, contamination, dedup-catches-planted; key tests |
| [dry_run_gpu_run.py](Dry_Run/dry_run_gpu_run.py) | model loads, probe, attribution, configure, smoke train, batching-equivalence, eval, verify, patchscope; key tests |
| [dry_run_analysis.py](Dry_Run/dry_run_analysis.py) | aggregation, statistics, figures, judge robustness, geometry on synthetic data; key tests |

---

## 12. Machine split and full run order

**Dataset_Prep, Dry_Run, and GPU_Run run on the H100 GPU VM; CPU_Run runs on the CPU
machine afterward.** `configure_layer_selective_lora.py` is CPU-light but stays in
GPU_Run because it runs between attribution and training. After the one-time download,
every script runs fully offline (`HF_HUB_OFFLINE=1`, `local_files_only=True`); only the
download script touches the network.

```
################################################################
# ON THE H100 GPU VM
################################################################

# 0. One-time setup
pip install -r requirements_global.txt --break-system-packages
python GPU_Run/common/flash_attn_setup.py
cp .env.example .env   # then fill in .env (or copy your existing .env)

# 1. Dataset_Prep: one-time download (the ONLY networked step) then offline prep
python Dataset_Prep/download_models_and_data.py
python Dataset_Prep/build_template_instances.py
python Dataset_Prep/build_counterfactual_pairs.py
python Dataset_Prep/prepare_prompts_deepseek.py
python Dataset_Prep/map_external_wang_datasets.py
python Dataset_Prep/contamination_check.py

# 2. Dry runs (fast, offline, verify integrity before expensive runs)
python Dry_Run/dry_run_dataset_prep.py
python Dry_Run/dry_run_gpu_run.py
python Dry_Run/dry_run_analysis.py

# 3. Start the crash-safe autosync daemon in the background BEFORE GPU_Run.
nohup python GPU_Run/autosync_results.py >> results/autosync.log 2>&1 &

# 4. GPU_Run (offline, in this order). All steps are resumable.
python GPU_Run/measure_base_competence.py
python GPU_Run/probe_subject_models.py
python GPU_Run/attribution_integrated_gradients.py
python GPU_Run/configure_layer_selective_lora.py
python GPU_Run/train_xlora_bias.py
python GPU_Run/train_baselines.py
python GPU_Run/evaluate_all.py
python GPU_Run/verify_bias_subspace.py
python GPU_Run/patchscope_bias_verification.py

#   If the VM is pre-empted, bring up a new VM, do step 0, then:
#     python GPU_Run/restore_artifacts.py
#     python Dataset_Prep/download_models_and_data.py   # re-fetch models/ only
#   relaunch the autosync daemon (step 3), and re-run the GPU_Run scripts (they resume).

################################################################
# ON THE CPU MACHINE (run after GPU_Run completes)
################################################################

# 5. Pull the artifacts GPU_Run pushed, then run the CPU stage (offline compute).
python GPU_Run/restore_artifacts.py     # restores results/, checkpoints/, data/
#   parameter_space_geometry additionally needs the base weights in models/. Either
#   copy models/ from the VM or run Dataset_Prep/download_models_and_data.py here once.
python CPU_Run/aggregate_results.py
python CPU_Run/statistics_tests.py
python CPU_Run/figures_and_tables.py
python CPU_Run/judge_robustness.py
python CPU_Run/parameter_space_geometry.py
```

---

## 13. Crash-safe execution on a pre-emptible VM

[GPU_Run/autosync_results.py](GPU_Run/autosync_results.py) runs as a background daemon
during GPU_Run and force-pushes `results/`, `checkpoints/`, and `data/` to a dedicated
`gpu-run-artifacts` branch every 10 minutes (configurable via
`GPU_RUN_SYNC_INTERVAL_SECONDS`) and once on exit. It uses a **separate git index** so
it never touches the `main` working tree or staging area, writes **orphan snapshot
commits** (force-pushed, so the branch always holds only the latest snapshot, and
pushes are incremental), and reads the GitHub token from `.env` without ever printing
it. `models/` is not synced because it is large and re-downloadable.

On pre-emption, [GPU_Run/restore_artifacts.py](GPU_Run/restore_artifacts.py) fetches the
branch and restores those three directories so the resumable GPU_Run scripts continue
from the last saved state. The same restore step brings the artifacts onto the CPU
machine before CPU_Run. Worst case on pre-emption: lose at most ~10 minutes plus the
in-progress epoch (training resumes per saved epoch; evaluation and API loops skip
completed ids).

---

## 14. Safe batching and determinism

Batching is a throughput optimization only; it is implemented so results stay
consistent for research use and is applied uniformly to every method, condition, and
seed.

- **Evaluation/generation batching** (`EVAL_BATCH_SIZE`, default 16) uses left-padding
  plus an attention mask, greedy decoding, and input-order processing, so a given batch
  size is reproducible and equivalent to the single-example path up to floating-point
  reduction order. Used by `evaluate_all`, `measure_base_competence`, and
  `probe_subject_models`.
- **Training batching** (`TRAIN_MICRO_BATCH_SIZE`, default 8) fuses examples within the
  existing gradient-accumulation window into one forward/backward. The effective batch
  (`grad_accum`) is unchanged; the per-example normalized loss and the
  condition/rationale weights are computed exactly per example, so the summed,
  grad_accum-scaled gradient equals the sequential accumulation up to floating-point
  order. The window grouping and replay draw order match the sequential path step for
  step. Setting this to 1 uses the exact sequential path.
- **Mamba-hybrid models** (Nemotron) are automatically clamped to batch size 1
  (`model_registry.safe_eval_batch_size` / `safe_train_micro_batch_size`).
- **Attribution is intentionally NOT batched**: it feeds discrete layer-selection and
  rank-rounding thresholds, so it stays single-example for exact reproducibility.
- **Determinism**: seeds are fixed, deterministic algorithms are enabled where
  available, batch composition is deterministic, and the batch sizes are pinned. Do not
  change batch sizes partway through a study.
- **Validation**: [Dry_Run/dry_run_gpu_run.py](Dry_Run/dry_run_gpu_run.py) includes
  equivalence checks that run on the real models before any expensive run: batched vs
  single generation must produce identical extracted letters on confident items, and
  the fused batched training loss must match the sequential accumulation within a small
  floating-point tolerance.
- **Caveat**: batched and single GPU kernels reduce floating point in different orders,
  so batched numbers are not bit-identical to a single-example run; a few borderline
  predictions may differ. This noise is non-systematic, far smaller than the
  seed-to-seed variance the study already reports, and identical across methods, so it
  does not bias any comparison.

---

## 15. Environment setup

- **No virtual environment.** Everything installs into the global Python environment:
  `pip install -r requirements_global.txt --break-system-packages`.
- **Flash-attention from a pre-built wheel only** (never compiled from source). Run
  `python GPU_Run/common/flash_attn_setup.py`. It detects the installed torch, CUDA,
  and Python versions, confirms the OS is compatible (Linux x86_64 with a CUDA build),
  selects and installs the matching pre-built wheel (preferring flash-attn 2.8.3,
  otherwise the nearest compatible release), and records the result to
  `results/flash_attn_setup.json`. If the OS or GPU is incompatible it skips
  flash-attention and falls back to `sdpa`.
- **`.env`**: copy `.env.example` to `.env` and fill in the keys (canonical names in
  [Section 10](#10-api-architecture-for-judgement-only); legacy aliases are accepted).
  Never commit `.env`.

---

## 16. Configuration reference (all environment variables)

| Variable | Default | Effect |
|---|---|---|
| `EVAL_BATCH_SIZE` | 16 | generation batch size (Mamba clamped to 1); 1 = single-example path |
| `TRAIN_MICRO_BATCH_SIZE` | 8 | training micro-batch within the accumulation window; 1 = sequential |
| `EVAL_SUBSET_SIZE` | 0 (all) | cap test/validation items in `evaluate_all` |
| `BASE_COMPETENCE_SUBSET_SIZE` | 0 (all) | cap items in base competence |
| `ATTRIBUTION_RIEMANN_STEPS` | 50 | IG Riemann steps |
| `ATTRIBUTION_MAX_ITEMS` | 64 | failure items used for attribution |
| `PROBE_MAX_ITEMS` | 200 | items for the linear identity probe in verification |
| `PATCHSCOPE_MAX_ITEMS` | 48 | items per condition for the Patchscope verification |
| `RATIONALE_JUDGE_FRACTION` | 0.1 | subsample fraction for rationale factual-correctness judging |
| `STRICT_BUDGET` | 1 | if 1, matched-budget mismatch raises; if 0, it only warns |
| `TRAIN_SMOKE_MAX_STEPS` | unset | cap optimizer steps (used by dry runs) |
| `GEOMETRY_LAYER_STRIDE` | 1 | subsample layers in the geometry analysis to speed a run |
| `GPU_RUN_SYNC_INTERVAL_SECONDS` | 600 | autosync interval |
| `GPU_RUN_ARTIFACTS_BRANCH` | gpu-run-artifacts | the artifacts branch name |
| `PRIMARY_DATASET_REPO` | a documented default | override the primary dataset HF repo id |
| `WANG_DATASET_REPO` | unset | optional HF repo id for the external Wang sets |
| `GENERAL_REPLAY_REPO` | databricks/databricks-dolly-15k | the general-replay source |
| `MULTILINGUAL_EMBED_MODEL` | unset | local sentence-embedding model dir for the contamination check |

For an RTX 3090 (24 GB) set `TRAIN_MICRO_BATCH_SIZE=1` and `EVAL_BATCH_SIZE=4` (try 8)
to avoid out-of-memory on the 4B models; see
[Section 19](#19-hardware-and-expected-wall-clock).

---

## 17. Outputs: the results-file dictionary

All intermediate and final outputs go to `results/`; datasets to `data/`; checkpoints
to `checkpoints/`; downloaded base models to `models/`; figures to `results/figures/`.
These four trees are git-ignored and recreated by the code.

**Data and models metadata**

- `models/model_revisions.json` - pinned model revision hashes.
- `data/dataset_revisions.json` - pinned dataset revisions.
- `data/templated_all_instances.jsonl`, `data/train_instances.jsonl`,
  `data/validation_instances.jsonl`, `data/test_instances_frozen.jsonl`,
  `data/counterfactual_pairs.jsonl` - the templated and split instances.
- `results/frozen_test_set_sha256.txt` - the frozen test-set hash.
- `results/data_hygiene_log.csv` - rows removed (duplicates, corrupted) per run.
- `results/contamination_report.csv` - the 8-gram and cosine contamination metrics.
- `results/flash_attn_setup.json`, `results/run_metadata.jsonl` - environment metadata.

**GPU_Run outputs**

- `results/base_competence_summary.csv` and
  `results/base_competence_predictions_<tier>.jsonl`.
- `data/probe_failures_<tier>.jsonl`, `results/probe_failure_summary.csv`.
- `results/attribution_<tier>.json` - per-layer/per-module attribution.
- `results/lora_config_<tier>.json` - attribution / uniform / random placements.
- `checkpoints/<tier>/<method>/seed_<n>/` - per-epoch and final LoRA adapters.
- `results/matched_budget_reference.json`, `results/train_xlora_bias_runs.json`,
  `results/train_baselines_runs.json`, `results/fairsteer_vector_<tier>.json`.
- `results/main_evaluation_results.csv` - the central per-(model, method, seed, scope,
  jurisdiction, form, dataset) results table (full column list in
  [Section 7](#7-metrics-and-statistics)).
- `results/cross_jurisdiction_transfer_results.csv`,
  `results/per_item_predictions_<tier>_<method>_seed<n>.jsonl`,
  `results/rationale_judgements_<tier>_<method>_seed<n>.jsonl`.
- `results/bias_subspace_verification.csv` - IG re-run plus linear-probe residual.
- `results/patchscope_bias_verification_per_layer.csv` and
  `results/patchscope_bias_verification_summary.csv`.
- `results/failed_rows_<script>.jsonl` - items where all judge providers failed.

**CPU_Run outputs**

- `results/aggregated_main_comparison_table.csv`,
  `results/difference_awareness_preservation_table.csv`,
  `results/aggregated_external_generalization_<dataset>.csv`,
  `results/quantization_study_table.csv`.
- `results/statistical_tests.csv`, `results/judge_robustness_agreement.csv`.
- `results/geometry/parameter_space_geometry_per_method.csv`,
  `results/geometry/per_layer_per_module_geometry.csv`,
  `results/geometry/geometry_vs_difference_awareness_correlation.csv`,
  `results/geometry/geometry_run_log.csv`.
- `results/figures/*.png` - main comparison, difference-awareness, placement ablation,
  rank sweep, and the two-panel `parameter_space_regime.png`.

---

## 18. Data hygiene, checkpointing, reproducibility

- **Data hygiene on every run.** Before any script consumes a dataset or intermediate
  file, the shared `validate_and_dedup`
  ([GPU_Run/common/hygiene.py](GPU_Run/common/hygiene.py)) drops exact-duplicate rows by
  content hash, detects corrupted rows (unparseable JSON, empty required fields), logs
  counts to `results/data_hygiene_log.csv`, and writes the cleaned file. It runs
  idempotently, including on reruns.
- **Checkpoint and resume.** Every API and evaluation loop saves progress every 50 rows
  and skips completed ids on restart. Every training script saves a LoRA checkpoint each
  epoch and supports resume.
- **Reproducibility.** Global seed 42 by default, with per-run seeds 42, 43, 44 where
  three seeds are required. Every seed, model revision hash, and API model string is
  logged into every results CSV. Deterministic algorithms are enabled where available.
- **No projected numbers.** Every reported value comes from a real run; synthetic
  numbers appear only in dry-run fixtures, clearly under `results/dry_run/`.

---

## 19. Hardware and expected wall-clock

**Target.** A single modern data-center GPU with bf16 and flash-attention 2 support
(A100 or H100 class), Linux, recent CUDA.

**On H100, with default batching (Section 14): roughly 15-25 H100-hours** for the full
GPU_Run stage, dominated by training and evaluation. Setting `EVAL_BATCH_SIZE=1` and
`TRAIN_MICRO_BATCH_SIZE=1` reverts to the single-example path (~50-65 H100-hours).
Approximate per-stage figures with batching:

- Dataset_Prep download (one time, network): 30-90 minutes.
- Dataset_Prep templating, counterfactuals, contamination check: minutes.
- Base competence, probe, attribution per model: minutes to tens of minutes.
- Training per method per model per seed: roughly 5-20 minutes batched.
- evaluate_all, verify, patchscope: minutes to tens of minutes.
- CPU_Run on the CPU machine: minutes for aggregation/statistics/figures; the
  parameter-space geometry module is CPU float64 SVD and can take several hours.

**On an RTX 3090 (24 GB).** It runs, but **not at the default batch sizes**: the 4B
models train past 24 GB at `TRAIN_MICRO_BATCH_SIZE=8` (the float32 logits over the
vocabulary plus activations dominate). Set `TRAIN_MICRO_BATCH_SIZE=1` and
`EVAL_BATCH_SIZE=4-8`. The 3090 supports bf16, flash-attention 2, and bitsandbytes
4-bit, and models load one at a time, so a single 4B model fits. Expect the full study
to take roughly **6-9 days of continuous GPU time** (dominated by the training runs at
micro-batch 1). The scope cut in [PREREGISTRATION.md](PREREGISTRATION.md) (drop the two
transfer models, optionally fewer seeds/ablations) brings this to roughly 2-3 days.

---

## 20. Full citation list

Method, benchmark, metric, and comparison citations are also embedded in the comment
header of each implementing file. arXiv ids and DOIs are marked UNVERIFIED where they
could not be confirmed offline at implementation time; verify each before publication.

```
Wang, A., Phan, M., Ho, D. E., Koyejo, S. "Fairness through Difference Awareness:
  Measuring Desired Group Discrimination in LLMs." ACL 2025, pp. 6867-6893. arXiv:2502.01926.
FairNet. "Dynamic Fairness Correction without Performance Loss via Contrastive
  Conditional LoRA." arXiv:2510.19421, 2025.
IGU-LoRA. "Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Aware
  Scoring." arXiv:2603.13792, 2026.
PEDAL. "Mitigating Fine-tuning Bias: A Parameter-Efficient Debiasing Framework for
  Large Language Models." ACM Web Conference 2026. doi:10.1145/3774904.3793029.
FairLoRA. 2025. [exact id UNVERIFIED]
ReGiFT. arXiv:2504.05632, 2025.
FairSteer. arXiv:2510.18914, 2025.
Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
Sundararajan, M., Taly, A., Yan, Q. "Axiomatic Attribution for Deep Networks." ICML 2017.
Banerjee, A., Ganguly, S., Mukherjee, I., Ganguly, N. "Explainability-Guided Hidden
  Content Sterilization for Trusted Multimedia Systems." ICSCCC 2026, IEEE.
Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L., Geva, M. "Patchscopes: A
  Unifying Framework for Inspecting Hidden Representations of Language Models."
  ICML 2024 (PMLR 235). arXiv:2401.06102.
Shen, Z. et al. "On the Geometry of On-Policy Distillation." arXiv:2606.07082, 2026.
Zhu, H. et al. "The Path Not Taken: RLVR Provably Learns Off the Principals."
  arXiv:2511.08567, 2025.
Liu, Z. et al. "Lift the Veil for the Truth: Principal Weights Emerge after Rank
  Reduction for Reasoning-Focused Supervised Fine-Tuning." arXiv:2506.00772, 2026.
Mukherjee, S. et al. "Reinforcement Learning Finetunes Small Subnetworks in Large
  Language Models." arXiv:2505.11711, 2025.
Shenfeld, I., Pari, J., Agrawal, P. "RL's Razor: Why Online Reinforcement Learning
  Forgets Less." arXiv:2509.04259, 2025.
Hill, B. M. "A Simple General Approach to Inference About the Tail of a Distribution."
  The Annals of Statistics, 3(5):1163-1174, 1975.
Conover, M. et al. "Free Dolly." Databricks, 2023. (databricks/databricks-dolly-15k)
Deb, K., Basu, A. "Language-Specific Bias Circuits in Multilingual Language Models."
  ACM Transactions on Asian and Low-Resource Language Processing, 2026.
```

---

## 21. Recency check

As of the implementation date, a review of fairness-specific parameter-efficient
debiasing methods did not surface a method that is clearly both fairness-specific and
PEFT-specific and newer than the six listed (FairNet 2025, IGU-LoRA 2026, PEDAL 2026,
FairLoRA 2025, ReGiFT 2025, FairSteer 2025). The two 2026 methods IGU-LoRA and PEDAL are
the most recent and are both included. If a newer 2026 fairness-and-PEFT method is
identified, add it as a seventh baseline by registering it in
[GPU_Run/train_baselines.py](GPU_Run/train_baselines.py) and noting it here.

---

## 22. Implementation decisions log

- Repository root is the `Codes/` folder, which holds `.env`, the four code folders, and
  the generated `results/`, `data/`, `checkpoints/`, `models/` trees.
- The four code folders are Dataset_Prep, Dry_Run, GPU_Run, and CPU_Run. The former
  `Analysis/` folder was renamed `CPU_Run/` so the CPU-only post-processing runs on a
  separate machine after the GPU stage. `parameter_space_geometry` additionally needs
  `models/` and `checkpoints/` on the CPU machine (copy `models/` from the VM or re-run
  the download script there).
- The env loader accepts both the canonical variable names and the legacy aliases in the
  provided `.env`. Keys are read only through `env_loader`.
- Unspecified low-level choices were resolved simply: HuggingFace text Causal LM SFT
  format; greedy temperature-0 decoding with `max_new_tokens` 256; LoRA dropout 0.05;
  AdamW; cosine-style schedule; bf16; default 3 training epochs.
- The general-replay subset defaults to a small fixed slice of
  `databricks/databricks-dolly-15k`, downloaded once and used offline.
- Integrated Gradients use a zero-embedding baseline and 50 Riemann steps (custom
  hook-based path conductance; Captum may be used to cross-check).
- External Wang D2/N3 sets are read from the official release if present locally,
  otherwise from a user-provided local path under `data/external_wang/`; never
  fabricated.
- Crash safety: the autosync daemon force-pushes `results/`, `checkpoints/`, and `data/`
  to the `gpu-run-artifacts` branch via a separate git index and orphan commits, so
  `main` is never touched; `models/` is excluded (re-downloadable).
- Safe batching (default on) is throughput-only: generation batching uses left-padding
  plus attention masks and deterministic ordering; training batching fuses the existing
  accumulation window without changing the effective batch, computing per-example
  normalized loss so the gradient matches the sequential path up to floating-point
  order. Mamba hybrids are clamped to batch size 1, attribution stays single-example,
  and Dry_Run validates batched-vs-single equivalence.
- Geometry analysis: Delta_W via PEFT `get_delta_weight`; CPU float64 SVD with the base
  SVD and reference masks cached per matrix; geometry diagnostics averaged over adapted
  matrices; regime thresholds (overlap below 0.5 and shift at most 0.5x the anchor) with
  a max-shift fallback anchor when no Full Fine-tune is present; preservation is
  post-repair CtxtAware divided by the base CtxtAware; correlations need at least three
  methods; `GEOMETRY_LAYER_STRIDE` subsamples layers.
- Patchscope verification is an additive, training-free Stage D+ readout (same model,
  identity mapping, letter-eliciting target prompt, single deterministic forward); it
  inspects the attribution-targeted layers by default with an all-layer fallback.
- Reference papers (PDFs) are git-ignored (`*.pdf`); kept local, not committed.

---

## 23. Limitations

- The primary benchmark is a constructed (synthetic-by-design) benchmark of legal
  scenarios; findings are about model behavior on that benchmark, not a guarantee about
  deployed legal systems.
- Training data is built only by deterministic templating and deterministic identity
  swaps of real rows. No synthetic training instances are generated; this bounds
  diversity to what the benchmark contains.
- Judgement (rationale factual correctness and the rare answer-extraction fallback)
  relies on LLM judges. A single-judge control re-runs 10 percent of judgements with the
  secondary judge and reports agreement and Cohen kappa.
- Several baselines approximate the original papers' full mechanisms; the comparison is
  at a matched parameter budget on the same data and harness, but is not a
  bit-for-bit reproduction of each method.
- Some arXiv ids and one DOI are marked UNVERIFIED and must be confirmed before
  publication.

---

## 24. Acceptance checklist

1. All three dry runs pass on a machine with a valid `.env` and pre-downloaded models
   and data.
2. No script other than `download_models_and_data.py` touches the network for
   downloads; all runtime loads are offline.
3. `git grep` finds no literal API key in any tracked file, including tests.
4. Every results CSV uses full descriptive column names with no abbreviations.
5. Answer extraction is deterministic JSON parsing with no judge; the judge chain is
   used only for rationale scoring and the rare extraction fallback, with no
   within-provider retry and correct round-robin keys.
6. All six SOTA baselines plus the two reference baselines are implemented, each with a
   citation comment; the two 2026 methods (IGU-LoRA, PEDAL) are present.
7. The recency check is recorded above.
8. Trainable-parameter budgets match across methods within 2 percent, asserted in code.
9. Flash-attention is installed from a pre-built wheel only, with automatic sdpa
   fallback recorded per model; OS compatibility is checked.
10. Duplicate and corrupted rows are detected and logged on every rerun.
11. No virtual environment is created; `requirements_global.txt` is complete and pinned.
12. No emoji anywhere in the repository.
13. No projected or simulated numbers exist anywhere; every reported value comes from a
    real run.
14. This README alone is sufficient to run the study from a fresh clone.
15. Batching is throughput-only, validated batched-vs-single in the dry run, applied
    uniformly across methods, and reversible to the single-example path.

---

## 25. Reproduce from a fresh clone

1. Clone the repository onto the H100 VM and `cd` into `Codes/`.
2. Do the one-time setup ([Section 15](#15-environment-setup)): install requirements,
   run the flash-attn setup, and create `.env` from `.env.example`.
3. Run Dataset_Prep, then the three dry runs, then GPU_Run in the order in
   [Section 12](#12-machine-split-and-full-run-order), with the autosync daemon running
   in the background.
4. On the CPU machine, restore the artifacts and run CPU_Run.
5. Read the tables and figures in `results/` and `results/figures/`. The primary
   endpoint is the contextual fairness score on the legal test set for the proposed
   method versus the strongest baseline at matched parameter budget, with the paired
   bootstrap and Wilcoxon tests in `results/statistical_tests.csv`.

The dry runs in step 3 verify keys, offline loading, flash-attention or sdpa selection,
batched-vs-single equivalence, and the absence of hardcoded keys before any expensive
run. No emoji appears anywhere in this repository.
