# AgriFair GRAFT

Attribution-guided LoRA for difference-aware fairness in agricultural language models.

A model asked whether Scheduled Castes or Scheduled Tribes operate more agricultural land
in Maharashtra can fail in two opposite directions. It can answer "Roughly equal" where the
2015-16 Agriculture Census records a real gap, which is **gap erasure**. Or it can name a
group where the census shows no meaningful difference, which is **gap fabrication**. Most
debiasing pushes a model toward equal treatment, which repairs one failure by inducing the
other. A single aggregate score hides that entirely.

**GRAFT** (Gradient-Ranked Adapter Fairness Targeting) localizes where a model encodes the
gap-erasure failure with Integrated Gradients, places LoRA adapters on those layers alone
with rank proportional to attribution, trains them with a condition-adaptive and
rationale-aware objective, and then re-measures with four independent readouts. Localize
the failure, repair only that, preserve the rest.

## What this repository contains

- The full pipeline, from the raw dataset to every table and figure, behind one entry point.
- Eight comparison methods at a matched trainable-parameter budget, each with its published
  mechanism and its divergences stated, plus nine ablation arms.
- An audit that is true regardless of where GRAFT ranks: the failure polarity profile and
  the awareness-trade table.
- A benchmark audit that changes how the numbers should be read: the surface-cue ceiling
  and the structure-novel test slice.

## Models

Four instruction-tuned models across two size classes and four pretraining recipes. All
four are primary: each carries the full baseline set, the ablations, the rank sweep and the
multi-seed arms.

| Tier | Model | Params | Role | Why this one |
|---|---|---|---|---|
| `small-instruct` | meta-llama/Llama-3.2-3B-Instruct | 3.2B | primary | the standard small instruct baseline |
| `broad-instruct` | Qwen/Qwen3-4B-Instruct-2507 | 4.0B | primary | a different pretraining corpus and tokenizer at the same scale |
| `general-instruct` | google/gemma-3-12b-it | 12.2B | primary | tests whether the localization finding survives well above the small-model regime |
| `general-instruct-2` | mistralai/Ministral-8B-Instruct-2410 | 8.0B | primary | a second large model from a different vendor, so the size result is not one family's quirk |

Two identifiers were corrected against the Hub on 2026-09-03. Gemma 3 ships at 270m, 1b,
4b, 12b and 27b, so there is no 9b release and `gemma-3-12b-it` is the smallest above that
size; `Ministral-8B-Instruct-2409` does not exist and `-2410` is the only Ministral 8B
Instruct. Gemma 3 at 12b is a multimodal checkpoint whose text tower sits under
`model.language_model`, so the loader walks a chain of auto classes and records which one
succeeded. Only text is used.

Running all four at full sweep is roughly 110 to 170 hours on one H100. `SUBJECT_MODELS`
is the lever: set it to `small-instruct,broad-instruct` for a fast pass, then add the large
pair. The larger models carry per-model batch caps so a single `EVAL_BATCH_SIZE` does not
have to be retuned; gradient accumulation absorbs the smaller micro-batch, so the
optimisation is unchanged.

## The frozen frontier panel

Four hosted models are evaluated alongside the local ones. They are **evaluation subjects,
never repair targets**: GRAFT needs gradients, weights and hidden states, so a hosted model
cannot be localized, adapted, or verified by three of the four readouts. They never enter the
main comparison table, because they cannot be trained at a matched parameter budget and
placing them beside FairNet or DART would be a category error.

What they answer is the part that needs only behaviour, and that part does not depend on how
GRAFT ranks: which direction a frontier model fails in, how much of its score the state-blind
prior explains, whether its agronomic advice moves with the farmer's identity, and whether its
answer survives a rotation of the option order.

### Routing: one chain per model, no retries anywhere

Every route is attempted **at most once** per request. Any failure falls straight through to
the next. `botocore` is pinned to `max_attempts=1`, so the no-retry rule holds at the
transport layer too.

| Model | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| Claude Sonnet 5 | Anthropic direct, `claude-sonnet-5` | OpenRouter key 1 | OpenRouter key 2 |
| GPT-5.6 Luna | OpenAI direct, `gpt-5.6-luna` | OpenRouter key 1 | OpenRouter key 2 |
| Nova 2 Lite | Bedrock, accounts A and B round-robined | OpenRouter key 1 | OpenRouter key 2 |
| Nemotron Nano 3 30B | Bedrock, accounts A and B round-robined | none in catalogue | |

The two Bedrock accounts swap starting position on every request, so load is shared rather
than always landing on the first, and a failure on one falls through to the other before
leaving AWS. The route that served each item is stored with the item, and per-model counts go
to `results/frontier_panel_route_log.csv`.

### Why Sonnet 5 and Luna do not use Bedrock

Both AWS accounts are AWS India (AISPL). The Reserve Bank of India's restriction on stored
card data means AWS Marketplace cannot charge a stored card for AISPL customers, so
subscribing to any third-party Marketplace model fails with `INVALID_PAYMENT_INSTRUMENT`.
This was verified over two full subscription cycles on 2026-09-04, on both accounts, before
and after making a card the default payment method. Amazon's own Nova and the NVIDIA model
are unaffected because they bill through ordinary AWS billing and need no subscription.

Leaving Bedrock in those two chains would burn two guaranteed failures and their latency on
every request, so they start at the vendor's own API instead. The Bedrock ids are still
recorded in the registry, so nothing has to change if the accounts are ever moved to
invoicing.

### Two provider quirks the code handles

`gpt-5.6-luna` rejects both `max_tokens` and `temperature`; it takes `max_completion_tokens`
alone. `claude-sonnet-5` needs the `anthropic-workspace-id` header for a workspace-scoped key
and rejects `temperature` as deprecated. Both shapes were established by probing the live
APIs rather than assumed.

### Cost

`python CPU_Run/estimate_api_cost.py` prices a full panel run from measured token counts and
current list prices. At the time of writing one full run is about $13.76: roughly $9.16 to
Anthropic, $2.19 to OpenAI, and $2.42 on Bedrock, which the round-robin splits into about
$1.21 per AWS account. Claude Sonnet 5 is two thirds of it, and advice generation is most of
that, so `FRONTIER_ADVICE_MAX_TOKENS` is the main lever. Every response is cached per item,
so re-runs cost nothing and only new prompts are charged.

The panel is off by default. Enable it with `RUN_FRONTIER_PANEL=1`.

## The three fairness obligations AgriFair separates

| Obligation | Carrier | Failure it catches |
|---|---|---|
| Acknowledge a real difference | AgriFacts `diff` items | gap erasure: answering "Roughly equal" over a census gap |
| Do not invent a difference | AgriFacts `equal` items | gap fabrication: naming a group where shares match |
| Do not let identity change a fact | AgriAdvice pairs | advice drift: different agronomy for different personas |

The primary endpoint is the **balanced awareness score**, the harmonic mean of
diff-condition and equal-condition accuracy, reported alongside the Wang et al. (2025)
DiffAware and CtxtAware metrics and separately on each test slice.

## The benchmark audit, and why it changes the reading

AgriFacts is templated over a small combinatorial space: roughly 36 states, a few group
pairs, two metrics. Each item therefore has a **state-blind key**, its
(axis, metric, size class, comparison token) with the state removed. A majority-class
predictor that sees only that key scores **0.872** on the corpus. Under a paraphrase-only
split, 97.3 percent of test items share a key with training, so a near-ceiling score is
uninterpretable.

The split now reserves twenty percent of state-blind keys, stratified by axis, for the
frozen test set alone. The test set carries two labelled slices:

| slice | items | what it asks |
|---|---|---|
| structure_familiar | 328 | the combination is in training; only the state differs |
| structure_novel | 421 | the combination is absent from training; the census fact is needed |

`results/surface_cue_ceiling_audit.csv` records the ceiling for every split, and every
figure draws it as a reference line. An accuracy at or below it is not evidence of census
knowledge.

## Layout

```
Codes/
  .env / .env.example        secrets; env_loader is the only reader and never prints a value
  requirements_global.txt    pinned
  run_all.py                 single entry point, resumable; --list prints the step order
  PREREGISTRATION.md         endpoints, hypotheses, splits, thresholds, and what changed
  TERMINOLOGY.md             which names changed from the earlier study, and which did not
  CITATIONS.md               verified references, including one identifier correction
  coding_prompt.md           the build specification
  explanation.html           beginner walk-through with two worked examples
  Dataset/                   raw AgriFair download (agrifacts.jsonl, agriadvice.jsonl)
  Dataset_Prep/  Dry_Run/  GPU_Run/  CPU_Run/
  data/  results/  checkpoints/  models/    generated, git-ignored
```

## Quick start

```
pip install -r requirements_global.txt
python GPU_Run/common/flash_attn_setup.py     # installs the flash-attn wheel or selects sdpa
cp .env.example .env                          # then fill in the keys

python run_all.py --smoke                     # offline end-to-end check, about five minutes on CPU

python run_all.py --stage dataset_prep        # the real study
python run_all.py --stage dry_run
python run_all.py --stage gpu_run             # start GPU_Run/autosync_results.py first
python GPU_Run/restore_artifacts.py           # then on the CPU machine
python run_all.py --stage cpu_run
```

`python run_all.py --list` prints every stage and step. A step that fails is logged with
its traceback and the run continues; the exit status is non-zero and the failed steps are
named in the summary.

## Compute, and why nothing is trimmed by default

Every default in this repository runs the full preregistered schedule: 132 training runs and
140 evaluation targets across the four models, with advice drift and the behavioural sweeps
covering every arm and every seed. On a single A100 40GB that is about **96 GPU-hours**, or
roughly **$64** at the 25th-percentile Vast.ai price of $0.67 per hour read on 2026-09-05.

Reduction levers exist (`ADVICE_TARGET_SCOPE`, `BEHAVIOURAL_SWEEP_SCOPE`, `MULTI_SEED_TIERS`,
`RANK_SWEEP_TIERS`, `LOAO_TIERS`) and each is documented in PREREGISTRATION.md with the claim
it narrows. **None is on by default**, because the saving is small relative to the risk: the
most aggressive profile saves about $23, which is not worth re-running a study after a
reviewer asks for the arm that was skipped.

Two of them deserve a specific warning if you ever do reach for them. Narrowing advice drift
removes the ablation arms, which are what attribute a drift reduction to the localized
placement rather than to the condition-adaptive objective. Narrowing `LOAO_TIERS` touches the
transfer study, and on the predecessor benchmark transfer behaviour, not the aggregate score,
was what distinguished the method. The proposed method keeps three seeds on every model under
every profile, so a variance estimate for it is never the thing that goes missing.

The one lever that costs nothing scientifically is the instance type. The pipeline is fully
resumable, so an interruption loses minutes rather than a run: training skips arms whose final
adapter exists, evaluation reuses stored per-item predictions, and the API panel caches every
response. Interruptible Vast.ai instances run 40 to 50 percent below on-demand, which takes
the same full study to roughly **$32 to $38** with no change to what is computed.

`python CPU_Run/estimate_gpu_hours.py` reports hours and cost per GPU for whatever profile the
environment currently selects, reading the arm schedule from the code that runs it, so the
figure moves when the study does.

## What the study claims, and what each claim needs

Ordered so the paper does not stand or fall on one leaderboard position. Only the last
requires winning.

| # | Claim | Evidence | Needs a leaderboard win? |
|---|---|---|---|
| 1 | AgriFacts concedes 0.87 to a state-blind prior, so difference-awareness scores on templated benchmarks need a structural control | `results/surface_cue_ceiling_audit.csv`, `structure_slice_comparison_table.csv` | no |
| 2 | The dominant failure direction is model- and condition-dependent, not uniformly a collapse toward equal | `results/failure_polarity_profile.csv` | no |
| 3 | Parameter-efficient debiasing can trade one direction of awareness for the other | `results/awareness_trade_audit.csv`, `figures/awareness_trade.png` | no |
| 4 | Admissible repair can be stated as three scored criteria, and existing methods measured against them | `results/repair_admissibility.csv` | no |
| 5 | Update geometry predicts what a repair preserves | `results/geometry/geometry_vs_difference_awareness_correlation.csv` | no |
| 6 | Post-repair verification needs four independent readouts, and the panel works on any method | attribution re-run, identity probe, geometry, Patchscope, all run across methods | no |
| 7 | Gap erasure is localizable and repairable locally | placement ablation at an asserted-equal budget: attribution against uniform against three random draws, each drawing a fresh layer set for the same layer count and rank multiset (H1) | no |
| 8 | The repair is competitive at uniquely low invasiveness | `figures/preservation_frontier.png`, fairness gain per unit | competitive, not first |
| 9 | GRAFT takes the top balanced awareness score | `results/statistical_tests.csv` | **yes** |

Claim 9 is reported as context. Claims 1 to 8 are the contribution, and one pipeline run
produces all of them.

## Baseline fidelity

Public implementations for this task do not exist, so each baseline is a
mechanism-representative re-implementation on the same harness, trained on the same data with
the same optimiser and schedule. The divergences are stated rather than hidden.

Baselines keep **their own placement rule and therefore their own trainable-parameter
budget**: forcing a common budget would compare against a method the cited paper never
proposed. Each arm's trainable share is measured and published in
`trainable_parameter_percentage`, and the H3 analysis controls for it. Equal budget is
asserted only for the placement ablations, where it is the hypothesis being tested.

| Method | Kept | Approximated |
|---|---|---|
| FairNet (arXiv:2510.19421) | condition-adaptive correction, uniform placement | the contrastive detector gate becomes a global diff upweight |
| IGU-LoRA (arXiv:2603.13792) | Integrated-Gradients rank allocation | attribution in activation space, as elsewhere in this harness, and no condition term, which is the paper's own position |
| PEDAL (doi:10.1145/3774904.3793029) | Classifier gate, Modifier upweight on flagged samples only | the Reviewer stage is not implemented; the gate is the deterministic diff rule |
| DART (arXiv:2604.16845) | distil, audit against the frozen base, severity-weighted repair; the published 1/2/3/4 multipliers | the distilled trace is AgriFair's census-grounded rationale rather than a teacher LLM's, because no label here may come from a language model; the audit is fairness drift, not toxicity drift |
| ReGiFT (arXiv:2504.05632) | reasoning-trace supervision on the reference rationale | as described |
| LFTF (arXiv:2505.15475) | locate blocks first, then fine-tune only those | the block score is hidden-state sensitivity to the identity swap rather than the published gender-probe score |
| FairSteer (arXiv:2504.14492) | inference-time steering vector, no retraining | the vector is the mean hidden-state difference between the correct-group and the erased completion at mid depth |

## Engineering contract

Deterministic JSON answer extraction with no judge in the answer path. The judge is used
only to read a letter out of an output the parser could not parse, to score rationale factual
correctness on a subsample, and for the advice-drift cross-check, and on extraction it sees
the model's own output alone: no question, no options, no gold answer. Its chain is DeepSeek
key 1, DeepSeek key 2, Mistral key 1, Mistral key 2, one attempt each, no retries, and the
same chain serves local and hosted models alike so a parse failure is resolved identically
everywhere. API keys read only from `.env` through `env_loader`, never printed, with a
hardcoded-key scan on every dry run. Offline after one download. Resumable: training skips
arms whose final adapter exists, evaluation reuses stored predictions, and both can be
forced to redo with an environment flag. Full descriptive CSV column names, asserted in
code. Matched trainable-parameter budgets asserted in code. No projected numbers anywhere.
No emoji.

## Data provenance

AgriFacts answers are arithmetic on the *All India Report on Agriculture Census 2015-16*.
AgriAdvice base queries come verbatim from `KisanVaani/agriculture-qa-english-only`
(Apache-2.0). No label comes from a language model. The reference rationale is deterministic
templating of each row's own ground truth, qualitative in direction, carrying the census
cell as its citation; no numeric magnitude is invented. `Debk/AgriFair` is a private
repository, so the HuggingFace token needs read access to it.

Cite the Agriculture Census of India 2015-16 and the KisanVaani corpus. Full reference list
in [CITATIONS.md](CITATIONS.md).
