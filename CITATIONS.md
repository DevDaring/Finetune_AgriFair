# References, verified 2026-09-03

Every identifier below was checked against the live record on 2026-09-03. Anything that
could not be verified is marked as such rather than carried forward silently.

## Correction carried into the code

**FairSteer.** The identifier used in the earlier draft, arXiv:2510.18914, does **not**
resolve to FairSteer. FairSteer is *FairSteer: Inference Time Debiasing for LLMs with
Dynamic Activation Steering*, Findings of ACL 2025, **arXiv:2504.14492**. Every citation
comment in the code now carries the correct identifier. arXiv:2510.18914 is a different
and separately relevant paper, *Fairness Evaluation and Inference Level Mitigation in
LLMs* (Nadeem, Dras, Naseem, October 2025), cited on its own footing in related work.

**XLoRA-Bias**, the earlier study's method name, returns no indexed record. It is treated
as unpublished prior work by the same author, cited from the manuscript rather than as an
externally verifiable reference. This is one more reason the method here carries a
different name (see TERMINOLOGY.md).

## The construct

- Wang, A., Phan, M., Ho, D. E., Koyejo, S. *Fairness through Difference Awareness:
  Measuring Desired Group Discrimination in Large Language Models.* ACL 2025.
  arXiv:2502.01926. The DiffAware and CtxtAware metrics reported here as primary.
- Pan, Z., Liang, Z., Kabbara, J., Emami, A. *DART: Mitigating Harm Drift in
  Difference-Aware LLMs via Distill-Audit-Repair Training.* Findings of ACL 2026.
  arXiv:2604.16845. The closest external competitor and a baseline here. DART reports
  base models over-predicting differential treatment, the opposite skew to the premise of
  this study, which is why failure polarity is measured rather than assumed.
- Tang, Z., Truong, S. T., Owens, D., Sharma, S., Zhang, Y. J., Miranda, B., Koyejo, S.
  *In-Situ Behavioral Evaluation for LLM Fairness, Not Standardized-Test Scores.*
  arXiv:2605.12530, April 2026. Argues static multiple-choice fairness benchmarks are
  unreliable. The free-text AgriAdvice half and the structure-novel slice are this
  study's answer to that critique.

## Baselines, as implemented

- Zhou, S., Liu, Z., Jiang, B. *FairNet: Dynamic Fairness Correction without Performance
  Loss via Contrastive Conditional LoRA.* NeurIPS 2025. arXiv:2510.19421.
- Cui, X., Li, H., Zeng, R., et al. *IGU-LoRA: Adaptive Rank Allocation via Integrated
  Gradients and Uncertainty-Aware Scoring.* ICLR 2026. arXiv:2603.13792.
- Li, Q., Zhang, K., Wu, L., et al. *Mitigating Fine-tuning Bias: A Parameter-Efficient
  Debiasing Framework for Large Language Models* (PEDAL). The ACM Web Conference 2026.
  doi:10.1145/3774904.3793029.
- Kabra, S., Jha, A., Reddy, C. K. *Reasoning towards Fairness: Mitigating Bias in
  Language Models through Reasoning-Guided Fine-Tuning* (ReGiFT). arXiv:2504.05632.
- *LFTF: Locating First and Then Fine-Tuning for Mitigating Gender Bias in Large Language
  Models.* arXiv:2505.15475, May 2025. The nearest published prior art to
  attribution-guided placement, and the control that isolates what the attribution signal
  contributes.
- Li, Y., Fan, Z., Chen, R., et al. *FairSteer: Inference Time Debiasing for LLMs with
  Dynamic Activation Steering.* Findings of ACL 2025. arXiv:2504.14492.
- Hu, E. J., Shen, Y., Wallis, P., et al. *LoRA: Low-Rank Adaptation of Large Language
  Models.* ICLR 2022. arXiv:2106.09685.
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. *QLoRA: Efficient Finetuning of
  Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.

FairLoRA is deliberately not used as a method name or a baseline: two unrelated papers
already carry it (arXiv:2410.17358 for vision models, and a separate 2025/2026 OpenReview
submission), so the name identifies nothing.

## Instruments

- Sundararajan, M., Taly, A., Yan, Q. *Axiomatic Attribution for Deep Networks.* ICML
  2017. arXiv:1703.01365.
- Ghandeharioun, A., Caciularu, A., Pearce, A., et al. *Patchscopes: A Unifying Framework
  for Inspecting Hidden Representations of Language Models.* ICML 2024. arXiv:2401.06102.
- Hendrycks, D., Burns, C., Basart, S., et al. *Measuring Massive Multitask Language
  Understanding* (MMLU). ICLR 2021. arXiv:2009.03300. The external capability probe.

## Parameter-space geometry

- Shen, Z., Li, Y., Yin, Q., Leong, C. T., Wang, Z., Chen, Y., Han, R., Lee, S., Fung,
  Y. R. *On the Geometry of On-Policy Distillation.* arXiv:2606.07082, June 2026.
- Liu, Z., Pang, T., Balabanov, O., et al. *Lift the Veil for the Truth: Principal Weights
  Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning.*
  arXiv:2506.00772, June 2025.
- Zhu, H., Zhang, Z., Huang, H., et al. *The Path Not Taken: RLVR Provably Learns Off the
  Principals.* arXiv:2511.08567, 2025.
- Mukherjee, S., Yuan, L., Hakkani-Tur, D., Peng, H. *Reinforcement Learning Finetunes
  Small Subnetworks in Large Language Models.* arXiv:2505.11711, 2025.
- Shenfeld, I., Pari, J., Agrawal, P. *RL's Razor: Why Online Reinforcement Learning
  Forgets Less.* arXiv:2509.04259, 2025. Cited for the link between where an update lands
  and how much is forgotten; the title does not itself claim a geometry framing.
- Hill, B. M. *A Simple General Approach to Inference About the Tail of a Distribution.*
  The Annals of Statistics 3(5):1163-1174, 1975.

## Bias localization and mechanistic context

- Floro, A., Benedetto, L., et al. *Cultural Binding Heads in Language Models.*
  arXiv:2605.28543, July 2026. Two to three mid-layer attention heads carry the binding of
  identity to contextually appropriate content, and they transfer from instruct to base
  models. Independent support for localizing rather than globally retuning.
- *Toward Localizing and Repairing Bias in Transformer Attention Heads* (ROBIN).
  arXiv:2607.12863, ICSME 2026 NIER track, July 2026. Head-level localization and surgical
  repair, concurrent with this work.
- Karvonen, A., Marks, S. *Robustly Improving LLM Fairness in Realistic Settings via
  Interpretability.* arXiv:2506.10922, June 2025. Prompt-only mitigation fails under
  realistic context, which is why this study measures free-text advice drift.
- *Activation-Guided Layer Selection for LoRA.* Information (MDPI) 17(3):283, March 2026.
  doi:10.3390/info17030283. Layer-selective LoRA outside fairness.
- Xu, X., He, X., Zhi, C., Chen, R., McAuley, J., He, Z. *BiasFreeBench: A Benchmark for
  Mitigating Bias in Large Language Model Responses.* ICLR 2026. arXiv:2510.00232.

## Indian context and agriculture

- Vashistha, A., Aneja, U., Gupta, A., et al. *Beyond Semantics: Examining Gender Bias in
  LLMs Deployed within Low-resource Contexts in India.* ACM FAccT 2025.
  doi:10.1145/3715275.3732180. Field study covering agricultural deployments in India.
- Nawale, J. A., Khan, M. S. U. R., D, J., Gupta, M., Pruthi, D., Khapra, M. M.
  *FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and
  Stereotypes* (Indic-Bias). ACL 2025. arXiv:2506.23111.
- *Sima AIunty: Caste Audit in LLM-Driven Matchmaking.* arXiv:2603.29288, 2026.
- *AgroBench: Vision-Language Model Benchmark in Agriculture.* arXiv:2507.20519, July
  2025. A capability benchmark, cited as scope contrast: it does not measure fairness.
- BharatGen's AgriParam-1 and BhashaBench-Krishi are agricultural instruction-tuning and
  knowledge benchmarks. Referenced as adjacent work with the caveat that only the
  vendor's own material could be verified, and that they measure knowledge accuracy
  rather than difference awareness.

No paper, model card, or leaderboard was found that uses or cites AgriFair. The dataset
repository Debk/AgriFair is private, created 8 June 2026, so this is expected rather than
a null result about the benchmark's reception.

## Multiple-choice shortcut and counterfactual evaluation

- *Do Large Language Models Plan Answer Positions? Position Bias in Multiple-Choice
  Question Generation.* arXiv:2605.01846, 2026. Motivates the option-rotation robustness
  check.
- *BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice
  Benchmarks.* arXiv:2602.06221, 2026. The same concern this study answers with the
  surface-cue ceiling and the structure-novel slice.
- Amiri-Margavi, A., et al. *Equal Access, Unequal Interaction: A Counterfactual Audit of
  LLM Fairness.* arXiv:2602.02932, February 2026. Counterfactual consistency scoring for
  identity-toggled free text, the closest published method to the AgriAdvice drift panel.

## Hosted models in the frozen frontier panel

Evaluated as frozen subjects, never repaired. Identifiers and access status verified against
the live APIs on 2026-09-04 and recorded in `GPU_Run/common/api_models.py`.

- Claude Sonnet 5, served by the Anthropic API as `claude-sonnet-5`, falling back to
  `anthropic/claude-sonnet-5` on OpenRouter.
- GPT-5.6 Luna, served by the OpenAI API as `gpt-5.6-luna`, falling back to
  `openai/gpt-5.6-luna` on OpenRouter.
- Nova 2 Lite, served by Bedrock as `amazon.nova-2-lite-v1:0` across two accounts, falling
  back to `amazon/nova-2-lite-v1` on OpenRouter.
- Nemotron Nano 3 30B, served by Bedrock as `nvidia.nemotron-nano-3-30b` across two accounts.
  Absent from the OpenRouter catalogue, so it has no fallback.

Hosted models are the one part of this study that cannot be version-pinned the way a
HuggingFace revision hash pins a local model. Every raw response is cached and released, and
the evaluation date is recorded, so the numbers remain checkable even after a provider
updates a model behind the same name. This is stated as a limitation rather than presented as
reproducible in the same sense as the local runs.

## Data sources

- All India Report on Agriculture Census 2015-16, Department of Agriculture and Farmers
  Welfare, Government of India. Every AgriFacts answer is arithmetic on these numbers.
- KisanVaani/agriculture-qa-english-only (Apache-2.0). The verbatim farmer questions
  behind AgriAdvice.
