# Terminology: what was renamed, what was not, and why

This study reuses an experimental design first applied to a legal difference-awareness
benchmark. The design carries over; the **names that study coined for its own contributions
do not**, because reusing them would present the same named contribution twice. Terms that
belong to other people's papers are left exactly as published, so the work stays legible to
anyone who knows that literature.

## Renamed, because the earlier study coined them

| Earlier study's term | This study | Why the new name |
|---|---|---|
| XLoRA-Bias | **GRAFT** (Gradient-Ranked Adapter Fairness Targeting) | The method name is the earlier paper's headline contribution. Grafting is the agricultural image for what the method does: a small, local, load-bearing repair that leaves the rest of the plant intact. |
| over-equalization | **gap erasure** | Names the AgriFair failure directly: a real census gap is erased when a diff item is answered "Roughly equal". No indexed paper uses "over-equalization", so it was the earlier study's coinage. |
| spurious differential treatment | **gap fabrication** | The mirror failure: a gap is invented on an equal item. |
| equalization ratchet | **awareness trading** | The phenomenon is a method buying one direction of awareness with the other. "Trading" says what happens without the earlier metaphor. |
| direction-of-failure census | **failure polarity profile** | Same measurement, reported per model, condition, axis and test slice. |
| Difference-Aware Repair Contract (DARC) | **repair admissibility criteria** | Three criteria with fixed thresholds, scored per method. The construct is retained; the contract branding is not. |
| PACF, preservation-adjusted contextual fairness | **non-regression-adjusted contextual fairness** | States the adjustment plainly: the score is docked for what a repair destroys. |
| repair efficiency ratio | **fairness gain per unit** | Score gained per unit of drift, per trainable-parameter percent, or per training minute. |
| contextual fairness score (CFS) | **balanced awareness score** | The harmonic mean of diff-condition and equal-condition accuracy. Same quantity, new name, because the earlier paper introduced the scalar under its name. |
| neq / eq condition labels | **diff / equal** | AgriFair ships these labels. Using them removes an imported legal artifact and makes every CSV read against the dataset card. |
| verification triangulation | **repair readout panel** | Four independent readouts around one claim; the panel is described rather than branded. |

## Kept, because they belong to other papers

Never renamed, never paraphrased into something that looks new:

- **difference awareness**, **contextual awareness**, **DiffAware**, **CtxtAware**, **desired group discrimination** - Wang, Phan, Ho and Koyejo, ACL 2025, arXiv:2502.01926.
- **harm drift**, **Distill-Audit-Repair (DART)**, the 1x/2x/3x/4x severity oversampling - Pan, Liang, Kabbara and Emami, Findings of ACL 2026, arXiv:2604.16845.
- **FairNet**, **IGU-LoRA**, **PEDAL** (Classifier / Modifier / Reviewer), **ReGiFT**, **LFTF** (Locating First and Then Fine-Tuning), **FairSteer** and its debiasing steering vector - the baseline papers listed in CITATIONS.md.
- **LoRA**, **QLoRA**, **Integrated Gradients**, **Patchscopes** - the method primitives.
- **principal weights**, **off-principal**, **spectrum-preserving**, **relaxed off-principal**, **subspace locking**, **stable rank**, **principal-angle rotation**, **normalized spectral shift**, **update-mask overlap**, **bf16-aware update sparsity**, **Hill tail exponent** - the parameter-space geometry literature.
- **MMLU** for the external capability probe.

## Names introduced here for things that had no name

These are new because the measurement is new, not because an old name was avoided.

- **state-blind key** - an item's (axis, metric, size class, comparison token): its answer-relevant structure with the state name removed.
- **surface-cue ceiling** - the accuracy a majority-class predictor reaches from the state-blind key alone. Every reported accuracy is read against it.
- **structure-familiar** and **structure-novel** test slices - whether an item's state-blind key also appears in training.
- **option-rotation robustness** - whether a method's canonical answer survives a cyclic rotation of the displayed option order.
