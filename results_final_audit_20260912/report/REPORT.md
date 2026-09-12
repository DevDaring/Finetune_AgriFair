# Final audit report (auto-generated; numbers are from artifacts, not typed in)
## Table 1 - source audit sample (frozen before inspecting model errors)
| source_cell | axis | frozen_condition | audit_stage |
|---|---|---|---|
| AgCensus2015-16 T14-16 All/Marginal/area/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 All/Medium/area/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 SC/Large/number/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 SC/Small/number/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 All/Small/number/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 ST/Large/number/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 ST/Small/number/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 All/Small/area/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 SC/Medium/area/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 All/Large/number/MvF | gender | diff | 1 |
| AgCensus2015-16 T14-16 SC/Medium/number/MvF | gender | diff | 2 |
| AgCensus2015-16 T14-16 ST/Marginal/number/MvF | gender | diff | 2 |

_120 comparisons in the frozen sample; see sources/discrepancy_report.csv once the ledger is filled._

## Table 2 - main results, novel-test B, seed-explicit, with CPU controls
| tier | method | n_seeds | novel_B_mean | erasure_on_diff | wrong_group_on_diff | fabrication_on_equal | invalid |
|---|---|---|---|---|---|---|---|
| broad-instruct | ablation_no_condition_adaptive | 1 | 0.9344 | 0.0 | 0.0329 | 0.0962 | 0.0 |
| broad-instruct | ablation_no_rationale_loss | 1 | 0.9122 | 0.0 | 0.0235 | 0.1442 | 0.0 |
| broad-instruct | ablation_placement_random | 1 | 0.949 | 0.0047 | 0.0188 | 0.0769 | 0.0 |
| broad-instruct | ablation_placement_random_draw2 | 1 | 0.9421 | 0.0094 | 0.0235 | 0.0817 | 0.0 |
| broad-instruct | ablation_placement_random_draw3 | 1 | 0.9392 | 0.0 | 0.0282 | 0.0913 | 0.0 |
| broad-instruct | ablation_placement_uniform | 1 | 0.9161 | 0.0 | 0.0329 | 0.1298 | 0.0 |
| broad-instruct | ablation_rank_sweep_16 | 1 | 0.9378 | 0.0 | 0.0141 | 0.1058 | 0.0 |
| broad-instruct | ablation_rank_sweep_32 | 1 | 0.923 | 0.0047 | 0.0188 | 0.125 | 0.0 |
| broad-instruct | ablation_rank_sweep_8 | 1 | 0.9357 | 0.0 | 0.0188 | 0.1058 | 0.0 |
| broad-instruct | baseline_dart | 3 | 0.9377 | 0.0047 | 0.0282 | 0.0865 | 0.0 |
| broad-instruct | baseline_fairnet | 3 | 0.9157 | 0.0188 | 0.0141 | 0.1154 | 0.0 |
| broad-instruct | baseline_fairsteer | 1 | 0.0189 | 0.0 | 0.3662 | 0.9904 | 0.0 |
| broad-instruct | baseline_igu_lora | 1 | 0.9568 | 0.0141 | 0.0141 | 0.0577 | 0.0 |
| broad-instruct | baseline_lftf | 1 | 0.924 | 0.0516 | 0.0235 | 0.0769 | 0.0 |
| broad-instruct | baseline_pedal | 1 | 0.9245 | 0.0094 | 0.0282 | 0.1106 | 0.0 |
| broad-instruct | baseline_regift | 1 | 0.9541 | 0.0047 | 0.0188 | 0.0673 | 0.0 |
| broad-instruct | frozen_base | 1 | 0.3829 | 0.0704 | 0.1596 | 0.7452 | 0.0 |
| broad-instruct | graft_proposed | 3 | 0.9315 | 0.0047 | 0.0282 | 0.0817 | 0.0 |
| broad-instruct | graft_proposed_loao_gender | 1 | 0.5161 | 0.6087 | 0.0435 | 0.0 | 0.0 |
| broad-instruct | graft_proposed_loao_landholding | 1 | 0.5844 | 0.0 | 0.4598 | 0.3636 | 0.0 |
| broad-instruct | graft_proposed_loao_social_group | 1 | 0.0 | 0.9417 | 0.0583 | 0.0357 | 0.0 |
| broad-instruct | graft_proposed_qlora_nf4 | 1 | 0.9141 | 0.0 | 0.0704 | 0.101 | 0.0 |
| broad-instruct | reference_vanilla_lora | 3 | 0.9324 | 0.0047 | 0.0282 | 0.0865 | 0.0 |
| broad-instruct | reference_vanilla_lora_loao_gender | 1 | 0.0833 | 0.7391 | 0.2174 | 0.0 | 0.0 |
| broad-instruct | reference_vanilla_lora_loao_landholding | 1 | 0.6547 | 0.0805 | 0.1954 | 0.4026 | 0.0 |
| broad-instruct | reference_vanilla_lora_loao_social_group | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| broad-instruct | reference_vanilla_qlora | 1 | 0.9513 | 0.0 | 0.0188 | 0.0769 | 0.0 |
| cpu-baseline | always_equal | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| cpu-baseline | majority | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| cpu-baseline | metadata | 1 | 0.8392 | 0.1033 | 0.0235 | 0.1923 | 0.0 |
| cpu-baseline | tfidf | 1 | 0.9333 | 0.0235 | 0.0329 | 0.0769 | 0.0 |
| general-instruct | ablation_no_condition_adaptive | 1 | 0.9308 | 0.0282 | 0.0282 | 0.0817 | 0.0 |
| general-instruct | ablation_no_rationale_loss | 1 | 0.9366 | 0.0047 | 0.0235 | 0.0962 | 0.0 |
| general-instruct | ablation_placement_random | 1 | 0.9355 | 0.0094 | 0.0423 | 0.0769 | 0.0 |
| general-instruct | ablation_placement_random_draw2 | 1 | 0.9388 | 0.0 | 0.0235 | 0.0962 | 0.0 |
| general-instruct | ablation_placement_random_draw3 | 1 | 0.94 | 0.0 | 0.0094 | 0.1058 | 0.0 |
| general-instruct | ablation_placement_uniform | 1 | 0.9314 | 0.0 | 0.0282 | 0.1058 | 0.0 |
| general-instruct | ablation_rank_sweep_16 | 1 | 0.9468 | 0.0094 | 0.0188 | 0.0769 | 0.0 |
| general-instruct | ablation_rank_sweep_32 | 1 | 0.8963 | 0.0141 | 0.0563 | 0.1346 | 0.0 |
| general-instruct | ablation_rank_sweep_8 | 1 | 0.9465 | 0.0141 | 0.0094 | 0.0817 | 0.0 |
| general-instruct | baseline_dart | 3 | 0.9435 | 0.0329 | 0.0329 | 0.0673 | 0.0 |
| general-instruct | baseline_fairnet | 3 | 0.9268 | 0.0047 | 0.0469 | 0.0769 | 0.0 |
| general-instruct | baseline_fairsteer | 1 | 0.0 | 0.0 | 0.23 | 1.0 | 0.0024 |
| general-instruct | baseline_igu_lora | 1 | 0.9357 | 0.0235 | 0.0329 | 0.0721 | 0.0 |
| general-instruct | baseline_lftf | 1 | 0.9446 | 0.0047 | 0.0282 | 0.0769 | 0.0 |
| general-instruct | baseline_pedal | 1 | 0.9108 | 0.0 | 0.0329 | 0.1394 | 0.0 |
| general-instruct | baseline_regift | 1 | 0.945 | 0.0657 | 0.0141 | 0.0288 | 0.0 |
| general-instruct | frozen_base | 1 | 0.3046 | 0.0329 | 0.2347 | 0.8077 | 0.0 |
| general-instruct | graft_proposed | 3 | 0.9071 | 0.0 | 0.0376 | 0.1154 | 0.0 |
| general-instruct | graft_proposed_loao_gender | 1 | 0.4138 | 0.5652 | 0.1739 | 0.0 | 0.0 |
| general-instruct | graft_proposed_loao_landholding | 1 | 0.33 | 0.0575 | 0.7126 | 0.4156 | 0.0 |
| general-instruct | graft_proposed_loao_social_group | 1 | 0.1268 | 0.4951 | 0.4369 | 0.0536 | 0.0 |
| general-instruct | graft_proposed_qlora_nf4 | 1 | 0.9288 | 0.0 | 0.0282 | 0.1106 | 0.0 |
| general-instruct | reference_vanilla_lora | 3 | 0.9211 | 0.0047 | 0.0188 | 0.0865 | 0.0 |
| general-instruct | reference_vanilla_lora_loao_gender | 1 | 0.6471 | 0.5217 | 0.0 | 0.0 | 0.0 |
| general-instruct | reference_vanilla_lora_loao_landholding | 1 | 0.3826 | 0.5172 | 0.2414 | 0.0779 | 0.0 |
| general-instruct | reference_vanilla_lora_loao_social_group | 1 | 0.4241 | 0.7184 | 0.0097 | 0.0357 | 0.0 |
| general-instruct | reference_vanilla_qlora | 1 | 0.9513 | 0.0094 | 0.0094 | 0.0769 | 0.0 |
| general-instruct-2 | ablation_no_condition_adaptive | 1 | 0.9154 | 0.0188 | 0.0329 | 0.1154 | 0.0 |
| general-instruct-2 | ablation_no_rationale_loss | 1 | 0.9068 | 0.0423 | 0.0329 | 0.1106 | 0.0 |
| general-instruct-2 | ablation_placement_random | 1 | 0.9446 | 0.0 | 0.0329 | 0.0769 | 0.0 |
| general-instruct-2 | ablation_placement_random_draw2 | 1 | 0.9216 | 0.0329 | 0.0469 | 0.0769 | 0.0 |
| general-instruct-2 | ablation_placement_random_draw3 | 1 | 0.9406 | 0.0376 | 0.0188 | 0.0625 | 0.0 |
| general-instruct-2 | ablation_placement_uniform | 1 | 0.9513 | 0.0047 | 0.0141 | 0.0769 | 0.0 |
| general-instruct-2 | ablation_rank_sweep_16 | 1 | 0.9298 | 0.0 | 0.0141 | 0.1202 | 0.0 |
| general-instruct-2 | ablation_rank_sweep_32 | 1 | 0.9448 | 0.0047 | 0.0329 | 0.0721 | 0.0 |
| general-instruct-2 | ablation_rank_sweep_8 | 1 | 0.9417 | 0.0047 | 0.0235 | 0.0865 | 0.0 |
| general-instruct-2 | baseline_dart | 3 | 0.8946 | 0.0141 | 0.0704 | 0.226 | 0.0 |
| general-instruct-2 | baseline_fairnet | 3 | 0.8573 | 0.0235 | 0.0986 | 0.3413 | 0.0 |
| general-instruct-2 | baseline_fairsteer | 1 | 0.5282 | 0.4085 | 0.1174 | 0.4038 | 0.0 |
| general-instruct-2 | baseline_igu_lora | 1 | 0.9216 | 0.0423 | 0.0329 | 0.0817 | 0.0 |
| general-instruct-2 | baseline_lftf | 1 | 0.7034 | 0.2488 | 0.1033 | 0.226 | 0.0024 |
| general-instruct-2 | baseline_pedal | 1 | 0.9015 | 0.0235 | 0.0469 | 0.125 | 0.0 |
| general-instruct-2 | baseline_regift | 1 | 0.9188 | 0.0235 | 0.0423 | 0.0962 | 0.0 |
| general-instruct-2 | frozen_base | 1 | 0.4862 | 0.3427 | 0.1925 | 0.4904 | 0.0 |
| general-instruct-2 | graft_proposed | 3 | 0.913 | 0.0329 | 0.0329 | 0.1058 | 0.0 |
| general-instruct-2 | graft_proposed_loao_gender | 1 | 0.2963 | 0.8261 | 0.0 | 0.0 | 0.0 |
| general-instruct-2 | graft_proposed_loao_landholding | 1 | 0.1275 | 0.5517 | 0.3793 | 0.1558 | 0.0 |
| general-instruct-2 | graft_proposed_loao_social_group | 1 | 0.0923 | 0.1748 | 0.7767 | 0.0625 | 0.0 |
| general-instruct-2 | graft_proposed_qlora_nf4 | 1 | 0.9378 | 0.0235 | 0.0235 | 0.0769 | 0.0 |
| general-instruct-2 | reference_vanilla_lora | 3 | 0.9089 | 0.0094 | 0.0516 | 0.1202 | 0.0 |
| general-instruct-2 | reference_vanilla_lora_loao_gender | 1 | 0.5161 | 0.6522 | 0.0 | 0.0 | 0.0 |
| general-instruct-2 | reference_vanilla_lora_loao_landholding | 1 | 0.3234 | 0.1724 | 0.6207 | 0.2597 | 0.0 |
| general-instruct-2 | reference_vanilla_lora_loao_social_group | 1 | 0.074 | 0.3204 | 0.6408 | 0.2143 | 0.0 |
| general-instruct-2 | reference_vanilla_qlora | 1 | 0.9407 | 0.0282 | 0.0376 | 0.0529 | 0.0 |
| small-instruct | ablation_no_condition_adaptive | 1 | 0.9383 | 0.0 | 0.0188 | 0.101 | 0.0 |
| small-instruct | ablation_no_rationale_loss | 1 | 0.9501 | 0.0188 | 0.0329 | 0.0481 | 0.0 |
| small-instruct | ablation_placement_random | 1 | 0.9589 | 0.0047 | 0.0141 | 0.0625 | 0.0 |
| small-instruct | ablation_placement_random_draw2 | 1 | 0.9516 | 0.0047 | 0.0188 | 0.0721 | 0.0 |
| small-instruct | ablation_placement_random_draw3 | 1 | 0.956 | 0.0047 | 0.0094 | 0.0721 | 0.0 |
| small-instruct | ablation_placement_uniform | 1 | 0.9493 | 0.0047 | 0.0235 | 0.0721 | 0.0 |
| small-instruct | ablation_rank_sweep_16 | 1 | 0.9409 | 0.0047 | 0.0141 | 0.0962 | 0.0 |
| small-instruct | ablation_rank_sweep_32 | 1 | 0.9373 | 0.0047 | 0.0329 | 0.0865 | 0.0 |
| small-instruct | ablation_rank_sweep_8 | 1 | 0.9406 | 0.0423 | 0.0188 | 0.0577 | 0.0 |
| small-instruct | baseline_dart | 3 | 0.9447 | 0.0 | 0.0235 | 0.0962 | 0.0 |
| small-instruct | baseline_fairnet | 3 | 0.9446 | 0.0047 | 0.0376 | 0.0721 | 0.0 |
| small-instruct | baseline_fairsteer | 1 | 0.4223 | 0.3239 | 0.3052 | 0.5096 | 0.0 |
| small-instruct | baseline_igu_lora | 1 | 0.9353 | 0.0094 | 0.0376 | 0.0817 | 0.0 |
| small-instruct | baseline_lftf | 1 | 0.8785 | 0.1033 | 0.0423 | 0.0962 | 0.0 |
| small-instruct | baseline_pedal | 1 | 0.9392 | 0.0 | 0.0282 | 0.0913 | 0.0 |
| small-instruct | baseline_regift | 1 | 0.9468 | 0.0047 | 0.0235 | 0.0769 | 0.0 |
| small-instruct | frozen_base | 1 | 0.4212 | 0.4038 | 0.2676 | 0.4135 | 0.0 |
| small-instruct | graft_proposed | 3 | 0.9486 | 0.0141 | 0.0376 | 0.0769 | 0.0 |
| small-instruct | graft_proposed_loao_gender | 1 | 0.2963 | 0.8261 | 0.0 | 0.0 | 0.0 |
| small-instruct | graft_proposed_loao_landholding | 1 | 0.5094 | 0.0 | 0.5862 | 0.3377 | 0.0 |
| small-instruct | graft_proposed_loao_social_group | 1 | 0.0745 | 0.3592 | 0.6019 | 0.0893 | 0.0 |
| small-instruct | graft_proposed_qlora_nf4 | 1 | 0.9471 | 0.0047 | 0.0282 | 0.0721 | 0.0 |
| small-instruct | reference_vanilla_lora | 3 | 0.9151 | 0.0 | 0.0282 | 0.0673 | 0.0 |
| small-instruct | reference_vanilla_lora_loao_gender | 1 | 0.0 | 0.913 | 0.087 | 0.0 | 0.0 |
| small-instruct | reference_vanilla_lora_loao_landholding | 1 | 0.5809 | 0.0 | 0.4828 | 0.3377 | 0.0 |
| small-instruct | reference_vanilla_lora_loao_social_group | 1 | 0.1273 | 0.8058 | 0.1262 | 0.0 | 0.0 |
| small-instruct | reference_vanilla_qlora | 1 | 0.9443 | 0.0094 | 0.0188 | 0.0817 | 0.0 |

### CPU baselines
| baseline | slice | harmonic_b | overall_accuracy |
|---|---|---|---|
| majority | structure_novel | 0.0 | 0.4941 |
| always_equal | structure_novel | 0.0 | 0.4941 |
| metadata | structure_novel | 0.8392 | 0.8409 |
| tfidf | structure_novel | 0.9333 | 0.9335 |

## Table 3 - transfer error taxonomy (LOAO)
| tier | held_out_axis | method | harmonic_b | erasure_rate_on_diff | wrong_group_rate_on_diff | fabrication_rate_on_equal | invalid_rate |
|---|---|---|---|---|---|---|---|
| broad-instruct | gender | graft_proposed | 0.3333 | 0.725 | 0.075 | 0.0 | 0.0 |
| broad-instruct | gender | reference_vanilla_lora | 0.0488 | 0.8 | 0.175 | 0.0 | 0.0 |
| broad-instruct | landholding | graft_proposed | 0.551 | 0.0 | 0.478 | 0.4167 | 0.0 |
| broad-instruct | landholding | reference_vanilla_lora | 0.6173 | 0.0503 | 0.2138 | 0.4615 | 0.0063 |
| broad-instruct | social_group | graft_proposed | 0.0116 | 0.9477 | 0.0465 | 0.0208 | 0.0 |
| broad-instruct | social_group | reference_vanilla_lora | 0.0116 | 0.9942 | 0.0 | 0.0 | 0.0 |
| general-instruct | gender | graft_proposed | 0.2609 | 0.75 | 0.1 | 0.0 | 0.0 |
| general-instruct | gender | reference_vanilla_lora | 0.6667 | 0.5 | 0.0 | 0.0 | 0.0 |
| general-instruct | landholding | graft_proposed | 0.2898 | 0.0503 | 0.7547 | 0.4359 | 0.0 |
| general-instruct | landholding | reference_vanilla_lora | 0.432 | 0.3648 | 0.3459 | 0.1474 | 0.0 |
| general-instruct | social_group | graft_proposed | 0.2165 | 0.5 | 0.3779 | 0.0469 | 0.0 |
| general-instruct | social_group | reference_vanilla_lora | 0.4887 | 0.6628 | 0.0116 | 0.0208 | 0.0 |
| general-instruct-2 | gender | graft_proposed | 0.1818 | 0.9 | 0.0 | 0.0 | 0.0 |
| general-instruct-2 | gender | reference_vanilla_lora | 0.3333 | 0.8 | 0.0 | 0.0 | 0.0 |
| general-instruct-2 | landholding | graft_proposed | 0.1278 | 0.5786 | 0.3522 | 0.1667 | 0.0 |
| general-instruct-2 | landholding | reference_vanilla_lora | 0.345 | 0.195 | 0.5786 | 0.2756 | 0.0 |
| general-instruct-2 | social_group | graft_proposed | 0.1588 | 0.2326 | 0.6802 | 0.1146 | 0.0 |
| general-instruct-2 | social_group | reference_vanilla_lora | 0.157 | 0.3663 | 0.5465 | 0.2135 | 0.0 |
| small-instruct | gender | graft_proposed | 0.1818 | 0.9 | 0.0 | 0.0 | 0.0 |
| small-instruct | gender | reference_vanilla_lora | 0.0 | 0.95 | 0.05 | 0.0 | 0.0 |
| small-instruct | landholding | graft_proposed | 0.4992 | 0.0 | 0.5912 | 0.359 | 0.0 |
| small-instruct | landholding | reference_vanilla_lora | 0.5509 | 0.0 | 0.5094 | 0.3718 | 0.0 |
| small-instruct | social_group | graft_proposed | 0.1295 | 0.3895 | 0.5407 | 0.099 | 0.0 |
| small-instruct | social_group | reference_vanilla_lora | 0.1405 | 0.7965 | 0.1279 | 0.0052 | 0.0 |

## Table 4 - human advice outcomes
_populated from advice/human_study/ratings_analysis.json when the study is complete_

## Figures
- fig_paired_contrasts.png
- fig_advice_length_vs_distance.png
- fig_loao_error_taxonomy.png

## Claims to evidence
| claim | artifact | status |
|---|---|---|
| Frozen test set is the preregistered one | inventory/manifest.json:frozen_test_hash_matches_config | supported |
| Census-derived labels were independently verified | sources/discrepancy_report.csv | unsupported |
| Simple train-only baselines reach a stated fraction of neural novel-test B | cpu_baselines/cpu_baseline_summary.csv | supported |
| GRAFT vs placement controls, with dependence-aware uncertainty (localisation family) | paired/comparison_families.csv | supported |
| Transfer failures are characterised by error type, not only accuracy | paired/loao_error_taxonomy.csv | supported |
| Advice drift reduction coincides with shorter, more duplicated answers (association) | advice/advice_paired_summary.json | supported |
| Blinded expert assessment links automated consistency to correctness/usefulness | advice/human_study/ratings_analysis.json | unsupported (study not completed - narrow the claim) |
| Evidence-sensitivity panel results | evidence_panel/evidence_panel_scores.csv | unsupported (P4 not run) |
| GRAFT is superior to ordinary adapters / localisation is necessary | — | do_not_claim (preregistered hypothesis unsupported; report as such) |
