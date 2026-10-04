#!/usr/bin/env bash
# Round 3 on the rented GPU (Future_PLan.md "Round 3": P1 MNLI, P2 WDI, P3 drifted v1, P5 adversarial, P6 recall).
# Waits for round 2, then: a check run (2 prompts per set, one family), then all 40 systems. Writes RUN_R3_DONE.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN_R2_DONE ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ ! -f /workspace/RUN_R2_DONE ]; then touch /workspace/RUN_R3_FAILED; exit 1; fi
cp /workspace/staged_r3/run_gpu.py Submission1_TMLR/run_gpu.py
cp /workspace/staged_r3/prompts_*.jsonl results_submission1_tmlr/
echo "round3 check start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --pilot 2 --families small-instruct --stage-name round3_check --sets round3 > logs/round3_check.log 2>&1 \
  || { echo "FAILED at: round3 check" >> logs/status.log; touch /workspace/RUN_R3_FAILED; exit 1; }
echo "round3 start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --all-arms --stage-name round3 --sets round3 > logs/round3.log 2>&1 \
  || { echo "FAILED at: round3" >> logs/status.log; touch /workspace/RUN_R3_FAILED; exit 1; }
echo "round3 done $(date -u)" >> logs/status.log
touch /workspace/RUN_R3_DONE
