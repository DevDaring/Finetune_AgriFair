#!/usr/bin/env bash
# Second job on the same GPU: the three follow-up conditions (norule, abstain, realtable), sdpa as the
# main run. Waits for the first job's marker, then writes RUN2_DONE or RUN2_FAILED.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN_DONE ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ -f /workspace/RUN_FAILED ]; then touch /workspace/RUN2_FAILED; exit 1; fi
cp /workspace/staged/run_gpu.py Submission1_TMLR/run_gpu.py
cp /workspace/staged/prompts_followups.jsonl results_submission1_tmlr/prompts_followups.jsonl
echo "followups start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --stage-name followups --sets followups > logs/followups.log 2>&1 \
  || { echo "FAILED at: followups" >> logs/status.log; touch /workspace/RUN2_FAILED; exit 1; }
echo "followups done $(date -u)" >> logs/status.log
touch /workspace/RUN2_DONE
