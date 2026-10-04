#!/usr/bin/env bash
# Sixth job (added 4 Oct 2026): answer-budget sensitivity. The two GPU systems with unparsed answers in the
# main run are re-run on all main prompts with 256 new tokens, the API models' budget. Waits for job 5.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN5_DONE ] || [ -f /workspace/RUN5_FAILED ] || [ -f /workspace/RUN4_FAILED ] || [ -f /workspace/RUN3_FAILED ] || [ -f /workspace/RUN2_FAILED ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ ! -f /workspace/RUN5_DONE ]; then touch /workspace/RUN6_FAILED; exit 1; fi
cp /workspace/staged6/run_gpu.py Submission1_TMLR/run_gpu.py
echo "retry256 start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --stage-name retry256 --sets main --max-new-tokens 256 \
  --families general-instruct-2 small-instruct \
  --only-systems "general-instruct-2|frozen_base|seed42" "small-instruct|reference_vanilla_lora|seed43" > logs/retry256.log 2>&1 \
  || { echo "FAILED at: retry256" >> logs/status.log; touch /workspace/RUN6_FAILED; exit 1; }
echo "retry256 done $(date -u)" >> logs/status.log
touch /workspace/RUN6_DONE
