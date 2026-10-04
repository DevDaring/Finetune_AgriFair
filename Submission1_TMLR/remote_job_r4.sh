#!/usr/bin/env bash
# Budget sensitivity for rounds 2 and 3 (low-parse GPU systems at 256 tokens). Waits for round 3; writes RUN_R4_DONE.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN_R3_DONE ] || [ -f /workspace/RUN_R3_FAILED ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ ! -f /workspace/RUN_R3_DONE ]; then touch /workspace/RUN_R4_FAILED; exit 1; fi
cp /workspace/staged_r4/retry_lowparse.py Submission1_TMLR/retry_lowparse.py
echo "retry r23 start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.retry_lowparse > logs/retry_r23.log 2>&1 || { echo "FAILED at: retry r23" >> logs/status.log; touch /workspace/RUN_R4_FAILED; exit 1; }
echo "retry r23 done $(date -u)" >> logs/status.log
touch /workspace/RUN_R4_DONE
