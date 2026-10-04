#!/usr/bin/env bash
# Fourth job: altered tables on the 36 remaining numerical scenarios, all 32 systems, sdpa.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN3_DONE ] || [ -f /workspace/RUN3_FAILED ] || [ -f /workspace/RUN2_FAILED ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ ! -f /workspace/RUN3_DONE ]; then touch /workspace/RUN4_FAILED; exit 1; fi
cp /workspace/staged4/run_gpu.py Submission1_TMLR/run_gpu.py
cp /workspace/staged4/prompts_altered48.jsonl results_submission1_tmlr/prompts_altered48.jsonl
echo "altered48 start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --stage-name altered48 --sets altered48 > logs/altered48.log 2>&1 \
  || { echo "FAILED at: altered48" >> logs/status.log; touch /workspace/RUN4_FAILED; exit 1; }
echo "altered48 done $(date -u)" >> logs/status.log
touch /workspace/RUN4_DONE
