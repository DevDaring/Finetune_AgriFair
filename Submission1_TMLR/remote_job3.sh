#!/usr/bin/env bash
# Third job: step-by-step reasoning condition on the 8 core systems (unmodified + attribution-guided
# seed 42 per family), sdpa. Waits for job 2, then writes RUN3_DONE or RUN3_FAILED.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN2_DONE ] || [ -f /workspace/RUN2_FAILED ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ ! -f /workspace/RUN2_DONE ]; then touch /workspace/RUN3_FAILED; exit 1; fi
cp /workspace/staged3/run_gpu.py Submission1_TMLR/run_gpu.py
cp /workspace/staged3/prompts_cot.jsonl results_submission1_tmlr/prompts_cot.jsonl
echo "cot start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --stage-name cot --sets cot --core-only > logs/cot.log 2>&1 \
  || { echo "FAILED at: cot" >> logs/status.log; touch /workspace/RUN3_FAILED; exit 1; }
echo "cot done $(date -u)" >> logs/status.log
touch /workspace/RUN3_DONE
