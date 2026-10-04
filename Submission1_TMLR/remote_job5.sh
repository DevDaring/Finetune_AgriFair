#!/usr/bin/env bash
# Fifth job (added 4 Oct 2026): the two further random-placement draws per family, on the main sets and
# the 36 new altered-table scenarios, sdpa. Waits for job 4, then writes RUN5_DONE or RUN5_FAILED.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
until [ -f /workspace/RUN4_DONE ] || [ -f /workspace/RUN4_FAILED ] || [ -f /workspace/RUN3_FAILED ] || [ -f /workspace/RUN2_FAILED ] || [ -f /workspace/RUN_FAILED ]; do sleep 20; done
if [ ! -f /workspace/RUN4_DONE ]; then touch /workspace/RUN5_FAILED; exit 1; fi
cp /workspace/staged5/run_gpu.py Submission1_TMLR/run_gpu.py
echo "draws start $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --stage-name draws --sets main+altered48 --random-draws > logs/draws.log 2>&1 \
  || { echo "FAILED at: draws" >> logs/status.log; touch /workspace/RUN5_FAILED; exit 1; }
echo "draws done $(date -u)" >> logs/status.log
touch /workspace/RUN5_DONE
