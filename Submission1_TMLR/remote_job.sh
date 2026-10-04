#!/usr/bin/env bash
# Runs on the rented GPU (plan: Submission1/TMLR_Research_Plan.md, section 6).
# Order: install pinned libraries -> main run (sdpa) -> FlashAttention-2 sensitivity run.
# Writes RUN_DONE or RUN_FAILED in /workspace so the local watchdog knows when to collect.
set -uo pipefail
cd /workspace/Codes
export PYTHONPATH="$PWD" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /workspace/.hf_token)"
mkdir -p logs
fail() { echo "FAILED at: $1" | tee -a logs/status.log; touch /workspace/RUN_FAILED; exit 1; }

{
  echo "start $(date -u)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  python -c "import torch,sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"
} > logs/status.log 2>&1

# transformers is held on the 4.5x line the study was validated on (see requirements_global.txt);
# einops is required for FlashAttention-2 to be accepted rather than silently replaced by sdpa.
pip install -q "transformers==4.56.2" "peft==0.17.1" "accelerate>=0.33" einops "jinja2>=3.1" \
    "huggingface_hub[hf_transfer]>=0.24" safetensors sentencepiece protobuf pyyaml numpy scipy \
    > logs/pip.log 2>&1 || fail "pip install"

# Main run with sdpa: reproduces the published outputs exactly (checked 4 Oct on the check run).
python -m Submission1_TMLR.run_gpu --attention sdpa --stage-name main > logs/main.log 2>&1 || fail "main"
echo "main (sdpa) done $(date -u)" >> logs/status.log
# Sensitivity run with FlashAttention-2 for every family that supports it (Gemma uses eager in both).
python -m Submission1_TMLR.run_gpu --attention auto --stage-name flash --families small-instruct broad-instruct general-instruct-2 > logs/flash.log 2>&1 || fail "flash"
echo "flash done $(date -u)" >> logs/status.log
pip freeze > logs/pip_freeze.txt 2>/dev/null
touch /workspace/RUN_DONE
