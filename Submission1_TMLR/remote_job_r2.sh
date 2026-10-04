#!/usr/bin/env bash
# Round 2 on the rented GPU (Submission1/Future_PLan.md E2, E3, E4, E6).
# Order: install pinned libraries -> logprobs pilot (2 prompts per set) -> full logprobs (40 systems)
#        -> round-2 generation (rule variants, few-shot, reordered options; 40 systems).
# Writes RUN_R2_DONE or RUN_FAILED in /workspace for the local watchdog.
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
pip install -q "transformers==4.56.2" "peft==0.17.1" "accelerate>=0.33" einops "jinja2>=3.1" \
    "huggingface_hub[hf_transfer]>=0.24" safetensors sentencepiece protobuf pyyaml numpy scipy \
    > logs/pip.log 2>&1 || fail "pip install"
echo "pip done $(date -u)" >> logs/status.log
# check run: 2 prompts per set, one family, the two cheapest arms -> format and token ids are inspected locally
python -m Submission1_TMLR.logprobs --attention sdpa --pilot 2 --stage-name logprobs_check --families small-instruct > logs/logprobs_check.log 2>&1 || fail "logprobs check"
echo "logprobs check done $(date -u)" >> logs/status.log
python -m Submission1_TMLR.logprobs --attention sdpa --all-arms --stage-name logprobs > logs/logprobs.log 2>&1 || fail "logprobs"
echo "logprobs done $(date -u)" >> logs/status.log
python -m Submission1_TMLR.run_gpu --attention sdpa --all-arms --stage-name round2 --sets round2 > logs/round2.log 2>&1 || fail "round2"
echo "round2 done $(date -u)" >> logs/status.log
pip freeze > logs/pip_freeze.txt 2>/dev/null
touch /workspace/RUN_R2_DONE
