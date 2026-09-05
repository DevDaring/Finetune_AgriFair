#!/usr/bin/env bash
# Pre-flight on real GPUs, before any expensive run.
#
# The smoke pipeline proves the plumbing on a 1M-parameter random model. It cannot prove the
# things that only appear on real weights: whether Gemma 3's multimodal checkpoint loads and
# its text tower is reachable, whether a 12B model fits at the configured micro-batch,
# whether flash-attention is actually being used, whether the gated downloads worked, and
# whether a real model's outputs parse.
#
# This runs the whole study end to end at a tiny subset size on the tiers given, so every
# code path executes against real weights in minutes rather than hours.
#
# Usage:  bash deploy/run_verification.sh small-instruct,broad-instruct
set -euo pipefail

TIERS="${1:?usage: run_verification.sh <tier[,tier]>}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"
mkdir -p results

# Expandable segments let the allocator grow and shrink a single arena instead of pinning
# fixed blocks, which is what stranded 15.7 GiB as "reserved but unallocated" between arms on
# the 12B model and made the next arm fail to allocate 2 MiB.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SUBJECT_MODELS="$TIERS"
# TWO DATA SAMPLES per stage. The point is coverage, not statistics: every GPU-bound code
# path executes against real weights, every arm type is built, and nothing is skipped. The
# subset helper is condition-balanced, so two items still cover both a "diff" item and an
# "equal" one and therefore both the gap-erasure and gap-fabrication paths.
#
# Two items cannot exercise memory pressure, which is the one failure a tiny run hides, so
# check_gpu_memory.py separately runs a real forward and backward pass at the production
# micro-batch and sequence length before the study is allowed to start.
export EVAL_SUBSET_SIZE=2
export BASE_COMPETENCE_SUBSET_SIZE=2
export ADVICE_SUBSET_SIZE=2
export CAPABILITY_SUBSET_SIZE=2
export SWAP_SUBSET_SIZE=2
export ROTATION_SUBSET_SIZE=2
export PROBE_MAX_ITEMS=2
export PROBE_SWAP_SUBSET_SIZE=2
export PATCHSCOPE_MAX_ITEMS=2
export ATTRIBUTION_MAX_ITEMS=2
export ATTRIBUTION_RIEMANN_STEPS=2
export ATTRIBUTION_BOOTSTRAP_RESAMPLES=10
export DART_AUDIT_SUBSET_SIZE=2
export LFTF_MAX_ITEMS=2
export FAIRSTEER_MAX_ITEMS=2
export TRAIN_SMOKE_MAX_STEPS=2
export GEOMETRY_MAX_MATRICES=4
export STATISTICS_BOOTSTRAP_RESAMPLES=50
export VERIFY_ATTRIBUTION_MAX_ITEMS=2
export VERIFY_ATTRIBUTION_RIEMANN_STEPS=2
export GRAFT_FULL_SWEEP=1        # build every arm type, so no arm is untested
export RUN_LEAVE_ONE_AXIS_OUT=1
export RUN_FRONTIER_PANEL=0      # hosted models are verified separately and cost money
export STRICT_BUDGET=0           # tiny runs cannot hit the real parameter budget

LOG="results/verification_${TIERS//,/_}.log"
echo "verification: tiers=$TIERS log=$LOG (two data samples per stage)"

echo
echo "=== memory headroom at production batch size ==="
# Fails fast, before hours of work, if the real micro-batch will not fit.
python3 deploy/check_gpu_memory.py "$TIERS" 2>&1 | tee "$LOG"
MEMSTATUS=${PIPESTATUS[0]}
if [ "$MEMSTATUS" -ne 0 ]; then
  echo "Memory probe failed: the study would run out of memory. Not starting verification." >&2
  exit 1
fi
set +e
python3 run_all.py --stage dry_run  2>&1 | tee -a "$LOG"
python3 run_all.py --stage gpu_run  2>&1 | tee -a "$LOG"
python3 run_all.py --stage cpu_run  2>&1 | tee -a "$LOG"
STATUS=$?
set -e

echo
echo "=== verification summary ==="
python3 deploy/check_verification.py "$TIERS" "$LOG"
