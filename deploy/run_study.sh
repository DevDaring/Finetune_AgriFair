#!/usr/bin/env bash
# Run the study for a set of model tiers, with autopush and resumability.
#
# Split across two instances so wall-clock halves at the same total cost:
#     instance A:  bash deploy/run_study.sh small-instruct,broad-instruct
#     instance B:  bash deploy/run_study.sh general-instruct,general-instruct-2
#
# Everything is resumable. A finished training arm is skipped, stored predictions are reused,
# and results are snapshotted to the artifacts branch every 30 minutes, so a preempted
# instance loses minutes rather than the run.
set -euo pipefail

TIERS="${1:?usage: run_study.sh <tier[,tier]>}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"
mkdir -p results

# Clear every workload cap the verification pass sets. If even one leaked into this
# environment, the "full" study would quietly run on two items per stage and report success,
# which is the single most expensive way this run could go wrong: the logs look healthy, the
# tables fill in, and the numbers are meaningless.
unset EVAL_SUBSET_SIZE BASE_COMPETENCE_SUBSET_SIZE ADVICE_SUBSET_SIZE CAPABILITY_SUBSET_SIZE \
      SWAP_SUBSET_SIZE ROTATION_SUBSET_SIZE PROBE_MAX_ITEMS PROBE_SWAP_SUBSET_SIZE \
      PATCHSCOPE_MAX_ITEMS ATTRIBUTION_MAX_ITEMS ATTRIBUTION_RIEMANN_STEPS \
      ATTRIBUTION_BOOTSTRAP_RESAMPLES DART_AUDIT_SUBSET_SIZE LFTF_MAX_ITEMS FAIRSTEER_MAX_ITEMS \
      TRAIN_SMOKE_MAX_STEPS GEOMETRY_MAX_MATRICES STATISTICS_BOOTSTRAP_RESAMPLES \
      VERIFY_ATTRIBUTION_MAX_ITEMS VERIFY_ATTRIBUTION_RIEMANN_STEPS TRAIN_MICRO_BATCH_SIZE \
      EVAL_BATCH_SIZE FRONTIER_EVAL_SUBSET_SIZE FRONTIER_ROTATION_SUBSET_SIZE \
      FRONTIER_SWAP_SUBSET_SIZE GEOMETRY_LAYER_STRIDE GEOMETRY_BOOTSTRAP_RESAMPLES \
      ADVICE_TARGET_SCOPE BEHAVIOURAL_SWEEP_SCOPE PATCHSCOPE_METHODS DISABLE_JUDGE \
      RATIONALE_JUDGE_FRACTION ADVICE_JUDGE_FRACTION ADVICE_MAX_NEW_TOKENS

# Expandable segments let the allocator grow and shrink a single arena instead of pinning
# fixed blocks, which is what stranded 15.7 GiB as "reserved but unallocated" between arms on
# the 12B model and made the next arm fail to allocate 2 MiB.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SUBJECT_MODELS="$TIERS"
export GRAFT_FULL_SWEEP=1          # every arm, not the reduced set
export RUN_LEAVE_ONE_AXIS_OUT=1    # the transfer study is part of the claim
export STRICT_BUDGET=1             # a budget violation must stop the run, not warn
export RUN_FRONTIER_PANEL="${RUN_FRONTIER_PANEL:-0}"   # hosted panel runs once, from one instance
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

TAG="${TIERS//,/_}"
LOG="results/study_${TAG}.log"

echo "study: tiers=$TIERS"
echo "study: log=$LOG"
python3 CPU_Run/estimate_gpu_hours.py 2>/dev/null | sed -n '1,12p' || true

if ! pgrep -f "deploy/autopush.sh" >/dev/null 2>&1; then
  echo "study: starting the 30-minute autopush"
  nohup bash deploy/autopush.sh > "results/autopush_${TAG}.log" 2>&1 &
  sleep 5
  tail -3 "results/autopush_${TAG}.log" || true
fi

echo "study: restoring any earlier artifacts from the branch"
python3 GPU_Run/restore_artifacts.py || true

# Refuse to start if this is not actually a full run.
python3 - <<'PY' || exit 1
import os, sys
SUFFIXES = ("_SUBSET_SIZE", "_MAX_ITEMS", "_MAX_STEPS", "_MAX_MATRICES", "_RIEMANN_STEPS")
# "0" means no cap for these, which is the full-fidelity setting, not a reduced run.
UNCAPPED = {"GEOMETRY_MAX_MATRICES", "GEOMETRY_LAYER_STRIDE"}
caps = [k for k, v in os.environ.items()
        if k.endswith(SUFFIXES) and not (k in UNCAPPED and v.strip() in ("0", ""))]
if caps:
    print("REFUSING TO START: workload caps are still set: " + ", ".join(sorted(caps)))
    print("This would run the study on a tiny subset and report it as a full result.")
    sys.exit(1)
print("scale check: no workload caps set, this is a full run")
PY

echo "study: starting"
date -u '+study: started %Y-%m-%dT%H:%M:%SZ' | tee -a "$LOG"
set +e
python3 run_all.py --stage gpu_run 2>&1 | tee -a "$LOG"
GPU_STATUS=${PIPESTATUS[0]}
python3 run_all.py --stage cpu_run 2>&1 | tee -a "$LOG"
CPU_STATUS=${PIPESTATUS[0]}
set -e
date -u '+study: finished %Y-%m-%dT%H:%M:%SZ' | tee -a "$LOG"

echo "study: final snapshot"
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from GPU_Run.common.artifact_sync import snapshot
print("final snapshot", "ok" if snapshot("final snapshot") else "failed")
PY

echo "study: gpu_run exit=$GPU_STATUS cpu_run exit=$CPU_STATUS"
exit $(( GPU_STATUS != 0 || CPU_STATUS != 0 ))
