#!/usr/bin/env bash
# Prepare a fresh Vast.ai GPU VM to run the AgriFair study.
#
# Everything here is idempotent: re-running it on a VM that is already prepared is safe and
# fast, which matters because an interruptible instance can be recreated mid-study.
#
# The .env file is NEVER fetched by this script. It carries every credential the study uses
# and it is gitignored, so it has to arrive out of band:
#     scp -P <port> Codes/.env root@<host>:/workspace/Finetune_AgriFair/.env
#
# Usage on the VM:
#     bash deploy/bootstrap.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/Finetune_AgriFair}"
REPO_URL="${REPO_URL:-https://github.com/DevDaring/Finetune_AgriFair.git}"
# The GitHub repository's root IS the code directory: CPU_Run/, GPU_Run/, deploy/ and the
# rest sit at the top level of DevDaring/Finetune_AgriFair, not under a Codes/ subdirectory.
# Overridable in case the layout is ever nested.
CODE_DIR="${CODE_DIR:-$REPO_DIR}"

say() { printf '\n=== %s ===\n' "$1"; }

say "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs curl rsync tmux jq python3-venv >/dev/null
git lfs install --skip-repo >/dev/null 2>&1 || true

say "Repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --all --quiet && git -C "$REPO_DIR" pull --ff-only --quiet || true
else
  git clone --quiet "$REPO_URL" "$REPO_DIR"
fi
cd "$CODE_DIR"

say "Credentials"
if [ ! -f "$CODE_DIR/.env" ]; then
  cat >&2 <<'MSG'
.env is missing. It is gitignored on purpose and must be copied in separately:
    scp -P <port> Codes/.env root@<host>:/workspace/Finetune_AgriFair/.env
Nothing else can run without it.
MSG
  exit 1
fi
chmod 600 "$CODE_DIR/.env"
echo ".env present, permissions tightened"

say "Python dependencies"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements_global.txt

say "FlashAttention from a pre-built wheel"
# Resolves the exact wheel for this torch, CUDA, Python and ABI combination and installs it.
# Never compiles from source; falls back to sdpa and records the choice.
python3 GPU_Run/common/flash_attn_setup.py

say "Environment report"
python3 - <<'PY'
import torch, transformers, peft, platform
print(f"  python              {platform.python_version()}")
print(f"  torch               {torch.__version__}")
print(f"  cuda available      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu                 {p.name}, {p.total_memory / 1024**3:.0f} GiB")
print(f"  transformers        {transformers.__version__}")
print(f"  peft                {peft.__version__}")
try:
    import flash_attn
    print(f"  flash-attn          {flash_attn.__version__}")
except Exception:
    print("  flash-attn          not installed, sdpa will be used")
PY

say "Credential self-check (values are never printed)"
python3 GPU_Run/common/key_selftest.py || true

say "Models and data"
# The only networked stage. Downloads the gated models, AgriFair, the replay set and the
# capability probe, then pins every revision.
python3 run_all.py --stage dataset_prep

say "Ready"
echo "Next: bash deploy/run_verification.sh    (pre-flight on real GPUs)"
echo "Then: bash deploy/run_study.sh <tier,tier>  (the study)"
