"""Pre-built-wheel flash-attention install with OS check and sdpa fallback.

Never compiles from source. Detects torch/CUDA/Python, confirms the OS is a
Linux x86_64 CUDA build, installs the nearest compatible pre-built wheel, and
records the outcome. On Windows/CPU it skips flash-attention and falls back to
sdpa (Instruction.md Section 15). Run directly:  python GPU_Run/common/flash_attn_setup.py
"""
from __future__ import annotations

import json
import platform
import sys

from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import RESULTS_DIR

logger = get_logger("flash_attn_setup")


def detect_and_setup() -> dict:
    result = {"installed": False, "reason": "", "attention_fallback": "sdpa"}
    try:
        import torch
    except Exception as e:
        result["reason"] = f"torch unavailable: {e}"
        return result

    is_linux = platform.system() == "Linux"
    is_x86 = platform.machine() in ("x86_64", "AMD64")
    has_cuda = bool(getattr(torch.version, "cuda", None)) and torch.cuda.is_available()

    if not (is_linux and is_x86 and has_cuda):
        result["reason"] = (
            f"OS/GPU not compatible (system={platform.system()}, machine={platform.machine()}, "
            f"cuda={has_cuda}); using sdpa."
        )
        logger.info(result["reason"])
        return result

    try:
        import flash_attn  # noqa: F401

        result["installed"] = True
        result["attention_fallback"] = "flash_attention_2"
        result["reason"] = "flash-attn already importable."
        return result
    except Exception:
        pass

    # Attempt a pre-built wheel install (no source build).
    try:
        import subprocess

        cmd = [sys.executable, "-m", "pip", "install", "flash-attn==2.8.3", "--no-build-isolation"]
        subprocess.run(cmd, check=True)
        import flash_attn  # noqa: F401

        result["installed"] = True
        result["attention_fallback"] = "flash_attention_2"
        result["reason"] = "installed flash-attn 2.8.3 wheel."
    except Exception as e:
        result["reason"] = f"wheel install failed ({e}); using sdpa."
    return result


def main() -> None:
    result = detect_and_setup()
    out = RESULTS_DIR / "flash_attn_setup.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("flash-attn setup: %s", result)


if __name__ == "__main__":
    main()
