"""Install FlashAttention from a pre-built wheel, never from source.

Compiling FlashAttention takes tens of minutes and several gigabytes of RAM, and on a rented
GPU that is billed time spent not doing science. This module resolves the exact pre-built
wheel for the running environment and installs that, or gives up cleanly and records that the
run used sdpa instead.

The wheels are published per (flash-attn version, CUDA major, torch major.minor, C++11 ABI,
Python version), and the filename encodes all five:

    flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

Every one of those has to match the interpreter and torch build actually present, so the
resolver reads them from the live process rather than assuming. Candidates are checked with a
HEAD request before anything is downloaded, so a miss costs a round trip rather than a build.

Attention falls back to sdpa when no wheel matches, which is correct everywhere and slower
only on the attention kernel. The choice is recorded in results/flash_attn_setup.json and in
the attention_implementation_used column of every results CSV, so a run is never ambiguous
about which kernel produced it.

Run directly:  python GPU_Run/common/flash_attn_setup.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Dict, List, Optional

from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import RESULTS_DIR

logger = get_logger("flash_attn_setup")

RELEASE_BASE = "https://github.com/Dao-AILab/flash-attention/releases/download"
# Newest first. A wheel exists only for some combinations, so several are tried.
CANDIDATE_VERSIONS = ["2.8.3", "2.8.2", "2.8.1", "2.8.0", "2.7.4.post1", "2.7.3", "2.6.3"]


def _environment() -> Optional[Dict[str, str]]:
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return None
    torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    return {
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_version": torch_version,
        "cuda_major": cuda.split(".")[0],
        "cuda_full": cuda,
        "cxx11abi": "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE",
        "machine": platform.machine(),
    }


def candidate_wheel_urls(env: Dict[str, str]) -> List[str]:
    """Every wheel URL worth trying, most preferred first."""
    urls = []
    for version in CANDIDATE_VERSIONS:
        for cuda_tag in (f"cu{env['cuda_major']}", f"cu{env['cuda_full'].replace('.', '')}"):
            name = (f"flash_attn-{version}+{cuda_tag}torch{env['torch_version']}"
                    f"cxx11abi{env['cxx11abi']}-{env['python_tag']}-{env['python_tag']}-linux_x86_64.whl")
            url = f"{RELEASE_BASE}/v{version}/{name}"
            if url not in urls:
                urls.append(url)
    return urls


def _wheel_exists(url: str, timeout: int = 20) -> bool:
    try:
        import requests

        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def detect_and_setup() -> dict:
    result = {"installed": False, "reason": "", "attention_fallback": "sdpa",
              "wheel_url": "", "built_from_source": False}

    env = _environment()
    if env is None:
        result["reason"] = "torch is missing or is a CPU build; FlashAttention does not apply."
        return result
    if platform.system() != "Linux" or env["machine"] not in ("x86_64", "AMD64"):
        result["reason"] = f"no pre-built wheels for {platform.system()}/{env['machine']}; using sdpa."
        return result
    result["environment"] = env

    try:
        import flash_attn  # noqa: F401

        result.update(installed=True, attention_fallback="flash_attention_2",
                      reason=f"flash-attn {getattr(flash_attn, '__version__', 'unknown')} already importable.")
        return result
    except Exception:
        pass

    for url in candidate_wheel_urls(env):
        if not _wheel_exists(url):
            continue
        logger.info("Installing the pre-built wheel %s", url.rsplit("/", 1)[-1])
        try:
            # --no-build-isolation and --no-deps keep pip from reaching for a source build or
            # disturbing the pinned torch already installed.
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                            "--no-build-isolation", "--no-deps", url], check=True)
            import flash_attn  # noqa: F401

            result.update(installed=True, attention_fallback="flash_attention_2",
                          wheel_url=url, reason="installed from a pre-built wheel.")
            return result
        except Exception as e:
            logger.warning("Wheel %s did not install (%s); trying the next candidate.",
                           url.rsplit("/", 1)[-1], type(e).__name__)

    result["reason"] = ("no pre-built wheel matched "
                        f"torch {env['torch_version']} / cu{env['cuda_major']} / "
                        f"{env['python_tag']} / cxx11abi{env['cxx11abi']}; using sdpa. "
                        "Source compilation is deliberately not attempted.")
    return result


def main() -> None:
    result = detect_and_setup()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "flash_attn_setup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("FlashAttention: installed=%s, attention=%s. %s",
                result["installed"], result["attention_fallback"], result["reason"])


if __name__ == "__main__":
    main()
