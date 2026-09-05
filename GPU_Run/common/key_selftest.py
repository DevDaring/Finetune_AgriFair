"""Key validation, hardcoded-key check, and an optional live per-key API test table.

Never prints a key value. Run directly:  python GPU_Run/common/key_selftest.py [--live]
"""
from __future__ import annotations

import sys
from typing import Dict, List

# Runnable as a script (deploy/bootstrap.sh calls it directly), so the repository root has
# to be on sys.path before the package imports below; running "python3 GPU_Run/common/x.py"
# puts only GPU_Run/common there.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from GPU_Run.common import env_loader
from GPU_Run.common.hygiene import scan_repo_for_hardcoded_keys
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("key_selftest")

_CANONICALS = [
    "HF_KEY",
    "GCP_Key1", "GCP_Key2", "GCP_Key3", "GCP_key4",
    "DEEPSEEK_KEY1", "DEEPSEEK_KEY2",
    "OPENROUTER_KEY1", "OPENROUTER_KEY2",
    "MISTRAL_KEY1", "MISTRAL_KEY2",
]


def presence_table() -> Dict[str, bool]:
    return {c: env_loader.has(c) for c in _CANONICALS}


def run(live: bool = False) -> bool:
    ok = True
    logger.info("Key presence (value never shown):")
    for c, present in presence_table().items():
        logger.info("  %-16s %s", c, "present" if present else "MISSING")

    hits = scan_repo_for_hardcoded_keys()
    if hits:
        ok = False
        logger.error("Hardcoded-key patterns found in tracked files (values not shown):")
        for path, pat in hits:
            logger.error("  %s matched /%s/", path, pat)
    else:
        logger.info("No hardcoded keys found in tracked source.")

    if live:
        from GPU_Run.common.clients import JudgeChain

        chain = JudgeChain()
        if chain.available():
            reply = chain.complete("Reply with one JSON object only: {\"ok\": 1}")
            logger.info("Live judge reachable via %s: %s", chain.last_model_string, bool(reply))
        else:
            logger.info("No judge keys present; skipping live test.")
    return ok


if __name__ == "__main__":
    success = run(live="--live" in sys.argv)
    sys.exit(0 if success else 1)
