"""Restore results/, checkpoints/, data/ from the gpu-run-artifacts branch.

Run on a fresh VM after pre-emption, or on the CPU machine before CPU_Run
(Instruction.md Sections 12, 13).

Run:  python GPU_Run/restore_artifacts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GPU_Run.common import artifact_sync
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("restore_artifacts")


def main():
    ok = artifact_sync.restore()
    logger.info("Restore %s", "ok" if ok else "skipped/failed (not a git repo or no branch yet)")


if __name__ == "__main__":
    main()
