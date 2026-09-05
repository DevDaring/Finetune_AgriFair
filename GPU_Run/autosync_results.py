"""Background daemon: snapshot results/, checkpoints/, data/ every N minutes.

Force-pushes to the gpu-run-artifacts branch via a separate git index and orphan commits,
so main is never touched; models/ is excluded (re-downloadable). Best-effort: if not a git
repository, it logs and idles.

Run (background):  nohup python GPU_Run/autosync_results.py >> results/autosync.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from GPU_Run.common import artifact_sync
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("autosync_results")


def main():
    interval = int(os.environ.get("GPU_RUN_SYNC_INTERVAL_SECONDS", "600"))
    logger.info("Autosync daemon every %d s (branch=%s).", interval, artifact_sync._branch())
    try:
        while True:
            ok = artifact_sync.snapshot("autosync periodic snapshot")
            logger.info("Snapshot %s", "ok" if ok else "skipped/failed")
            time.sleep(interval)
    except KeyboardInterrupt:
        artifact_sync.snapshot("autosync exit snapshot")
        logger.info("Autosync daemon stopped.")


if __name__ == "__main__":
    main()
