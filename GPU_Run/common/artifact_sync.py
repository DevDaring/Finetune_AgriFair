"""Crash-safe orphan-branch snapshot/restore of results/, checkpoints/, data/.

Uses a separate git index so it never touches the main working tree or staging area,
writes orphan snapshot commits, and force-pushes so the branch holds only the latest
snapshot. models/ is not synced (large, re-downloadable). Reads the GitHub token from
.env without printing it (Instruction.md Section 13). Best-effort: if the directory is
not a git repository, snapshot/restore are no-ops with a clear log line.
"""
from __future__ import annotations

import os
import subprocess
from typing import List, Optional

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import REPO_ROOT

logger = get_logger("artifact_sync")

SYNCED_DIRS = ["results", "checkpoints", "data"]


def _branch() -> str:
    return os.environ.get("GPU_RUN_ARTIFACTS_BRANCH") or env_loader.get(
        "GPU_RUN_ARTIFACTS_BRANCH", "gpu-run-artifacts"
    )


def _is_git_repo() -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


def _git(args: List[str], extra_env: Optional[dict] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["git"] + args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def snapshot(message: str = "autosync snapshot") -> bool:
    """Force-push results/checkpoints/data to the artifacts branch via a separate index."""
    if not _is_git_repo():
        logger.info("Not a git repository; skipping artifact snapshot.")
        return False
    index_file = str(REPO_ROOT / ".git" / "artifact_index")
    env = {"GIT_INDEX_FILE": index_file}
    try:
        _git(["read-tree", "--empty"], env)
        for d in SYNCED_DIRS:
            if (REPO_ROOT / d).exists():
                _git(["add", "-A", d], env)
        tree = _git(["write-tree"], env).stdout.strip()
        commit = _git(["commit-tree", tree, "-m", message], env).stdout.strip()
        if not commit:
            return False
        _git(["update-ref", f"refs/heads/{_branch()}", commit], env)
        token = env_loader.get("GITHUB_TOKEN")
        push = _git(["push", "-f", "origin", f"{_branch()}"], env)
        return push.returncode == 0
    except Exception as e:
        logger.warning("Snapshot failed: %s", type(e).__name__)
        return False


def restore() -> bool:
    if not _is_git_repo():
        logger.info("Not a git repository; skipping artifact restore.")
        return False
    try:
        _git(["fetch", "origin", _branch()])
        for d in SYNCED_DIRS:
            _git(["checkout", f"origin/{_branch()}", "--", d])
        return True
    except Exception as e:
        logger.warning("Restore failed: %s", type(e).__name__)
        return False
