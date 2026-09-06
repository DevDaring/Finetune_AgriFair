"""Crash-safe snapshots of results/, checkpoints/ and data/ to an orphan git branch.

Uses a separate git index so it never touches the working tree or the staging area, writes
orphan commits, and force-pushes so the branch holds only the latest snapshot. models/ is
excluded because it is large and re-downloadable.

Two details that decide whether this works at all:

  * Those three directories are in .gitignore, because they are generated. `git add` obeys
    .gitignore, so the snapshot has to force-add them. Without the force flag every push
    succeeds and contains nothing, which is the worst possible failure: silent, and only
    discovered when the results are needed.
  * The push has to authenticate without a credential helper on a fresh VM, so the token is
    injected into the remote URL for the duration of the call and never written to disk or
    logged.

Reads the GitHub token from .env through env_loader and never prints it.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import List, Optional, Tuple

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import REPO_ROOT

logger = get_logger("artifact_sync")

# What the 30-minute snapshot pushes to GitHub. Deliberately NOT checkpoints/: the study
# trains 132 adapters totalling about 4.6 GB, or roughly 18 GB once per-epoch checkpoints are
# counted, and force-pushing that to a branch every 30 minutes exceeds what GitHub accepts and
# would take longer than the interval it runs on. Adapters are model weights and belong in the
# HuggingFace repository, which deploy/push_models_to_huggingface.py writes; results/ and
# data/ are the scientific record and are small enough to push often.
# ARTIFACT_SYNC_DIRS overrides this, space- or comma-separated.
SYNCED_DIRS = [d for d in re.split(r"[,\s]+", os.environ.get("ARTIFACT_SYNC_DIRS", "results data"))
               if d]
# A single snapshot larger than this is refused rather than attempted, so a misconfiguration
# cannot wedge the run behind a multi-gigabyte push that can never succeed.
MAX_SNAPSHOT_MEGABYTES = float(os.environ.get("MAX_SNAPSHOT_MEGABYTES", "1500"))


def _directory_megabytes(path) -> float:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total / 1024 ** 2


def _branch() -> str:
    return os.environ.get("GPU_RUN_ARTIFACTS_BRANCH") or env_loader.get(
        "GPU_RUN_ARTIFACTS_BRANCH", "gpu-run-artifacts")


def _is_git_repo() -> bool:
    r = _git(["rev-parse", "--is-inside-work-tree"])
    return r.returncode == 0 and r.stdout.strip() == "true"


SNAPSHOT_IDENTITY_NAME = os.environ.get("GIT_SNAPSHOT_NAME", "AgriFair run")
SNAPSHOT_IDENTITY_EMAIL = os.environ.get("GIT_SNAPSHOT_EMAIL", "agrifair-run@localhost")


def _git(args: List[str], extra_env: Optional[dict] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    # Snapshots are written with commit-tree, which refuses to run without a committer
    # identity ("Author identity unknown"). A freshly provisioned GPU box has no git config,
    # so every snapshot failed there while the preflight reported ready. Supplying the
    # identity here makes the snapshot independent of machine configuration.
    env.setdefault("GIT_AUTHOR_NAME", SNAPSHOT_IDENTITY_NAME)
    env.setdefault("GIT_AUTHOR_EMAIL", SNAPSHOT_IDENTITY_EMAIL)
    env.setdefault("GIT_COMMITTER_NAME", SNAPSHOT_IDENTITY_NAME)
    env.setdefault("GIT_COMMITTER_EMAIL", SNAPSHOT_IDENTITY_EMAIL)
    return subprocess.run(["git"] + args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def _redact(text: str) -> str:
    """Strip anything token-shaped out of git output before it reaches a log."""
    return re.sub(r"(gh[pousr]_[A-Za-z0-9]{10,}|://[^@\s/]+@)", "<redacted>@", text or "")


def _authenticated_remote() -> Optional[str]:
    """The origin URL with the token embedded, for one push. Never persisted."""
    token = env_loader.get("GITHUB_TOKEN")
    if not token:
        return None
    r = _git(["remote", "get-url", "origin"])
    url = r.stdout.strip()
    if not url.startswith("https://"):
        return None
    return "https://" + token + "@" + url[len("https://"):]


def snapshot(message: str = "autosync snapshot") -> bool:
    """Force-push results, checkpoints and data to the artifacts branch."""
    if not _is_git_repo():
        logger.info("Not a git repository; skipping artifact snapshot.")
        return False
    index_file = str(REPO_ROOT / ".git" / "artifact_index")
    env = {"GIT_INDEX_FILE": index_file}
    try:
        _git(["read-tree", "--empty"], env)
        present = []
        for d in SYNCED_DIRS:
            path = REPO_ROOT / d
            if not path.exists():
                continue
            mb = _directory_megabytes(path)
            if mb > MAX_SNAPSHOT_MEGABYTES:
                logger.error(
                    "Refusing to snapshot %s: %.0f MB exceeds the %.0f MB limit. GitHub would "
                    "reject or stall on this. Model weights belong in the HuggingFace "
                    "repository, not the artifacts branch.", d, mb, MAX_SNAPSHOT_MEGABYTES)
                continue
            present.append(d)
        if not present:
            logger.warning("No directory small enough to snapshot; nothing pushed.")
            return False
        for d in present:
            # -f is essential: these directories are gitignored, and without it the snapshot
            # would be an empty commit that pushes cleanly and contains nothing.
            add = _git(["add", "-f", "-A", d], env)
            if add.returncode != 0:
                logger.warning("Could not stage %s: %s", d, _redact(add.stderr)[:160])
        staged = _git(["diff", "--cached", "--name-only"], env).stdout.split()
        if not staged:
            logger.warning("Snapshot staged 0 files from %s; nothing to push.", present)
            return False

        tree = _git(["write-tree"], env).stdout.strip()
        if not tree:
            logger.warning("write-tree produced nothing; snapshot aborted.")
            return False
        made = _git(["commit-tree", tree, "-m", message], env)
        commit = made.stdout.strip()
        if not commit:
            logger.error("commit-tree produced nothing; snapshot aborted. git said: %s",
                         _redact(made.stderr).strip()[:200] or "(no stderr)")
            return False
        _git(["update-ref", f"refs/heads/{_branch()}", commit], env)

        remote = _authenticated_remote() or "origin"
        push = _git(["push", "-f", remote, f"refs/heads/{_branch()}:refs/heads/{_branch()}"], env)
        if push.returncode != 0:
            logger.error("Snapshot push failed: %s", _redact(push.stderr)[:200])
            return False
        logger.info("Snapshot pushed: %d files to branch %s.", len(staged), _branch())
        return True
    except Exception as e:
        logger.warning("Snapshot failed: %s", type(e).__name__)
        return False


def restore() -> bool:
    """Restore results, checkpoints and data from the artifacts branch."""
    if not _is_git_repo():
        logger.info("Not a git repository; skipping artifact restore.")
        return False
    try:
        remote = _authenticated_remote() or "origin"
        fetch = _git(["fetch", remote, f"refs/heads/{_branch()}:refs/remotes/origin/{_branch()}"])
        if fetch.returncode != 0:
            logger.warning("Could not fetch the artifacts branch: %s", _redact(fetch.stderr)[:160])
            return False
        restored = 0
        for d in SYNCED_DIRS:
            r = _git(["checkout", f"origin/{_branch()}", "--", d])
            if r.returncode == 0:
                restored += 1
        logger.info("Restored %d of %d artifact directories.", restored, len(SYNCED_DIRS))
        return restored > 0
    except Exception as e:
        logger.warning("Restore failed: %s", type(e).__name__)
        return False


def snapshot_status() -> Tuple[bool, str]:
    """Whether a snapshot could run right now, and why not if it could not."""
    if not _is_git_repo():
        return False, "not a git repository"
    if not env_loader.get("GITHUB_TOKEN"):
        return False, "no GitHub token in .env"
    if _git(["remote", "get-url", "origin"]).returncode != 0:
        return False, "no origin remote configured"
    return True, "ready"
