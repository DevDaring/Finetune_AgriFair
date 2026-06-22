"""One-shot downloader for the private HF dataset Debk/AgriFair.

Reads HUGGINGFACE_TOKEN from ../.env (never printed) and snapshots the full
dataset repo (data files + README) into this folder.
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE.parent / ".env"
REPO_ID = "Debk/AgriFair"


def read_token(env_path: Path) -> str:
    if not env_path.exists():
        sys.exit(f"ERROR: .env not found at {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*HUGGINGFACE_TOKEN\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    sys.exit("ERROR: HUGGINGFACE_TOKEN not found in .env")


def main() -> None:
    from huggingface_hub import snapshot_download

    token = read_token(ENV_PATH)
    target = HERE  # download into Codes/Dataset
    print(f"Downloading dataset repo '{REPO_ID}' -> {target}")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(target),
        token=token,
    )
    print(f"DONE. Files materialized under: {local_dir}")
    for p in sorted(Path(local_dir).rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            print(f"  {p.relative_to(local_dir)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
