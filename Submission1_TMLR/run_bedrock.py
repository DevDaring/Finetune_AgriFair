"""Large open-weight reference models through AWS Bedrock (plan: section 3, "Large references").

    python -m Submission1_TMLR.run_bedrock --probe        # two prompts per model on each account
    python -m Submission1_TMLR.run_bedrock                # the full design, resumable

Same prompts, the same JSON-letter parser and the same scoring as the GPU systems. Temperature is 0.
Calls alternate between the two AWS accounts in Codes/.env; a throttled call waits and retries on
the same account. Keys are read from the environment loader and never printed or written.
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from GPU_Run.common import env_loader
from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2.run_inference import score_row
from Submission1_TMLR.run_gpu import OUT_DIR, load_prompts

MODELS = {"bedrock|qwen3-next-80b-a3b|unmodified": "qwen.qwen3-next-80b-a3b",
          "bedrock|deepseek-v3.2|unmodified": "deepseek.v3.2"}
# round 2 (Future_PLan.md E5): the larger sibling of each GPU family, then three frontier open-weight models
NEW_MODELS = {"bedrock|gemma-3-27b|unmodified": "google.gemma-3-27b-it",
              "bedrock|qwen3-32b|unmodified": "qwen.qwen3-32b-v1:0",
              "bedrock|llama-3.3-70b|unmodified": "us.meta.llama3-3-70b-instruct-v1:0",
              "bedrock|mistral-large-3|unmodified": "mistral.mistral-large-3-675b-instruct",
              "bedrock|gpt-oss-120b|unmodified": "openai.gpt-oss-120b-1:0",
              "bedrock|kimi-k2.5|unmodified": "moonshotai.kimi-k2.5",
              "bedrock|glm-5|unmodified": "zai.glm-5"}
ALL_MODELS = {**MODELS, **NEW_MODELS}
# reasoning-style models spend output tokens on reasoning before the JSON answer: the 4-prompt probe
# (4 Oct 2026) cut off 3 of 4 answers of each at 256 tokens, so they get 1,024 for every set
BUDGET = {"bedrock|gpt-oss-120b|unmodified": 1024, "bedrock|kimi-k2.5|unmodified": 1024}
MAX_TOKENS = 256          # room for a short preamble before the JSON; the parser reads the letter
REGION = env_loader.get("AWS_REGION", "us-east-1")
_lock = threading.Lock()
_clients: Dict[str, object] = {}


def accounts() -> List[tuple]:
    acc = [("aws-1", env_loader.get("AWS_ACCESS_KEY"), env_loader.get("AWS_SECRET_KEY")),
           ("aws-2", env_loader.get("AWS_ACCESS_KEY2"), env_loader.get("AWS_SECRET_KEY2"))]
    acc = [a for a in acc if a[1] and a[2]]
    if not acc:
        raise SystemExit("no AWS credentials found in Codes/.env")
    return acc


def client(name: str, ak: str, sk: str):
    import boto3
    from botocore.config import Config
    with _lock:
        if name not in _clients:
            _clients[name] = boto3.client("bedrock-runtime", region_name=REGION, aws_access_key_id=ak,
                                          aws_secret_access_key=sk,
                                          config=Config(retries={"max_attempts": 2}, read_timeout=120))
        return _clients[name]


def call(account: tuple, model_id: str, prompt: str, max_tokens: int = MAX_TOKENS) -> Dict:
    name, ak, sk = account
    for attempt in range(8):
        try:
            resp = client(name, ak, sk).converse(
                modelId=model_id, messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0})
            blocks = resp["output"]["message"]["content"]
            text = "".join(b.get("text", "") for b in blocks)
            reasoning = "".join(b["reasoningContent"].get("reasoningText", {}).get("text", "")
                                for b in blocks if "reasoningContent" in b)
            u = resp.get("usage", {})
            return {"raw": text, "input_tokens": u.get("inputTokens", 0), "output_tokens": u.get("outputTokens", 0),
                    "account": name, "stop_reason": resp.get("stopReason"),
                    **({"reasoning_chars": len(reasoning)} if reasoning else {})}
        except Exception as e:                                   # throttling or a transient error
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            if "Throttl" in msg or "TooManyRequests" in msg or "ServiceUnavailable" in msg or "timed out" in msg:
                time.sleep(min(60, 2 ** attempt))
                continue
            return {"raw": "", "error": msg, "account": name}
    return {"raw": "", "error": "gave up after 8 throttled attempts", "account": name}


def run(probe: bool, workers: int, sets: str = "main", which: str = "original") -> Dict:
    models = {"original": MODELS, "new": NEW_MODELS, "all": ALL_MODELS}[which]
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = "probe" if probe else ("main" if sets == "main" else sets)
    path = out / f"bedrock_{stem}_predictions.jsonl"
    done = {(r["prompt_id"], r["system"]) for r in C.read_jsonl(path) if not r.get("error")} if path.exists() else set()
    prompts = load_prompts(sets)
    accs = accounts()
    jobs = []
    for sid, mid in models.items():
        todo = [p for p in prompts if (p["prompt_id"], sid) not in done]
        if probe:
            todo = todo[:2 * len(accs)]
        jobs += [(sid, mid, p) for p in todo]
    # interleave models so concurrent calls spread over the per-model quotas instead of throttling one model
    import random
    random.Random(0).shuffle(jobs)
    cycle = itertools.cycle(accs)
    started, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex, path.open("a", encoding="utf-8") as f:
        futs = {ex.submit(call, next(cycle), mid, p["prompt"], max(BUDGET.get(sid, MAX_TOKENS), int(p.get("max_new_tokens", 0)) + 100)): (sid, mid, p)
                for sid, mid, p in jobs}
        for fut in as_completed(futs):
            sid, mid, p = futs[fut]
            res = fut.result()
            row = {**{k: v for k, v in p.items() if k != "prompt"}, "system": sid, "tier": sid.split("|")[1],
                   "method": "unmodified", "seed": None, "bedrock_model_id": mid,
                   "prompt_sha256": C.freeze(p["prompt"]), **{k: v for k, v in res.items() if k != "raw"},
                   **score_row(p, res["raw"])}
            with _lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)} calls, {(time.time() - started) / 60:.1f} min", flush=True)
    rows = list(C.read_jsonl(path))
    summary = {sid: {"rows": sum(r["system"] == sid for r in rows),
                     "errors": sum(bool(r.get("error")) for r in rows if r["system"] == sid),
                     "parse_ok": round(sum(r["parse_ok"] for r in rows if r["system"] == sid) /
                                       max(1, sum(r["system"] == sid for r in rows)), 4),
                     "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows if r["system"] == sid)}
               for sid in models}
    C.write_json(out / f"bedrock_{stem}_manifest.json",
                 {"models": models, "region": REGION, "max_tokens": MAX_TOKENS, "temperature": 0.0,
                  "accounts_used": [a[0] for a in accs], "summary": summary,
                  "minutes": round((time.time() - started) / 60, 2)})
    print(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sets", choices=["main", "followups", "cot", "altered48", "round2", "round3", "renamed"], default="main")
    ap.add_argument("--models", choices=["original", "new", "all"], default="original")
    a = ap.parse_args(argv)
    run(a.probe, a.workers, a.sets, a.models)


if __name__ == "__main__":
    main()
