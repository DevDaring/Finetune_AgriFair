"""llm_client.py — unified, resumable, cached LLM access layer for AgriFair.

Design goals (see Dataset_building_prompt.md section 2):
  * DeepSeek = PRIMARY, Mistral = FALLBACK, OpenRouter = SPECIALIST / 3rd judge.
  * Round-robin across the two keys of each provider to spread rate limits.
  * Fallback chain on error/429/timeout: deepseek key A -> key B -> mistral -> openrouter.
  * sha256 disk cache: a repeated call costs nothing and is crash-safe.
  * Cost logged to logs/costs.csv on every *live* call. Keys never printed.

All three providers expose an OpenAI-compatible /chat/completions endpoint, so a
single openai.OpenAI client (re-pointed per provider/key) drives everything.

Model slugs are NOT hard-coded here; they come from config.yaml, which was
verified live on 2026-06-08. Keys come only from .env via python-dotenv.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    raise SystemExit("openai package missing — run: pip install -r requirements.txt") from e

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- approximate USD per 1M tokens (verified 2026-06-08); for budgeting only ---
PRICE_PER_MTOK = {
    "deepseek-v4-flash": {"in": 0.14, "out": 0.28},
    "deepseek-v4-pro":   {"in": 0.435, "out": 0.87},
    "mistral-small-2603": {"in": 0.15, "out": 0.30},
    # OpenRouter slugs vary; fall back to a conservative estimate if unknown.
    "_default": {"in": 0.30, "out": 0.60},
}


def _env(name: str) -> str | None:
    v = os.getenv(name)
    return v.strip() if v else None


@dataclass
class _Provider:
    name: str
    base_url: str
    keys: list[str]
    models: dict[str, str]
    _rr: itertools.cycle = field(init=False)

    def __post_init__(self):
        # round-robin iterator over available keys
        self._rr = itertools.cycle(range(len(self.keys)))

    def next_key_index(self) -> int:
        return next(self._rr)


class LLMClient:
    def __init__(self, config_path: str | os.PathLike = ROOT / "config.yaml"):
        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.cache_dir = ROOT / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir = ROOT / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.cost_csv = self.logs_dir / "costs.csv"
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(int(self.cfg.get("concurrency", 6)))
        self.providers = self._build_providers()
        self.temps = self.cfg["temperature"]
        self.max_retries = int(self.cfg.get("max_retries", 5))
        self.timeout_s = int(self.cfg.get("timeout_s", 90))
        if not self.cost_csv.exists():
            with self.cost_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "provider", "model", "in_tok", "out_tok", "est_usd", "cached"]
                )

    # ----------------------------------------------------------------- setup
    def _build_providers(self) -> dict[str, _Provider]:
        out: dict[str, _Provider] = {}
        for pname, pc in self.cfg["providers"].items():
            base = pc.get("base_url") or _env(pc.get("base_url_env", ""))
            keys = [k for e in pc.get("key_envs", []) if (k := _env(e))]
            if not base or not keys:
                # provider simply unavailable; skip silently (no key leak)
                continue
            out[pname] = _Provider(pname, base, keys, pc.get("models", {}))
        if "deepseek" not in out:
            raise SystemExit("DeepSeek keys/base_url not found in .env — cannot proceed.")
        return out

    # ----------------------------------------------------------------- cache
    def _cache_key(self, provider: str, model: str, messages, params) -> str:
        blob = json.dumps(
            {"p": provider, "m": model, "msg": messages, "prm": params},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> str | None:
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["content"]
        return None

    def _cache_write(self, key: str, content: str, meta: dict) -> None:
        f = self.cache_dir / f"{key}.json"
        f.write_text(json.dumps({"content": content, **meta}, ensure_ascii=False), encoding="utf-8")

    # ----------------------------------------------------------------- cost
    def _log_cost(self, provider, model, in_tok, out_tok, cached):
        price = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["_default"])
        est = (in_tok / 1e6) * price["in"] + (out_tok / 1e6) * price["out"]
        with self._lock, self.cost_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [int(time.time()), provider, model, in_tok, out_tok, round(est, 6), int(cached)]
            )

    # ----------------------------------------------------------------- tiers
    def _route(self, tier: str) -> list[tuple[str, str]]:
        """Return ordered [(provider, model)] attempts for a tier."""
        ds = self.providers.get("deepseek")
        ms = self.providers.get("mistral")
        orr = self.providers.get("openrouter")
        chain: list[tuple[str, str]] = []
        if tier == "specialist" and orr:
            chain.append(("openrouter", orr.models.get("specialist")))
        if tier == "reason" and ds:
            chain.append(("deepseek", ds.models.get("reason", ds.models["default"])))
        # default path + universal fallbacks
        if ds:
            chain.append(("deepseek", ds.models["default"]))
        if ms:
            chain.append(("mistral", ms.models["default"]))
        if orr and tier != "specialist":
            chain.append(("openrouter", orr.models.get("specialist")))
        # de-dup while preserving order
        seen, uniq = set(), []
        for p, m in chain:
            if m and (p, m) not in seen:
                seen.add((p, m)); uniq.append((p, m))
        return uniq

    # ----------------------------------------------------------------- call
    def complete(self, messages: list[dict], tier: str = "default",
                 temperature: float | None = None, max_tokens: int = 1024,
                 task: str = "judge") -> str:
        if temperature is None:
            temperature = self.temps.get(task, 0.0)
        # DeepSeek v4 is a hybrid model: thinking ON only for the reason tier.
        thinking = (tier == "reason")
        params = {"max_tokens": max_tokens, "thinking": thinking}
        # Thinking mode rejects temperature/top_p/penalties; omit when enabled.
        if not thinking:
            params["temperature"] = temperature
        attempts = self._route(tier)
        last_err = None
        for provider, model in attempts:
            ck = self._cache_key(provider, model, messages, params)
            hit = self._cache_read(ck)
            if hit is not None:
                self._log_cost(provider, model, 0, 0, cached=True)
                return hit
            try:
                with self._sem:
                    content, in_tok, out_tok = self._call_live(provider, model, messages, params)
                self._cache_write(ck, content, {"provider": provider, "model": model})
                self._log_cost(provider, model, in_tok, out_tok, cached=False)
                return content
            except Exception as e:  # noqa: BLE001 — deliberately broad: try next provider
                last_err = e
                continue
        raise RuntimeError(f"All providers failed for tier={tier}: {last_err}")

    def judge_call(self, provider: str, model_key: str, messages: list[dict],
                   max_tokens: int = 16) -> str:
        """Call ONE specific provider/family (no cross-family fallback) for the
        3-judge panel. Cached + cost-logged like complete(). Raises if the chosen
        provider is unavailable or fails, so a judge is never silently swapped."""
        prov = self.providers.get(provider)
        if prov is None:
            raise RuntimeError(f"judge provider '{provider}' unavailable")
        model = prov.models.get(model_key) or prov.models.get("default")
        params = {"max_tokens": max_tokens, "temperature": 0.0, "thinking": False}
        ck = self._cache_key(provider, model, messages, params)
        hit = self._cache_read(ck)
        if hit is not None:
            self._log_cost(provider, model, 0, 0, cached=True)
            return hit
        with self._sem:
            content, in_tok, out_tok = self._call_live(provider, model, messages, params)
        self._cache_write(ck, content, {"provider": provider, "model": model})
        self._log_cost(provider, model, in_tok, out_tok, cached=False)
        return content

    def _call_live(self, provider: str, model: str, messages, params):
        prov = self.providers[provider]
        idx = prov.next_key_index()
        last_err = None
        # Translate the portable `thinking` flag into provider-specific wire form.
        params = dict(params)
        thinking = params.pop("thinking", False)
        call_kwargs = dict(params)
        if provider == "deepseek":
            call_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }
        # try both keys of this provider with backoff before falling through
        for attempt in range(self.max_retries):
            key = prov.keys[(idx + attempt) % len(prov.keys)]
            try:
                client = OpenAI(api_key=key, base_url=prov.base_url, timeout=self.timeout_s)
                resp = client.chat.completions.create(model=model, messages=messages, **call_kwargs)
                msg = resp.choices[0].message
                content = msg.content or ""
                # Thinking mode may put the usable answer in reasoning_content if the
                # token budget was tight; prefer content, fall back to reasoning.
                if not content and getattr(msg, "reasoning_content", None):
                    content = msg.reasoning_content
                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
                out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
                return content, in_tok, out_tok
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt + (attempt * 0.1), 30))
        raise last_err  # surfaces to complete(), which advances the fallback chain


if __name__ == "__main__":
    # Smoke test — uses cache after first run, costs ~nothing.
    c = LLMClient()
    print("Providers available:", list(c.providers))
    out = c.complete(
        [{"role": "user", "content": "Reply with exactly the word: ok"}],
        tier="default", task="extraction", max_tokens=8,
    )
    print("Model replied:", repr(out))
