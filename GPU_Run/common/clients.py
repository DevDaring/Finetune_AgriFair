"""Round-robin API clients and the no-retry judge chain.

Judgement only (Instruction.md Section 10): rationale factual-correctness scoring, the
rare answer-extraction fallback, and the AgriAdvice drift cross-check. Answers themselves
never use a judge. Chain order: Gemini -> DeepSeek (+ OpenRouter alternate) -> Mistral.
Keys rotate round-robin per request; on any provider error the chain moves once to the
next (no within-provider retry). Keys are read only from .env via env_loader.
"""
from __future__ import annotations

import itertools
import json
from typing import Callable, Dict, List, Optional

import requests

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("clients")
_TIMEOUT = 60


class _RoundRobin:
    def __init__(self, keys: List[str]):
        self.keys = [k for k in keys if k]
        self._cycle = itertools.cycle(self.keys) if self.keys else None

    def __bool__(self):
        return bool(self.keys)

    def next(self) -> Optional[str]:
        return next(self._cycle) if self._cycle else None


def _gemini_keys() -> _RoundRobin:
    return _RoundRobin(env_loader.get_round_robin(["GCP_Key1", "GCP_Key2", "GCP_Key3", "GCP_key4"]))


def _deepseek_keys() -> _RoundRobin:
    return _RoundRobin(env_loader.get_round_robin(["DEEPSEEK_KEY1", "DEEPSEEK_KEY2"]))


def _openrouter_keys() -> _RoundRobin:
    return _RoundRobin(env_loader.get_round_robin(["OPENROUTER_KEY1", "OPENROUTER_KEY2"]))


def _mistral_keys() -> _RoundRobin:
    return _RoundRobin(env_loader.get_round_robin(["MISTRAL_KEY1", "MISTRAL_KEY2"]))


def _call_openai_compatible(base_url: str, key: str, model: str, prompt: str) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(key: str, model: str, prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    )
    resp = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


class JudgeChain:
    """Provider chain with round-robin keys and no within-provider retry."""

    def __init__(self):
        self.gemini = _gemini_keys()
        self.deepseek = _deepseek_keys()
        self.openrouter = _openrouter_keys()
        self.mistral = _mistral_keys()
        self.gemini_model = env_loader.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self.deepseek_model = env_loader.get("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")
        self.deepseek_base = env_loader.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
        self.openrouter_model = env_loader.get("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini")
        self.openrouter_base = env_loader.get("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
        self.mistral_model = env_loader.get("MISTRAL_MODEL_NAME", "mistral-small-latest")
        self.last_model_string = ""

    def available(self) -> bool:
        import os

        if os.environ.get("DISABLE_JUDGE") == "1":
            return False
        return bool(self.gemini or self.deepseek or self.openrouter or self.mistral)

    def complete(self, prompt: str) -> Optional[str]:
        attempts: List[Callable[[], str]] = []
        if self.gemini:
            k = self.gemini.next()
            attempts.append((lambda k=k: ("gemini:" + self.gemini_model, _call_gemini(k, self.gemini_model, prompt))))
        if self.deepseek:
            k = self.deepseek.next()
            attempts.append((lambda k=k: ("deepseek:" + self.deepseek_model, _call_openai_compatible(self.deepseek_base, k, self.deepseek_model, prompt))))
        if self.openrouter:
            k = self.openrouter.next()
            attempts.append((lambda k=k: ("openrouter:" + self.openrouter_model, _call_openai_compatible(self.openrouter_base, k, self.openrouter_model, prompt))))
        if self.mistral:
            k = self.mistral.next()
            attempts.append((lambda k=k: ("mistral:" + self.mistral_model, _call_openai_compatible("https://api.mistral.ai/v1", k, self.mistral_model, prompt))))

        for fn in attempts:
            try:
                model_string, text = fn()
                self.last_model_string = model_string
                return text
            except Exception as e:
                logger.warning("Judge provider failed (%s); moving to next.", type(e).__name__)
                continue
        return None

    def score_int(self, prompt: str, field: str, lo: int, hi: int) -> Optional[int]:
        text = self.complete(prompt)
        if text is None:
            return None
        from GPU_Run.common.parsing import json_repair_parse

        obj = json_repair_parse(text) or {}
        try:
            v = int(round(float(obj.get(field))))
            return max(lo, min(hi, v))
        except Exception:
            return None
