"""The judge chain: a strictly ordered fallback with no retries.

The judge never sees a gold answer and never produces a label. It does exactly three jobs:

  1. Answer extraction, and only when the deterministic JSON parser has already failed on a
     specific item. It is shown the model's own output and nothing else: no question, no
     options, no gold answer. It cannot therefore turn a wrong answer into a right one; it
     can only read a letter out of prose the parser could not.
  2. Rationale factual-correctness scoring on a stratified subsample.
  3. The AgriAdvice drift cross-check, which is never used for any headline number.

The chain, in order, with NO retries anywhere. Each route is attempted at most once per
request and any failure falls straight through to the next:

    1. DeepSeek, key 1
    2. DeepSeek, key 2
    3. Mistral, key 1
    4. Mistral, key 2

The same chain serves every subject model, local or hosted, so a parse failure is resolved
the same way everywhere and the resolution route is recorded. Keys are read only from .env
through env_loader and are never printed.

Set DISABLE_JUDGE=1 to make every judge call a no-op, which is what the smoke run does so it
stays fully offline.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("clients")

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("JUDGE_REQUEST_TIMEOUT_SECONDS", "60"))
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


@dataclass(frozen=True)
class JudgeRoute:
    name: str
    provider: str
    base_url: str
    model: str
    api_key: str

    @property
    def model_string(self) -> str:
        return f"{self.provider}:{self.model}"


def _call_openai_compatible(base_url: str, key: str, model: str, prompt: str) -> str:
    """One chat completion against any OpenAI-compatible endpoint, at temperature zero."""
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if "choices" not in payload:
        raise RuntimeError(str(payload.get("error", payload))[:200])
    return payload["choices"][0]["message"].get("content") or ""


def build_judge_routes() -> List[JudgeRoute]:
    """The four routes, in the fixed order, skipping any whose key is absent."""
    env_loader.load_env()
    deepseek_base = env_loader.get("DEEPSEEK_API_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL)
    deepseek_model = env_loader.get("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")
    mistral_model = env_loader.get("MISTRAL_MODEL_NAME", "mistral-small-latest")
    wanted = [
        ("deepseek-key-1", "deepseek", deepseek_base, deepseek_model, env_loader.get("DEEPSEEK_KEY1")),
        ("deepseek-key-2", "deepseek", deepseek_base, deepseek_model, env_loader.get("DEEPSEEK_KEY2")),
        ("mistral-key-1", "mistral", MISTRAL_BASE_URL, mistral_model, env_loader.get("MISTRAL_KEY1")),
        ("mistral-key-2", "mistral", MISTRAL_BASE_URL, mistral_model, env_loader.get("MISTRAL_KEY2")),
    ]
    return [JudgeRoute(n, p, b, m, k) for n, p, b, m, k in wanted if k]


class JudgeChain:
    """DeepSeek key 1, DeepSeek key 2, Mistral key 1, Mistral key 2. One attempt each."""

    def __init__(self, routes: Optional[List[JudgeRoute]] = None):
        self.routes = routes if routes is not None else build_judge_routes()
        self.last_model_string = ""
        self.last_route = ""
        self._lock = threading.Lock()
        self.route_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}

    def available(self) -> bool:
        if os.environ.get("DISABLE_JUDGE") == "1":
            return False
        return bool(self.routes)

    def route_names(self) -> List[str]:
        return [r.name for r in self.routes]

    def complete(self, prompt: str) -> Optional[str]:
        """Walk the chain once. Returns None only when every route has failed."""
        if not self.available():
            return None
        errors = []
        for route in self.routes:
            try:
                text = _call_openai_compatible(route.base_url, route.api_key, route.model, prompt)
                with self._lock:
                    self.route_counts[route.name] = self.route_counts.get(route.name, 0) + 1
                self.last_model_string = route.model_string
                self.last_route = route.name
                return text
            except Exception as e:  # no retry: record and fall straight through
                detail = _error_detail(e)
                errors.append(f"{route.name}={detail}")
                with self._lock:
                    self.failure_counts[route.name] = self.failure_counts.get(route.name, 0) + 1
                logger.warning("Judge route %s failed (%s); falling through to the next route.",
                               route.name, detail)
        logger.error("Judge: every route failed (%s).", ", ".join(errors))
        return None

    def extract_answer_letter(self, model_output: str, letters=("a", "b", "c")) -> Optional[str]:
        """Read the chosen option letter out of a model's own output.

        The judge sees the output alone, so it cannot supply an answer the subject model did
        not give. Used only after the deterministic parser has failed on that item."""
        from GPU_Run.common import prompts as P
        from GPU_Run.common.parsing import extract_answer_letter as parse_letter

        text = self.complete(P.judge_answer_extraction_prompt(model_output, letters))
        if text is None:
            return None
        letter, _ = parse_letter(text, letters)
        return letter

    def score_int(self, prompt: str, field: str, lo: int, hi: int) -> Optional[int]:
        text = self.complete(prompt)
        if text is None:
            return None
        from GPU_Run.common.parsing import json_repair_parse

        obj = json_repair_parse(text) or {}
        try:
            value = int(round(float(obj.get(field))))
        except (TypeError, ValueError):
            return None
        return max(lo, min(hi, value))

    def usage_summary(self) -> Dict[str, Dict[str, int]]:
        return {"judge_requests_served_by_route": dict(self.route_counts),
                "judge_failures_by_route": dict(self.failure_counts)}


def secondary_judge_chain() -> JudgeChain:
    """The Mistral half of the chain, used by the judge-agreement check so the two graders
    are genuinely different providers rather than the same one twice."""
    return JudgeChain([r for r in build_judge_routes() if r.provider == "mistral"])


def primary_judge_chain() -> JudgeChain:
    """The DeepSeek half, the counterpart of secondary_judge_chain."""
    return JudgeChain([r for r in build_judge_routes() if r.provider == "deepseek"])


def _error_detail(exc: Exception) -> str:
    """A short, safe error label. Never includes a key or a full response body."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return f"HTTP{status}"
    return type(exc).__name__
