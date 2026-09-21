"""Router for Submission 2: the frozen route chain from GPU_Run plus three additions.

1. `temperature` for self-consistency sampling (the chain hard-codes 0).
2. `force_route` so the route-consistency study (E7) can send the same item through each route.
3. A deterministic `FakeRouter` used by --smoke and the tests, which never spends.

Every call records the model id string actually served, so a silent provider update shows up
as a change in the `model_id` column rather than as an unexplained shift in the results.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from GPU_Run.common import api_models as AM


@dataclass
class Reply:
    text: str
    route: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    failed_routes: str = ""
    ok: bool = True


class Router(AM.ApiRouter):
    """ApiRouter with a settable temperature and an optional forced route."""

    def __init__(self):
        super().__init__()
        self.temperature = 0.0
        self.force_route: Optional[str] = None

    # temperature-aware transports (bodies mirror the parent; only the sampling field changes)
    def _call_xai(self, spec, prompt, max_tokens):
        r = requests.post(f"{AM.XAI_BASE_URL.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {self.xai_key}", "Content-Type": "application/json"},
                          json={"model": spec.native_model_id, "messages": [{"role": "user", "content": prompt}],
                                "temperature": self.temperature, "max_tokens": max_tokens},
                          timeout=AM.REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status(); p = r.json()
        if "choices" not in p:
            raise RuntimeError(str(p.get("error", p))[:200])
        u = p.get("usage", {})
        return p["choices"][0]["message"].get("content") or "", int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))

    def _call_openrouter(self, key, spec, prompt, max_tokens):
        r = requests.post(f"{self.openrouter_base.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": spec.openrouter_model_id, "messages": [{"role": "user", "content": prompt}],
                                "temperature": self.temperature, "max_tokens": max_tokens},
                          timeout=AM.REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status(); p = r.json()
        if "choices" not in p:
            raise RuntimeError(str(p.get("error", p))[:200])
        u = p.get("usage", {})
        return p["choices"][0]["message"].get("content") or "", int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))

    def _call_bedrock(self, account, spec, prompt, max_tokens):
        name, ak, sk = account
        client = self._bedrock_client(name, ak, sk)
        resp = client.converse(modelId=spec.bedrock_model_id,
                               messages=[{"role": "user", "content": [{"text": prompt}]}],
                               inferenceConfig={"maxTokens": max_tokens, "temperature": float(self.temperature)})
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        u = resp.get("usage", {})
        return text, int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))

    def _ordered_routes(self, spec, advance=True):
        routes = super()._ordered_routes(spec, advance)
        if self.force_route:
            routes = [r for r in routes if r[0] == self.force_route]
        return routes

    def available_routes(self, tier: str) -> List[str]:
        return self.describe_routes(tier)

    def ask(self, tier: str, prompt: str, max_tokens: int, temperature: float = 0.0,
            force_route: Optional[str] = None) -> Reply:
        self.temperature, self.force_route = temperature, force_route
        spec = AM.REGISTRY[tier]
        resp = self.complete(tier, prompt, max_tokens=max_tokens)
        self.temperature, self.force_route = 0.0, None
        if resp is None:
            return Reply(text="", route="none", model_id="", ok=False)
        mid = {"openai-direct": spec.native_model_id, "xai-direct": spec.native_model_id,
               "anthropic-direct": spec.native_model_id}.get(resp.route)
        if mid is None:
            mid = spec.bedrock_model_id if resp.route.startswith("aws") else spec.openrouter_model_id
        return Reply(text=resp.text, route=resp.route, model_id=mid or "", input_tokens=resp.input_tokens,
                     output_tokens=resp.output_tokens, latency_seconds=resp.latency_seconds,
                     failed_routes=resp.failed_routes)


class FakeRouter:
    """Deterministic offline stand-in. Answers are a hash of the prompt, so they are stable
    across runs and differ between items; nothing is sent anywhere."""

    def __init__(self, flaky_route: Optional[str] = None):
        self.calls = 0; self.flaky_route = flaky_route

    def available_routes(self, tier: str) -> List[str]:
        return ["fake-direct", "fake-openrouter-1"]

    def ask(self, tier: str, prompt: str, max_tokens: int, temperature: float = 0.0,
            force_route: Optional[str] = None) -> Reply:
        self.calls += 1
        h = hashlib.sha256(f"{tier}|{prompt}|{temperature}".encode()).digest()
        letter = "abc"[h[0] % 3]
        if "Reply with JSON" in prompt or "JSON object" in prompt:
            text = f'{{"answer_choice_letter": "{letter}"}}'
            if "Let's think step by step" in prompt or "Step 1" in prompt:
                text = "Step 1: compare the two shares. Step 2: decide.\n" + text
        else:
            words = ["apply", "neem", "oil", "spray", "at", "dusk", "and", "remove", "infected", "leaves"]
            text = " ".join(words[(h[i % 32] % len(words))] for i in range(8, 8 + 20 + h[3] % 40)) + "."
        route = force_route or self.available_routes(tier)[0]
        return Reply(text=text, route=route, model_id=f"fake-{tier}", input_tokens=len(prompt.split()) * 2,
                     output_tokens=len(text.split()) * 2, latency_seconds=0.01 + (h[4] % 50) / 100)
