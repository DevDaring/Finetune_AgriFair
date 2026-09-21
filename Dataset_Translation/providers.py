"""Chat clients for the translation pipeline.

Every hosted model except Nova speaks the OpenAI chat-completions dialect, so one function
covers DeepSeek, Kimi (Moonshot), xAI, Mistral and OpenRouter. Nova 2 Lite goes through the
Bedrock route chain already in GPU_Run/common/api_models.py (account A/B round-robin with an
OpenRouter fallback). Keys are read through env_loader and never logged.

Round-robin: a provider with two keys (DeepSeek, Mistral, OpenRouter) alternates keys on every
call; a failure on one key falls through to the other before the provider-level fallback.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from GPU_Run.common import api_models as AM
from GPU_Run.common import env_loader

_KEY_ALIASES = {
    "deepseek": ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"],
    "kimi": ["KIMI_2009_API_KEY"],
    "xai": ["Grok_API_KEY", "XAI_DIRECT_KEY"],
    "mistral": ["MISTRAL_API_KEY1", "MISTRAL_API_KEY2"],
    "openrouter": ["OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"],
    "openai": ["Open_AI_2009_Key", "OPENAI_API_KEY"],
    "linkapi": ["LINKAPI_AzureOpenAI_KEY"],          # OpenAI models at 0.5x through LinkAPI; primary for GPT-4o
    "linkapi_gemini": ["LINKAPI_Gemini_Cheap_KEY"],
}


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    key_label: str = ""


class ProviderError(RuntimeError):
    pass


class Clients:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.endpoints = cfg["endpoints"]
        self.timeout = int(cfg.get("request_timeout_seconds", 120))
        self._keys: Dict[str, List[Tuple[str, str]]] = {}
        for prov, names in _KEY_ALIASES.items():
            env_loader.load_env()
            self._keys[prov] = [(n, env_loader.get(n)) for n in names if env_loader.get(n)]
        self._cycles = {p: itertools.cycle(range(len(ks))) for p, ks in self._keys.items() if ks}
        self._lock = threading.Lock()
        self._bedrock: Optional[AM.ApiRouter] = None
        self.calls: Dict[str, int] = {}

    # ---------------------------------------------------------------- helpers
    def _next_key(self, provider: str) -> List[Tuple[str, str]]:
        """Keys for a provider, starting from the round-robin position."""
        ks = self._keys.get(provider, [])
        if not ks:
            return []
        with self._lock:
            start = next(self._cycles[provider])
        return ks[start:] + ks[:start]

    def _openai_chat(self, provider: str, model: str, system: str, user: str, max_tokens: int, temperature: float) -> Completion:
        keys = self._next_key(provider)
        if not keys:
            raise ProviderError(f"{provider}: no key configured")
        last = ""
        for label, key in keys:
          for attempt in range(4):
            t0 = time.time()
            try:
                r = requests.post(f"{self.endpoints[provider].rstrip('/')}/chat/completions",
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                  json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                                  timeout=self.timeout)
                if r.status_code == 429 and attempt < 3:        # rate limit: back off on the same key before switching
                    time.sleep(2 ** (attempt + 1)); continue
                if r.status_code >= 400:
                    raise ProviderError(f"HTTP {r.status_code}: {r.text[:160]}")
                p = r.json()
                if "choices" not in p:
                    raise ProviderError(str(p.get("error", p))[:160])
                text = p["choices"][0]["message"].get("content") or ""
                u = p.get("usage", {})
                with self._lock:
                    self.calls[f"{provider}/{model}"] = self.calls.get(f"{provider}/{model}", 0) + 1
                return Completion(text, provider, model, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)),
                                  time.time() - t0, label)
            except (requests.RequestException, ProviderError, ValueError) as e:
                last = f"{type(e).__name__}: {str(e)[:160]}"
                break                                            # next key
        raise ProviderError(f"{provider}/{model}: all keys failed ({last})")

    def _nova(self, model: str, system: str, user: str, max_tokens: int) -> Completion:
        if self._bedrock is None:
            self._bedrock = AM.ApiRouter()
        spec = AM.REGISTRY["frontier-nova-2-lite"]
        if model and model != spec.bedrock_model_id:
            spec = AM.ApiModelSpec(tier=spec.tier, display_name=spec.display_name, vendor=spec.vendor, route_order=spec.route_order,
                                   bedrock_model_id=model, openrouter_model_id=spec.openrouter_model_id)
        t0 = time.time()
        prompt = f"{system}\n\n{user}"
        last = ""
        for account in self._bedrock.aws_accounts:
            try:
                text, i, o = self._bedrock._call_bedrock(account, spec, prompt, max_tokens)
                with self._lock:
                    self.calls["bedrock/nova-2-lite"] = self.calls.get("bedrock/nova-2-lite", 0) + 1
                return Completion(text, "bedrock", spec.bedrock_model_id, i, o, time.time() - t0, account[0])
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
        raise ProviderError(f"bedrock nova: both accounts failed ({last})")

    # ---------------------------------------------------------------- public
    def complete(self, role: Dict, system: str, user: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> Completion:
        """role = {provider, model, fallback:{provider, model}} from config.yaml."""
        mt = max_tokens or int(self.cfg["max_tokens"]); temp = self.cfg["temperature"] if temperature is None else temperature
        chain = [role] + list(role.get("fallbacks", [])) + ([role["fallback"]] if role.get("fallback") else [])
        last = ""
        for step in chain:
            try:
                if step["provider"] == "bedrock":
                    return self._nova(step["model"], system, user, mt)
                return self._openai_chat(step["provider"], step["model"], system, user, mt, temp)
            except ProviderError as e:
                last = str(e)
                continue
        raise ProviderError(f"every route failed for {role['provider']}/{role['model']}: {last}")
