"""Thin, provider-agnostic chat wrapper with response caching and a hard cost cap.

Backends: "ollama" (local, free, no key), "openai" (any OpenAI-compatible
endpoint; key read from the LLM_API_KEY environment variable or a gitignored
.env file) and "mock" (scripted function, for tests).

The API key is never stored on objects that get printed/logged and never
written to the cache or any log.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

Message = Dict[str, str]


class BudgetExceeded(RuntimeError):
    """Raised before a fresh call when the spend cap has been reached."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    latency: float


class CostTracker:
    """Thread-safe usage/cost accounting. Cached replays cost nothing."""

    def __init__(self, price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0,
                 cap_usd: float = 5.0, max_fresh_tokens: Optional[int] = None):
        self.price_in, self.price_out = price_in_per_mtok, price_out_per_mtok
        self.cap_usd, self.max_fresh_tokens = cap_usd, max_fresh_tokens
        self._lock = threading.Lock()
        self.fresh_prompt = self.fresh_completion = self.fresh_calls = 0
        self.cached_prompt = self.cached_completion = self.cached_calls = 0
        self.fresh_seconds = 0.0

    @property
    def cost_usd(self) -> float:
        return (self.fresh_prompt * self.price_in + self.fresh_completion * self.price_out) / 1e6

    def check(self) -> None:
        with self._lock:
            cost = self.cost_usd
            if cost > 0 and cost >= self.cap_usd:
                raise BudgetExceeded(f"Cost cap reached: ${cost:.4f} >= ${self.cap_usd:.2f}")
            if self.max_fresh_tokens and self.fresh_prompt + self.fresh_completion >= self.max_fresh_tokens:
                raise BudgetExceeded("Fresh-token cap reached")

    def record(self, r: LLMResponse) -> None:
        with self._lock:
            if r.cached:
                self.cached_prompt += r.prompt_tokens
                self.cached_completion += r.completion_tokens
                self.cached_calls += 1
            else:
                self.fresh_prompt += r.prompt_tokens
                self.fresh_completion += r.completion_tokens
                self.fresh_calls += 1
                self.fresh_seconds += r.latency

    def snapshot(self) -> dict:
        with self._lock:
            return {"fresh_calls": self.fresh_calls, "fresh_prompt_tokens": self.fresh_prompt,
                    "fresh_completion_tokens": self.fresh_completion,
                    "cached_calls": self.cached_calls,
                    "cached_prompt_tokens": self.cached_prompt,
                    "cached_completion_tokens": self.cached_completion,
                    "fresh_llm_seconds": round(self.fresh_seconds, 1),
                    "cost_usd": round(self.cost_usd, 6), "cap_usd": self.cap_usd}


# ---------------------------------------------------------------- backends
class OllamaBackend:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", num_ctx: int = 4096):
        self.model, self.base_url, self.num_ctx = model, base_url.rstrip("/"), num_ctx

    def complete(self, messages, temperature, max_tokens, seed, json_mode):
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "60m",
                "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens,
                            "num_ctx": self.num_ctx}}
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=900)
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str, key_env: str = "LLM_API_KEY"):
        self.model, self.base_url, self._key_env = model, base_url.rstrip("/"), key_env

    def __repr__(self) -> str:  # never leak anything key-related
        return f"OpenAICompatBackend(model={self.model!r}, base_url={self.base_url!r})"

    def complete(self, messages, temperature, max_tokens, seed, json_mode):
        key = os.environ.get(self._key_env)
        if not key:
            raise RuntimeError(f"Set {self._key_env} in the environment or a .env file")
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "seed": seed}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300,
                          headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


class MockBackend:
    """fn(messages, json_mode) -> text. Token counts are a crude word count."""

    def __init__(self, fn: Callable[[List[Message], bool], str], model: str = "mock"):
        self.fn, self.model = fn, model

    def complete(self, messages, temperature, max_tokens, seed, json_mode):
        text = self.fn(messages, json_mode)
        pt = sum(len(m["content"].split()) for m in messages)
        return text, pt, len(text.split())


# --------------------------------------------------------------------- LLM
class LLM:
    def __init__(self, backend, tracker: Optional[CostTracker] = None,
                 cache_path: Optional[Path] = None, retries: int = 3):
        self.backend = backend
        self.tracker = tracker or CostTracker()
        self.retries = retries
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            for line in self._cache_path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._cache[e["k"]] = e

    @property
    def model(self) -> str:
        return self.backend.model

    def _key(self, messages, temperature, max_tokens, seed, json_mode) -> str:
        blob = json.dumps([self.backend.model, messages, temperature, max_tokens, seed, json_mode],
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: List[Message], temperature: float = 0.0, max_tokens: int = 700,
             seed: int = 0, json_mode: bool = False) -> LLMResponse:
        key = self._key(messages, temperature, max_tokens, seed, json_mode)
        with self._lock:
            hit = self._cache.get(key)
        if hit:
            resp = LLMResponse(hit["text"], hit["pt"], hit["ct"], True, hit["lat"])
            self.tracker.record(resp)
            return resp
        self.tracker.check()
        last_err = None
        for attempt in range(self.retries):
            try:
                t0 = time.time()
                text, pt, ct = self.backend.complete(messages, temperature, max_tokens, seed, json_mode)
                lat = time.time() - t0
                break
            except Exception as e:  # network / server hiccup
                last_err = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"LLM call failed after {self.retries} attempts: {type(last_err).__name__}")
        entry = {"k": key, "text": text, "pt": pt, "ct": ct, "lat": round(lat, 3)}
        with self._lock:
            self._cache[key] = entry
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        resp = LLMResponse(text, pt, ct, False, lat)
        self.tracker.record(resp)
        return resp


def llm_from_config(cfg: dict, cache_path: Optional[Path] = None) -> LLM:
    tracker = CostTracker(cfg.get("price_in_per_mtok", 0.0), cfg.get("price_out_per_mtok", 0.0),
                          cfg.get("cost_cap_usd", 5.0), cfg.get("max_fresh_tokens"))
    prov = cfg["provider"]
    if prov == "ollama":
        backend = OllamaBackend(cfg["model"], cfg.get("base_url", "http://127.0.0.1:11434"),
                                cfg.get("num_ctx", 4096))
    elif prov == "openai":
        backend = OpenAICompatBackend(cfg["model"], cfg["base_url"], cfg.get("key_env", "LLM_API_KEY"))
    else:
        raise ValueError(f"unknown provider {prov!r}")
    return LLM(backend, tracker, cache_path)
