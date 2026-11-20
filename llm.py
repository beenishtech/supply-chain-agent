"""Thin LLM client (Gemini or Groq over plain REST) with a safe offline fallback.

The agents compute all numbers with tools. The LLM is used for reasoning text and for
parsing free-text disruption signals. If no key is set, or the API fails / is rate
limited, `complete()` returns None and the agents fall back to rule-based text.
"""
from __future__ import annotations

import os
import threading
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional
    pass


def _clean(key):
    if not key or key.strip().lower().startswith(("yahan", "your", "paste")):
        return None
    return key.strip()


class LLM:
    GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    def __init__(self, enabled=True, timeout=25):
        self.gemini_key = _clean(os.getenv("GEMINI_API_KEY"))
        self.groq_key = _clean(os.getenv("GROQ_API_KEY"))
        self.gemini_model = os.getenv("GEMINI_MODEL", self.GEMINI_FALLBACK_MODELS[0])
        # llama-3.3-70b-versatile was retired by Groq on 2026-08-16; override with GROQ_MODEL in .env if needed
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.timeout = timeout
        self.calls = 0
        self.failures = 0
        self._consecutive_failures = 0
        self.last_error = None
        self.disabled = not enabled
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", "6"))
        self._lock = threading.Lock()
        self._last_call = 0.0

    @property
    def provider(self):
        if self.disabled:
            return None
        if self.gemini_key:
            return "gemini"
        if self.groq_key:
            return "groq"
        return None

    @property
    def enabled(self):
        # circuit breaker: after 2 failures in a row stop calling for this run
        return self.provider is not None and self._consecutive_failures < 2

    def status(self):
        if self.provider is None:
            return "offline (no API key) - rule-based reasoning"
        model = self.gemini_model if self.provider == "gemini" else self.groq_model
        extra = f", last error: {self.last_error}" if self.last_error else ""
        return f"{self.provider}:{model} ({self.calls} calls, {self.failures} failures{extra})"

    def complete(self, system: str, prompt: str, max_tokens: int = 350):
        if not self.enabled:
            return None
        # One call at a time with a minimum gap: agents run in parallel but the free tier
        # allows only a few requests per minute (tune with LLM_MIN_INTERVAL in .env).
        with self._lock:
            if not self.enabled:
                return None
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self.calls += 1
            try:
                text = self._gemini(system, prompt, max_tokens) if self.provider == "gemini" \
                    else self._groq(system, prompt, max_tokens)
                self._consecutive_failures = 0
                return text.strip() if text else None
            except Exception as e:  # network, quota, bad model name, etc.
                self.failures += 1
                self._consecutive_failures += 1
                self.last_error = str(e)[:120]
                return None
            finally:
                self._last_call = time.time()

    # ---- providers ----
    def _gemini(self, system, prompt, max_tokens):
        models = [self.gemini_model] + [m for m in self.GEMINI_FALLBACK_MODELS if m != self.gemini_model]
        last = None
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            gen = {"temperature": 0.2, "maxOutputTokens": max(max_tokens, 800)}
            if "2.5-flash" in model:          # 2.5 "thinking" tokens would otherwise eat the answer budget
                gen["thinkingConfig"] = {"thinkingBudget": 0}
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gen,
            }
            r = requests.post(url, json=body, headers={"x-goog-api-key": self.gemini_key}, timeout=self.timeout)
            if r.status_code == 429:          # free-tier rate limit: wait once, retry once
                time.sleep(8)
                r = requests.post(url, json=body, headers={"x-goog-api-key": self.gemini_key}, timeout=self.timeout)
            if r.status_code == 404:          # model name not available -> try the next one
                last = f"model {model} not found"
                continue
            r.raise_for_status()
            self.gemini_model = model
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        raise RuntimeError(last or "no Gemini model available")

    def _groq(self, system, prompt, max_tokens):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.groq_key}"}
        body = {"model": self.groq_model, "temperature": 0.2, "max_tokens": max(max_tokens, 1000),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if "gpt-oss" in self.groq_model:
            body["reasoning_effort"] = "low"      # reasoning models: keep thinking short so the answer is not cut off
        r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if r.status_code == 400 and "reasoning_effort" in body:   # model/API does not accept it -> retry plain
            body.pop("reasoning_effort")
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if r.status_code == 429:                  # rate limit: wait once, retry once
            time.sleep(8)
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]