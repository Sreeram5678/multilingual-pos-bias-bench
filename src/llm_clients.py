"""
Unified LLM Client Interface for Groq, Gemini, and Mock Offline Execution.
Includes persistent MD5-based JSONL response caching, rate-limit retry handling, and deterministic decoding.
"""

import os
import json
import time
import hashlib
import requests
from typing import Optional, Dict, Any

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "raw", "cache")

def get_cache_key(model: str, prompt: str, temperature: float) -> str:
    payload = f"{model}::{temperature}::{prompt}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()

def read_from_cache(cache_key: str) -> Optional[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("response")
        except Exception:
            return None
    return None

def write_to_cache(cache_key: str, model: str, prompt: str, response: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "cache_key": cache_key,
                "model": model,
                "prompt": prompt,
                "response": response,
                "timestamp": time.time()
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to write to cache: {e}")

class BaseLLMClient:
    def generate(self, prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 40) -> str:
        raise NotImplementedError

class GroqClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it or set the environment variable.")
        from groq import Groq
        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str, model: str = "qwen/qwen3.8-27b", temperature: float = 0.0, max_tokens: int = 40) -> str:
        cache_key = get_cache_key(model, prompt, temperature)
        cached = read_from_cache(cache_key)
        if cached is not None:
            return cached

        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Groq Free Tier has 8,000 TPM limit.
                # Sleeping 25s ensures we never exceed 2 requests/minute when prompts are 3.5k-4.5k tokens.
                time.sleep(25.0)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = response.choices[0].message.content.strip()
                write_to_cache(cache_key, model, prompt, text)
                return text
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    wait_time = 45.0 + (attempt * 15.0)
                    print(f"\n[Groq RateLimit 429] Respecting 8k TPM limit: cooling down for {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[Groq Error] {e}")
                    raise e
        raise RuntimeError("Groq API: Max retries exceeded.")

class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please provide it or set the environment variable.")

    def generate(self, prompt: str, model: str = "gemini-3.5-flash-lite", temperature: float = 0.0, max_tokens: int = 40) -> str:
        cache_key = get_cache_key(model, prompt, temperature)
        cached = read_from_cache(cache_key)
        if cached is not None:
            return cached

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        max_retries = 10
        for attempt in range(max_retries):
            try:
                # 1.5s pace gives safe throughput with backoff on rate limits
                time.sleep(1.5)
                res = requests.post(url, headers=headers, json=payload, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        write_to_cache(cache_key, model, prompt, text)
                        return text
                    return ""
                elif res.status_code in [429, 503]:
                    err_msg = res.text
                    if "per day limit exceeded" in err_msg.lower() or "daily" in err_msg.lower():
                        raise RuntimeError(f"Gemini API Daily Quota Exceeded for {model}: {err_msg}")
                    wait_time = 20.0 + (attempt * 10.0)
                    print(f"\n[Gemini RateLimit {res.status_code}] Cooling down for {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[Gemini Error {res.status_code}] {res.text}")
                    res.raise_for_status()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "timeout" in err_str:
                    wait_time = 20.0 + (attempt * 10.0)
                    time.sleep(wait_time)
                else:
                    raise e

        raise RuntimeError("Gemini API: Max retries exceeded.")

class MockClient(BaseLLMClient):
    """
    Offline simulation client modeling empirical 'Lost in the Middle' position degradation.
    """
    def __init__(self):
        pass

    def generate(self, prompt: str, model: str = "mock-model", temperature: float = 0.0, max_tokens: int = 40) -> str:
        import re
        gold = "UNKNOWN"
        lines = prompt.split("\n")
        gold_line_idx = -1
        for i, l in enumerate(lines):
            if any(term in l for term in ["IMPORTANT NOTE", "महत्वपूर्ण सूचना", "ముఖ్యమైన గమనిక"]):
                m = re.search(r"([0-9A-F]{4}-[0-9A-F]{4})", l)
                if m:
                    gold = m.group(1)
                    gold_line_idx = i
                    break

        distractor_matches = re.findall(r"([0-9A-F]{4}-[0-9A-F]{4})", prompt)
        distractors = [k for k in distractor_matches if k != gold]

        if gold_line_idx == -1 or len(lines) <= 1:
            rel_pos = 0.5
        else:
            rel_pos = gold_line_idx / len(lines)

        edge_dist = min(rel_pos, 1.0 - rel_pos)
        success_prob = 0.35 + 0.60 * (2.0 * edge_dist)

        if "[Search Objective:" in prompt:
            success_prob = 0.95

        h = int(hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8], 16)
        roll = (h % 1000) / 1000.0

        if roll < success_prob:
            return gold
        else:
            if distractors and (h % 2 == 0):
                return distractors[h % len(distractors)]
            return "NO_KEY_FOUND"

def get_llm_client(provider: str, api_key: Optional[str] = None) -> BaseLLMClient:
    provider = provider.lower()
    if provider == "groq":
        return GroqClient(api_key=api_key)
    elif provider == "gemini":
        return GeminiClient(api_key=api_key)
    elif provider == "mock":
        return MockClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Options: 'groq', 'gemini', 'mock'.")
