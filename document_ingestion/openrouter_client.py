"""Minimal OpenRouter REST client (for Llama-3-70B-Instruct extraction).

A drop-in alternative to gemini_client with the same `.generate(prompt,
temperature)` surface, so the extraction pipeline can swap models without any
other change. The API key is read from OPENROUTER_API_KEY and never written to
disk. Talks to the OpenAI-compatible chat-completions endpoint.

Model is configurable via OPENROUTER_MODEL (default: meta-llama/llama-3-70b-instruct).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-70b-instruct")


class OpenRouterError(RuntimeError):
    pass


def api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def is_available() -> bool:
    return bool(api_key())


class OpenRouterClient:
    def __init__(self, model: str | None = None, key: str | None = None):
        self.model = model or DEFAULT_MODEL
        self.key = key or api_key()
        if not self.key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_retries: int = 5,
        response_json: bool = True,
    ) -> str:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        # NB: Llama-3-70B via OpenRouter does NOT support a json_object
        # response_format, so we don't request one — the prompt already asks for
        # a JSON array and _extract_json_block parses raw/fenced output robustly.

        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
            # OpenRouter attribution headers (optional but recommended):
            "HTTP-Referer": "https://localhost/case-timeline",
            "X-Title": "case-timeline-ingestion",
        }
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(_ENDPOINT, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return _first_text(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code in (408, 429, 500, 502, 503) and attempt < max_retries:
                    logger.warning(
                        "OpenRouter HTTP %s (attempt %d/%d); backing off %.0fs",
                        exc.code, attempt, max_retries, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail[:300]}") from exc
            except (urllib.error.URLError, OSError) as exc:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
        raise OpenRouterError("OpenRouter: exhausted retries")


def _first_text(payload: dict) -> str:
    try:
        choices = payload["choices"]
        if not choices:
            raise OpenRouterError(f"OpenRouter returned no choices: {payload}")
        return choices[0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {payload}") from exc
