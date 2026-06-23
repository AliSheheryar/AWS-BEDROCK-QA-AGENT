"""Minimal Google Gemini REST client (free tier).

Talks directly to the Generative Language API so no SDK dependency is
required. The API key is read from the GEMINI_API_KEY (or GOOGLE_API_KEY)
environment variable and is never written to disk.

Free-tier note: the API enforces per-minute request limits; `generate`
retries on HTTP 429/503 with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiError(RuntimeError):
    pass


def api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def is_available() -> bool:
    return bool(api_key())


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, key: str | None = None):
        self.model = model
        self.key = key or api_key()
        if not self.key:
            raise GeminiError("GEMINI_API_KEY is not set")

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_retries: int = 5,
        response_json: bool = True,
    ) -> str:
        url = _ENDPOINT.format(model=self.model) + f"?key={self.key}"
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if response_json:
            body["generationConfig"]["responseMimeType"] = "application/json"

        data = json.dumps(body).encode("utf-8")
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return _first_text(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code in (429, 500, 503) and attempt < max_retries:
                    wait = _retry_delay(detail) or delay
                    logger.warning(
                        "Gemini HTTP %s (attempt %d/%d); backing off %.0fs",
                        exc.code, attempt, max_retries, wait,
                    )
                    time.sleep(wait)
                    delay = min(delay * 2, 60)
                    continue
                raise GeminiError(f"Gemini HTTP {exc.code}: {detail[:300]}") from exc
            except (urllib.error.URLError, OSError) as exc:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise GeminiError(f"Gemini request failed: {exc}") from exc
        raise GeminiError("Gemini: exhausted retries")


def _retry_delay(detail: str) -> float | None:
    """Pull the server-suggested retry delay (seconds) from a 429 body."""
    try:
        err = json.loads(detail).get("error", {})
        for d in err.get("details", []):
            if d.get("@type", "").endswith("RetryInfo"):
                secs = d.get("retryDelay", "").rstrip("s")
                return float(secs) + 1.0 if secs else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def _first_text(payload: dict) -> str:
    try:
        candidates = payload["candidates"]
        if not candidates:
            raise GeminiError(f"Gemini returned no candidates: {payload}")
        parts = candidates[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected Gemini response shape: {payload}") from exc
