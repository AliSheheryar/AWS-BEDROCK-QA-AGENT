"""Thin HTTP client for a local Ollama instance.

Uses the stdlib only (urllib) so the pipeline has no cloud SDK dependency.
Talks to Ollama's native /api/chat endpoint, which accepts base64 images
on a message for vision models.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import OllamaConfig, load_config

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when the local Ollama instance cannot fulfil a request."""


@dataclass
class OllamaClient:
    config: OllamaConfig | None = None
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_config()

    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        model: str,
        images_png: list[bytes] | None = None,
        options: dict | None = None,
    ) -> str:
        """Send a single-turn chat request, return the assistant text.

        `images_png` are attached to the message for vision models.
        """
        message: dict = {"role": "user", "content": prompt}
        if images_png:
            message["images"] = [
                base64.b64encode(img).decode("ascii") for img in images_png
            ]

        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
        }
        if options:
            payload["options"] = options

        data = self._post("/api/chat", payload)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected Ollama response shape: {data}") from exc

    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        # Use streaming so each token keeps the socket alive, avoiding timeouts
        # during slow vision inference on large page images.
        payload = {**payload, "stream": True}
        url = f"{self.config.host}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            chunks: list[str] = []
            final: dict = {}
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        chunks.append(content)
                    if chunk.get("done"):
                        final = chunk
            # Reconstruct a non-streaming response shape
            final.setdefault("message", {})["content"] = "".join(chunks)
            return final
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise OllamaError(
                f"Ollama {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.host}. Is `ollama serve` "
                f"running? ({exc})"
            ) from exc
