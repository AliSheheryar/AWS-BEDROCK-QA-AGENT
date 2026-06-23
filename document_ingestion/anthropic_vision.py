"""Cloud VLM backend: OCR a page image to Markdown with Claude (Anthropic API).

This is a drop-in alternative to the local `OllamaVisionClient` — it implements
the same `VLMClient` protocol (`transcribe(image_png: bytes) -> str`), so the
`OCREngine` / `IngestionPipeline` can use it without any other changes.

Unlike the Ollama backend this calls Anthropic's hosted models, so it is *not*
local — it needs network access and an API key. Credentials resolve the standard
SDK way: `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login`
profile). Nothing is hardcoded here.

The OCR instructions live in the (stable) system prompt with a cache breakpoint,
so they can be served from Anthropic's prompt cache across pages; the per-page
image is the only part that changes.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field

from .ocr import _VLM_PROMPT  # reuse the exact OCR instruction used for Ollama

logger = logging.getLogger(__name__)

# "Use sonnet" — see https://platform.claude.com models table. Bare alias, no date suffix.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


@dataclass
class AnthropicVisionClient:
    """VLM backend backed by Anthropic's hosted Claude models.

    Set the model via the constructor or the `DOCING_ANTHROPIC_MODEL` env var;
    defaults to Claude Sonnet 4.6.
    """

    model: str | None = None
    prompt: str = _VLM_PROMPT
    max_tokens: int = 8192
    _client: "object | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = os.getenv("DOCING_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        if self._client is None:
            import anthropic

            # Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ant profile.
            self._client = anthropic.Anthropic()

    def transcribe(self, image_png: bytes) -> str:
        image_b64 = base64.standard_b64encode(image_png).decode("ascii")

        # Stream: a dense page can produce long Markdown, and streaming avoids
        # the SDK's long-request HTTP timeout guard.
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self.prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": "Transcribe this page."},
                    ],
                }
            ],
        ) as stream:
            message = stream.get_final_message()

        text = "".join(block.text for block in message.content if block.type == "text")
        return text.strip()


def anthropic_vlm() -> AnthropicVisionClient:
    """Return the Anthropic-backed vision client."""
    return AnthropicVisionClient()
