"""Bedrock VLM backend: OCR a page image to Markdown via AWS Bedrock.

Drop-in alternative to OllamaVisionClient and AnthropicVisionClient — implements
the same `VLMClient` protocol (`transcribe(image_png: bytes) -> str`) so the
OCREngine / IngestionPipeline can use it without any other changes.

Uses Claude 3.5 Sonnet on Bedrock by default. Override via BEDROCK_MODEL_ID env var.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field

from .ocr import _VLM_PROMPT
from .bedrock_client import BedrockClient, DEFAULT_MODEL, DEFAULT_REGION

logger = logging.getLogger(__name__)


@dataclass
class BedrockVisionClient:
    """VLM backend backed by AWS Bedrock (Claude 3.5 Sonnet)."""

    model: str | None = None
    prompt: str = _VLM_PROMPT
    _client: BedrockClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL)
        if self._client is None:
            self._client = BedrockClient(model=self.model)

    def transcribe(self, image_png: bytes) -> str:
        image_b64 = base64.standard_b64encode(image_png).decode("ascii")
        return self._client.generate_with_image(
            prompt="Transcribe this page.",
            image_b64=image_b64,
            media_type="image/png",
            temperature=0.0,
        ).strip()


def bedrock_vlm() -> BedrockVisionClient:
    """Return the Bedrock-backed vision client."""
    return BedrockVisionClient()
