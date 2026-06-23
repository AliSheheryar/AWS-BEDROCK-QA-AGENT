"""AWS Bedrock LLM client for event extraction.

Drop-in replacement for gemini_client / openrouter_client — exposes the same
`.generate(prompt, temperature=...)` surface so the extraction pipeline can
swap backends without any other change.

Credentials resolve via the standard boto3 chain (env vars, ~/.aws/credentials,
instance profile, etc.). Region from AWS_DEFAULT_REGION or us-east-1.

Model defaults to Claude 3.5 Sonnet; override via BEDROCK_MODEL_ID env var.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
)
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


class BedrockError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=DEFAULT_REGION)
        return True
    except Exception:
        return False


class BedrockClient:
    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
    ):
        import boto3

        self.model = model or DEFAULT_MODEL
        self.region = region or DEFAULT_REGION
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_retries: int = 3,
        response_json: bool = True,
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.invoke_model(
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                result = json.loads(response["body"].read())
                return _extract_text(result)
            except self._client.exceptions.ThrottlingException:
                if attempt < max_retries:
                    logger.warning(
                        "Bedrock throttled (attempt %d/%d); backing off %.0fs",
                        attempt, max_retries, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise BedrockError("Bedrock: throttled after all retries")
            except Exception as exc:
                if attempt < max_retries and _is_retryable(exc):
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise BedrockError(f"Bedrock invoke failed: {exc}") from exc
        raise BedrockError("Bedrock: exhausted retries")

    def generate_with_image(
        self,
        prompt: str,
        image_b64: str,
        media_type: str = "image/png",
        *,
        temperature: float = 0.0,
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        response = self._client.invoke_model(
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return _extract_text(result)


def _extract_text(result: dict) -> str:
    try:
        content = result["content"]
        return "".join(block["text"] for block in content if block["type"] == "text")
    except (KeyError, IndexError, TypeError) as exc:
        raise BedrockError(f"Unexpected Bedrock response: {result}") from exc


def _is_retryable(exc: Exception) -> bool:
    exc_name = type(exc).__name__
    return any(
        keyword in exc_name
        for keyword in ("Throttling", "ServiceUnavailable", "InternalServer", "Timeout")
    )
