"""Local Ollama configuration.

The pipeline talks ONLY to a local Ollama instance — no cloud APIs. This
module resolves the endpoint and model names from (in order):

1. Environment variables.
2. A local JSON file at $DOCING_CONFIG_FILE or `./.docing.json`.
3. Built-in defaults.

Expected JSON shape (all keys optional):
    {
        "ollama_host":  "http://localhost:11434",
        "vision_model": "llama3.2-vision",
        "text_model":   "llama3.1"
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(".docing.json")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_VISION_MODEL = "llama3.2-vision"
DEFAULT_TEXT_MODEL = "llama3.1"


@lru_cache(maxsize=1)
def _load_file() -> dict:
    path = Path(os.getenv("DOCING_CONFIG_FILE", _DEFAULT_PATH))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read config file %s: %s", path, exc)
        return {}


def _resolve(env_name: str, file_key: str, default: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    return _load_file().get(file_key, default)


@dataclass(frozen=True)
class OllamaConfig:
    host: str = DEFAULT_HOST
    vision_model: str = DEFAULT_VISION_MODEL
    text_model: str = DEFAULT_TEXT_MODEL


def load_config() -> OllamaConfig:
    return OllamaConfig(
        host=_resolve("OLLAMA_HOST", "ollama_host", DEFAULT_HOST).rstrip("/"),
        vision_model=_resolve("DOCING_VISION_MODEL", "vision_model", DEFAULT_VISION_MODEL),
        text_model=_resolve("DOCING_TEXT_MODEL", "text_model", DEFAULT_TEXT_MODEL),
    )


def is_available(host: str | None = None, timeout: float = 2.0) -> bool:
    """Return True if a local Ollama instance answers at `host`."""
    import urllib.error
    import urllib.request

    host = (host or load_config().host).rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False
