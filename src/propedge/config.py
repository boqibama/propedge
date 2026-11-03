"""Load settings from config.yaml and secrets from .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read config.yaml into a plain dict (cached after the first call)."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_secret(name: str) -> str | None:
    """Return a secret from the environment or .env, or None if it is unset/blank."""
    load_dotenv(ROOT / ".env", override=False)
    value = os.getenv(name, "").strip()
    return value or None


def resolve_path(relative: str) -> Path:
    """Turn a path from config.yaml (relative to the project root) into an absolute Path."""
    return ROOT / relative
