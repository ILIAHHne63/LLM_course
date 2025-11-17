from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def load_env(env_path: Optional[Path] = None) -> None:
    """Load key=value pairs from .env if they are not already set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = env_path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        _ENV_LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    _ENV_LOADED = True
