"""Minimal .env loader.

Reads ``KEY=value`` lines from a ``.env`` file at the repository root the
first time AttrForge is imported, and populates ``os.environ`` *only for
keys that are not already set*. Shell exports always win.

Kept as a 30-line dependency-free helper so the runtime doesn't need
``python-dotenv``. Supports comments (``#`` prefix), blank lines, and
single- or double-quoted values. Does NOT support multi-line values or
shell expansion.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> int:
    """Populate ``os.environ`` from a ``.env`` file. Returns count loaded."""
    if path is None:
        # Search the repo root, which is two levels up from this file.
        here = Path(__file__).resolve().parent.parent
        candidates = [here / ".env", Path.cwd() / ".env"]
    else:
        candidates = [Path(path)]

    loaded = 0
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1
        break
    return loaded
