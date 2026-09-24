"""Loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}