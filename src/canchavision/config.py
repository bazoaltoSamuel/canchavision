"""Carga de configuración desde YAML."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
