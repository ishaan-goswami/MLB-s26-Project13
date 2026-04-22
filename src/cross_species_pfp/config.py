from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in [
        "raw_dir",
        "processed_dir",
        "embeddings_dir",
        "models_dir",
        "search_dir",
        "eval_dir",
    ]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

