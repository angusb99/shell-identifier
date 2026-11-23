"""Configuration loading: packaged defaults deep-merged with an optional user file."""

from __future__ import annotations

import copy
import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def default_config() -> dict[str, Any]:
    text = resources.files("shell_identifier").joinpath("default_config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_config(path: str | Path | None = None, weights_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config, overlay ``path`` (YAML) and then ``weights_path``.

    ``weights_path`` is the JSON written by ``shell-identifier train`` and holds
    ``{"intercept": float, "weights": {feature: float}}``.
    """
    cfg = default_config()
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if weights_path:
        with open(weights_path, encoding="utf-8") as fh:
            learned = json.load(fh)
        cfg["scoring"]["intercept"] = float(learned["intercept"])
        cfg["scoring"]["weights"] = _deep_merge(cfg["scoring"]["weights"], learned["weights"])
    return cfg
