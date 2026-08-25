"""Config loading. Thin on purpose -- a dict is fine, I don't need pydantic
for eleven keys."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = {
    "seed": 7,
    "data": {
        "fs": 3200,
        "nominal_rpm": 1455.0,
        "duration_s": 4.0,
        "n_healthy": 220,
        "n_per_fault": 45,
    },
    "model": {"latent_dim": 8, "hidden": [32, 16], "dropout": 0.05, "l2": 1e-5},
    "train": {
        "epochs": 120,
        "batch_size": 64,
        "lr": 1e-3,
        "patience": 15,
        "target_fpr": 0.01,
    },
    "edge": {"debounce_n": 3, "debounce_m": 5, "spool_path": "artifacts/spool.jsonl"},
    "cloud": {
        "bucket": "CHANGEME-motor-anomaly",
        "prefix": "raw",
        "region": "eu-west-2",
        "role_arn": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        return DEFAULTS
    p = Path(path)
    if not p.exists():
        # Not fatal. Defaults are a working config; the yaml is a convenience.
        print(f"warning: {p} not found, using built-in defaults")
        return DEFAULTS
    return _deep_merge(DEFAULTS, yaml.safe_load(p.read_text()) or {})
