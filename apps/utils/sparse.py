"""Checkpoint and index paths for the sparse models."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def catalog():
    """Checkpoint and index for each sparse model in configs/sparse.yaml."""
    listed = yaml.safe_load((ROOT / "configs" / "sparse.yaml").read_text())
    found = {}
    for name, item in listed.items():
        local = ROOT / item["checkpoint"]
        found[name] = {
            "checkpoint": local if local.is_dir() else item["checkpoint"],
            "index": ROOT / item["index"],
            "directory": local,
        }
    return found
