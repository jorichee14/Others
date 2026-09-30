# -*- coding: utf-8 -*-
"""
Configuration: ``configs/defaults.yaml`` overlaid by ``configs/sites/<site>.yaml``.

Everything a stage may read is in one merged dict. Frozen parameters (the
paper's Table "Pipeline parameters, frozen across sequences") live in
defaults.yaml; a site file holds only what differs per site: its bags, its
seed trajectories, and the choices a person makes after looking at the data
(floor/ceiling bounds, the anchor board).

Each stage reads ``cfg["stages"][<stage name>]`` plus any top-level sections it
declares in the registry (``uses``). Only those parts are hashed into the
stage's stamp, so editing one stage's parameters re-runs that stage and what
depends on it, and nothing else.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]           # scoop_pipeline/
CONFIG_DIR = ROOT / "configs"


class ConfigError(RuntimeError):
    pass


def deep_merge(base: Dict, over: Dict) -> Dict:
    """``over`` on top of ``base``; dicts merge recursively, anything else
    (lists included) is replaced whole."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path) -> Dict:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_config(site: str, config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """Merged config for one site. ``SCOOP_DATA`` overrides ``data_root``."""
    config_dir = Path(config_dir)
    cfg = _read(config_dir / "defaults.yaml")
    site_file = config_dir / "sites" / f"{site}.yaml"
    if not site_file.exists():
        have = sorted(p.stem for p in (config_dir / "sites").glob("*.yaml"))
        raise ConfigError(f"no config for site {site!r} at {site_file} "
                          f"(have: {have})")
    cfg = deep_merge(cfg, _read(site_file))
    cfg["site"] = site
    if os.environ.get("SCOOP_DATA"):
        cfg["data_root"] = os.environ["SCOOP_DATA"]
    if not cfg.get("data_root"):
        raise ConfigError("data_root is not set (defaults.yaml or $SCOOP_DATA)")
    cfg["data_root"] = str(Path(cfg["data_root"]).expanduser())
    cfg.setdefault("stages", {})
    cfg.setdefault("passes", {})
    return cfg


def stage_view(cfg: Dict, stage: str, uses: Iterable[str] = ()) -> Dict:
    """The part of the config a stage depends on: its own block plus the
    top-level sections it declared. This is what gets hashed and stamped."""
    view = {"stage": copy.deepcopy(cfg.get("stages", {}).get(stage, {}))}
    for key in uses:
        view[key] = copy.deepcopy(cfg.get(key))
    return view


def digest(obj: Any) -> str:
    """Stable sha256 of a JSON-able object (key order does not matter)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
