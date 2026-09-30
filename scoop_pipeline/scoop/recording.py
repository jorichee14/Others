# -*- coding: utf-8 -*-
"""
One robot recording, from its raw bag to a GLIM trajectory, with every output
at a fixed place.

    data/raw/<rel>/                       as recorded -- only ever read
        <bag folder>/ (metadata.yaml + *.mcap), *.svo2, ...
    data/work/<rel>/                      everything derived from it
        <bag>_decoded/  1  packets -> points, sensor time    (scoop.replay)
        <bag>_retimed/  2  sensor -> capture time            (scoop.retime)
        glim/           3  GLIM dump: traj_lidar.txt, map    (scripts/run_glim.sh)
        clock.json         the clock fit of step 2 (drift, residual, ...)
        process.yaml       the settings the steps ran with

``<bag>`` is the original bag's folder name, so a derived bag says which
recording it came from wherever it is copied. ``<rel>`` is the recording's
path below ``raw/`` (e.g. 20260924/mapping_A/
mobile_1), so raw and work mirror each other and nothing derived is ever
written into ``raw/``.

A step writes into ``.partial/<its folder name>/`` and is moved into place only when it
succeeded, so a folder with a step's name is always complete. A step whose
output exists is skipped; ``redo`` deletes it and every later step.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import bag, replay, retime
from .bag import BagError

__all__ = ["STEPS", "Recording", "find_recording", "load_settings", "process"]

STEPS = ["decoded", "retimed", "glim"]
ROOT = Path(__file__).resolve().parents[1]                 # scoop_pipeline/
DEFAULT_SETTINGS = ROOT / "configs" / "recording.yaml"


@dataclass
class Recording:
    raw: Path                       # data/raw/<rel>
    bag: Path                       # the original rosbag2 folder inside it
    work: Path                      # data/work/<rel>

    def step(self, name: str) -> Path:
        """Output folder of a step: bags are named after the original bag
        (``<bag>_decoded``, ``<bag>_retimed``), the GLIM dump is ``glim``."""
        if name not in STEPS:
            raise ValueError(f"unknown step {name!r}; steps: {STEPS}")
        return self.work / (name if name == "glim" else f"{self.bag.name}_{name}")

    @property
    def clock_json(self) -> Path:
        return self.work / "clock.json"

    def done(self, name: str) -> bool:
        p = self.step(name)
        if name == "glim":
            return (p / "traj_lidar.txt").is_file()
        return (p / "metadata.yaml").is_file()

    def status(self) -> Dict[str, bool]:
        return {s: self.done(s) for s in STEPS}


def _bag_folders(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_dir() and (p / "metadata.yaml").is_file())


def find_recording(raw_dir, work_root=None, packets_ns: str = "/ouster") -> Recording:
    """The recording in ``raw_dir`` (a folder below ``.../raw/``). Its bag is
    the rosbag2 folder inside that carries ``<packets_ns>/lidar_packets``.
    ``work_root`` defaults to the ``work`` folder next to ``raw``."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise BagError(f"{raw_dir} is not a folder")
    parts = raw_dir.parts
    if "raw" not in parts:
        raise BagError(f"{raw_dir} is not below a 'raw' folder; pass work_root")
    i = len(parts) - 1 - parts[::-1].index("raw")          # the last 'raw'
    rel = Path(*parts[i + 1:]) if i + 1 < len(parts) else Path()
    work = (Path(work_root).expanduser().resolve() if work_root
            else Path(*parts[:i]) / "work") / rel

    topic = f"{packets_ns}/lidar_packets"
    candidates = [b for b in _bag_folders(raw_dir)
                  if topic in bag.open_bag(b).topics()]
    if not candidates:
        raise BagError(f"no bag with {topic} in {raw_dir} "
                       f"(bag folders: {[b.name for b in _bag_folders(raw_dir)]})")
    if len(candidates) > 1:
        raise BagError(f"several bags with {topic} in {raw_dir}: "
                       f"{[b.name for b in candidates]}")
    return Recording(raw_dir, candidates[0], work)


def load_settings(path=None) -> dict:
    with open(Path(path or DEFAULT_SETTINGS).expanduser()) as fh:
        s = yaml.safe_load(fh) or {}
    s.setdefault("ouster", {})
    s.setdefault("glim", {})
    if os.environ.get("GLIM_CONFIG"):
        s["glim"]["config"] = os.environ["GLIM_CONFIG"]
    return s


def _clear(p: Path):
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def _publish(tmp: Path, final: Path):
    """Move a finished step from .partial/ into place."""
    _clear(final)
    os.replace(tmp, final)
    try:
        tmp.parent.rmdir()                   # .partial/, once empty
    except OSError:
        pass


def process(rec: Recording, settings: dict, until: Optional[str] = None,
            redo: Optional[str] = None, glim_script: Optional[Path] = None,
            log=print) -> Dict[str, bool]:
    """Run the steps of ``rec`` that are not done yet, up to ``until``.
    ``redo`` first removes that step's output and every later one."""
    for name in (until, redo):
        if name is not None and name not in STEPS:
            raise ValueError(f"unknown step {name!r}; steps: {STEPS}")
    last = STEPS.index(until) if until else len(STEPS) - 1
    if redo:
        for s in STEPS[STEPS.index(redo):]:
            _clear(rec.step(s))
            if s == "retimed":
                _clear(rec.clock_json)
    rec.work.mkdir(parents=True, exist_ok=True)
    with open(rec.work / "process.yaml", "w") as fh:
        yaml.safe_dump({"raw_bag": str(rec.bag), **settings}, fh, sort_keys=False)

    o = settings["ouster"]
    packets_ns = o.get("packets_ns", "/ouster")
    out_ns = o.get("out_ns", "/mobile_1/ouster")
    for s in STEPS[:last + 1]:
        final = rec.step(s)
        tmp = rec.work / ".partial" / final.name   # same name, so the .mcap inside is too
        if rec.done(s):
            log(f"[skip] {s}: done ({final})")
            continue
        _clear(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        log(f"[run ] {s} -> {final}")
        if s == "decoded":
            replay.decode_ouster_bag(
                rec.bag, tmp, packets_ns=packets_ns, out_ns=out_ns,
                driver_min_range=float(o.get("driver_min_range", 0.0)),
                driver_max_range=float(o.get("driver_max_range", 1000.0)),
                frame=o.get("frame", "lidar"), imu=bool(o.get("imu", True)),
                remap=replay.parse_remaps(o.get("remap")),
                copy_all=bool(o.get("copy_all", False)), exclude=o.get("exclude") or (),
                log=log)
        elif s == "retimed":
            res = retime.retime_bag(rec.bag, rec.step("decoded"), tmp,
                                    packets_ns=packets_ns, points_ns=out_ns, log=log)
            res.clock.save(rec.clock_json)
            log(f"    clock fit -> {rec.clock_json}")
        elif s == "glim":
            g = settings["glim"]
            script = glim_script or ROOT / "scripts" / "run_glim.sh"
            env = dict(os.environ)
            if g.get("image"):
                env["GLIM_IMAGE"] = str(g["image"])
            cfg = str(Path(g.get("config", "~/workspaces/isaac_ros-dev/glim_config"))
                      .expanduser())
            tmp.mkdir()                          # made here, owned by you, empty
            code = subprocess.call([str(script), str(rec.step("retimed")), cfg, str(tmp)],
                                   env=env)
            if code != 0 or not (tmp / "traj_lidar.txt").is_file():
                raise BagError(f"GLIM did not produce {tmp / 'traj_lidar.txt'} "
                               f"(exit {code}); the partial run is left in {tmp}")
        _publish(tmp, final)
        log(f"[done] {s}")
    return rec.status()
