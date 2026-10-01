# -*- coding: utf-8 -*-
"""
One robot recording, from its raw bag to a GLIM trajectory and its ZED SVO to
a bag, with every output at a fixed place.

    data/raw/<rel>/                       as recorded -- only ever read
        <bag folder>/ (metadata.yaml + *.mcap), *.svo2, ...
    data/work/<rel>/                      everything derived from it
        <bag>_decoded/  1  packets -> points, sensor time    (scoop.replay)
        <bag>_retimed/  2  sensor -> capture time            (scoop.retime)
        glim/           3  GLIM dump: traj_lidar.txt, map    (scripts/run_glim.sh)
        <svo>_zed/      4  the SVO2 as bags, capture time    (scoop.zed)
        <svo>_zed_right/   one replay per bag (settings zed.bags)
        clock.json         the clock fit of step 2 (drift, residual, ...)
        process.yaml       the settings the steps ran with

``<bag>`` is the original bag's folder name (``<svo>`` the SVO's file name), so a derived bag says which
recording it came from wherever it is copied. ``<rel>`` is the recording's
path below ``raw/`` (e.g. 20260924/mapping_A/
mobile_1), so raw and work mirror each other and nothing derived is ever
written into ``raw/``.

A step writes into ``.partial/<its folder name>/`` and is moved into place only when it
succeeded, so a folder with a step's name is always complete. A step whose
output exists is skipped, and so is one whose outputs are only inputs to
steps that are all done (decoded can be deleted once retimed and glim exist;
``until`` that step builds it again). ``redo`` deletes a step and the steps
built from it (``LATER``): decoded -> retimed -> glim; zed stands alone.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from . import bag, replay, retime, zed
from .bag import BagError

__all__ = ["STEPS", "LATER", "Recording", "work_dir_for", "find_recording", "load_settings", "zed_bags",
           "zed_args", "process"]

STEPS = ["decoded", "retimed", "glim", "zed"]
LATER = {"decoded": ["retimed", "glim"], "retimed": ["glim"], "glim": [], "zed": []}
ROOT = Path(__file__).resolve().parents[1]                 # scoop_pipeline/
DEFAULT_SETTINGS = ROOT / "configs" / "recording.yaml"


@dataclass
class Recording:
    raw: Path                       # data/raw/<rel>
    bag: Optional[Path]             # the Ouster packets bag inside it (None: SVO only)
    work: Path                      # data/work/<rel>
    svo: Optional[Path] = None      # the ZED .svo2 inside raw, if there is one
    zed_bags: Tuple[str, ...] = ("zed",)   # the zed step's bags (settings zed.bags)

    def step(self, name: str) -> Path:
        """Output folder of a step: bags are named after their original
        (``<bag>_decoded``, ``<bag>_retimed``, ``<svo>_zed``), the GLIM dump
        is ``glim``. For zed, the first of its bags (:meth:`zed_bag`)."""
        if name not in STEPS:
            raise ValueError(f"unknown step {name!r}; steps: {STEPS}")
        if name == "glim":
            return self.work / "glim"
        if name == "zed":
            return self.zed_bag(self.zed_bags[0])
        if self.bag is None:
            raise BagError(f"no Ouster packets bag in {self.raw}: no {name} step")
        return self.work / f"{self.bag.name}_{name}"

    def zed_bag(self, name: str) -> Path:
        """``<svo>_<name>``, e.g. ``<svo>_zed``, ``<svo>_zed_right``."""
        return self.work / (f"{self.svo.stem}_{name}" if self.svo else name)

    @property
    def clock_json(self) -> Path:
        return self.work / "clock.json"

    def has(self, name: str) -> bool:
        """Whether the recording has what the step needs (packets bag, SVO)."""
        return self.svo is not None if name == "zed" else self.bag is not None

    def done(self, name: str) -> bool:
        if not self.has(name):
            return False
        if name == "zed":
            return all((self.zed_bag(n) / "metadata.yaml").is_file() for n in self.zed_bags)
        p = self.step(name)
        if name == "glim":
            return (p / "traj_lidar.txt").is_file()
        return (p / "metadata.yaml").is_file()

    def status(self) -> Dict[str, Optional[bool]]:
        """Step -> done; None where the recording lacks what the step needs
        (zed without an SVO, the Ouster steps without a packets bag)."""
        return {s: self.done(s) if self.has(s) else None for s in STEPS}


def _bag_folders(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_dir() and (p / "metadata.yaml").is_file())


def work_dir_for(raw_dir, work_root=None) -> Path:
    """The folder below ``work/`` that mirrors ``raw_dir`` (a folder below
    ``.../raw/``); ``work_root`` replaces that ``work`` folder."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise BagError(f"{raw_dir} is not a folder")
    parts = raw_dir.parts
    if "raw" not in parts:
        raise BagError(f"{raw_dir} is not below a 'raw' folder; pass work_root")
    i = len(parts) - 1 - parts[::-1].index("raw")          # the last 'raw'
    rel = Path(*parts[i + 1:]) if i + 1 < len(parts) else Path()
    return (Path(work_root).expanduser().resolve() if work_root
            else Path(*parts[:i]) / "work") / rel


def find_recording(raw_dir, work_root=None, packets_ns: str = "/ouster",
                   zed_bags=("zed",)) -> Recording:
    """The recording in ``raw_dir`` (a folder below ``.../raw/``). Its bag is
    the rosbag2 folder inside that carries ``<packets_ns>/lidar_packets``.
    ``work_root`` defaults to the ``work`` folder next to ``raw``;
    ``zed_bags`` are the names of the zed step's bags (:func:`zed_bags`)."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    work = work_dir_for(raw_dir, work_root)

    topic = f"{packets_ns}/lidar_packets"
    candidates = [b for b in _bag_folders(raw_dir)
                  if topic in bag.open_bag(b).topics()]
    svo = zed.find_svo(raw_dir)
    if not candidates and svo is None:
        raise BagError(f"no bag with {topic} and no .svo2 in {raw_dir} "
                       f"(bag folders: {[b.name for b in _bag_folders(raw_dir)]}, "
                       f"files: {sorted(p.name for p in raw_dir.iterdir() if p.is_file())})")
    if len(candidates) > 1:
        raise BagError(f"several bags with {topic} in {raw_dir}: "
                       f"{[b.name for b in candidates]}")
    return Recording(raw_dir, candidates[0] if candidates else None, work, svo,
                     tuple(zed_bags))


def load_settings(path=None) -> dict:
    with open(Path(path or DEFAULT_SETTINGS).expanduser()) as fh:
        s = yaml.safe_load(fh) or {}
    s.setdefault("ouster", {})
    s.setdefault("glim", {})
    s.setdefault("zed", {})
    if os.environ.get("GLIM_CONFIG"):
        s["glim"]["config"] = os.environ["GLIM_CONFIG"]
    return s


def zed_bags(z: dict) -> Dict[str, List[str]]:
    """Bag name -> topics, one ZED replay per bag: the ``zed.bags`` setting
    (or ``zed.topics`` as a single bag ``zed``)."""
    bags = z.get("bags") or {"zed": z.get("topics") or []}
    return {str(k): list(v or []) for k, v in bags.items()}


def zed_args(z: dict, topics: Optional[List[str]] = None) -> dict:
    """The ``zed`` settings as :func:`scoop.zed.svo_to_bag` arguments, for
    ``topics`` (default: the first bag's)."""
    topics = list(topics if topics is not None else next(iter(zed_bags(z).values())))
    rename = z.get("rename")
    remap = zed.prefix_remap(topics, *rename) if rename else {}
    configs = [ROOT / Path(c).expanduser() for c in z.get("wrapper_config") or ()]
    return dict(topics=topics, remap=remap, camera_model=z.get("camera_model", "zed2i"),
                params=z.get("params") or {}, realtime=bool(z.get("realtime", False)),
                wrapper_config=configs)


def _clear(p: Path):
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def _zed_outputs(rec: Recording, bags) -> List[Path]:
    """The zed step's bags, also those of an earlier bags setting
    (<svo>_zed_right after going back to one bag)."""
    out = [rec.zed_bag(n) for n in bags]
    if rec.svo is not None:
        out += [p for p in rec.work.glob(f"{rec.svo.stem}_zed_*")
                if p.is_dir() and (p / "metadata.yaml").is_file() and p not in out]
    return out


def _clear_leftovers(tmp: Path):
    """Remove what a failed run left in .partial/ for this output."""
    for p in (tmp, tmp.with_name(tmp.name + ".record"),
              tmp.with_name(tmp.name + ".params.yaml")):
        _clear(p)
    tmp.parent.mkdir(parents=True, exist_ok=True)


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
            zed_script: Optional[Path] = None, only: Optional[str] = None,
            log=print) -> Dict[str, Optional[bool]]:
    """Run the steps of ``rec`` that are not done yet, up to ``until`` (or
    just ``only``). ``redo`` first removes that step's output and the ones
    built from it."""
    for name in (until, redo, only):
        if name is not None and name not in STEPS:
            raise ValueError(f"unknown step {name!r}; steps: {STEPS}")
    last = STEPS.index(until) if until else len(STEPS) - 1
    bags = zed_bags(settings["zed"])
    rec.zed_bags = tuple(bags)
    if redo:
        for s in [redo] + LATER[redo]:
            if not rec.has(s):
                continue
            for p in (_zed_outputs(rec, bags) if s == "zed" else [rec.step(s)]):
                _clear(p)
            if s == "retimed":
                _clear(rec.clock_json)
    rec.work.mkdir(parents=True, exist_ok=True)
    with open(rec.work / "process.yaml", "w") as fh:
        yaml.safe_dump({"raw_bag": str(rec.bag), **settings}, fh, sort_keys=False)

    o = settings["ouster"]
    packets_ns = o.get("packets_ns", "/ouster")
    out_ns = o.get("out_ns", "/mobile_1/ouster")
    for s in ([only] if only else STEPS[:last + 1]):
        if not rec.has(s):
            log(f"[skip] {s}: " + (f"no .svo2 in {rec.raw}" if s == "zed" else
                                   f"no {packets_ns}/lidar_packets bag in {rec.raw}"))
            continue
        final = rec.step(s)
        tmp = rec.work / ".partial" / final.name   # same name, so the .mcap inside is too
        if rec.done(s):
            log(f"[skip] {s}: done ({final})")
            continue
        if not only and s != until and LATER[s] and all(rec.done(x) for x in LATER[s]):
            log(f"[skip] {s}: not needed, {' and '.join(LATER[s])} done")   # e.g. deleted
            continue
        if s == "zed":                           # one replay per bag, each published alone
            log(f"[run ] zed -> {', '.join(rec.zed_bag(n).name for n in bags)}")
            for name, topics in bags.items():
                fb = rec.zed_bag(name)
                if (fb / "metadata.yaml").is_file():
                    log(f"    {fb.name}: done")
                    continue
                tb = rec.work / ".partial" / fb.name
                _clear_leftovers(tb)
                log(f"    {fb.name}: {len(topics)} topics")
                zed.svo_to_bag(rec.svo, tb, script=zed_script, log=log,
                               **zed_args(settings["zed"], topics))
                _publish(tb, fb)
            log("[done] zed")
            continue
        _clear_leftovers(tmp)
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
