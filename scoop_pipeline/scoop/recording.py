# -*- coding: utf-8 -*-
"""
One robot recording, from its raw bag to a GLIM trajectory and its ZED SVO to
a bag, with every output at a fixed place.

    data/raw/<rel>/                       as recorded -- only ever read
        <bag folder>/ (metadata.yaml + *.mcap), *.svo2, ...
    data/work/<rel>/                      everything derived from it
        <bag>_decoded/  1  packets -> points, sensor time    (scoop.replay)
        <bag>_retimed/  2  sensor -> capture time            (scoop.retime)
        glim/           3  GLIM dump: traj_lidar.txt, map    (processing/run_glim.sh)
        <svo>_zed/      4  the SVO2 as bags, capture time    (scoop.zed)
        <svo>_zed_right/   one replay per bag (settings zed.bags)
        clock.json         the clock fit of step 2 (drift, residual, ...)
        process.yaml       the settings the steps ran with
    data/processed/<date>/<pass>/odometry/<machine>/   what the map stages use of GLIM
        traj_lidar.txt  traj_imu.txt  glim_config/  glim.yaml     (:func:`export_glim`)

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

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from . import bag, replay, retime, zed
from .bag import BagError

__all__ = ["STEPS", "LATER", "Recording", "work_dir_for", "find_recording", "load_settings",
           "settings_for", "zed_bags", "export_glim", "glim_config_dir", "glim_loop_check",
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
    def odometry(self) -> Optional[Path]:
        """data/processed/<date>/<pass>/odometry/<machine>/ for a work folder
        data/work/<date>/<pass>/<machine>/ -- where the map stages look for this
        robot's trajectory first; None for another layout."""
        parts = self.work.parts
        if "work" not in parts:
            return None
        i = len(parts) - 1 - parts[::-1].index("work")
        rel = parts[i + 1:]
        if len(rel) < 3:
            return None
        return Path(*parts[:i]) / "processed" / rel[0] / rel[1] / "odometry" / rel[-1]

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
        if work_root:                  # e.g. straight from record.sh's Dataset/ folder
            return Path(work_root).expanduser().resolve() / raw_dir.name
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


def _deep(base, over):
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep(base.get(k), v) if k in base else v
    return out


def settings_for(settings: dict, machine: Optional[str]) -> dict:
    """The settings for one machine: the top level (mobile_1's) with that
    machine's block of ``machines:`` merged over it (dicts merged key by key,
    lists and values replaced). A machine without a block gets the top level."""
    base = {k: v for k, v in settings.items() if k != "machines"}
    over = (settings.get("machines") or {}).get(machine) if machine else None
    s = _deep(base, over or {})
    for k in ("ouster", "glim", "zed"):
        s.setdefault(k, {})
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


def glim_config_dir(settings: dict) -> Path:
    """GLIM's config folder: settings glim.config, relative to scoop_pipeline/."""
    return ROOT / Path(settings["glim"].get("config", "configs/glim/mobile_1")).expanduser()


def _traj_summary(path: Path) -> dict:
    """Poses, time span, rate and path length of a TUM trajectory."""
    import numpy as np
    a = np.loadtxt(path, ndmin=2)
    if a.size == 0:
        return {"poses": 0}
    t, xyz = a[:, 0], a[:, 1:4]
    span = float(t[-1] - t[0])
    return {"poses": int(len(a)), "t_start": float(t[0]), "t_end": float(t[-1]),
            "duration_s": round(span, 3),
            "rate_hz": round((len(a) - 1) / span, 3) if span > 0 else None,
            "path_length_m": round(float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()), 3)}


def export_glim(rec: "Recording", settings: dict, log=print) -> Optional[Path]:
    """GLIM's results into processed/<date>/<pass>/odometry/<machine>/, where
    the map stages read this robot's trajectory: traj_lidar.txt (the poses of
    the cloud's frame, TUM), traj_imu.txt when GLIM wrote one, glim_config/
    (the config it ran with) and glim.yaml (bag, config, image, and the
    trajectory's poses / span / rate / length). Its dump with the map stays
    in work/.../glim/. Returns the folder, or None for another layout."""
    out = rec.odometry
    if out is None:
        log(f"    (glim: {rec.work} is not data/work/<date>/<pass>/<machine>; not exported)")
        return None
    dump = rec.step("glim")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("traj_lidar.txt", "traj_imu.txt"):
        if (dump / name).is_file():
            shutil.copy2(dump / name, out / name)
            files[name] = _traj_summary(out / name)
    cfg = glim_config_dir(settings)
    if cfg.is_dir():
        _clear(out / "glim_config")
        shutil.copytree(cfg, out / "glim_config")
    info = {"bag": str(rec.step("retimed")), "dump": str(dump),
            "config": str(cfg), "image": settings["glim"].get("image"),
            "frame": "the points' frame (traj_lidar.txt: T_world_lidar, TUM: t x y z qx qy qz qw)",
            "files": files}
    with open(out / "glim.yaml", "w") as fh:
        yaml.safe_dump(info, fh, sort_keys=False)
    tl = files.get("traj_lidar.txt", {})
    log(f"    glim -> {out}: {tl.get('poses', 0)} poses over {tl.get('duration_s')} s "
        f"({tl.get('rate_hz')} Hz), {tl.get('path_length_m')} m")
    return out


def glim_loop_check(rec: "Recording", log=print, config=None) -> Optional[dict]:
    """GLIM's loop closure, measured on the boards: stage 05 in its read-only
    check mode (map_stages/05_anchor.py --traj --check) on GLIM's own
    trajectory, with this pass's map-stage config (the one whose dataset is
    this pass and this machine; ``config`` picks another). The report goes to
    odometry/<machine>/glim_loop_closure.json and its summary into glim.yaml:
    per board seen at the start and again later, how far apart GLIM puts it
    (mm / deg, against the detections' own noise). None when there is no
    such config, the export is missing, or the check fails (GLIM's result
    stands either way)."""
    import sys
    out = rec.odometry
    if out is None or not (out / "traj_lidar.txt").is_file():
        return None
    stages = ROOT / "map_stages"
    if config is None:
        sys.path.insert(0, str(stages))
        import pass_tf
        config = pass_tf.find_config(str(rec.raw.parent))
        if config is None:
            log(f"    glim loop closure: no map-stage config for {rec.raw.parent}; not checked")
            return None
        import json as _json
        with open(config) as fh:
            mach = (_json.load(fh).get("dataset") or {}).get("machine")
        if mach and mach != rec.raw.name:
            log(f"    glim loop closure: {Path(config).name} maps with {mach}, "
                f"not {rec.raw.name}; not checked")
            return None
    report = out / "glim_loop_closure.json"
    log(f"    glim loop closure: stage 05 on GLIM's trajectory ({Path(config).name}) ...")
    code = subprocess.call([sys.executable, str(stages / "05_anchor.py"), str(config),
                            "--traj", str(out / "traj_lidar.txt"), "--check", str(report)])
    if code != 0 or not report.is_file():
        log(f"    glim loop closure: the check failed (exit {code}); GLIM's result stands")
        return None
    with open(report) as fh:
        r = json.load(fh)
    boards = {}
    for name, b in (r.get("boards") or {}).items():
        lc = b.get("loop_closure")
        boards[name] = ({"sections": b["sections"], "from_section": lc["from_section"],
                         "to_section": lc["to_section"], "over_s": lc["dt_s"],
                         "mm": lc["mm"], "deg": lc["deg"], "noise_mm": lc["noise_mm"],
                         "significant": lc["significant"]} if lc else
                        {"sections": b["sections"], "mm": None})
    summary = {"report": str(report), "anchor_board": r.get("anchor_board"), "boards": boards,
               "cross_checks": [{k: c[k] for k in ("a", "b", "disagreement_mm",
                                                   "disagreement_deg", "significant")}
                                for c in r.get("cross_checks") or []]}
    info_p = out / "glim.yaml"
    info = yaml.safe_load(open(info_p)) if info_p.is_file() else {}
    info["loop_closure"] = summary
    with open(info_p, "w") as fh:
        yaml.safe_dump(info, fh, sort_keys=False)
    for name, b in boards.items():
        if b.get("mm") is None:
            log(f"    loop closure {name}: seen in {b['sections']} section(s), not revisited")
        else:
            log(f"    loop closure {name}: s{b['from_section']}->s{b['to_section']} over "
                f"{b['over_s']:.0f} s: {b['mm']:.1f} mm / {b['deg']:.3f} deg "
                f"(noise {b['noise_mm']:.1f} mm) -> "
                f"{'DRIFT' if b['significant'] else 'not significant'}")
    return summary


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
    until = until or settings.get("until")       # a machine's own last step (recording.yaml)
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
            if s == "glim" and rec.odometry is not None and \
                    not (rec.odometry / "glim.yaml").is_file():
                export_glim(rec, settings, log)          # done before the export existed
                glim_loop_check(rec, log)
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
            cals = []
            if o.get("static_tf"):                    # the robot's calibrations, into its bag
                from . import tftree
                with open(ROOT / Path(o["static_tf"]).expanduser()) as fh:
                    cals = tftree.load_calibrations(yaml.safe_load(fh))
            replay.decode_ouster_bag(
                rec.bag, tmp, packets_ns=packets_ns, out_ns=out_ns,
                driver_min_range=float(o.get("driver_min_range", 0.0)),
                driver_max_range=float(o.get("driver_max_range", 1000.0)),
                frame=o.get("frame", "lidar"), imu=bool(o.get("imu", True)),
                remap=replay.parse_remaps(o.get("remap")),
                copy_all=bool(o.get("copy_all", False)), exclude=o.get("exclude") or (),
                frame_suffix=o.get("frame_suffix") or "", static_tf=cals, log=log)
        elif s == "retimed":
            res = retime.retime_bag(rec.bag, rec.step("decoded"), tmp,
                                    packets_ns=packets_ns, points_ns=out_ns,
                                    imu_sync=o.get("imu_sync") or (), log=log)
            res.clock.save(rec.clock_json)
            log(f"    clock fit -> {rec.clock_json}")
            if res.imu_sync:
                with open(rec.work / "imu_sync.json", "w") as fh:
                    json.dump([r.as_dict() for r in res.imu_sync], fh, indent=1)
                log(f"    imu sync -> {rec.work / 'imu_sync.json'}")
        elif s == "glim":
            g = settings["glim"]
            script = glim_script or ROOT / "processing" / "run_glim.sh"
            env = dict(os.environ)
            if g.get("image"):
                env["GLIM_IMAGE"] = str(g["image"])
            cfg = str(glim_config_dir(settings))   # configs/glim/<machine>, or absolute
            tmp.mkdir()                          # made here, owned by you, empty
            code = subprocess.call([str(script), str(rec.step("retimed")), cfg, str(tmp)],
                                   env=env)
            if code != 0 or not (tmp / "traj_lidar.txt").is_file():
                raise BagError(f"GLIM did not produce {tmp / 'traj_lidar.txt'} "
                               f"(exit {code}); the partial run is left in {tmp}")
        _publish(tmp, final)
        log(f"[done] {s}")
        if s == "glim":
            export_glim(rec, settings, log)
            glim_loop_check(rec, log)
    return rec.status()
