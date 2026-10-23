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
READ_BY = {"decoded": ["retimed"], "retimed": ["glim"], "glim": [], "zed": []}   # direct readers
ROOT = Path(__file__).resolve().parents[1]                 # scoop_pipeline/
DEFAULT_SETTINGS = ROOT / "configs" / "recording.yaml"


@dataclass
class Recording:
    raw: Path                       # data/raw/<rel>
    bag: Optional[Path]             # the Ouster packets bag inside it (None: SVO only)
    work: Path                      # data/work/<rel>
    svo: Optional[Path] = None      # the ZED .svo2 inside raw, if there is one
    zed_bags: Tuple[str, ...] = ("zed",)   # the zed step's bags (settings zed.bags)
    packets_ns: str = "/ouster"     # where its bag has the packets (of ouster.packets_ns)

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


def find_recording(raw_dir, work_root=None, packets_ns="/ouster",
                   zed_bags=("zed",)) -> Recording:
    """The recording in ``raw_dir`` (a folder below ``.../raw/``). Its bag is
    the rosbag2 folder inside that carries ``<packets_ns>/lidar_packets``;
    ``packets_ns`` may be a list (a robot recorded under one name before,
    another since: mobile_1's /ouster, /mobile_1/ouster since 20261015), the
    one the bag has is the recording's ``packets_ns``. ``work_root`` defaults
    to the ``work`` folder next to ``raw``; ``zed_bags`` are the names of the
    zed step's bags (:func:`zed_bags`)."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    work = work_dir_for(raw_dir, work_root)

    nss = [packets_ns] if isinstance(packets_ns, str) else list(packets_ns)
    candidates = []                                        # (bag, its namespace)
    for b in _bag_folders(raw_dir):
        have = bag.open_bag(b).topics()
        ns = next((n for n in nss if f"{n}/lidar_packets" in have), None)
        if ns is not None:
            candidates.append((b, ns))
    topic = " or ".join(f"{n}/lidar_packets" for n in nss)
    svo = zed.find_svo(raw_dir)
    if not candidates and svo is None:
        raise BagError(f"no bag with {topic} and no .svo2 in {raw_dir} "
                       f"(bag folders: {[b.name for b in _bag_folders(raw_dir)]}, "
                       f"files: {sorted(p.name for p in raw_dir.iterdir() if p.is_file())})")
    if len(candidates) > 1:
        raise BagError(f"several bags with {topic} in {raw_dir}: "
                       f"{[b.name for b, _ in candidates]}")
    b, ns = candidates[0] if candidates else (None, nss[0])
    return Recording(raw_dir, b, work, svo, tuple(zed_bags), ns)


def load_settings(path=None) -> dict:
    """configs/recording.yaml: ``base:`` what every robot shares, ``machines:``
    each robot's own. The result's top level is ``default_machine``'s (base +
    its block), for readers that name no robot; :func:`settings_for` gives any
    robot's. A file without ``base:`` (the older layout) is its top level as
    written, ``machines:`` merged over it."""
    with open(Path(path or DEFAULT_SETTINGS).expanduser()) as fh:
        raw = yaml.safe_load(fh) or {}
    machines = raw.get("machines") or {}
    default = raw.get("default_machine") if "base" in raw else None
    shared = (raw.get("base") or {}) if "base" in raw else \
        {k: v for k, v in raw.items() if k not in ("machines", "default_machine")}
    s = _deep(shared, machines.get(default) or {}) if default else dict(shared)
    s["machines"] = machines
    if default:
        s["_shared"], s["_default"] = shared, default
    return _filled(s, True)


def _filled(s: dict, default: bool) -> dict:
    """The three sections there; $GLIM_CONFIG overrides the default machine's
    GLIM config (a machine with a config of its own keeps it)."""
    for k in ("ouster", "glim", "zed"):
        s.setdefault(k, {})
    if default and os.environ.get("GLIM_CONFIG"):
        s["glim"] = dict(s["glim"], config=os.environ["GLIM_CONFIG"])
    return s


def _deep(base, over):
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep(base.get(k), v) if k in base else v
    return out


def settings_for(settings: dict, machine: Optional[str]) -> dict:
    """The settings for one machine: the shared ones with that machine's block
    of ``machines:`` merged over them (dicts merged key by key, lists and values
    replaced). A machine without a block, or none, gets the default machine's
    (the older layout: the top level, the block merged over it)."""
    machines = settings.get("machines") or {}
    top = {k: v for k, v in settings.items() if k not in ("machines", "_shared", "_default")}
    if machine not in machines:
        return _filled(dict(top), True)
    s = _deep(settings.get("_shared", top), machines[machine])
    return _filled(s, machine == settings.get("_default"))


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


def shift_traj(src: Path, dst: Path, dt: float) -> None:
    """A TUM trajectory moved ``dt`` s later in time, on its own stamps: the
    pose written for stamp t is the input's pose at t - dt (spherical
    linear interpolation of the rotation, linear of the position; held at
    the first pose before it). Moving the stamps instead would leave every
    pose dt away from its scan, and the map stages pair a scan with a pose
    only within 01_build_map.time_tol (40 ms). Stamps are kept as text."""
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    rows = [ln.split() for ln in Path(src).read_text().splitlines()
            if ln.split() and not ln.lstrip().startswith("#")]
    stamps = [r[0] for r in rows]
    if len(rows) < 2:                        # nothing to interpolate between
        Path(dst).write_text("".join(" ".join(r) + "\n" for r in rows))
        return
    a = np.array([[float(x) for x in r] for r in rows])
    t0 = a[0, 0]
    t = a[:, 0] - t0                         # relative: no float rounding of 1.8e9 s
    q = np.clip(t - dt, t[0], t[-1])
    rot = Slerp(t, Rotation.from_quat(a[:, 4:8]))(q).as_quat()
    pos = np.stack([np.interp(q, t, a[:, k]) for k in (1, 2, 3)], 1)
    with open(dst, "w") as fh:
        for st, p_, r_ in zip(stamps, pos, rot):
            fh.write("%s %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (st, *p_, *r_))


def export_glim(rec: "Recording", settings: dict, log=print) -> Optional[Path]:
    """GLIM's results into processed/<date>/<pass>/odometry/<machine>/, where
    the map stages read this robot's trajectory: traj_lidar.txt (the poses of
    the cloud's frame, TUM), traj_imu.txt when GLIM wrote one, glim_config/
    (the config it ran with) and glim.yaml (bag, config, image, and the
    trajectory's poses / span / rate / length). Its dump with the map stays
    in work/.../glim/. Returns the folder, or None for another layout.

    glim.pose_time_offset (s, per machine): traj_lidar.txt's poses are
    moved by it in time, on GLIM's own stamps (shift_traj), so that the pose
    given for t is the scans' pose at t. On
    mobile_2 GLIM's pose for stamp t is the scans' pose at t + 46 ms
    (20261015: heading error against the BA reference follows the yaw rate
    as 45.8 ms, r 0.92; GLIM's scans are thinnest at +46 ms; unchanged with
    an IMU time offset in GLIM). The dump keeps GLIM's own stamps."""
    out = rec.odometry
    if out is None:
        log(f"    (glim: {rec.work} is not data/work/<date>/<pass>/<machine>; not exported)")
        return None
    dump = rec.step("glim")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    offset = float(settings["glim"].get("pose_time_offset") or 0.0)
    for name in ("traj_lidar.txt", "traj_imu.txt"):
        if (dump / name).is_file():
            if name == "traj_lidar.txt" and offset:
                shift_traj(dump / name, out / name, offset)
            else:
                shutil.copy2(dump / name, out / name)
            files[name] = _traj_summary(out / name)
    cfg = glim_config_dir(settings)
    if cfg.is_dir():
        _clear(out / "glim_config")
        shutil.copytree(cfg, out / "glim_config")
    info = {"bag": str(rec.step("retimed")), "dump": str(dump),
            "config": str(cfg), "image": settings["glim"].get("image"),
            "frame": "the points' frame (traj_lidar.txt: T_world_lidar, TUM: t x y z qx qy qz qw)",
            "pose_time_offset_s": offset,
            "files": files}
    with open(out / "glim.yaml", "w") as fh:
        yaml.safe_dump(info, fh, sort_keys=False)
    tl = files.get("traj_lidar.txt", {})
    log(f"    glim -> {out}: {tl.get('poses', 0)} poses over {tl.get('duration_s')} s "
        f"({tl.get('rate_hz')} Hz), {tl.get('path_length_m')} m"
        + (f"; poses moved {offset * 1000:+.1f} ms in time (glim.pose_time_offset)" if offset else ""))
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
    sys.path.insert(0, str(stages))
    import pipeline_common
    bag = pipeline_common.load_pipeline(str(config)).dataset.get("bag")
    if bag and not Path(bag).exists():
        # e.g. the merged bag (with the ZED's images), made after processing:
        # check.py report measures it then
        log(f"    glim loop closure: no {Path(bag).name} yet (the merge makes it); "
            "measured by the mapping report")
        return None
    report = out / "glim_loop_closure.json"
    log(f"    glim loop closure: stage 05 on GLIM's trajectory ({Path(config).name}) ...")
    sys.path.insert(0, str(ROOT / "analysis"))
    import trajs
    summary = trajs.loop_run(config, out / "traj_lidar.txt", report, log)
    if summary is None:
        log("    glim loop closure: not measured; GLIM's result stands")
        return None
    info_p = out / "glim.yaml"
    info = yaml.safe_load(open(info_p)) if info_p.is_file() else {}
    info["loop_closure"] = summary
    with open(info_p, "w") as fh:
        yaml.safe_dump(info, fh, sort_keys=False)
    for ln in trajs.loop_lines(summary):
        log("    " + ln)
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
    packets_ns = rec.packets_ns
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
        # not needed when every step that reads it is done (e.g. decoded, once
        # retimed exists -- also while glim is being redone)
        if not only and s != until and READ_BY[s] and all(rec.done(x) for x in READ_BY[s]):
            log(f"[skip] {s}: not needed, {' and '.join(READ_BY[s])} done")   # e.g. deleted
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
            remap = replay.parse_remaps(o.get("remap"))
            have = bag.open_bag(rec.bag).topics()       # renames of topics it does not have
            gone = sorted(t for t in remap if t not in have)
            if gone:
                log("    not in this recording, not renamed: %s" % ", ".join(gone))
                remap = {k: v for k, v in remap.items() if k in have}
            replay.decode_ouster_bag(
                rec.bag, tmp, packets_ns=packets_ns, out_ns=out_ns,
                driver_min_range=float(o.get("driver_min_range", 0.0)),
                driver_max_range=float(o.get("driver_max_range", 1000.0)),
                frame=o.get("frame", "lidar"), imu=bool(o.get("imu", True)),
                remap=remap,
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
