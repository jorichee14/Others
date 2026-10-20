#!/usr/bin/env python3
"""Is a mapping pass's chain what it should be, each step once, nothing stale?

  python3 check.py chain <pipeline config>

Reads the pass's files only (no bag, no map is loaded) and checks, step by step:

  1 pose time   traj_lidar.txt against GLIM's own dump: the shift between
                them is the export's glim.pose_time_offset, applied once (not
                0, not twice); GLIM's own imu / points time offsets are 0, so
                no second correction ran inside GLIM
  2 BA          02's report: its input is the seed (traj_lidar.txt) and its
                output the reference; the reference is newer than its input
  3 keyframes   02's cache (in place or in _old/): made from the same input
                file as it is now (path and modification time in its key)
  4 time check  pose_time_check.json is newer than the reference; what is left
  5 maps        01's clouds of each chain (the map, the union, the seed map),
                each newer than the trajectory it was built from and in step
                order (merged, static, denoised, colored, final). 01 resumes
                from a cloud already on disk, so a cloud older than its
                trajectory is a map of earlier poses, reused
  6 02b         its trajectory is newer than the seed map it registered to
  7 report      newer than the reference and the maps
  then every trajectory and cloud in odometry/ and mapping/ that is in no step
  (an earlier version, a side run): not wrong, but not what the chain used.

OK / ! per line; ! says what to redo. Exit 1 when any ! .
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "map_stages"))
sys.path.insert(0, HERE)

from pipeline_common import load_pipeline, load_traj                     # noqa: E402

BAD = []


def say(ok, msg):
    print(("  OK  " if ok else "  !   ") + msg)
    if not ok:
        BAD.append(msg)


def mtime(p):
    return os.path.getmtime(p) if p and os.path.exists(p) else None


def rel(p, base):
    return os.path.relpath(p, base) if p else "-"


def shift_between(src, dst, span=0.2, step=0.002):
    """dt (s) with dst(t) = src(t - dt), from the positions while moving."""
    ts, Ts = load_traj(src)
    td, Td = load_traj(dst)
    ps, pd = Ts[:, :3, 3], Td[:, :3, 3]
    v = np.linalg.norm(np.gradient(pd, td, axis=0), axis=1)
    m = (v > 0.05) & (td > ts[0] + span) & (td < ts[-1] - span)
    if m.sum() < 50:
        return None
    t, p = td[m], pd[m]
    grid = np.arange(-span, span + step / 2, step)
    err = [np.sqrt(np.mean(np.sum((np.stack([np.interp(t - s, ts, ps[:, k]) for k in range(3)], 1) - p) ** 2, 1)))
           for s in grid]
    i = int(np.argmin(err))
    if 0 < i < len(grid) - 1:                       # parabola through the minimum
        a, b, c = err[i - 1], err[i], err[i + 1]
        den = a - 2 * b + c
        return float(grid[i] + (0.5 * step * (a - c) / den if den > 0 else 0.0)), float(err[i])
    return float(grid[i]), float(err[i])


def glim_yaml(odo):
    f = os.path.join(odo, "glim.yaml")
    out = {}
    if os.path.isfile(f):
        for ln in open(f):
            m = re.match(r"^(\w+):\s*(.*)$", ln.rstrip())
            if m:
                out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def glim_offsets(odo):
    """imu / points time offsets of the GLIM config the export copied."""
    out = {}
    for f in glob.glob(os.path.join(odo, "glim_config", "config_ros.json")):
        txt = re.sub(r"//.*", "", open(f).read())
        for k in ("imu_time_offset", "points_time_offset"):
            m = re.search(r'"%s"\s*:\s*([-+0-9.eE]+)' % k, txt)
            if m:
                out[k] = float(m.group(1))
    return out


def chain_files(P, kind):
    """01's clouds of one chain, in step order."""
    tag = P.tag
    if kind == "map":
        name = lambda b: P.pcd(b)                                         # noqa: E731
        steps = ("merged", "static", "denoised", "colored")
    else:
        name = lambda b: ("%s_%s_%s.pcd" % (b, kind, tag) if tag else "%s_%s.pcd" % (b, kind))  # noqa: E731
        steps = ("merged", "static", "denoised")
    files = [(b, P.outp(name(b))) for b in steps]
    if kind == "map":
        s = P.cfg.get("01_build_map") or {}
        files.append(("final", P.outp(s.get("output") or P.pcd("map_final"))))
    return [(b, f) for b, f in files if os.path.exists(f)]


def main_chain(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    a = ap.parse_args(argv)
    BAD.clear()
    P = load_pipeline(a.config)
    base = P.out_dir
    ba = P.cfg.get("02_bundle_adjust") or P.cfg.get("02b_bundle_adjust") or {}
    seed = P.outp(ba["input"]) if ba.get("input") else P.seed_traj()
    ref = P.outp(ba.get("output") or "traj_lidar_ba.txt")
    odo = os.path.dirname(P.outp("traj_lidar.txt"))
    used = {os.path.abspath(p) for p in (seed, ref)}
    print(f"chain of {P.tag or a.config}: {base}")
    print(f"  seed {rel(seed, base)}    reference {rel(ref, base)}    dataset.traj {P.dataset.get('traj')}")
    say(os.path.abspath(P.outp(P.dataset["traj"])) == os.path.abspath(ref),
        "the map is built from the reference (dataset.traj is 02's output)")

    print("\n1 pose time")
    gy = glim_yaml(odo)
    applied = float(gy.get("pose_time_offset_s") or 0.0)
    dump = os.path.join(gy["dump"], "traj_lidar.txt") if gy.get("dump") else P.glim_traj()
    if os.path.abspath(dump) != os.path.abspath(seed) and os.path.isfile(dump) and os.path.isfile(seed):
        r = shift_between(dump, seed)
        if r is None:
            say(False, "too little motion to measure the export's shift")
        else:
            dt, e = r
            say(abs(dt - applied) < 0.004,
                f"traj_lidar.txt is GLIM's dump moved {dt * 1000:+.1f} ms (applied {applied * 1000:+.0f} ms, "
                f"residual {e * 1000:.1f} mm)" + ("" if abs(dt - applied) < 0.004 else
                "; " + ("applied twice?" if abs(dt - 2 * applied) < 0.004 and applied else
                        "not applied?" if abs(dt) < 0.004 and applied else "not the export of this dump?")
                        + " re-export: process_recording.py export"))
    else:
        say(False, f"GLIM's dump trajectory not found ({dump}); cannot check the export's shift")
    off = glim_offsets(odo)
    say(all(abs(v) < 1e-9 for v in off.values()) if off else True,
        "GLIM's own time offsets: " + (", ".join(f"{k} {v:g}" for k, v in off.items()) or "config not copied")
        + ("" if all(abs(v) < 1e-9 for v in off.values()) else " -- a second correction inside GLIM"))

    print("\n2 bundle adjustment")
    rep = os.path.splitext(ref)[0] + "_report.json"
    if os.path.isfile(rep):
        r = json.load(open(rep))
        say(os.path.abspath(r.get("input", "")) == os.path.abspath(seed),
            f"02's input {rel(r.get('input'), base)} is the seed")
        say(os.path.abspath(r.get("output", "")) == os.path.abspath(ref), f"02's output is the reference")
        rounds = r.get("rounds") or []
        if rounds:
            x = rounds[-1]
            print(f"        {len(rounds)} rounds; last: {x.get('planes')} planes, keyframes moved median "
                  f"{x.get('moved_median_cm', float('nan')):.1f} cm, p95 {x.get('moved_p95_cm', float('nan')):.1f} cm")
    else:
        say(False, f"no 02 report beside the reference ({rel(rep, base)})")
    say(bool(mtime(ref)) and mtime(ref) > (mtime(seed) or 0), "the reference is newer than its input")

    print("\n3 keyframe cache")
    kc = os.path.splitext(ref)[0] + "_keyframes.npz"
    old = os.path.join(base, "_old", os.path.relpath(kc, base))
    kc = kc if os.path.isfile(kc) else old if os.path.isfile(old) else None
    if kc:
        key = json.loads(str(np.load(kc, allow_pickle=False)["key"]))
        same = (os.path.abspath(key.get("input", "")) == os.path.abspath(seed)
                and abs(key.get("mtime", 0) - (mtime(seed) or -1)) < 1e-3)
        say(same, f"{rel(kc, base)}: from {rel(key.get('input'), base)}"
            + ("" if same else " -- not the seed as it is now (02 would read the bag again)"))
    else:
        print("        no cache (in place or in _old/): nothing to check")

    print("\n4 pose-time check")
    tc = os.path.join(odo, "pose_time_check.json")
    if os.path.isfile(tc):
        say(mtime(tc) > (mtime(ref) or 0), "pose_time_check.json is newer than the reference")
        try:
            h = json.load(open(tc))
            h = h.get("checks", h) if isinstance(h, dict) else h
            txt = json.dumps(h)
            m = re.search(r'"dt_ms":\s*([-+0-9.eE]+)', txt)
            if m:
                print(f"        left after the correction: {float(m.group(1)):+.1f} ms")
        except ValueError:
            pass
    else:
        say(False, "no pose_time_check.json (check.py time)")

    print("\n5 maps (01 resumes from clouds on disk: each must be newer than its poses)")
    for kind, traj in (("map", ref), ("union", ref), ("seed", seed)):
        files = chain_files(P, kind)
        if not files:
            print(f"        {kind}: none")
            continue
        tt = mtime(traj) or 0
        prev = 0
        for b, f in files:
            used.add(os.path.abspath(f))
            t = mtime(f)
            ok = t > tt and t >= prev
            say(ok, f"{kind:<5} {b:<8} {rel(f, base)}" + ("" if t > tt else
                f" -- OLDER than {os.path.basename(traj)}: a map of earlier poses; delete it and run 01 again")
                + ("" if t >= prev else " -- older than the step before it"))
            prev = t

    print("\n6 scan-to-map (02b)")
    s2m = P.outp((P.cfg.get("02b_scan_to_map") or {}).get("output", "traj_lidar_refined.txt"))
    seed_map = [f for b, f in chain_files(P, "seed") if b == "denoised"]
    if os.path.isfile(s2m):
        used.add(os.path.abspath(s2m))
        say(not seed_map or mtime(s2m) > mtime(seed_map[0]), f"{rel(s2m, base)} is newer than the seed map")
    else:
        print("        not run")

    print("\n7 report")
    rp = os.path.join(base, "mapping_report_%s.md" % P.tag if P.tag else "mapping_report.md")
    if os.path.isfile(rp):
        newest = max([mtime(ref) or 0] + [mtime(f) for k in ("map", "union") for _, f in chain_files(P, k)])
        say(mtime(rp) > newest, f"{rel(rp, base)} is newer than the reference and the maps")
    else:
        print("        no report yet")

    print("\nnot in the chain (earlier versions, side runs; kept, not used):")
    loose = []
    for pat in (os.path.join(odo, "traj_*.txt"), os.path.join(os.path.dirname(P.outp(P.pcd("denoised"))), "*.pcd")):
        for f in sorted(glob.glob(pat)):
            if os.path.abspath(f) not in used:
                loose.append(f)
    for f in loose:
        print(f"        {rel(f, base)}")
    if not loose:
        print("        none")
    print(f"\n{len(BAD)} problem(s)" if BAD else "\nthe chain is consistent: every step once, nothing stale")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main_chain())
