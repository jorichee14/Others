#!/usr/bin/env python3
"""Several methods' trajectories on one pass, drawn as they unfold, in one video.

    python scoop_pipeline/slam/odometry/traj_video.py <run dir> [<run dir> ...] --out compare.mp4
        [--map <cloud or mesh, the reference's frame>] [--ref TUM] [--align fit|start|none]
        [--speed 10] [--fps 30] [--size 1920x1080] [--labels "2Fast-2Lamaa,FAST-LIO2,..."]

Each run is a folder run_odometry.sh wrote (run.json + trajectory.tum). Only the
trajectories are read: the reference is the one score.py recorded in the run's
accuracy.json, and each estimate is drawn in the frame its method reported
(FAST-LIO2's IMU, a camera for RTAB-Map: a fixed lever arm of a few cm to tens
of cm from the LiDAR). --use-bag reads the bag's /tf_static to put every estimate
in the LiDAR frame first, exactly as score.py scores it. Placement on the reference:

  fit    (default) the best rigid fit of the estimate's poses onto the reference
         (score.py's ATE); a run localized in the map (run.json "world": "map",
         2Fast-2Lamaa loc) is drawn as it is, with no fit, as a consumer sees it
  start  the estimate put on the reference at its first pose: drift as it builds up
  none   as each method reported it

Left: the plan view. The reference (stage 04's LiDAR track) in black, each
method in its own colour up to the current moment, its position a dot; under
them the map, binned into a grey floor plan when --map is given. Right: each
method's position error against the reference over time, on one axis, in
metres. The legend gives each method's score from accuracy.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as SC                                                  # noqa: E402

NAMES = {"fast2lamaa": "2Fast-2Lamaa", "fast2lamaa_loc": "2Fast-2Lamaa (in map)",
         "kiss_icp": "KISS-ICP", "rko_lio": "RKO-LIO", "fast_lio": "FAST-LIO2",
         "openvins": "OpenVINS", "rtabmap_rgbd": "RTAB-Map RGB-D", "rtabmap_loc": "RTAB-Map (in map)",
         "rtabmap_map": "RTAB-Map (mapping)", "cuvslam": "cuVSLAM"}
# the validated categorical palette (dataviz reference instance, light mode), fixed order
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def reference_of(run, rj, ref, use_bag):
    """The reference TUM: --ref, else the run's own record of it (run.json, then the one
    score.py used, in accuracy.json), else stage 04's found from the bag (use_bag only)."""
    cands = [ref, rj.get("ref")]
    acc = os.path.join(run, "accuracy.json")
    if os.path.exists(acc):
        cands.append(json.load(open(acc)).get("reference"))
    for c in cands:
        if c:
            for p in (c, os.path.join(os.getcwd(), c), os.path.join(run, c)):
                if os.path.exists(p):
                    return p
    if use_bag:
        return SC.default_ref(rj["bag"], rj["machine"])
    raise SystemExit(f"{run}: no reference recorded in run.json or accuracy.json (give --ref)")


def load_run(run, ref=None, align="fit", use_bag=False):
    """One run, on the reference: times, plan positions, error to the reference (m), label facts."""
    rj = json.load(open(os.path.join(run, "run.json")))
    tum = os.path.join(run, "trajectory.tum")
    if not os.path.exists(tum) or os.path.getsize(tum) == 0:
        raise SystemExit(f"{run}: no trajectory.tum")
    ref = reference_of(run, rj, ref, use_bag)
    if use_bag:                    # the estimate put in the LiDAR frame, as score.py does (reads /tf_static)
        paths = SC.MK.mc._paths(rj["bag"])
        T_fl, _ = SC.MK.to_lidar(paths, SC.MK.mc.build_tf_static(paths), rj["machine"], rj["tum_frame"])
    else:                          # as the method reported it: its frame, a fixed lever arm from the LiDAR
        T_fl = np.eye(4)
    t, T = SC.load_traj(tum)
    E = np.einsum("nij,jk->nik", T, T_fl)
    tr, Tr = SC.load_traj(ref)
    lo, hi = max(t[0], tr[0]), min(t[-1], tr[-1])
    m = (t >= lo) & (t <= hi)
    if m.sum() < 2:
        raise SystemExit(f"{run}: the estimate and the reference do not overlap in time")
    t, E = t[m], E[m]
    R = SC.trajs.interp(tr, Tr, t)
    in_map = rj.get("world") == "map"
    if align == "fit" and not in_map:
        Rf, tf = SC.pose_fit(E, R)
        P = E[:, :3, 3] @ np.asarray(Rf).T + np.asarray(tf).reshape(1, 3)
        how = "rigid fit"
    elif align == "start" and not in_map:
        A = R[0] @ np.linalg.inv(E[0])
        P = E[:, :3, 3] @ A[:3, :3].T + A[:3, 3]
        how = "start aligned"
    else:
        P = E[:, :3, 3]
        how = "in map, no fit" if in_map else "as reported"
    err = np.linalg.norm(P - R[:, :3, 3], axis=1)
    score_txt, kind, val = "", ("in map" if in_map else "ATE"), None
    acc = os.path.join(run, "accuracy.json")
    if os.path.exists(acc):
        a = json.load(open(acc))
        key = ("in_map", "position_cm") if in_map else ("ate", "position_cm")
        v = a.get(key[0], {}).get(key[1], {}).get("rmse")
        if v is not None:
            val = float(v)
            score_txt = f"{kind} {v:.1f} cm"
    return {"run": run, "method": rj.get("method", os.path.basename(run.rstrip("/"))),
            "machine": rj.get("machine", ""), "ref": ref, "t": t, "P": P, "err": err,
            "how": how, "score": score_txt, "kind": kind, "val": val}


def floor_plan(path, bounds, cell=0.1):
    """The map's points binned into a plan (log count), for imshow."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets"))
    from fly_overhead import read_xy
    xy = read_xy(path, 1)
    (x0, x1), (y0, y1) = bounds
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[max(2, int((x1 - x0) / cell)), max(2, int((y1 - y0) / cell))],
                             range=[[x0, x1], [y0, y1]])
    return np.log1p(H.T)


def video_writer(out, fps, size):
    if shutil.which("ffmpeg"):
        p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                              "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                             stdin=subprocess.PIPE)
        return (lambda rgb: p.stdin.write(np.ascontiguousarray(rgb).tobytes())), (lambda: (p.stdin.close(), p.wait()))
    raise SystemExit("ffmpeg is needed to write the video")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (run.json + trajectory.tum)")
    ap.add_argument("--out", required=True, help="the video (.mp4)")
    ap.add_argument("--ref", default=None, help="reference TUM for every run (default: each run's stage 04 reference)")
    ap.add_argument("--map", default=None, help="a cloud or mesh in the reference's frame, drawn as the floor plan")
    ap.add_argument("--align", choices=["fit", "start", "none"], default="fit")
    ap.add_argument("--speed", type=float, default=10.0, help="seconds of the pass per second of video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080", help="WxH")
    ap.add_argument("--labels", default="", help="comma-separated names, in the order of the runs")
    ap.add_argument("--title", default="", help="title above the plan view")
    ap.add_argument("--use-bag", action="store_true",
                    help="read each run's bag to put its estimate in the LiDAR frame exactly as score.py does "
                         "(default: the trajectories alone, each in the frame its method reported)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "font.size": 13, "axes.linewidth": 0.6})

    runs = [load_run(r, a.ref, a.align, a.use_bag) for r in a.runs]
    labels = [s.strip() for s in a.labels.split(",")] if a.labels else []
    # colour = the method (the same across robots), line style = the robot
    methods = list(dict.fromkeys(r["method"] for r in runs))
    machines = list(dict.fromkeys(r["machine"] for r in runs))
    STYLES = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (8, 2, 1, 2))]
    for k, r in enumerate(runs):
        name = labels[k] if k < len(labels) and labels[k] else NAMES.get(r["method"], r["method"])
        r["name"] = name
        r["colour"] = COLOURS[methods.index(r["method"]) % len(COLOURS)]
        r["style"] = STYLES[machines.index(r["machine"]) % len(STYLES)]
    for m in methods:                                   # one legend entry per method, its score per robot
        rs = [r for r in runs if r["method"] == m]
        vals = [r for r in rs if r["val"] is not None]
        if not vals:
            score = ""
        elif len(machines) > 1:
            score = f"  ({rs[0]['kind']}: " + ", ".join(
                f"{r['machine'].replace('mobile_', 'm')} {r['val']:.1f}" for r in vals) + " cm)"
        else:
            score = f"  ({vals[0]['kind']} {vals[0]['val']:.1f} cm)"
        rs[0]["label"] = rs[0]["name"] + score
        for r in rs[1:]:
            r["label"] = None
    refs = {}
    for r in runs:
        if r["ref"] not in refs:
            tr, Tr = SC.load_traj(r["ref"])
            refs[r["ref"]] = (tr, Tr[:, :3, 3])

    t0 = min(r["t"][0] for r in runs)
    t1 = max(r["t"][-1] for r in runs)
    allxy = np.concatenate([r["P"][:, :2] for r in runs] + [p[(t >= t0) & (t <= t1), :2] for t, p in refs.values()])
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.06 * (hi - lo).max() + 1.0
    bounds = ((lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad))

    W, H = (int(v) for v in a.size.lower().split("x"))
    W, H = W - W % 2, H - H % 2
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    axp = fig.add_axes([0.055, 0.07, 0.56, 0.84])
    axe = fig.add_axes([0.68, 0.50, 0.30, 0.34])
    axp.set_aspect("equal")
    axp.set_xlim(*bounds[0]); axp.set_ylim(*bounds[1])
    axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)")
    for s in axp.spines.values():
        s.set_linewidth(0.5)
    if a.map:
        img = floor_plan(a.map, bounds)
        axp.imshow(img, extent=(*bounds[0], *bounds[1]), origin="lower", cmap="Greys",
                   vmin=0, vmax=max(1e-9, np.percentile(img[img > 0], 99)) if (img > 0).any() else 1,
                   alpha=0.55, interpolation="nearest", zorder=0)
    ref_lines = []
    for k, (tr, pr) in enumerate(refs.values()):
        sel = (tr >= t0) & (tr <= t1)
        axp.plot(pr[sel, 0], pr[sel, 1], color="#c9c9c4", lw=0.8, zorder=1)
        ln, = axp.plot([], [], color="#1a1a19", lw=1.0, alpha=0.85, zorder=5,
                       label="Reference (stage 04 LiDAR)" if k == 0 else None)
        ref_lines.append((ln, tr, pr))
    tracks = []
    for r in runs:
        ln, = axp.plot([], [], color=r["colour"], ls=r["style"], lw=1.8, zorder=3)
        dot, = axp.plot([], [], "o" if r["style"] == "-" else "s", color=r["colour"], ms=8, mec="white", mew=1.5, zorder=4)
        el, = axe.plot([], [], color=r["colour"], ls=r["style"], lw=1.5)
        tracks.append((r, ln, dot, el))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#1a1a19", lw=1.0, label="Reference (stage 04 LiDAR)")]
    handles += [Line2D([], [], color=r["colour"], lw=2.4, label=r["label"]) for r in runs if r["label"]]
    if len(machines) > 1:
        handles += [Line2D([], [], color="#555550", lw=1.8, ls=STYLES[k % len(STYLES)],
                           marker="o" if k == 0 else "s", ms=6, label=m) for k, m in enumerate(machines)]
    axp.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 0.40), frameon=False, fontsize=13)
    hows = sorted({r["how"] for r in runs})
    fig.text(0.68, 0.075, "Estimates placed on the reference by: " + "; ".join(
        f"{h} ({', '.join(dict.fromkeys(r['name'] for r in runs if r['how'] == h))})" for h in hows),
        fontsize=11, color="#555550", wrap=True)
    emax = max(float(np.percentile(r["err"], 99.5)) for r in runs)
    axe.set_xlim(0, t1 - t0); axe.set_ylim(0, max(0.05, emax * 1.1))
    axe.set_xlabel("time (s)"); axe.set_ylabel("position error (m)")
    axe.set_title("Position error against the reference", fontsize=14, loc="left")
    axe.grid(True, color="#e3e3e0", lw=0.6); axe.set_axisbelow(True)
    for s in ("top", "right"):
        axe.spines[s].set_visible(False)
    cursor = axe.axvline(0, color="#9a9a9a", lw=0.8)
    title = fig.text(0.03, 0.955, "", fontsize=17, fontweight="bold")
    clock = fig.text(0.68, 0.92, "", fontsize=15)

    n = max(2, int(round((t1 - t0) / a.speed * a.fps)))
    write, release = video_writer(os.path.expanduser(a.out), a.fps, (W, H))
    print(f"{len(runs)} runs, {t1 - t0:.0f} s of the pass at {a.speed:g}x: {n} frames -> {a.out}", flush=True)
    for i in range(n):
        tn = t0 + (t1 - t0) * i / (n - 1)
        for ln, tr, pr in ref_lines:
            s = (tr >= t0) & (tr <= tn)
            ln.set_data(pr[s, 0], pr[s, 1])
        for r, ln, dot, el in tracks:
            s = r["t"] <= tn
            if s.any():
                ln.set_data(r["P"][s, 0], r["P"][s, 1])
                dot.set_data([r["P"][s][-1, 0]], [r["P"][s][-1, 1]])
                el.set_data(r["t"][s] - t0, r["err"][s])
            else:
                ln.set_data([], []); dot.set_data([], []); el.set_data([], [])
        cursor.set_xdata([tn - t0, tn - t0])
        title.set_text(a.title or "Trajectories against the reference")
        clock.set_text(f"t = {tn - t0:6.1f} s   ({a.speed:g}× speed)")
        fig.canvas.draw()
        write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if i % (10 * a.fps) == 0:
            print(f"  {i} of {n} frames", flush=True)
    release()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
