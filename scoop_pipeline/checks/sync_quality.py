#!/usr/bin/env python3
"""Do a robot's sensors agree on time, per frame? LiDAR, RGB cameras, depth
images and IMUs of one recording, each measured against the robot's own
motion, from the merged bag.

    python3 check.py sensors <bag> [--config <pipeline_config.json>] [--robot mobile_1 ...]
        [--out <dir>] [--span FROM TO] [--segment 60] [--max-shift 2] [--tol 0.03]
        [--dataset <OPV2V split dir>]

Two layers, both per stream:

  stamps   what the clocks say: the stream's rate, its jitter (p95 of the
           period's deviation), its gaps, and per LiDAR frame how far the
           nearest message of the stream is from the frame time (median and
           p95), i.e. what a frame built on the LiDAR grid pays to include it.
  content  what the data says: how fast the robot turns about the vertical,
           measured from the stream itself (an image's horizontal shift
           between frames over fx; a depth image's likewise; a scan's shift in
           azimuth; an IMU's |angular velocity|), lined up against the
           reference, the registered LiDAR trajectory's yaw rate (stage 04,
           from --config) or, without one, the LiDAR scans'. "lag" is how far
           the stream's stamps run behind the reference's: +30 ms means the
           stream's messages describe the world 30 ms before their stamps say.
           Measured over the whole run and per --segment seconds, so a lag
           that drifts shows as a spread.

A stream PASSES when its whole-run lag is within --tol of 0 and it correlates
with the reference (above 0.5) over a run in which the robot turns. IMUs are
compared as |w| against |yaw rate| (a cart turns about the vertical); images
and scans as signed rates. Writes <out>/sync_quality.json, .md and .png.
--dataset adds the per-frame cloud_dt and camera_dt the converter wrote
(ros_sync in every yaml), the realised offsets of the dataset as built.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import camera_lidar_sync as CL                                     # noqa: E402
from scoop import bag as B                                          # noqa: E402
from scoop import imu_sync                                          # noqa: E402

MIN_CORR, MIN_TURN = 0.5, 0.05


# ------------------------------------------------------------------- streams --
def discover(reader, robots=None):
    """{robot: [(name, kind, topic, info_topic)]}: every LiDAR, IMU, RGB and
    depth stream of each robot namespace that has a LiDAR."""
    out = {}
    for topic, info in sorted(reader.topics().items()):
        parts = topic.strip("/").split("/")
        if len(parts) < 2:
            continue
        robot, mt = parts[0], info.msgtype
        if robots and robot not in robots:
            continue
        low = topic.lower()
        if "PointCloud2" in mt and "ouster" in low and low.endswith("points"):
            out.setdefault(robot, []).append(("lidar", "lidar", topic, None))
        elif mt.endswith("msg/Imu"):
            name = "imu_" + ("ouster" if "ouster" in low else "zed" if "zed" in low else
                             "realsense" if parts[-1] == "imu" else parts[-2])
            out.setdefault(robot, []).append((name, "imu", topic, None))
        elif mt.endswith("msg/Image"):
            if "infra" in parts[-2:] and "infra" not in robot:
                continue                                   # IR pairs: not assessed
            kind = "depth" if "depth" in low else "rgb"
            name = kind + "_" + ("zed" if "zed" in low else "realsense" if robot == "mobile_2" or
                                 "color" in low or parts[-2] == "depth" else parts[-2])
            out.setdefault(robot, []).append((name, kind, topic, os.path.dirname(topic) + "/camera_info"))
    return {r: s for r, s in out.items() if any(k == "lidar" for _, k, _, _ in s)}


def first_stamp(reader, topic):
    return CL.first_stamp(reader, topic)


# -------------------------------------------------------------------- stamps --
def stamps_of(reader, topic):
    idx = reader.index([topic])[topic]
    return np.sort(np.asarray(idx.header_stamps, np.int64))


def stamp_stats(st):
    if len(st) < 3:
        return {"messages": int(len(st))}
    d = np.diff(st) * 1e-9
    per = float(np.median(d))
    gaps = d > 3 * per
    return {"messages": int(len(st)), "rate_hz": round(1 / per, 3) if per > 0 else None,
            "jitter_p95_ms": round(float(np.percentile(np.abs(d - per), 95)) * 1e3, 2),
            "gaps": int(gaps.sum()), "longest_gap_s": round(float(d.max()), 3),
            "span_s": round(float((st[-1] - st[0]) * 1e-9), 1)}


def grid_offsets(st, grid):
    """Per grid time, the nearest stamp's signed distance (ms)."""
    if not len(st) or not len(grid):
        return np.zeros(0)
    i = np.clip(np.searchsorted(st, grid), 1, len(st) - 1)
    lo, hi = st[i - 1], st[i]
    near = np.where(np.abs(grid - lo) <= np.abs(hi - grid), lo, hi)
    return (near - grid) * 1e-6


# ------------------------------------------------------------------- content --
def traj_yaw_rate(path, t0):
    """(s after t0, signed yaw rate rad/s) from a TUM trajectory."""
    from scoop.tftree import read_tum
    P = read_tum(path)
    t = np.array([p[0] for p in P], np.float64)
    yaw = np.unwrap(np.array([np.arctan2(p[1][1, 0], p[1][0, 0]) for p in P]))
    dt = np.diff(t) * 1e-9
    ok = (dt > 0) & (dt < 0.5)
    return ((t[1:] + t[:-1]) / 2 - t0)[ok] * 1e-9, (np.diff(yaw) / np.where(dt > 0, dt, 1))[ok]


def lidar_rate(reader, topic, t0, span, at_stamp=True):
    """(s, rad/s) from consecutive scans; timed at the scan STAMP (as the
    pipeline's trajectories are) or at the sweep's middle."""
    lo, hi = t0 + int((span[0] - 1) * 1e9), t0 + int((span[1] + 1) * 1e9)
    ts, rate, prev = [], [], None
    for sc in B.iter_scans(reader, topic, select=lambda s: 0 if lo <= s <= hi else None, min_points=50,
                           with_time=True, log=lambda *_: None):
        p = CL.lidar_profile(np.asarray(sc.xyz, np.float64))
        mid = sc.stamp_ns + (0 if at_stamp or sc.t is None or not len(sc.t)
                             else int(np.median(sc.t) * 1e9))
        if prev is not None:
            dt = (mid - prev[0]) * 1e-9
            if 0 < dt < 0.5:
                ts.append(((mid + prev[0]) / 2 - t0) * 1e-9)
                rate.append(-CL.shift_1d(prev[1], p) * 2 * np.pi / CL.BINS / dt)
        prev = (mid, p)
    return np.array(ts), np.array(rate)


def depth_profile(im):
    """log range per image column (the median over the rows that have a
    return), holes interpolated, de-meaned and tapered: a depth image as a
    range-per-azimuth profile, like a scan's."""
    ok = np.isfinite(im) & (im > 0)
    with np.errstate(invalid="ignore"):
        col = np.array([np.median(c[k]) if k.sum() >= 3 else np.nan for c, k in zip(im.T, ok.T)])
    good = np.isfinite(col)
    if good.sum() < 8:
        return None
    x = np.arange(len(col))
    col = np.interp(x, x[good], col[good])
    p = np.log1p(col)
    p = p - p.mean()
    return p * np.hanning(len(p))


def depth_rate(reader, topic, fx, t0, span):
    """(s, rad/s) from consecutive depth images: the horizontal shift of the
    image's range-per-column profile over fx. Phase correlation of the whole
    image, which works for RGB texture, fails on depth: planar walls make a
    smooth image whose whitened spectrum is noise and holes, so the 1-D
    profile (what a scan is for the LiDAR) is correlated instead."""
    lo, hi = t0 + int((span[0] - 1) * 1e9), t0 + int((span[1] + 1) * 1e9)
    ts, rate, prev = [], [], None
    for fr in B.iter_images(reader, topic, select=lambda s: 0 if lo <= s <= hi else None, mode="raw"):
        im = np.asarray(fr.image, np.float32)
        if im.ndim == 3:
            im = im[:, :, 0]
        p = depth_profile(im)
        if p is None:
            prev = None
            continue
        if prev is not None:
            dt = (fr.stamp_ns - prev[0]) * 1e-9
            if 0 < dt < 0.5:
                dx = CL.shift_1d(prev[1], p)              # p(x) ~ prev(x - dx): the scene moved right
                if abs(dx) < len(p) / 4:
                    ts.append(((fr.stamp_ns + prev[0]) / 2 - t0) * 1e-9)
                    rate.append(dx / fx / dt)
        prev = (fr.stamp_ns, p)
    return np.array(ts), np.array(rate)


def imu_rate(reader, topic, t0, span):
    t, w = imu_sync.angular_rates(reader, topic)
    t = (t - t0) * 1e-9
    k = (t >= span[0] - 1) & (t <= span[1] + 1)
    return t[k], w[k]


def fx_of(reader, info_topic):
    ci = CL.first_msg(reader, info_topic) if info_topic in reader.topics() else None
    if ci is None:
        return None
    return float(ci.p[0] or ci.k[0])


def lag(t_ref, x_ref, t, x, max_shift, segment, signed=True):
    """Whole-run and per-segment lag of (t, x) behind (t_ref, x_ref)."""
    if not signed:
        x_ref, x = np.abs(x_ref), np.abs(x)
    if len(t) < 10 or len(t_ref) < 10:
        return {"lag_s": None, "correlation": None, "counted": False, "segments": []}
    s, c, sc, n, turn = imu_sync.line_up(t_ref, x_ref, t, x, max_shift)
    segs = []
    a = max(t_ref[0], t[0])
    end = min(t_ref[-1], t[-1])
    while a + segment <= end + 1e-9 or (not segs and end - a > 2 * max_shift + 12):
        b = min(a + segment, end)
        k_r = (t_ref >= a) & (t_ref <= b)
        k = (t >= a - max_shift) & (t <= b + max_shift)
        if k_r.sum() > 20 and k.sum() > 20:
            ss, cc, _, nn, tt = imu_sync.line_up(t_ref[k_r], x_ref[k_r], t[k], x[k], max_shift)
            segs.append({"from_s": round(float(a), 1), "to_s": round(float(b), 1), "lag_s": round(ss, 3),
                         "correlation": round(cc, 3), "turn_rms": round(tt, 3),
                         "counted": bool(nn and cc >= MIN_CORR and tt >= MIN_TURN)})
        a = b
    counted = [g["lag_s"] for g in segs if g["counted"]]
    out = {"lag_s": round(s, 3), "correlation": round(c, 3), "scale": round(sc, 3), "samples": int(n),
           "turn_rms": round(turn, 3), "counted": bool(n and c >= MIN_CORR and turn >= MIN_TURN),
           "segments": segs}
    if counted:
        out["segment_lag_median_s"] = round(float(np.median(counted)), 3)
        out["segment_lag_spread_s"] = round(float(np.percentile(counted, 95) - np.percentile(counted, 5)), 3)
    return out


# -------------------------------------------------------------------- assess --
def trajectories(config):
    """{robot: tum} from the pass's stage 04 folder."""
    if not config:
        return {}
    from pipeline_common import load_pipeline
    P = load_pipeline(config)
    out = {}
    for f in glob.glob(os.path.join(P.reference_dir(), "traj_*_lidar.tum")):
        out[os.path.basename(f)[len("traj_"):-len("_lidar.tum")]] = f
    return out


def assess(bag, config=None, robots=None, trajs=None, span=None, segment=60.0, max_shift=2.0,
           tol=0.03, log=print):
    reader = B.open_bag(bag)
    streams = discover(reader, robots)
    if not streams:
        raise SystemExit("no robot with an Ouster in %s" % bag)
    trajs = dict(trajectories(config), **(trajs or {}))
    rep = {"bag": os.path.abspath(bag), "tol_s": tol, "segment_s": segment, "max_shift_s": max_shift,
           "robots": {}}
    for robot, items in streams.items():
        lidar_topic = next(tp for _, k, tp, _ in items if k == "lidar")
        t0 = first_stamp(reader, lidar_topic)
        grid = stamps_of(reader, lidar_topic)
        sp = span or (0.0, float((grid[-1] - t0) * 1e-9))
        gsel = grid[(grid >= t0 + sp[0] * 1e9) & (grid <= t0 + sp[1] * 1e9)]
        if robot in trajs:
            t_ref, x_ref = traj_yaw_rate(trajs[robot], t0)
            ref = {"kind": "trajectory", "path": trajs[robot]}
        else:
            t_ref, x_ref = lidar_rate(reader, lidar_topic, t0, sp)
            ref = {"kind": "lidar_scans", "topic": lidar_topic}
        log(f"{robot}: reference {ref['kind']}, {len(t_ref)} rates; LiDAR grid {len(gsel)} frames")
        R = {"reference": ref, "t0_ns": int(t0), "frames": int(len(gsel)), "streams": {}}
        for name, kind, topic, info in items:
            st = stamps_of(reader, topic)
            entry = {"kind": kind, "topic": topic, "stamps": stamp_stats(st)}
            off = grid_offsets(st, gsel)
            if len(off):
                entry["to_frame"] = {"median_abs_ms": round(float(np.median(np.abs(off))), 2),
                                     "p95_abs_ms": round(float(np.percentile(np.abs(off), 95)), 2),
                                     "mean_ms": round(float(off.mean()), 2)}
            try:
                if kind == "lidar":
                    t, x = lidar_rate(reader, topic, t0, sp, at_stamp=(ref["kind"] == "trajectory"))
                    signed = True
                elif kind == "imu":
                    t, x = imu_rate(reader, topic, t0, sp)
                    signed = False
                else:
                    fx = fx_of(reader, info)
                    if fx is None:
                        raise RuntimeError("no %s for fx" % info)
                    t, x = (CL.camera_rate if kind == "rgb" else depth_rate)(reader, topic, fx, t0, sp)
                    signed = True
                if kind == "lidar" and ref["kind"] == "lidar_scans":
                    entry["content"] = {"lag_s": 0.0, "correlation": 1.0, "counted": True, "segments": [],
                                        "note": "the reference itself"}
                else:
                    entry["content"] = lag(t_ref, x_ref, t, x, max_shift, segment, signed)
                entry["content"]["rates"] = int(len(t))
                entry["_trace"] = (t, x, signed)
            except Exception as exc:                                  # noqa: BLE001
                entry["content"] = {"error": str(exc), "counted": False, "lag_s": None, "correlation": None}
            c = entry["content"]
            entry["pass"] = bool(c.get("counted") and c.get("lag_s") is not None and abs(c["lag_s"]) <= tol)
            R["streams"][name] = entry
            log(f"  {name:<16} {fmt_row(entry)}")
        R["_ref"] = (t_ref, x_ref)
        rep["robots"][robot] = R
    rep["pass"] = all(e["pass"] for R in rep["robots"].values() for e in R["streams"].values()
                      if e["content"].get("counted"))
    return rep


def fmt_row(e):
    s, c, f = e["stamps"], e["content"], e.get("to_frame", {})
    lag_ = "n/a" if c.get("lag_s") is None else f"{c['lag_s'] * 1e3:+6.0f} ms"
    corr = "n/a" if c.get("correlation") is None else f"{c['correlation']:.2f}"
    spread = f" (segments {c['segment_lag_median_s'] * 1e3:+.0f} ms, spread {c['segment_lag_spread_s'] * 1e3:.0f})" \
        if "segment_lag_spread_s" in c else ""
    verdict = "PASS" if e["pass"] else ("not counted" if not c.get("counted") else "FAIL")
    return (f"{s.get('rate_hz', 0) or 0:6.1f} Hz  jitter p95 {s.get('jitter_p95_ms', 0):5.1f} ms  "
            f"gaps {s.get('gaps', 0):3d}  to frame |dt| median {f.get('median_abs_ms', 0):5.1f} / p95 "
            f"{f.get('p95_abs_ms', 0):5.1f} ms  lag {lag_}  corr {corr}{spread}  {verdict}")


def dataset_offsets(split_dir):
    """The converter's ros_sync per frame: {agent: {cloud_dt: stats, camera_dt: stats}}."""
    import yaml
    out = {}
    for ydir in sorted(glob.glob(os.path.join(split_dir, "*", "*"))):
        if not os.path.isdir(ydir):
            continue
        agent = os.path.basename(ydir)
        rows = {}
        for y in sorted(glob.glob(os.path.join(ydir, "*.yaml"))):
            p = yaml.safe_load(open(y)) or {}
            rs = p.get("ros_sync") or {}
            if "cloud_dt_ms" in rs:
                rows.setdefault("cloud_dt_ms", []).append(float(rs["cloud_dt_ms"]))
            for cam, v in (rs.get("camera_dt_ms") or {}).items():
                rows.setdefault(cam + "_dt_ms", []).append(float(v))
        if rows:
            out[agent] = {k: {"frames": len(v), "median_abs_ms": round(float(np.median(np.abs(v))), 2),
                              "p95_abs_ms": round(float(np.percentile(np.abs(v), 95)), 2),
                              "max_abs_ms": round(float(np.max(np.abs(v))), 2)}
                          for k, v in rows.items()}
    return out


# -------------------------------------------------------------------- output --
def markdown(rep):
    L = ["# Sensor sync quality", "", f"bag: `{rep['bag']}`  tolerance {rep['tol_s'] * 1e3:.0f} ms, "
         f"segments {rep['segment_s']:.0f} s", ""]
    for robot, R in rep["robots"].items():
        L += [f"## {robot}", "", f"reference: {R['reference']['kind']} "
              f"(`{R['reference'].get('path') or R['reference'].get('topic')}`), {R['frames']} LiDAR frames", "",
              "| stream | topic | rate Hz | jitter p95 ms | gaps | to frame median / p95 ms | lag ms | corr | segments median / spread ms | result |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for name, e in R["streams"].items():
            s, c, f = e["stamps"], e["content"], e.get("to_frame", {})
            lag_ = "n/a" if c.get("lag_s") is None else f"{c['lag_s'] * 1e3:+.0f}"
            corr = "n/a" if c.get("correlation") is None else f"{c['correlation']:.2f}"
            seg = (f"{c['segment_lag_median_s'] * 1e3:+.0f} / {c['segment_lag_spread_s'] * 1e3:.0f}"
                   if "segment_lag_spread_s" in c else "n/a")
            verdict = "PASS" if e["pass"] else ("not counted" if not c.get("counted") else "FAIL")
            L.append(f"| {name} | `{e['topic']}` | {s.get('rate_hz', '')} | {s.get('jitter_p95_ms', '')} | "
                     f"{s.get('gaps', '')} | {f.get('median_abs_ms', '')} / {f.get('p95_abs_ms', '')} | {lag_} | "
                     f"{corr} | {seg} | {verdict} |")
        L.append("")
    if rep.get("dataset"):
        L += ["## Dataset frames (ros_sync as written)", "",
              "| agent | field | frames | median abs ms | p95 abs ms | max abs ms |", "|---|---|---|---|---|---|"]
        for agent, fields in rep["dataset"].items():
            for k, v in fields.items():
                L.append(f"| {agent} | {k} | {v['frames']} | {v['median_abs_ms']} | {v['p95_abs_ms']} | {v['max_abs_ms']} |")
        L.append("")
    L += ["lag: the stream's stamps behind the reference (+ = its data is older than its stamp says). "
          "to frame: the nearest message's distance from each LiDAR frame time. "
          "IMUs compared as |w| against |yaw rate|; images and scans as signed yaw rates.", ""]
    return "\n".join(L)


def draw(rep, path, window=40.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    robots = list(rep["robots"])
    fig, axes = plt.subplots(len(robots), 2, figsize=(14, 3.6 * len(robots)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, robot in enumerate(robots):
        R = rep["robots"][robot]
        t_ref, x_ref = R["_ref"]
        if not len(t_ref):
            continue
        # the window where the robot turns most
        G = np.arange(t_ref[0], t_ref[-1], 0.02)
        a = np.abs(np.interp(G, t_ref, x_ref))
        n = max(1, int(window / 0.02))
        e = np.convolve(a, np.ones(n) / n, "same")
        c = G[int(np.argmax(e))]
        lo, hi = c - window / 2, c + window / 2
        ax = axes[row, 0]
        k = (t_ref >= lo) & (t_ref <= hi)
        ax.plot(t_ref[k], np.abs(x_ref[k]), "k", lw=1.5, label="reference |yaw rate|")
        names, lags, corrs = [], [], []
        for name, en in R["streams"].items():
            tr = en.get("_trace")
            cn = en["content"]
            if tr is None or cn.get("lag_s") is None:
                continue
            t, x, _ = tr
            k = (t + cn["lag_s"] >= lo) & (t + cn["lag_s"] <= hi)
            sc = cn.get("scale") or 1.0
            ax.plot(t[k] - cn["lag_s"], np.abs(x[k]) / (sc if sc > 0 else 1.0), lw=0.8, alpha=0.8,
                    label=f"{name} ({cn['lag_s'] * 1e3:+.0f} ms, r {cn['correlation']:.2f})")
            names.append(name)
            lags.append(cn["lag_s"] * 1e3)
            corrs.append(cn["correlation"])
        ax.set_title(f"{robot}: turn rate, each stream shifted by its lag")
        ax.set_xlabel("s after the first scan")
        ax.set_ylabel("rad/s")
        ax.legend(fontsize=7, loc="upper right")
        ax = axes[row, 1]
        y = np.arange(len(names))
        ax.barh(y, lags, color=["#2a7" if abs(v) <= rep["tol_s"] * 1e3 else "#c33" for v in lags])
        for yi, (v, r) in enumerate(zip(lags, corrs)):
            ax.text(v, yi, f" {v:+.0f} ms, r {r:.2f}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(-rep["tol_s"] * 1e3, rep["tol_s"] * 1e3, color="#2a7", alpha=0.12)
        ax.set_xlabel("lag behind the reference (ms)")
        ax.set_xlim(min(-60, min(lags) - 20) if lags else -60, max(60, max(lags) + 60) if lags else 60)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def strip(rep):
    out = json.loads(json.dumps({k: v for k, v in rep.items()}, default=lambda o: None))
    for R in out["robots"].values():
        R.pop("_ref", None)
        for e in R["streams"].values():
            e.pop("_trace", None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--config", help="the pass's pipeline config: its stage 04 trajectories are the reference")
    ap.add_argument("--robot", action="append", help="only these robots (default: every one with an Ouster)")
    ap.add_argument("--traj", action="append", default=[], metavar="ROBOT=TUM",
                    help="a reference trajectory by hand")
    ap.add_argument("--out", help="folder for sync_quality.{json,md,png} (default: next to the bag)")
    ap.add_argument("--span", nargs=2, type=float, metavar=("FROM", "TO"), help="seconds after the first scan")
    ap.add_argument("--segment", type=float, default=60.0)
    ap.add_argument("--max-shift", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--dataset", help="an OPV2V split folder: adds the converter's per-frame ros_sync")
    a = ap.parse_args()
    trajs = dict(x.split("=", 1) for x in a.traj)
    rep = assess(a.bag, a.config, a.robot, trajs, tuple(a.span) if a.span else None, a.segment,
                 a.max_shift, a.tol)
    if a.dataset:
        rep["dataset"] = dataset_offsets(a.dataset)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.bag.rstrip("/"))), "sync_quality")
    os.makedirs(out, exist_ok=True)
    draw(rep, os.path.join(out, "sync_quality.png"))
    with open(os.path.join(out, "sync_quality.json"), "w") as fh:
        json.dump(strip(rep), fh, indent=2)
    md = markdown(rep)
    with open(os.path.join(out, "sync_quality.md"), "w") as fh:
        fh.write(md)
    print("\n" + md)
    print("wrote %s/sync_quality.{json,md,png}" % out)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
