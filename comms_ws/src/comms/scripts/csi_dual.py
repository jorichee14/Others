#!/usr/bin/env python3
"""Two-platform CSI figure: both robots' channels on one shared time axis.

The RoboMNIST-style figure contrasts one platform moving against the same
platform still, which works when the still period is the control. Here BOTH
platforms move, so that contrast is not available and the figure has to do
something else: show the two channels against each other and against the
ground-truth motion that supposedly drives them.

Layout, all sharing one time axis:

    (a) mobile_1 CSI amplitude          time x subcarrier, |H|
    (b) mobile_2 CSI amplitude          same, same colour scale
    (c) speed of both platforms         from LiDAR pose
    (d) channel change rate             1 - corr(|H|) between consecutive
                                        frames, the CSI-side counterpart of (c)

Panels (c) and (d) are the point. A channel measured from a moving transmitter
must change faster when that transmitter moves faster and go quiet when it
stops. If (d) tracks (c), the CSI is measuring the robots. If (d) is flat
while (c) varies, it is measuring the receiver, and no downstream metric means
anything. With two platforms this is a stronger test than with one, because
each robot's channel should follow ITS OWN speed and not the other's -- a
receiver artefact would show up in both identically.

Stationary stretches are shaded, detected from pose rather than typed in, so
the segmentation costs nothing during collection.

    python3 csi_dual.py --bag ./session_bag
    python3 csi_dual.py --bag ./bag --window 0 60 --frames ba

Frames are split on `seq` (0xFFFF = Block Ack), never on `frame_control`:
that field holds the first byte of the PPDU, an MPDU delimiter for aggregated
data, so data frames never appear as 0x88.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_analysis import slot_classes, stack, CONTROL_FC   # noqa: E402

BA_SEQ = 0xFFFF
GRID = "#d8d5d0"
TEXT = "#2b2926"
COLOR = {"mobile_1": "#1f4e9c", "mobile_2": "#e08214"}


def read_bag(bag: Path, csi_topics: dict, odom_topics: dict):
    """CSI rows per robot, plus (t, x, y) pose arrays per robot."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise SystemExit("needs: pip install rosbags")
    csi = {k: [] for k in csi_topics}
    pose = {k: [] for k in odom_topics}
    want = {v: k for k, v in csi_topics.items()}
    want_p = {v: k for k, v in odom_topics.items()}
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections
                 if c.topic in want or c.topic in want_p]
        if not conns:
            have = sorted({c.topic for c in reader.connections})
            raise SystemExit("none of the expected topics are in the bag.\n"
                             f"looked for: {sorted(want) + sorted(want_p)}\n"
                             f"present: {have}")
        for conn, ts, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            if conn.topic in want:
                csi[want[conn.topic]].append({
                    "t_ns": ts, "seq": int(m.seq), "rssi": int(m.rssi),
                    "frame_control": int(m.frame_control),
                    "src_mac": str(m.src_mac), "bw": int(m.bandwidth_mhz),
                    "idx": np.asarray(m.subcarrier_index, dtype=np.int32),
                    "re": np.asarray(m.csi_real, dtype=np.float64),
                    "im": np.asarray(m.csi_imag, dtype=np.float64)})
            else:
                # Odometry carries pose.pose; PoseStamped carries pose.
                pz = m.pose
                p = pz.pose.position if hasattr(pz, "pose") else pz.position
                pose[want_p[conn.topic]].append((ts, p.x, p.y))
    pose = {k: np.array(v, dtype=np.float64) for k, v in pose.items() if v}
    return csi, pose


def speed_from_pose(P: np.ndarray, t0_ns: int, smooth_s: float = 0.5):
    """(t_s, speed) from an (N,3) array of (t_ns, x, y)."""
    if P is None or len(P) < 3:
        return None, None
    o = np.argsort(P[:, 0])
    t, xy = P[o, 0], P[o, 1:3]
    dt = np.diff(t) / 1e9
    ok = dt > 1e-6
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1)[ok] / dt[ok]
    tm = (0.5 * (t[:-1] + t[1:]))[ok]
    rate = len(v) / max((t[-1] - t[0]) / 1e9, 1e-9)
    n = max(int(smooth_s * rate), 1)
    v = pd.Series(v).rolling(n, center=True, min_periods=1).median().to_numpy()
    return (tm - t0_ns) / 1e9, v


def change_rate(absH: np.ndarray, t_s: np.ndarray, smooth_s: float = 0.5):
    """1 - corr(|H|) between consecutive frames: how fast the channel moves."""
    if absH.shape[0] < 3:
        return None, None
    a, b = absH[:-1], absH[1:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)
    tm = 0.5 * (t_s[:-1] + t_s[1:])
    rate = len(r) / max(t_s[-1] - t_s[0], 1e-9)
    n = max(int(smooth_s * rate), 1)
    ch = pd.Series(1.0 - r).rolling(n, center=True, min_periods=1).median()
    return tm, ch.to_numpy()


def still_spans(t: np.ndarray, v: np.ndarray, thr: float, min_s: float = 2.0):
    """Contiguous stretches below `thr` lasting at least `min_s`."""
    if t is None:
        return []
    m = v < thr
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                out.append((t[i], t[j]))
            i = j + 1
        else:
            i += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("csi_dual.png"))
    ap.add_argument("--frames", choices=["ba", "data", "all"], default="ba",
                    help="ba is the dense stream (~116 Hz) and the right "
                         "choice for a figure; data is ~8.5 Hz and too sparse "
                         "to see structure in")
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    metavar=("T0", "T1"), help="seconds to plot")
    ap.add_argument("--still-mps", type=float, default=0.05)
    ap.add_argument("--panels", choices=["amp", "full"], default="amp",
                    help="amp = just the two amplitude heat maps with the "
                         "motion phases marked (the figure to publish). "
                         "full = adds speed and channel-change-rate traces "
                         "(diagnostic: use it to check the CSI follows the "
                         "robots before trusting the amplitude panels)")
    ap.add_argument("--robots", nargs=2, default=["mobile_1", "mobile_2"])
    ap.add_argument("--csi-topics", nargs=2,
                    default=["/mobile1/csi", "/mobile2/csi"])
    ap.add_argument("--odom-topics", nargs=2,
                    default=["/mobile_1/global_pose", "/mobile_2/global_pose"],
                    help="Odometry or PoseStamped, one per robot")
    args = ap.parse_args(argv)

    csi_t = dict(zip(args.robots, args.csi_topics))
    odom_t = dict(zip(args.robots, args.odom_topics))
    csi_rows, pose = read_bag(args.bag, csi_t, odom_t)
    for r, tp in odom_t.items():
        n = len(pose.get(r, []))
        print(f"{r}: {n} poses from {tp}" if n else
              f"{r}: NO poses from {tp} -- stationary shading will be missing",
              file=sys.stderr)
    t0 = min(r[0]["t_ns"] for r in csi_rows.values() if r)
    return render(csi_rows, pose, t0, args)


def render(csi_rows, pose, t0_ns, args):
    per = {}
    for robot in args.robots:
        rows = csi_rows.get(robot) or []
        H, idx, meta = stack(rows)
        if H is None:
            print(f"{robot}: no frames", file=sys.stderr)
            continue
        if args.frames != "all":
            m = meta["is_ba"].to_numpy()       # frame_control-based, not seq
            if args.frames == "data":
                m = ~m
            if m.sum() < 5:
                print(f"{robot}: too few '{args.frames}' frames",
                      file=sys.stderr)
                continue
            H, meta = H[m], meta[m].reset_index(drop=True)
        amp = np.abs(H)
        # zero frames out first -- they break the frozen-slot test
        good = (amp <= 0).mean(axis=1) < 0.5
        if (~good).any() and good.sum() >= 5:
            H, amp, meta = H[good], amp[good], meta[good].reset_index(drop=True)
        usable, artefact, null, outband = slot_classes(amp)
        Hg, idxg = H[:, usable], idx[usable]
        n_excl = int((~usable).sum())
        t = (meta["t_ns"].to_numpy() - t0_ns) / 1e9
        pt, pv = speed_from_pose(pose.get(robot), t0_ns)
        ct, cv = change_rate(np.abs(Hg), t)
        per[robot] = dict(H=Hg, idx=idxg, t=t, n_bad=n_excl,
                          n_out=int(outband.sum()), n_null=int(null.sum()),
                          n_art=int(artefact.sum()),
                          rate=(len(t) - 1) / max(t[-1] - t[0], 1e-9),
                          pt=pt, pv=pv, ct=ct, cv=cv)
    if not per:
        raise SystemExit("nothing to plot")

    lo = args.window[0] if args.window else min(p["t"][0] for p in per.values())
    hi = args.window[1] if args.window else max(p["t"][-1] for p in per.values())

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "text.color": TEXT})
    names = [r for r in args.robots if r in per]
    extra = 0 if getattr(args, "panels", "full") == "amp" else 2
    fig, axes = plt.subplots(
        len(names) + extra, 1,
        figsize=(11, 3.0 * len(names) + 2.3 * extra + 0.8),
        sharex=True, squeeze=False,
        gridspec_kw=dict(height_ratios=[2.0] * len(names) + [1.0] * extra,
                         hspace=0.30))
    axes = axes[:, 0]
    # One colour scale across both platforms, or the panels cannot be
    # compared. From finite values only: a zero slot is -240 dB and would
    # otherwise own the scale.
    allv = np.concatenate([20 * np.log10(np.maximum(np.abs(p["H"]), 1e-12)).ravel()
                           for p in per.values()])
    allv = allv[allv > -100]
    vlo, vhi = np.percentile(allv, [1, 99])

    spans = []
    for r in names:
        p = per[r]
        spans += still_spans(p["pt"], p["pv"], args.still_mps) if p["pt"] is not None else []

    for i, r in enumerate(names):
        ax, p = axes[i], per[r]
        m = (p["t"] >= lo) & (p["t"] <= hi)
        img = 20 * np.log10(np.maximum(np.abs(p["H"][m]), 1e-12))
        step = max(img.shape[0] // 1500, 1)
        im = ax.imshow(img[::step].T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vlo, vmax=vhi,
                       extent=[p["t"][m][0], p["t"][m][-1],
                               p["idx"][0] - 0.5, p["idx"][-1] + 0.5])
        ax.set_ylabel("subcarrier")
        ax.set_title(f"({chr(97+i)}) {r}: CSI amplitude — "
                     f"{p['H'].shape[0]} frames at {p['rate']:.0f} Hz, "
                     f"{p['H'].shape[1]} usable subcarriers "
                     f"(band {p['idx'][0]}-{p['idx'][-1]}; "
                     f"{p['n_out']} out of band, {p['n_null']} null, "
                     f"{p['n_art']} artefact excluded)",
                     loc="left", fontsize=8.5)
        fig.colorbar(im, ax=ax, pad=0.01, label="|H| [dB]")

    if extra:
        ax = axes[len(names)]
        for r in names:
            p = per[r]
            if p["pt"] is not None:
                ax.plot(p["pt"], p["pv"], lw=1.3, color=COLOR.get(r), label=r)
        ax.axhline(args.still_mps, color=GRID, lw=0.8, ls="--")
        ax.set_ylabel("speed [m/s]")
        ax.set_title(f"({chr(97+len(names))}) platform speed from LiDAR pose",
                     loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, ncol=2)
        ax.grid(True, color=GRID, lw=0.5)

        ax = axes[len(names) + 1]
        for r in names:
            p = per[r]
            if p["ct"] is not None:
                ax.plot(p["ct"], p["cv"], lw=1.3, color=COLOR.get(r), label=r)
        ax.set_ylabel("1 - corr(|H|)")
        ax.set_title(f"({chr(98+len(names))}) channel change rate -- should "
                     "fall to zero while the platforms are stopped",
                     loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, ncol=2)
        ax.grid(True, color=GRID, lw=0.5)

    axes[-1].set_xlabel("time [s]")

    for ax in axes:
        for a, b in spans:
            if b >= lo and a <= hi:
                ax.axvspan(max(a, lo), min(b, hi), color="0.55", alpha=0.16,
                           lw=0, zorder=0)
    axes[-1].set_xlim(lo, hi)

    fig.savefig(args.out, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    for r in names:
        p = per[r]
        if p["pt"] is None:
            continue
        v = np.interp(p["ct"], p["pt"], p["pv"])
        still, moving = p["cv"][v < args.still_mps], p["cv"][v > 3 * args.still_mps]
        if len(still) > 5 and len(moving) > 5:
            rho = pd.Series(p["cv"]).corr(pd.Series(v), method="spearman")
            print(f"  {r}: change while still {np.median(still):.3f}, "
                  f"moving {np.median(moving):.3f} "
                  f"(x{np.median(moving)/max(np.median(still),1e-9):.1f}), "
                  f"rho={rho:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
