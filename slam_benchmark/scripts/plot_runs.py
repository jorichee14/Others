#!/usr/bin/env python3
"""Two figures per agent: the trajectories on the map, and error against time.

    python3 scripts/plot_runs.py --runs runs/coop2_20260828 --config configs/coop2.yaml \
        --agent mobile_1 --reference runs/reference/mobile_1.tum \
        --map /path/to/map.ply --out results/figs

  <prefix>_<agent>_xy.png     top-down, every method over the map cloud, boards
                              marked where the dataset config says they are.
  <prefix>_<agent>_error.png  distance from the reference against time, log y,
                              with the board dwell windows shaded. A method that
                              is fine until it is not shows the moment here and
                              nowhere else -- an ATE is one number for the run.

The drawn trajectories are the SCORED ones: the extrinsic and the alignment
come out of the run's own `metrics.json`, so a figure cannot disagree with the
table beside it. Runs that were never scored do not appear; run
`scripts/rescore.py` first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import load_cloud, load_dataset, load_method, load_tum  # noqa: E402
from slambench.config import to_reference_frame                        # noqa: E402
from slambench.trajectory import Trajectory                            # noqa: E402


def pick_runs(runs_root: Path, agent: str | None, newest_only: bool = True,
              include: tuple[str, ...] = ("metrics.json",)) -> list[dict]:
    """Scored runs for an agent, newest per (method, stream) unless told otherwise.

    One line per cell keeps a legend readable; `--all-runs` draws the spread.
    """
    found = []
    for p in sorted(runs_root.rglob("metrics*.json")):
        if p.name not in include:
            continue
        m = json.loads(p.read_text())
        if agent and m.get("agent") != agent:
            continue
        if not m.get("ate") or "_alignment_T" not in m:
            continue
        found.append({"method": m["method"], "stream": m.get("stream"),
                      "agent": m.get("agent"), "run": Path(m["run_dir"]),
                      "T": np.array(m["_alignment_T"]),
                      "scale": float(m.get("_alignment_scale", 1.0)),
                      "ate": m["ate"]["ate_trans_m"]["rmse"],
                      "from": "graph" if p.name == "metrics.json"
                              else p.stem.replace("metrics_", "")})
    if not newest_only:
        return found
    best: dict[tuple, dict] = {}
    for f in found:
        key = (f["method"], f["stream"], f["from"])
        if key not in best or f["run"].name > best[key]["run"].name:
            best[key] = f
    return sorted(best.values(), key=lambda f: f["ate"])


def place(traj: Trajectory, E: np.ndarray, T: np.ndarray, scale: float,
          frame: str = "") -> Trajectory:
    """Sensor poses -> the reference world frame, exactly as the scorer did it."""
    out = to_reference_frame(traj, E, frame)
    poses = out.poses.copy()
    poses[:, :3, 3] *= scale
    return Trajectory(out.stamps, np.einsum("ij,njk->nik", T, poses),
                      traj.name, out.frame, out.world)


def error_curve(est: Trajectory, ref: Trajectory, max_gap_s: float = 0.25):
    """(stamps, distance) of the estimate from the reference, per estimate pose.

    The reference is interpolated onto the estimate's stamps, and it drops any
    that fall outside its span or inside a real dropout, so the two arrays are
    the poses that could honestly be compared and nothing else.
    """
    if len(est) < 1 or len(ref) < 2:
        return np.zeros(0), np.zeros(0)
    keep = (est.stamps >= ref.stamps[0]) & (est.stamps <= ref.stamps[-1])
    t, p = est.stamps[keep], est.positions[keep]
    if not len(t):
        return np.zeros(0), np.zeros(0)
    try:
        r = ref.interpolate(t, max_gap_s=max_gap_s)
    except ValueError:                       # no query stamp fell inside it
        return np.zeros(0), np.zeros(0)
    idx = np.searchsorted(t, r.stamps)
    return r.stamps, np.linalg.norm(p[idx] - r.positions, axis=1)


def dwell_windows(cfg, agent: str) -> list[tuple[float, float, str]]:
    """The board dwells for this agent, for shading. Empty is legal and means
    the config declares no window there — which is itself worth seeing."""
    out = []
    for a in cfg.reference.get("anchors", []):
        w = (a.get("windows_by_agent") or {}).get(agent) or a.get("window")
        if w:
            out.append((float(w[0]), float(w[1]), a["name"]))
    return out


def main() -> int:                                           # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--map", default=None, help="map cloud; defaults to reference.map_cloud")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--all-runs", action="store_true", help="every run, not newest per cell")
    ap.add_argument("--include-odometry", action="store_true")
    ap.add_argument("--max-error-m", type=float, default=None,
                    help="drop methods whose ATE exceeds this, to keep the plot readable")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: pip3 install matplotlib", file=sys.stderr)
        return 2

    cfg = load_dataset(args.config)
    ref = load_tum(args.reference, name=f"reference/{args.agent}")
    inc = ("metrics.json", "metrics_odometry.json") if args.include_odometry else ("metrics.json",)
    runs = pick_runs(Path(args.runs), args.agent, not args.all_runs, inc)
    if not runs:
        print(f"no scored run for {args.agent} under {args.runs}; "
              f"run scripts/rescore.py first", file=sys.stderr)
        return 1
    if args.max_error_m is not None:
        runs = [r for r in runs if r["ate"] <= args.max_error_m]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    # ------------------------------------------------------------------ xy
    fig, ax = plt.subplots(figsize=(9, 10))
    cloud_path = args.map or cfg.reference.get("map_cloud")
    if cloud_path:
        pts = load_cloud(Path(cloud_path).expanduser())
        step = max(1, len(pts) // 400_000)
        ax.scatter(pts[::step, 0], pts[::step, 1], s=0.2, c="0.8", linewidths=0,
                   rasterized=True, zorder=0)
    else:
        ax.set_title("no map cloud: reference.map_cloud is null and --map was not given",
                     fontsize=8, loc="right", color="0.4")
    ax.plot(ref.positions[:, 0], ref.positions[:, 1], "k-", lw=2.5,
            label=f"{args.agent} reference", zorder=5)
    for i, r in enumerate(runs):
        mcfg = load_method(ROOT / "configs" / "methods" / f"{r['method']}.yaml")
        est = place(load_tum(r["run"] / ("odometry.tum" if r["from"] == "odometry"
                                         else "trajectory.tum")),
                    mcfg.sensor_extrinsic(cfg, r["stream"]), r["T"], r["scale"],
                    cfg.agent_reference(args.agent).get("frame", ""))
        lab = f"{r['method']}" + (f" [{r['from']}]" if r["from"] != "graph" else "")
        lab += f"  ATE {r['ate']*1e3:.0f} mm"
        ax.plot(est.positions[:, 0], est.positions[:, 1], lw=1.4, color=cmap(i % 10),
                label=lab, zorder=4)
        r["_est"] = est
    for a in cfg.reference.get("anchors", []):
        x, y = a["position"][0], a["position"][1]
        ax.plot(x, y, "*", ms=16, color="gold", mec="0.3", zorder=6)
        ax.annotate(a["name"], (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p1 = f"{out}_{args.agent}_xy.png"
    fig.savefig(p1, dpi=160); plt.close(fig)

    # --------------------------------------------------------------- error
    fig, ax = plt.subplots(figsize=(9, 7))
    t0 = float(ref.stamps[0])
    for a0, a1, name in dwell_windows(cfg, args.agent):
        ax.axvspan(a0 - t0, a1 - t0, color="gold", alpha=0.25, lw=0, zorder=0)
    for i, r in enumerate(runs):
        t, d = error_curve(r["_est"], ref)
        if not len(t):
            continue
        lab = f"{r['method']}" + (f" [{r['from']}]" if r["from"] != "graph" else "")
        ax.plot(t - t0, d, lw=1.3, color=cmap(i % 10), label=lab)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("t [s]"); ax.set_ylabel(f"distance from the {args.agent} reference [m]")
    ax.set_title("shaded: declared board dwell windows", fontsize=9, loc="right", color="0.4")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = f"{out}_{args.agent}_error.png"
    fig.savefig(p2, dpi=160); plt.close(fig)

    print(f"{len(runs)} runs -> {p1}\n{' ' * len(str(len(runs)))}        {p2}")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
