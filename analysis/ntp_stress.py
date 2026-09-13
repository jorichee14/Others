#!/usr/bin/env python3
"""NTP stress test: does the clock stay within bounds while the machine is pushed?

    python3 ntp_stress.py --run stress --bag stress.mcap --side-log 'stress/ntp_side_*.csv'

Two sources, joined by wall-clock time:
  * the bag's NtpStatus topics (v2 layout: last_offset_seconds, skew_ppm,
    fit_samples, reference_time = chrony's Ref time). Older bags without those
    fields still work; polls are then found from changes in offset_seconds.
  * the CSVs from tools/ntp_side_log.sh, one per agent per phase, which carry
    the SoC temperature and name the stressor (thermal, coldstart, wifiload,
    motion, idle). Without them there is no temperature and one phase "run".

What is tested, per agent and phase (ntp_issues.md):
  * the measured offset at every poll stays under --offset-limit-ms;
  * the skew settles under --skew-limit-ppm once the temperature stops moving;
  * the poll interval is the configured one, so the sample count is as planned;
  * no clock step after warm-up.

Outputs in results/<run>/ntp_stress/: ntp_stress_polls.csv (one row per poll),
ntp_stress_phases.csv (the table above), fig_ntp_stress.{pdf,png}, ntp_stress.md.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRID, TEXT, TEXT2, BAD_COLOR, color_for, node_of_topic  # noqa: E402
from extract_bag import extract  # noqa: E402

PHASE_SHADE = {"thermal": "#f5a173", "cool": "#9ec5f4", "coldstart": "#c9b8e8",
               "wifiload": "#f0d37a", "motion": "#a8d5a2", "idle": "#dddbd6"}


def load_bag_polls(extracts: Path):
    """One row per chrony poll per agent, from the NtpStatus topics.

    The glob is `*ntp__*status.parquet` because the topics are namespaced and
    split by role: /mobile_2/ntp/client/status and /mobile_1/ntp/server/status
    become mobile_2__ntp__client__status.parquet and
    mobile_1__ntp__server__status.parquet. A pattern anchored on `ntp__status`
    matches neither."""
    frames = []
    for f in sorted(glob.glob(str(extracts / "*ntp__*status.parquet"))):
        df = pd.read_parquet(f)
        topic = "/" + Path(f).stem.replace("__", "/")
        agent = node_of_topic(topic)
        if "role" in df and (df["role"] != "client").all():
            continue
        df = df.sort_values("log_time_ns").reset_index(drop=True)
        v2 = "last_offset_seconds" in df.columns
        if v2:
            ref = df["reference_time.sec"].astype(np.int64) * 10**9 + df["reference_time.nanosec"].astype(np.int64)
            new = ref.ne(ref.shift(1, fill_value=-1))
        else:
            new = df["offset_seconds"].ne(df["offset_seconds"].shift(1))
            stepped = df["clock_stepped"].astype(bool)
            if stepped.any():
                new &= df.index >= stepped.idxmax()
        p = df.loc[new].copy()
        out = pd.DataFrame({
            "agent": agent,
            "t_epoch_s": p["log_time_ns"] / 1e9,
            "offset_ms": (p["last_offset_seconds"] if v2 else p["offset_seconds"]) * 1e3,
            "system_offset_ms": p["offset_seconds"] * 1e3,
            "freq_ppm": p["frequency_error_ppm"],
            "skew_ppm": p["skew_ppm"] if v2 else np.nan,
            "fit_samples": p["fit_samples"] if v2 else np.nan,
            "delay_ms": p["delay_seconds"] * 1e3,
            "jitter_ms": p["jitter_seconds"] * 1e3,
            "poll_interval_s": p["poll_interval_seconds"],
            "stepped": p["clock_stepped"].astype(bool),
            "delay_is_bound": p["warnings"].map(
                lambda w: any("upper bound" in str(x) for x in (w if w is not None else []))),
            "source": "bag", "layout": "v2" if v2 else "v1",
        })
        # The monitor carries these now, so a stress run needs no side log.
        for col in ("temperature_c", "cpu_load_1min"):
            out[col] = p[col].to_numpy() if col in p.columns else np.nan
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_side_logs(pattern: str):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        if "t_utc" not in d.columns:
            continue
        # resolution of the parsed datetime varies (ns or us) between pandas versions,
        # so go through a fixed epoch rather than dividing an integer view
        d["t_epoch_s"] = (pd.to_datetime(d["t_utc"], utc=True)
                          - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        d["file"] = Path(f).name
        rows.append(d)
    if not rows:
        return None
    s = pd.concat(rows, ignore_index=True)
    for c in ("temp_c", "system_offset_s", "last_offset_s", "freq_ppm", "skew_ppm", "interval_s", "np", "stddev_s"):
        if c in s:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=None)
    ap.add_argument("--extracts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run", default="run")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--side-log", default=None, help="glob of ntp_side_*.csv files (default: next to the bag)")
    ap.add_argument("--offset-limit-ms", type=float, default=1.0)
    ap.add_argument("--skew-limit-ppm", type=float, default=1.0)
    ap.add_argument("--poll-s", type=int, default=8, help="the configured poll interval")
    ap.add_argument("--warmup-s", type=float, default=60.0,
                    help="steps within this many seconds of the first sample are start-up, not failures")
    args = ap.parse_args()

    extracts = args.extracts or Path("extracts") / args.run
    out = args.out or Path("results") / args.run / "ntp_stress"
    out.mkdir(parents=True, exist_ok=True)
    polls = None
    if args.bag is not None and (args.force_extract or not (extracts / "metadata.json").exists()):
        extract(args.bag, extracts, topics=None, include_heavy=False)   # light topics only
    if (extracts / "metadata.json").exists():
        polls = load_bag_polls(extracts)
    side = load_side_logs(args.side_log or str((args.bag.parent if args.bag else extracts) / "ntp_side_*.csv"))
    if polls is None and side is None:
        raise SystemExit("neither NtpStatus topics nor side logs found")

    # ---- the time base and the phases --------------------------------------------
    starts = []
    if polls is not None:
        starts.append(float(polls["t_epoch_s"].min()))
    if side is not None:
        starts.append(float(side["t_epoch_s"].min()))
    t0 = min(starts)
    if polls is not None:
        polls["t_s"] = polls["t_epoch_s"] - t0
    phases = []                                   # (agent, phase, t_start, t_end)
    if side is not None:
        side["t_s"] = side["t_epoch_s"] - t0
        side = side.sort_values(["agent", "t_s"]).reset_index(drop=True)   # files arrive in name order
        for (a, ph), g in side.groupby(["agent", "phase"]):
            phases.append((a, ph, float(g["t_s"].min()), float(g["t_s"].max()) + args.poll_s))
    agents = sorted(set(polls["agent"]) if polls is not None else set(side["agent"]))
    if not phases:
        for a in agents:
            g = polls[polls["agent"] == a]
            phases.append((a, "run", float(g["t_s"].min()), float(g["t_s"].max())))

    # side logs stand in for the bag when there is none
    if polls is None:
        polls = pd.DataFrame({
            "agent": side["agent"], "t_s": side["t_s"], "t_epoch_s": side["t_epoch_s"],
            "offset_ms": side["last_offset_s"] * 1e3, "system_offset_ms": side["system_offset_s"] * 1e3,
            "freq_ppm": side["freq_ppm"], "skew_ppm": side["skew_ppm"], "fit_samples": side.get("np"),
            "delay_ms": np.nan, "jitter_ms": side.get("stddev_s", np.nan) * 1e3,
            "poll_interval_s": side["interval_s"], "stepped": False, "delay_is_bound": False,
            "source": "side_log", "layout": "side_log"})
    polls.to_csv(out / "ntp_stress_polls.csv", index=False)

    # ---- per agent and phase --------------------------------------------------------
    rows = []
    for a, ph, lo, hi in phases:
        p = polls[(polls["agent"] == a) & (polls["t_s"] >= lo) & (polls["t_s"] < hi)]
        s = side[(side["agent"] == a) & (side["phase"] == ph)] if side is not None else None
        if len(p) == 0 and (s is None or len(s) == 0):
            continue
        # temperature from the message first, from a side log only if absent
        temp = p["temperature_c"].dropna() if ("temperature_c" in p and p["temperature_c"].notna().any()) \
            else (s["temp_c"].dropna() if s is not None else pd.Series(dtype=float))
        fq = p["freq_ppm"].dropna() if len(p) else (s["freq_ppm"].dropna() if s is not None else pd.Series(dtype=float))
        sk = p["skew_ppm"].dropna() if len(p) and p["skew_ppm"].notna().any() else (s["skew_ppm"].dropna() if s is not None else pd.Series(dtype=float))
        tail = sk.tail(10)
        row = {
            "agent": a, "phase": ph, "t_start_s": round(lo, 1), "duration_s": round(hi - lo, 1),
            "polls": int(len(p)),
            "poll_interval_s": float(p["poll_interval_s"].mode().iloc[0]) if len(p) else np.nan,
            "temp_min_c": round(float(temp.min()), 1) if len(temp) else np.nan,
            "temp_max_c": round(float(temp.max()), 1) if len(temp) else np.nan,
            "freq_start_ppm": round(float(fq.iloc[0]), 3) if len(fq) else np.nan,
            "freq_end_ppm": round(float(fq.iloc[-1]), 3) if len(fq) else np.nan,
            "skew_median_ppm": round(float(sk.median()), 3) if len(sk) else np.nan,
            "skew_max_ppm": round(float(sk.max()), 3) if len(sk) else np.nan,
            "skew_settled_ppm": round(float(tail.median()), 3) if len(tail) else np.nan,
            "abs_offset_median_ms": round(float(p["offset_ms"].abs().median()), 4) if len(p) else np.nan,
            "abs_offset_max_ms": round(float(p["offset_ms"].abs().max()), 4) if len(p) else np.nan,
            "abs_system_offset_max_ms": round(float(p["system_offset_ms"].abs().max()), 4) if len(p) else np.nan,
            "jitter_median_ms": round(float(p["jitter_ms"].median()), 4) if len(p) else np.nan,
            "delay_median_ms": round(float(p["delay_ms"].median()), 3) if len(p) and p["delay_ms"].notna().any() else np.nan,
            "delay_is_upper_bound": bool(p["delay_is_bound"].any()) if len(p) else False,
            "clock_steps": int(p.loc[p["t_s"] >= args.warmup_s, "stepped"].sum()) if len(p) else 0,
        }
        if len(temp) and len(fq) > 1:
            row["freq_per_degree_ppm"] = round(float((fq.iloc[-1] - fq.iloc[0]) /
                                                     max(temp.iloc[-1] - temp.iloc[0], 1e-9)), 3) \
                if abs(temp.iloc[-1] - temp.iloc[0]) > 2 else np.nan
        row["pass_offset"] = bool(row["abs_offset_max_ms"] < args.offset_limit_ms) if len(p) else None
        row["pass_skew_settled"] = bool(row["skew_settled_ppm"] < args.skew_limit_ppm) if len(tail) else None
        row["pass_interval"] = bool(abs(row["poll_interval_s"] - args.poll_s) < 1) if len(p) else None
        row["pass_no_steps"] = row["clock_steps"] == 0
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "ntp_stress_phases.csv", index=False)

    # ---- figure ----------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT})
    n = len(agents)
    fig, axes = plt.subplots(4, n, figsize=(4.6 * n + 0.6, 7.6), sharex="col", squeeze=False,
                             constrained_layout=True)
    for j, a in enumerate(agents):
        c = color_for(a)
        p = polls[polls["agent"] == a]
        s = side[side["agent"] == a] if side is not None else None
        ax_t, ax_f, ax_k, ax_o = axes[:, j]
        for ax in (ax_t, ax_f, ax_k, ax_o):
            for (aa, ph, lo, hi) in phases:
                if aa == a:
                    ax.axvspan(lo, hi, color=PHASE_SHADE.get(ph, "#eeeeee"), alpha=0.25, lw=0)
            ax.grid(True, color=GRID, lw=0.5)
        for (aa, ph, lo, hi) in phases:
            if aa == a:
                ax_t.text((lo + hi) / 2, 0.97, ph, transform=ax_t.get_xaxis_transform(),
                          ha="center", va="top", fontsize=7, color=TEXT2)
        if "temperature_c" in p and p["temperature_c"].notna().any():
            ax_t.plot(p["t_s"], p["temperature_c"], color=c, lw=1.4)
        elif s is not None and s["temp_c"].notna().any():
            ax_t.plot(s["t_s"], s["temp_c"], color=c, lw=1.4)
        else:
            ax_t.text(0.5, 0.5, "no temperature in the message or a side log",
                      transform=ax_t.transAxes, ha="center", va="center", fontsize=7, color=TEXT2)
        ax_t.set_ylabel("SoC temperature [°C]")
        ax_t.set_title(f"({chr(97 + j)}) {a}", loc="left", fontsize=8)
        ax_f.plot(p["t_s"], p["freq_ppm"], color=c, lw=1.4)
        ax_f.set_ylabel("frequency error [ppm]")
        if p["skew_ppm"].notna().any():
            ax_k.plot(p["t_s"], p["skew_ppm"], color=c, lw=1.4)
        ax_k.axhline(args.skew_limit_ppm, color=BAD_COLOR, lw=0.8, ls="--")
        ax_k.set_ylabel("skew [ppm]")
        ax_k.set_ylim(bottom=0)
        ax_o.scatter(p["t_s"], p["offset_ms"], s=8, color=c, alpha=0.8, linewidths=0, label="measured at poll")
        ax_o.plot(p["t_s"], p["system_offset_ms"], color=c, lw=0.9, alpha=0.6, label="steered estimate")
        ax_o.axhline(args.offset_limit_ms, color=BAD_COLOR, lw=0.8, ls="--")
        ax_o.axhline(-args.offset_limit_ms, color=BAD_COLOR, lw=0.8, ls="--")
        for ts in p.loc[p["stepped"], "t_s"]:
            ax_o.axvline(ts, color=BAD_COLOR, lw=0.8)
        ax_o.set_ylabel("offset to server [ms]")
        ax_o.set_xlabel("time since first sample [s]")
        if j == 0:
            ax_o.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(out / "fig_ntp_stress.pdf")
    fig.savefig(out / "fig_ntp_stress.png", dpi=200)
    plt.close(fig)

    # ---- markdown ----------------------------------------------------------------------
    md = [f"# NTP stress test — run `{args.run}`", "",
          f"Pass criteria: |measured offset| < {args.offset_limit_ms} ms at every poll; skew settles "
          f"under {args.skew_limit_ppm} ppm (median of the last 10 polls of the phase); poll interval "
          f"{args.poll_s} s; no clock step after the first {args.warmup_s:.0f} s.", "",
          table.to_markdown(index=False), ""]
    if table.get("delay_is_upper_bound", pd.Series(dtype=bool)).any():
        md += ["> `delay_median_ms` is an **upper bound** (twice the sources error term) where the node "
               "had no socket access to `chronyc ntpdata`; it includes upstream dispersion and is not "
               "the leg to the server.", ""]
    md += ["`freq_per_degree_ppm` is the crystal's temperature coefficient over the phase, from the "
           "change in chrony's frequency estimate against the change in SoC temperature; it is only "
           "computed where the temperature moved by more than 2 °C.", ""]
    (out / "ntp_stress.md").write_text("\n".join(md))
    cols = ["agent", "phase", "duration_s", "polls", "temp_min_c", "temp_max_c", "freq_start_ppm",
            "freq_end_ppm", "skew_max_ppm", "skew_settled_ppm", "abs_offset_max_ms", "clock_steps",
            "pass_offset", "pass_skew_settled", "pass_interval", "pass_no_steps"]
    print(table[[c for c in cols if c in table]].to_string(index=False))
    print(f"\nwrote {out}/ntp_stress_phases.csv, ntp_stress_polls.csv, fig_ntp_stress.pdf/png, ntp_stress.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
