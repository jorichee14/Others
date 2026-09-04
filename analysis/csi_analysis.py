#!/usr/bin/env python3
"""Wi-Fi CSI (channel state information) analysis of one run.

WHAT CSI IS, AND WHAT COMES OUT OF IT
-------------------------------------
One `CsiFrame` is the complex channel response of ONE received 802.11 frame,
sampled per OFDM subcarrier: `csi_real[i] + j*csi_imag[i]` at raw firmware slot
`subcarrier_index[i]`. Where RSSI is a single number (total received power),
CSI is a vector across frequency -- roughly 240 usable values at 80 MHz. It
describes the PROPAGATION ENVIRONMENT, not link performance.

Three things can be measured from it, all of them ratios and therefore immune to
the receiver's automatic gain control:

  frequency selectivity   the spread of |H| across subcarriers, in dB. A flat
                          response means one dominant path; deep notches mean
                          multipath components cancelling at some frequencies.
  RMS delay spread        from the power delay profile, |IFFT(H)|^2: how far the
                          multipath energy is spread in time. Short (tens of ns)
                          means a dominant direct path; long means rich scatter.
  Rician K-factor         the power ratio of the dominant path to everything
                          else, from the 2nd and 4th moments of |H|. High K is
                          the signature of an unobstructed path.

WHAT CANNOT BE MEASURED HERE
----------------------------
The PHASE is not usable as recorded. Every frame carries an unknown carrier and
sampling frequency offset plus a random packet-detection delay, so raw phase is
meaningless across frames without sanitisation or a second antenna to reference
against. Nothing below uses phase; only |H| and the delay profile, both of which
survive those offsets.

Absolute amplitude is also not meaningful (AGC), so amplitudes are reported in dB
relative to each frame's own median.

RESOLUTION
----------
The delay profile's tap spacing is 1/bandwidth: 12.5 ns at 80 MHz, 25 ns at
40 MHz, 50 ns at 20 MHz. Indoor delay spreads are tens of nanoseconds, so at
20 MHz the measurement is one tap wide and the delay-spread number should not be
quoted; the script says so when it happens.

Usage
-----
    python csi_analysis.py --bag BAG.mcap --run coop2
    python csi_analysis.py --run coop2 --map <anchored>.pcd

Outputs (in --out, default results/<run>/csi)
---------------------------------------------
    csi_inventory.csv     one row per agent: rate, config, subcarriers, MIMO streams
    csi_transmitters.csv  one row per (agent, source MAC, frame type): count, RSSI
    csi_frames.parquet    per frame: amplitude, selectivity, delay spread, K-factor
    csi_summary.md        all of the above as readable tables
    csi_subsection.tex    a paragraph for the paper with the numbers filled in
    fig_csi.{pdf,png}     amplitude heat map per agent, then delay spread and
                          K-factor over the run
    fig_csi_map.{pdf,png} the trajectories coloured by K-factor (needs poses)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    GRID, TEXT, TEXT2, AGENT_COLOR, color_for, load_poses, node_of_topic, read_pcd_xy,
)
from csi_core import (  # noqa: E402
    amplitude_db, delay_profile, rician_k, rms_delay_spread,
)
from extract_bag import extract  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections  # noqa: E402
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 802.11 frame control values the capture firmware reports
FRAME_TYPES = {0x88: "QoS Data", 0x94: "Block Ack", 0x80: "Beacon", 0x08: "Data"}
# Same direction as the RSSI coverage map -- dark is the better channel -- so the
# two spatial figures of a run read the same way.
K_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
# Amplitude is reported relative to each frame's own median, so it is a signed
# quantity: a diverging pair with a neutral middle puts the deep fades and the
# peaks on opposite poles instead of burying both in one ramp.
AMP_DIVERGING = ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f5a173", "#e34948", "#8f1f1e"]


def load_csi(extracts: Path):
    """{agent: DataFrame} from the per-topic CSI tables."""
    out = {}
    for f in sorted(glob.glob(str(extracts / "*csi.parquet"))):
        topic = "/" + Path(f).stem.replace("__", "/")
        df = pd.read_parquet(f)
        df["topic"] = topic
        out[node_of_topic(topic)] = df
    return out


def stack_H(df: pd.DataFrame):
    """The frames' channel responses as one complex array (n_frames, n_subcarriers).

    Frames whose subcarrier layout differs from the run's dominant one are
    dropped rather than padded: mixing bandwidths in one array would silently
    change the delay-profile scale."""
    counts = df["subcarrier_index"].map(len)
    n_sub = int(counts.mode().iloc[0])
    keep = (counts == n_sub).to_numpy()
    re = np.stack(df.loc[keep, "csi_real"].to_numpy())
    im = np.stack(df.loc[keep, "csi_imag"].to_numpy())
    idx = np.asarray(df.loc[keep, "subcarrier_index"].iloc[0], dtype=int)
    return re.astype(np.float64) + 1j * im.astype(np.float64), idx, keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=None)
    ap.add_argument("--extracts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run", default="run")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--map", type=Path, default=None, help="anchored .pcd drawn as the map background")
    ap.add_argument("--pose-topic", default="global_pose")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame for the per-frame metrics")
    ap.add_argument("--smooth-s", type=float, default=1.0, help="window for the smoothed traces")
    args = ap.parse_args()

    extracts = args.extracts or Path("extracts") / args.run
    out = args.out or Path("results") / args.run / "csi"
    out.mkdir(parents=True, exist_ok=True)

    if args.bag is not None and (args.force_extract or not (extracts / "metadata.json").exists()):
        extract(args.bag, extracts)
    elif not (extracts / "metadata.json").exists():
        raise SystemExit(f"no extraction in {extracts}; pass --bag BAG.mcap to create it")
    else:
        print(f"using existing extraction in {extracts}")

    csi = load_csi(extracts)
    if not csi:
        raise SystemExit(f"no *csi.parquet in {extracts}")
    t0_ns = min(int(d["log_time_ns"].min()) for d in csi.values())

    inv_rows, tx_rows, frames = [], [], []
    per_agent = {}
    for agent in sorted(csi):
        df = csi[agent].sort_values("log_time_ns").reset_index(drop=True)
        df["t_s"] = (df["log_time_ns"] - t0_ns) / 1e9
        dur = float(df["t_s"].max() - df["t_s"].min())
        bw = int(df["bandwidth_mhz"].mode().iloc[0])
        raw_slots = int(df["raw_slots"].mode().iloc[0])
        streams = sorted({(int(a), int(b)) for a, b in zip(df["core"], df["spatial_stream"])})

        # ---- who transmitted the measured frames -------------------------------
        for (mac, fc), g in df.groupby(["src_mac", "frame_control"]):
            tx_rows.append({
                "agent": agent, "src_mac": mac,
                "frame_control": f"0x{int(fc):02x}",
                "frame_type": FRAME_TYPES.get(int(fc), "other"),
                "frames": len(g), "share": len(g) / len(df),
                "rssi_median_dbm": float(g["rssi"].median()),
                "t_first_s": float(g["t_s"].min()), "t_last_s": float(g["t_s"].max()),
            })

        # ---- per-frame channel metrics ------------------------------------------
        sub = df.iloc[:: max(args.stride, 1)].reset_index(drop=True)
        H, idx, keep = stack_H(sub)
        sub = sub.loc[keep].reset_index(drop=True)
        n_sub = H.shape[1]
        dt_s = 1.0 / (bw * 1e6)                      # tap spacing of the delay profile
        window = max(raw_slots // 4, 8)
        P = delay_profile(H, idx, raw_slots)
        tau = rms_delay_spread(P, dt_s, window)
        K = rician_k(H)
        amp_db = amplitude_db(H)                             # AGC out, shape kept
        sel_db = amp_db.std(axis=1)

        frames.append(pd.DataFrame({
            "run": args.run, "agent": agent, "t_s": sub["t_s"].to_numpy(),
            "rssi_dbm": sub["rssi"].to_numpy(),
            "selectivity_db": sel_db,
            "delay_spread_ns": tau * 1e9,
            "k_factor": K,
            "k_factor_db": 10 * np.log10(np.maximum(K, 1e-3)),
        }))
        per_agent[agent] = dict(t=sub["t_s"].to_numpy(), amp_db=amp_db, idx=idx,
                                bw=bw, dt_ns=dt_s * 1e9)

        inv_rows.append({
            "run": args.run, "agent": agent, "topic": df["topic"].iloc[0],
            "frames": len(df), "rate_hz": (len(df) - 1) / max(dur, 1e-9), "duration_s": dur,
            "channel": int(df["channel"].mode().iloc[0]), "bandwidth_mhz": bw,
            "chanspec": f"0x{int(df['chanspec'].mode().iloc[0]):04x}",
            "chip_version": f"0x{int(df['chip_version'].mode().iloc[0]):04x}",
            "raw_slots": raw_slots, "subcarriers_kept": n_sub,
            "trimmed": bool(df["trimmed"].mode().iloc[0]),
            "streams": "; ".join(f"core{c}/ss{s}" for c, s in streams),
            "n_streams": len(streams),
            "transmitters": int(df["src_mac"].nunique()),
            "tap_spacing_ns": dt_s * 1e9,
            "frames_used": len(sub),
        })

    inventory = pd.DataFrame(inv_rows)
    transmitters = pd.DataFrame(tx_rows).sort_values(["agent", "frames"], ascending=[True, False])
    fr = pd.concat(frames, ignore_index=True)
    inventory.to_csv(out / "csi_inventory.csv", index=False)
    transmitters.to_csv(out / "csi_transmitters.csv", index=False)
    fr.to_parquet(out / "csi_frames.parquet", index=False)

    # smoothed traces: a single frame's K is noisy, so nothing is quoted un-smoothed
    def smooth(g):
        n = max(int(args.smooth_s * len(g) / max(g["t_s"].max() - g["t_s"].min(), 1e-9)), 1)
        return g.set_index("t_s")[["delay_spread_ns", "k_factor_db", "selectivity_db"]] \
                .rolling(n, center=True, min_periods=max(n // 4, 1)).median()

    stats = fr.groupby("agent").agg(
        delay_spread_median_ns=("delay_spread_ns", "median"),
        delay_spread_p95_ns=("delay_spread_ns", lambda x: x.quantile(0.95)),
        k_factor_median_db=("k_factor_db", "median"),
        k_factor_p05_db=("k_factor_db", lambda x: x.quantile(0.05)),
        selectivity_median_db=("selectivity_db", "median"),
        rssi_median_dbm=("rssi_dbm", "median"),
    ).round(2).reset_index()

    # ---- figure 1: what the channel looks like ---------------------------------
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT})
    agents = sorted(per_agent)
    ncol = len(agents)
    t_max = float(fr["t_s"].max())
    fig = plt.figure(figsize=(4.2 * ncol + 1.0, 7.4))
    gs = fig.add_gridspec(3, ncol + 1, height_ratios=[1.25, 1.0, 1.0],
                          width_ratios=[1.0] * ncol + [0.05],
                          hspace=0.42, wspace=0.26,
                          left=0.075, right=0.93, top=0.93, bottom=0.09)
    amp_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("amp", AMP_DIVERGING)
    # symmetric about 0 dB so the neutral midpoint really is "at the frame median"
    vabs = float(np.percentile(np.abs(np.concatenate(
        [p["amp_db"].ravel() for p in per_agent.values()])), 98))
    im = None
    for i, a in enumerate(agents):
        ax = fig.add_subplot(gs[0, i])
        p = per_agent[a]
        step = max(len(p["t"]) // 1200, 1)          # one column per output pixel
        im = ax.imshow(p["amp_db"][::step].T, aspect="auto", origin="lower", cmap=amp_cmap,
                       vmin=-vabs, vmax=vabs,
                       extent=[p["t"][0], p["t"][-1], p["idx"].min(), p["idx"].max()])
        ax.set_xlabel("time in run [s]")
        ax.set_ylabel("subcarrier index" if i == 0 else "")
        ax.set_title(f"({chr(97 + i)}) {a}: channel amplitude\n"
                     f"{p['bw']} MHz, {p['amp_db'].shape[1]} subcarriers, "
                     f"{p['dt_ns']:.1f} ns tap spacing", loc="left", fontsize=8)
    if im is not None:
        cax = fig.add_subplot(gs[0, ncol].subgridspec(3, 1, height_ratios=[0.16, 1, 0.05])[1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("|H| relative to frame median [dB]", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    for a in agents:
        g = fr[fr["agent"] == a]
        sm = smooth(g)
        ax.plot(g["t_s"], g["delay_spread_ns"], lw=0.5, color=color_for(a), alpha=0.22)
        ax.plot(sm.index, sm["delay_spread_ns"], lw=1.5, color=color_for(a), label=a)
    tap = max(per_agent[a]["dt_ns"] for a in agents)
    ax.axhline(tap, color=TEXT2, lw=0.8, ls="--")
    ax.text(0.995, tap, f" one tap ({tap:.1f} ns) ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.5, color=TEXT2)
    ax.set_xlim(0, t_max); ax.set_ylabel("RMS delay spread [ns]")
    ax.set_title(f"({chr(97 + ncol)}) multipath spread  (faint: per frame, bold: {args.smooth_s:.0f} s median)",
                 loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=len(agents)); ax.grid(True, color=GRID, lw=0.5)

    ax = fig.add_subplot(gs[2, :])
    for a in agents:
        g = fr[fr["agent"] == a]
        sm = smooth(g)
        ax.plot(g["t_s"], g["k_factor_db"], lw=0.5, color=color_for(a), alpha=0.22)
        ax.plot(sm.index, sm["k_factor_db"], lw=1.5, color=color_for(a), label=a)
    ax.axhline(0, color=GRID, lw=0.8)
    k_lo = max(float(np.percentile(fr["k_factor_db"], 1)) - 1.0, -12.0)
    k_hi = float(np.percentile(fr["k_factor_db"], 99.8)) + 1.5
    n_floor = int((fr["k_factor_db"] < k_lo).sum())
    ax.set_ylim(k_lo, k_hi)
    ax.set_xlim(0, t_max); ax.set_xlabel("time in run [s]")
    ax.set_ylabel("Rician K [dB]")
    floor_note = (f".  {100 * n_floor / len(fr):.0f}% of frames fall below the axis: no dominant path at all"
                  if n_floor else "")
    ax.set_title(f"({chr(98 + ncol)}) dominant-path strength  "
                 f"(high = one strong path, low = diffuse){floor_note}",
                 loc="left", fontsize=8)
    ax.grid(True, color=GRID, lw=0.5)

    fig.savefig(out / "fig_csi.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_csi.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: where the channel is what it is ------------------------------
    poses = load_poses(extracts, args.pose_topic)
    map_xy = read_pcd_xy(args.map) if args.map else None
    placed = [a for a in agents if a in poses]
    if placed:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("k", K_RAMP)
        kv = fr[fr["agent"].isin(placed)]["k_factor_db"]
        v_lo, v_hi = np.percentile(kv, [5, 95])
        norm = matplotlib.colors.Normalize(v_lo, v_hi)
        fig = plt.figure(figsize=(4.0 * len(placed) + 1.2, 4.4))
        gs = fig.add_gridspec(1, len(placed) + 1, width_ratios=[1.0] * len(placed) + [0.05],
                              wspace=0.28, left=0.08, right=0.9, top=0.86, bottom=0.14)
        sm_ = None
        for i, a in enumerate(placed):
            ax = fig.add_subplot(gs[0, i])
            g = fr[fr["agent"] == a]
            n = max(int(args.smooth_s * len(g) / max(g["t_s"].max() - g["t_s"].min(), 1e-9)), 1)
            kdb = g["k_factor_db"].rolling(n, center=True, min_periods=1).median().to_numpy()
            pt, pxy = poses[a]
            o = np.argsort(pt)
            tq = g["t_s"].to_numpy() * 1e9 + t0_ns
            ins = (tq >= pt[o][0]) & (tq <= pt[o][-1])
            xy = np.column_stack([np.interp(tq, pt[o], pxy[o, 0]), np.interp(tq, pt[o], pxy[o, 1])])[ins]
            if len(xy) < 2:
                continue
            if map_xy is not None:
                ax.scatter(map_xy[:, 0], map_xy[:, 1], s=0.12, c="0.88", linewidths=0, zorder=0)
            pts = xy.reshape(-1, 1, 2)
            seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
            for lw, col, z in ((5.4, "white", 2), (3.2, None, 3)):
                lc = matplotlib.collections.LineCollection(
                    seg, linewidths=lw, capstyle="round",
                    **({"colors": col} if col else {"cmap": cmap, "norm": norm}))
                if col is None:
                    lc.set_array(kdb[ins][:-1]); sm_ = lc
                lc.set_zorder(z); ax.add_collection(lc)
            ax.plot(*xy[0], "o", ms=10, mfc="white", mec=TEXT, mew=1.8, zorder=7)
            ax.plot(*xy[0], "o", ms=2.6, mfc=TEXT, mec="none", zorder=8)
            ax.plot(*xy[-1], "s", ms=9, mfc=TEXT, mec="white", mew=1.6, zorder=7)
            cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
            half = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) / 2 * 1.35
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal"); ax.grid(True, color=GRID, lw=0.4)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]" if i == 0 else "")
            ax.set_title(f"({chr(97 + i)}) {a}", loc="left", fontsize=8)
        if sm_ is not None:
            cax = fig.add_subplot(gs[0, len(placed)].subgridspec(3, 1, height_ratios=[0.12, 1, 0.05])[1])
            cb = fig.colorbar(sm_, cax=cax)
            cb.set_label("Rician K [dB]", fontsize=7.5)
            cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
            fig.savefig(out / "fig_csi_map.pdf", bbox_inches="tight")
            fig.savefig(out / "fig_csi_map.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- markdown ----------------------------------------------------------------
    coarse = inventory[inventory["bandwidth_mhz"] <= 20]
    md = [f"# Wi-Fi CSI — run `{args.run}`", "",
          "One CSI frame is the complex channel response of one received 802.11 frame across "
          "OFDM subcarriers. Everything below uses |H| and its delay profile only: the recorded "
          "phase carries an unknown per-frame carrier/sampling offset and packet-detection delay, "
          "so it is not usable without sanitisation. Amplitudes are in dB relative to each frame's "
          "own median, which removes the receiver's automatic gain control.", "",
          "## Capture inventory", "",
          inventory.round(2).to_markdown(index=False), ""]
    if len(coarse):
        md += ["> **Delay spread not quotable** for "
               + ", ".join(f"`{r.agent}`" for r in coarse.itertuples())
               + f": at {int(coarse['bandwidth_mhz'].max())} MHz the delay profile's tap spacing is "
               + f"{coarse['tap_spacing_ns'].max():.0f} ns, and indoor delay spreads are of that order. "
               "The K-factor and the frequency selectivity are unaffected.", ""]
    md += ["## Transmitters measured", "",
           "CSI is captured from whatever frames the radio receives, so the source MAC says whose "
           "channel is being measured. Frames from the access point measure the agent→AP path; "
           "frames from the other robot measure the robot→robot path.", "",
           transmitters.round(3).to_markdown(index=False), "",
           "## Channel metrics", "",
           stats.to_markdown(index=False), "",
           "`delay_spread` is the RMS spread of the power delay profile past its strongest tap. "
           "`k_factor_db` is the Rician K in dB, from the 2nd and 4th moments of |H| across "
           "subcarriers: high means one path dominates, near 0 dB or below means diffuse "
           "multipath. `selectivity_db` is the standard deviation of |H| across subcarriers — the "
           "simplest and most robust of the three, and the one that needs no assumptions.", ""]
    (out / "csi_summary.md").write_text("\n".join(md))

    # ---- LaTeX --------------------------------------------------------------------
    def tt(x):
        return "\\texttt{" + str(x).replace("_", "\\_") + "}"

    r0 = inventory.iloc[0]
    macs = transmitters.groupby("src_mac")["frames"].sum().sort_values(ascending=False)
    per = "; ".join(
        f"{tt(r.agent)} at {r.k_factor_median_db:.1f}\\,dB median Rician K and "
        f"{r.delay_spread_median_ns:.0f}\\,ns median RMS delay spread"
        for r in stats.itertuples())
    coarse_note = ""
    if len(coarse):
        coarse_note = (f" At {int(coarse['bandwidth_mhz'].max())}\\,MHz the delay profile resolves "
                       f"{coarse['tap_spacing_ns'].max():.0f}\\,ns per tap, so the delay-spread figures "
                       f"are reported as indicative rather than calibrated.")
    tex = f"""\\subsubsection{{Wi-Fi Channel State Information}}
Both mobile agents capture per-frame CSI with a Nexmon-patched radio at about
{inventory['rate_hz'].median():.0f}\\,Hz, on channel {int(r0['channel'])} at {int(r0['bandwidth_mhz'])}\\,MHz,
retaining {int(r0['subcarriers_kept'])} of {int(r0['raw_slots'])} subcarrier slots after the constant and
DC-null slots are trimmed. Frames from {len(macs)} transmitter{'s' if len(macs) != 1 else ''} are measured,
so the source address identifies which path each measurement describes. From the channel amplitude we
report the Rician K-factor (the power of the dominant path relative to the diffuse component) and the RMS
delay spread of the power delay profile: {per}. Recorded phase carries an unknown per-frame carrier and
sampling offset and is therefore not used; amplitudes are normalised per frame to remove the receiver's
automatic gain control, so both quantities are ratios and independent of it.{coarse_note}
"""
    (out / "csi_subsection.tex").write_text(tex)

    print(inventory[["agent", "frames", "rate_hz", "channel", "bandwidth_mhz", "subcarriers_kept",
                     "n_streams", "transmitters", "tap_spacing_ns"]].round(2).to_string(index=False))
    print()
    print(transmitters[["agent", "src_mac", "frame_type", "frames", "share", "rssi_median_dbm"]]
          .round(3).to_string(index=False))
    print()
    print(stats.to_string(index=False))
    print(f"\nwrote {out}/csi_summary.md, csi_subsection.tex, fig_csi.pdf/png"
          + (", fig_csi_map.pdf/png" if placed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
