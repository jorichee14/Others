#!/usr/bin/env python3
"""Wi-Fi link-quality analysis of one run.

Answers, from the tables produced by extract_bag.py:
  * which radios were recorded, on which band/channel, associated to which AP,
  * per radio: RSSI, PHY rate, MCS/NSS/width, retry and failure rate, channel
    occupancy -- rates derived from the cumulative counters, not read raw,
  * association changes and roaming (BSSID switches),
  * iperf throughput, retransmits and RTT, including the robot-to-robot tests,
  * for an agent carrying two radios: the measured joint-loss fraction and the
    inter-link correlation rho of the DLC thread, against what independent links
    would have produced.

Every agent's link is to the same access point, so the links analysed are
agent->AP; the robot-to-robot iperf still traverses the AP and is reported as the
two-hop path it is.

Usage
-----
    python wifi_analysis.py --bag BAG.mcap --run coop2
    python wifi_analysis.py --extracts extracts/coop2 --run coop2

Outputs (in --out, default results/<run>/wifi)
---------------------------------------------
    wifi_links.csv        one row per radio: identity, band, association, RSSI, rates
    wifi_events.csv       association / BSSID / channel changes with their time
    wifi_iperf.csv        one row per iperf test
    wifi_rho.csv          dual-radio joint-loss and correlation (if two radios found)
    wifi_summary.md       all of the above as readable tables
    wifi_subsection.tex   a paragraph for the paper with the numbers filled in
    fig_wifi_link.{pdf,png}   RSSI / PHY rate / failure rate / channel busy over time
    fig_wifi_rho.{pdf,png}    dual-radio bad-state overlap (only if two radios found)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bag import extract  # noqa: E402

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# A statistic over a field the adapter never reported is NaN, which is the right answer;
# the field-availability table below says which fields those were, so the numpy warning
# about it adds nothing.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

NODE_ALIASES = {"mobile1": "mobile_1", "mobile2": "mobile_2"}
# Links are the entities in these figures, so they get the categorical slots in a
# fixed order; the order is by link label so a link keeps its colour across runs.
LINK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
BAD_COLOR = "#e34948"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# WifiLinkStatus documents whole groups of fields as unavailable when `iw station dump`
# or `iw survey dump` is denied: floats come back NaN, several ints come back -1, strings
# come back empty. Those must be reported as missing, not silently summarised as zero.
FIELD_GROUPS = {
    "association": ["essid", "bssid", "mode", "frequency_ghz", "channel", "bit_rate_mbps", "tx_power_dbm"],
    "signal": ["link_quality", "link_quality_ratio", "signal_dbm", "signal_avg_dbm", "noise_dbm", "snr_db"],
    "phy rate": ["rx_bitrate_mbps", "tx_bitrate_mbps", "rx_mcs", "tx_mcs", "rx_nss", "tx_nss",
                 "rx_width_mhz", "tx_width_mhz", "rx_phy_mode", "tx_phy_mode"],
    "station dump": ["tx_retries", "tx_failed", "expected_mbps", "connected_time_s",
                     "sta_rx_bytes", "sta_tx_bytes", "sta_rx_packets", "sta_tx_packets"],
    "channel survey": ["channel_active_ms", "channel_busy_ms", "channel_busy_ratio"],
    "error counters": ["rx_invalid_nwid", "rx_invalid_crypt", "rx_invalid_frag",
                       "tx_excessive_retries", "invalid_misc", "missed_beacon"],
    "interface traffic": ["rx_packets", "rx_bytes", "rx_errors", "rx_dropped", "tx_packets",
                          "tx_bytes", "tx_errors", "tx_dropped", "collisions"],
}
# fields whose documented "unknown" value is -1 rather than NaN
SENTINEL_NEG1 = {
    "channel", "rx_mcs", "tx_mcs", "rx_nss", "tx_nss", "rx_width_mhz", "tx_width_mhz",
    "tx_retries", "tx_failed", "connected_time_s",
    "sta_rx_bytes", "sta_tx_bytes", "sta_rx_packets", "sta_tx_packets",
}


def known_fraction(col: pd.Series, name: str) -> float:
    """Fraction of samples in which this field carries a real value.

    NaN means unknown for the float fields, -1 for the integer fields the message
    definition documents that way, and "" for the string fields."""
    if col.dtype == object or str(col.dtype).startswith("str"):
        return float((col.astype(str).str.len() > 0).mean())
    ok = col.notna()
    if name in SENTINEL_NEG1:
        ok &= col != -1
    return float(ok.mean())


def node_of_topic(topic: str) -> str:
    node = topic.strip("/").split("/")[0]
    return NODE_ALIASES.get(node, node)


def band_of(ghz: float) -> str:
    if not np.isfinite(ghz) or ghz <= 0:
        return "?"
    return "6 GHz" if ghz > 5.9 else ("5 GHz" if ghz > 3 else "2.4 GHz")


def load_glob(extracts: Path, pattern: str):
    frames = []
    for f in sorted(glob.glob(str(extracts / pattern))):
        df = pd.read_parquet(f)
        df["topic"] = "/" + Path(f).stem.replace("__", "/")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def counter_rate(series: pd.Series, denom: pd.Series | None = None) -> pd.Series:
    """Differentiate a cumulative counter, dropping the steps where it reset.

    A per-association counter restarts at zero when the station reassociates, which
    shows up as a negative difference; those samples become NaN rather than a large
    negative rate. With `denom` the result is a ratio of the two increments."""
    d = series.diff()
    d[d < 0] = np.nan
    if denom is None:
        return d
    dd = denom.diff()
    dd[dd <= 0] = np.nan
    return d / dd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=None)
    ap.add_argument("--extracts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run", default="run")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--bad-rssi-dbm", type=float, default=-70.0, help="RSSI at or below which a link counts as Bad")
    ap.add_argument("--bad-failure-rate", type=float, default=0.05, help="TX failure rate above which a link counts as Bad")
    ap.add_argument("--rho-grid-ms", type=float, default=250.0, help="grid the radios are resampled onto to compare their states")
    args = ap.parse_args()

    extracts = args.extracts or Path("extracts") / args.run
    out = args.out or Path("results") / args.run / "wifi"
    out.mkdir(parents=True, exist_ok=True)

    if args.bag is not None and (args.force_extract or not (extracts / "metadata.json").exists()):
        extract(args.bag, extracts)
    elif not (extracts / "metadata.json").exists():
        raise SystemExit(f"no extraction in {extracts}; pass --bag BAG.mcap to create it")
    else:
        print(f"using existing extraction in {extracts}")

    status = load_glob(extracts, "*wifi__status.parquet")
    if status is None:
        raise SystemExit(f"no *wifi__status.parquet in {extracts}")
    iperf = load_glob(extracts, "*wifi__iperf*.parquet")

    t0_ns = int(status["log_time_ns"].min())
    if iperf is not None and len(iperf):
        t0_ns = min(t0_ns, int(iperf["log_time_ns"].min()))
    status["t_s"] = (status["log_time_ns"] - t0_ns) / 1e9
    status["agent"] = status["topic"].map(node_of_topic)
    status["band"] = status["frequency_ghz"].map(band_of)

    # ---- 1. split the topic into radios -------------------------------------------
    # An agent running two radios publishes both on one topic; `interface` tells them
    # apart. Label by agent and band, which is what the dual-link story is about.
    label_of = {}
    for (agent, iface), g in status.groupby(["agent", "interface"]):
        band = g["band"].mode().iloc[0]
        label_of[(agent, iface)] = f"{agent} {band}"
    if len(set(label_of.values())) < len(label_of):  # two radios on one band
        label_of = {k: f"{v} ({k[1]})" for k, v in label_of.items()}
    status["link"] = [label_of[(a, i)] for a, i in zip(status["agent"], status["interface"])]
    links = sorted(status["link"].unique())
    color_of = {lk: LINK_COLORS[i % len(LINK_COLORS)] for i, lk in enumerate(links)}

    # ---- 2. derived rates from the cumulative counters ----------------------------
    parts = []
    for lk, g in status.groupby("link"):
        g = g.sort_values("t_s").copy()
        g["tx_retry_rate"] = counter_rate(g["tx_retries"], g["sta_tx_packets"])
        g["tx_failure_rate"] = counter_rate(g["tx_failed"], g["sta_tx_packets"])
        g["tx_pps"] = counter_rate(g["sta_tx_packets"]) / g["t_s"].diff()
        g["rx_pps"] = counter_rate(g["sta_rx_packets"]) / g["t_s"].diff()
        g["tx_mbps_iface"] = counter_rate(g["tx_bytes"]) * 8 / 1e6 / g["t_s"].diff()
        g["rx_mbps_iface"] = counter_rate(g["rx_bytes"]) * 8 / 1e6 / g["t_s"].diff()
        # the survey counters are cumulative too: the instantaneous occupancy is the
        # ratio of the increments, not the ratio of the totals the message carries
        g["busy_ratio_inst"] = counter_rate(g["channel_busy_ms"], g["channel_active_ms"])
        g["bad"] = (
            (~g["associated"].astype(bool))
            | (g["signal_dbm"] <= args.bad_rssi_dbm)
            | (g["tx_failure_rate"] > args.bad_failure_rate)
        )
        parts.append(g)
    status = pd.concat(parts, ignore_index=True).sort_values("t_s")

    # ---- 3. per-radio summary ------------------------------------------------------
    rows = []
    for lk, g in status.groupby("link"):
        rssi = g["signal_dbm"]
        rows.append(
            {
                "run": args.run,
                "link": lk,
                "agent": g["agent"].iloc[0],
                "interface": g["interface"].iloc[0],
                "mac": g["mac_address"].iloc[0],
                "essid": g["essid"].mode().iloc[0],
                "bssid": g["bssid"].mode().iloc[0],
                "n_bssid": int(g["bssid"].nunique()),
                "band": g["band"].mode().iloc[0],
                "channel": int(g["channel"].mode().iloc[0]),
                "width_mhz": int(g["tx_width_mhz"].mode().iloc[0]),
                "phy_mode": g["tx_phy_mode"].mode().iloc[0],
                "n": len(g),
                "rate_hz": float((len(g) - 1) / max(g["t_s"].max() - g["t_s"].min(), 1e-9)),
                "associated_frac": float(g["associated"].astype(bool).mean()),
                "rssi_median_dbm": float(rssi.median()),
                "rssi_p05_dbm": float(rssi.quantile(0.05)),
                "rssi_p95_dbm": float(rssi.quantile(0.95)),
                "rssi_min_dbm": float(rssi.min()),
                "rssi_range_db": float(rssi.max() - rssi.min()),
                "snr_valid_frac": float(g["snr_valid"].astype(bool).mean()),
                "tx_phy_median_mbps": float(g["tx_bitrate_mbps"].median()),
                "tx_phy_min_mbps": float(g["tx_bitrate_mbps"].min()),
                "tx_mcs_median": float(g["tx_mcs"].median()),
                "tx_nss_mode": int(g["tx_nss"].mode().iloc[0]),
                "tx_retry_rate_median": float(g["tx_retry_rate"].median()),
                "tx_retry_rate_p95": float(g["tx_retry_rate"].quantile(0.95)),
                "tx_failure_rate_median": float(g["tx_failure_rate"].median()),
                "tx_failure_rate_p95": float(g["tx_failure_rate"].quantile(0.95)),
                "busy_ratio_median": float(g["busy_ratio_inst"].median()),
                "busy_ratio_p95": float(g["busy_ratio_inst"].quantile(0.95)),
                "missed_beacon_total": int(g["missed_beacon"].max() - g["missed_beacon"].min()),
                "bad_frac": float(g["bad"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("link").reset_index(drop=True)
    summary.to_csv(out / "wifi_links.csv", index=False)

    # ---- 3b. which fields the adapter actually reported ----------------------------
    avail_rows = []
    for lk, g in status.groupby("link"):
        for group, fields in FIELD_GROUPS.items():
            for f in fields:
                if f not in g.columns:
                    continue
                avail_rows.append({"link": lk, "group": group, "field": f,
                                   "known_frac": known_fraction(g[f], f)})
    avail = pd.DataFrame(avail_rows)
    avail.to_csv(out / "wifi_field_availability.csv", index=False)
    group_avail = avail.pivot_table(index="group", columns="link", values="known_frac", aggfunc="mean")
    missing = avail[avail["known_frac"] < 0.99]
    # a group nobody reported at all -- the whole `iw` subcommand was unavailable
    dead_groups = sorted({g for g, sub in avail.groupby("group") if sub["known_frac"].max() == 0.0})

    # ---- 4. association / roaming events -------------------------------------------
    ev = []
    for lk, g in status.groupby("link"):
        g = g.sort_values("t_s")
        for field in ["associated", "bssid", "essid", "channel"]:
            chg = g[field].ne(g[field].shift())
            chg.iloc[0] = False
            for _, r in g[chg].iterrows():
                ev.append({"t_s": round(float(r["t_s"]), 2), "link": lk, "field": field, "value": r[field]})
    events = pd.DataFrame(ev).sort_values("t_s").reset_index(drop=True) if ev else None
    if events is not None:
        events.to_csv(out / "wifi_events.csv", index=False)

    # ---- 5. iperf -------------------------------------------------------------------
    iperf_df = None
    if iperf is not None and len(iperf):
        iperf = iperf.copy()
        iperf["t_s"] = (iperf["log_time_ns"] - t0_ns) / 1e9
        iperf["agent"] = iperf["topic"].map(node_of_topic)
        iperf["kind"] = np.where(iperf["topic"].str.contains("r2r"), "robot-to-robot", "to server")
        iperf["direction"] = np.where(iperf["reverse"].astype(bool), "downlink", "uplink")
        iperf_df = iperf[
            ["t_s", "topic", "agent", "kind", "direction", "protocol", "server_address", "success",
             "bitrate_mbps", "bytes", "retransmits", "rtt_ms_mean", "rtt_ms_max", "duration_s", "error"]
        ].sort_values("t_s").reset_index(drop=True)
        iperf_df.to_csv(out / "wifi_iperf.csv", index=False)

    # ---- 6. dual-radio correlation --------------------------------------------------
    # Two radios only share a "state" at a common instant, so both are resampled onto
    # one grid before their Bad states are compared.
    rho_df = None
    dual = [a for a, g in status.groupby("agent") if g["link"].nunique() >= 2]
    rho_rows, rho_series = [], {}
    for agent in dual:
        sub = status[status["agent"] == agent]
        lks = sorted(sub["link"].unique())[:2]
        grid = np.arange(0.0, sub["t_s"].max(), args.rho_grid_ms / 1000.0)
        states = {}
        for lk in lks:
            g = sub[sub["link"] == lk].sort_values("t_s")
            idx = np.searchsorted(g["t_s"].to_numpy(), grid).clip(0, len(g) - 1)
            near = np.abs(g["t_s"].to_numpy()[idx] - grid) <= args.rho_grid_ms / 1000.0
            st = g["bad"].to_numpy()[idx].astype(float)
            st[~near] = np.nan
            states[lk] = st
        a, b = states[lks[0]], states[lks[1]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[ok].astype(bool), b[ok].astype(bool)
        rho_series[agent] = (grid[ok], a, b, lks)
        p1, p2 = float(a.mean()), float(b.mean())
        pj = float((a & b).mean())
        ind = p1 * p2
        rho_rows.append(
            {
                "run": args.run,
                "agent": agent,
                "link_a": lks[0],
                "link_b": lks[1],
                "n_samples": int(ok.sum()),
                "grid_ms": args.rho_grid_ms,
                "p_bad_a": p1,
                "p_bad_b": p2,
                "p_joint_measured": pj,
                "p_joint_if_independent": ind,
                # P(the other link is also Bad | this one is). Independence gives the
                # other link's own Bad rate; full correlation gives 1.
                "rho_b_given_a": pj / p1 if p1 > 0 else np.nan,
                "rho_a_given_b": pj / p2 if p2 > 0 else np.nan,
                # 0 = independent, 1 = one link is Bad whenever the other is
                "excess_correlation": (pj - ind) / (min(p1, p2) - ind) if min(p1, p2) > ind else np.nan,
                "phi": float(np.corrcoef(a.astype(float), b.astype(float))[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan,
            }
        )
    if rho_rows:
        rho_df = pd.DataFrame(rho_rows)
        rho_df.to_csv(out / "wifi_rho.csv", index=False)

    # ---- 7. figures ------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT})
    t_max = float(status["t_s"].max())

    def group_known(group: str) -> bool:
        sub = avail[avail["group"] == group]
        return bool(len(sub)) and sub["known_frac"].max() > 0.0

    panels = ["rssi", "rate"]
    if group_known("station dump"):
        panels.append("failures")
    if group_known("channel survey"):
        panels.append("occupancy")
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.16, 1.6 * len(panels) + 0.6),
                             sharex=True, constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    at = {name: axes[i] for i, name in enumerate(panels)}
    letter = {name: "(" + chr(ord("a") + i) + ")" for i, name in enumerate(panels)}

    ax = at["rssi"]
    for lk in links:
        g = status[status["link"] == lk]
        ax.plot(g["t_s"], g["signal_dbm"], lw=1.1, color=color_of[lk], label=lk)
    lo, hi = float(status["signal_dbm"].min()), float(status["signal_dbm"].max())
    pad = max((hi - lo) * 0.12, 1.5)
    legend_handles = list(ax.get_lines()[: len(links)])
    legend_labels = list(links)
    rssi_thr_drawn = args.bad_rssi_dbm >= lo - pad
    if rssi_thr_drawn:
        ax.axhline(args.bad_rssi_dbm, color=BAD_COLOR, lw=0.8, ls="--")
        ax.set_ylim(min(lo - pad, args.bad_rssi_dbm - pad), hi + pad)
    else:
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("RSSI [dBm]")
    margin = "" if rssi_thr_drawn else f"  (weakest sample {lo - args.bad_rssi_dbm:.0f} dB above the Bad threshold)"
    ax.set_title(f"{letter['rssi']} received signal strength{margin}", loc="left", fontsize=8)
    ax.grid(True, color=GRID, lw=0.5)

    ax = at["rate"]
    for lk in links:
        g = status[status["link"] == lk]
        ax.plot(g["t_s"], g["tx_bitrate_mbps"], lw=1.1, color=color_of[lk])
    if iperf_df is not None:
        for kind, mk in [("to server", "o"), ("robot-to-robot", "^")]:
            sub = iperf_df[(iperf_df["kind"] == kind) & iperf_df["success"].astype(bool)]
            if len(sub):
                (h,) = ax.plot(sub["t_s"], sub["bitrate_mbps"], mk, ms=5, mfc="none", mec=TEXT, mew=1.1, ls="none")
                legend_handles.append(h)
                legend_labels.append(f"iperf, {kind}")
    # PHY rate and goodput share a unit but can differ by a large factor on a fast link;
    # a log axis keeps both legible and leaves the gap between them visible as an offset
    vals = [status["tx_bitrate_mbps"]]
    if iperf_df is not None and len(iperf_df):
        vals.append(iperf_df.loc[iperf_df["success"].astype(bool), "bitrate_mbps"])
    allv = pd.concat(vals).dropna()
    allv = allv[allv > 0]
    if len(allv) and allv.max() / allv.min() > 8:
        ax.set_yscale("log")
    ax.set_ylabel("rate [Mbit/s]")
    ax.set_title(f"{letter['rate']} lines: negotiated PHY rate.  markers: measured iperf goodput",
                 loc="left", fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, which="both")

    if "failures" in at:
        ax = at["failures"]
        for lk in links:
            g = status[status["link"] == lk]
            ax.plot(g["t_s"], g["tx_failure_rate"] * 100, lw=1.1, color=color_of[lk], label=lk)
        fmax = float(status["tx_failure_rate"].max() * 100) if status["tx_failure_rate"].notna().any() else 0.0
        note = ""
        if fmax <= 0.0:
            ax.set_ylim(-0.05, 1.0)
            note = "  (no frame failed after retries anywhere in this run)"
        elif fmax < args.bad_failure_rate * 100:
            ax.set_ylim(-0.05 * fmax, fmax * 1.15)
        else:
            ax.axhline(args.bad_failure_rate * 100, color=BAD_COLOR, lw=0.8, ls="--")
        ax.set_ylabel("TX failures [%]")
        ax.set_title(f"{letter['failures']} frames that failed after all retries{note}", loc="left", fontsize=8)
        ax.grid(True, color=GRID, lw=0.5)

    if "occupancy" in at:
        ax = at["occupancy"]
        for lk in links:
            g = status[status["link"] == lk]
            ax.plot(g["t_s"], g["busy_ratio_inst"], lw=1.1, color=color_of[lk], label=lk)
        ax.set_ylabel("channel busy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{letter['occupancy']} channel occupancy", loc="left", fontsize=8)
        ax.grid(True, color=GRID, lw=0.5)

    axes[-1].set_xlabel("time in run [s]")

    thr = plt.Line2D([], [], color=BAD_COLOR, lw=0.8, ls="--")
    legend_handles.append(thr)
    legend_labels.append(f"Bad: RSSI ≤ {args.bad_rssi_dbm:.0f} dBm or failures > {args.bad_failure_rate:.0%}"
                         + ("" if rssi_thr_drawn else " (off scale)"))
    fig.legend(legend_handles, legend_labels, frameon=False, fontsize=7,
               ncol=min(len(legend_labels), 3), loc="outside upper center")

    for a_ in axes:
        a_.set_xlim(0, t_max)
    fig.savefig(out / "fig_wifi_link.pdf")
    fig.savefig(out / "fig_wifi_link.png", dpi=200)
    plt.close(fig)

    if rho_series:
        agent = sorted(rho_series)[0]
        grid, a, b, lks = rho_series[agent]
        r = rho_df[rho_df["agent"] == agent].iloc[0]
        fig, axes = plt.subplots(3, 1, figsize=(7.16, 4.2), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [2, 1, 1.4]})

        ax = axes[0]
        for lk in lks:
            g = status[status["link"] == lk]
            ax.plot(g["t_s"], g["signal_dbm"], lw=1.1, color=color_of[lk], label=lk)
        ax.axhline(args.bad_rssi_dbm, color=BAD_COLOR, lw=0.8, ls="--")
        ax.set_ylabel("RSSI [dBm]")
        ax.set_title(f"(a) {agent}: both radios", loc="left", fontsize=8)
        ax.tick_params(labelbottom=False)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
        ax.grid(True, color=GRID, lw=0.5)

        ax = axes[1]
        for i, (lk, st) in enumerate(zip(lks, [a, b])):
            ax.fill_between(grid, i, i + 0.8, where=st, color=color_of[lk], lw=0, step="mid")
        ax.fill_between(grid, 2, 2.8, where=a & b, color=BAD_COLOR, lw=0, step="mid")
        ax.set_yticks([0.4, 1.4, 2.4])
        ax.set_yticklabels([lks[0], lks[1], "both"])
        ax.set_ylim(-0.1, 2.9)
        ax.set_xlabel("time in run [s]")
        ax.set_title("(b) intervals in the Bad state", loc="left", fontsize=8)
        ax.grid(True, axis="x", color=GRID, lw=0.5)

        ax = axes[2]
        vals = [r["p_joint_if_independent"] * 100, r["p_joint_measured"] * 100]
        bars = ax.barh([0, 1], vals, height=0.5, color=[GRID, BAD_COLOR])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["if the links were\nindependent", "measured"])
        for y, v in zip([0, 1], vals):
            ax.text(v, y, f"  {v:.1f}%", va="center", fontsize=7, color=TEXT)
        ax.set_xlim(0, max(max(vals) * 1.35, 1.0))
        ax.set_xlabel("percentage of the run with BOTH links Bad")
        ax.set_title("(c) does the second radio actually diversify?", loc="left", fontsize=8)
        ax.grid(True, axis="x", color=GRID, lw=0.5)

        for ax_ in axes[:2]:
            ax_.set_xlim(0, t_max)
        fig.savefig(out / "fig_wifi_rho.pdf")
        fig.savefig(out / "fig_wifi_rho.png", dpi=200)
        plt.close(fig)

    # ---- 8. markdown ------------------------------------------------------------------
    md = [f"# Wi-Fi link quality — run `{args.run}`", ""]
    if dead_groups:
        md += [
            "> **Not measured in this run.** The adapter reported nothing for: **"
            + ", ".join(dead_groups)
            + "**. `WifiLinkStatus` documents these groups as NaN or −1 when the underlying "
            "`iw station dump` / `iw survey dump` is unavailable or denied, so any statistic over "
            "them would be an artefact. They are excluded from the tables and figures below.",
            "",
        ]
    md += [
        f"All links are agent → access point. Bad state: not associated, or RSSI ≤ "
        f"{args.bad_rssi_dbm:.0f} dBm, or TX failure rate > {args.bad_failure_rate:.0%}.",
        "",
        "## Radios",
        "",
        summary[["link", "agent", "interface", "essid", "bssid", "n_bssid", "band", "channel", "width_mhz",
                 "phy_mode", "n", "rate_hz", "associated_frac"]].round(2).to_markdown(index=False),
        "",
        "## Link quality",
        "",
        summary[["link", "rssi_median_dbm", "rssi_p05_dbm", "rssi_min_dbm", "rssi_range_db",
                 "tx_phy_median_mbps", "tx_mcs_median", "tx_retry_rate_median", "tx_failure_rate_p95",
                 "busy_ratio_median", "missed_beacon_total", "bad_frac"]].round(3).to_markdown(index=False),
        "",
        "Retry, failure and occupancy figures are differentiated from the cumulative counters, so they "
        "are per-interval rates rather than run totals. `snr_valid_frac` is "
        + ", ".join(f"{r['link']} {r['snr_valid_frac']:.0%}" for _, r in summary.iterrows())
        + " — where it is zero the adapter never reported a noise floor, so RSSI is the only signal measure available.",
        "",
    ]
    md += [
        "## Field availability",
        "",
        "Fraction of samples in which each group of `WifiLinkStatus` fields carried a real value "
        "(not NaN, not −1, not an empty string).",
        "",
        group_avail.round(3).to_markdown(),
        "",
    ]
    if len(missing):
        md += ["Fields below full availability:", "",
               missing.sort_values(["known_frac", "link"]).round(3).to_markdown(index=False), ""]
    if events is not None and len(events):
        md += ["## Association and roaming events", "", events.to_markdown(index=False), ""]
    else:
        md += ["No association, BSSID, ESSID or channel changes during the run: every radio stayed on one AP.", ""]
    if iperf_df is not None:
        md += ["## iperf", "", iperf_df.round(2).to_markdown(index=False), ""]
    if rho_df is not None:
        md += [
            "## Dual-radio correlation",
            "",
            "`p_joint_measured` is the fraction of the run in which **both** radios were Bad at once. "
            "`p_joint_if_independent` is what two links failing independently would have produced. "
            "Measured well above independent means the two radios fade together, so a second copy of a "
            "message buys little; near independent means the diversity is real. `rho_b_given_a` is the "
            "probability the second link is Bad given the first is — it equals the second link's own Bad "
            "rate under independence and 1 under full correlation.",
            "",
            rho_df.round(4).to_markdown(index=False),
            "",
        ]
    (out / "wifi_summary.md").write_text("\n".join(md))

    # ---- 9. LaTeX -----------------------------------------------------------------------
    def tt(x):
        return "\\texttt{" + str(x).replace("_", "\\_") + "}"

    def pct(x):
        # % is a comment character in LaTeX and must be escaped
        return f"{x * 100:.1f}\\%"

    per_link = "; ".join(
        f"{tt(r['link'])} on channel {r['channel']} ({r['width_mhz']}\\,MHz {r['phy_mode']}) held a median "
        f"RSSI of {r['rssi_median_dbm']:.0f}\\,dBm (5th percentile {r['rssi_p05_dbm']:.0f}\\,dBm) at a median "
        f"PHY rate of {r['tx_phy_median_mbps']:.0f}\\,Mbit/s"
        for _, r in summary.iterrows()
    )
    roam = ("Every radio stayed associated to a single access point for the whole run."
            if events is None or len(events) == 0
            else f"{len(events)} association, BSSID, ESSID or channel changes occurred during the run.")
    iperf_sentence = ""
    if iperf_df is not None and len(iperf_df):
        ok = iperf_df[iperf_df["success"].astype(bool)]
        srv = ok[ok["kind"] == "to server"]
        r2r = ok[ok["kind"] == "robot-to-robot"]
        iperf_sentence = (
            f" Active throughput tests to the fixed server returned a median of {srv['bitrate_mbps'].median():.0f}\\,Mbit/s "
            f"over {len(srv)} tests"
        )
        if len(r2r):
            iperf_sentence += (
                f", and robot-to-robot tests -- which traverse the access point in both directions -- "
                f"{r2r['bitrate_mbps'].median():.0f}\\,Mbit/s over {len(r2r)} tests"
            )
        iperf_sentence += "."
    rho_sentence = ""
    if rho_df is not None:
        r = rho_df.iloc[0]
        rho_sentence = (
            f" {tt(r['agent'])} carries two radios ({tt(r['link_a'])}, {tt(r['link_b'])}), which lets the "
            f"inter-link correlation be measured rather than assumed: each was in the Bad state for "
            f"{pct(r['p_bad_a'])} and {pct(r['p_bad_b'])} of the run, and both were Bad simultaneously for "
            f"{pct(r['p_joint_measured'])}, against the {pct(r['p_joint_if_independent'])} that independent "
            f"links would have produced."
        )
    # what the adapters actually reported decides which caveats belong in the paper
    snr_frac = float(avail[avail["field"] == "snr_db"]["known_frac"].max()) if len(avail) else 0.0
    noise_sentence = (
        " The adapters report no noise floor, so RSSI rather than SNR is the signal measure throughout."
        if snr_frac < 0.01 else ""
    )
    dead_sentence = ""
    if dead_groups:
        dead_sentence = (
            f" The {' and '.join(dead_groups)} field group{'s' if len(dead_groups) != 1 else ''} "
            f"{'were' if len(dead_groups) != 1 else 'was'} not reported by the adapters in this run "
            f"(the message definition marks them unavailable when the underlying \\texttt{{iw}} query is denied "
            f"or unsupported by the adapter's driver), so no statistic is quoted over them."
        )
    derived_sentence = ""
    if "station dump" not in dead_groups:
        derived_sentence = (
            " Retry and failure figures are differentiated from the driver's cumulative counters and are"
            " therefore per-interval rates."
        )
    tex = f"""\\subsubsection{{Wi-Fi Link Quality}}
Both mobile agents associate with the same access point, so the link measured on each robot is its own
uplink to that AP. {per_link}. {roam}{iperf_sentence}{rho_sentence}{derived_sentence}{dead_sentence}{noise_sentence}
"""
    (out / "wifi_subsection.tex").write_text(tex)

    if dead_groups:
        print(f"\nNOT REPORTED by the adapter in this run: {', '.join(dead_groups)}"
              f"\n  -> statistics over those fields are omitted; see wifi_field_availability.csv\n")
    print(summary[["link", "band", "channel", "n", "rate_hz", "rssi_median_dbm", "rssi_min_dbm",
                   "tx_phy_median_mbps", "tx_failure_rate_p95", "bad_frac"]].round(3).to_string(index=False))
    if rho_df is not None:
        print()
        print(rho_df[["agent", "p_bad_a", "p_bad_b", "p_joint_measured", "p_joint_if_independent",
                      "rho_b_given_a", "phi"]].round(4).to_string(index=False))
    print(f"\nwrote {out}/wifi_summary.md, wifi_subsection.tex, fig_wifi_link.pdf/png"
          + (", fig_wifi_rho.pdf/png" if rho_series else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
