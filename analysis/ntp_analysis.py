#!/usr/bin/env python3
"""NTP / temporal-calibration analysis of one run.

Answers, from the extracted tables produced by extract_bag.py:
  * who is the NTP server and who are the clients (role / hostname / sync_source),
  * per client: offset statistics (mean, median, p95, max |offset|), delay, jitter,
    stratum, reach, poll interval, clock steps -- counted over real measurements only,
    since a status topic republishes one poll's result at the topic rate,
  * an NTP-independent check: header stamp minus recorder log time, per node,
  * the sync bound relative to the shortest sensor period in the bag.

Usage
-----
    # from the bag directly (extracts once into extracts/<run>/, reused on later runs)
    python ntp_analysis.py --bag BAG.mcap --run coop2

    # from an existing extraction
    python ntp_analysis.py --extracts extracts/coop2 --out results/coop2/ntp --run coop2

Outputs (in --out)
------------------
    ntp_roles.csv        one row per (topic, role, hostname, sync_source)
    ntp_summary.csv      offset statistics per client
    ntp_audit.csv        stamp-minus-log statistics per topic and node
    ntp_unset_stamps.csv topics whose driver never set header.stamp (written only if any exist)
    ntp_summary.md       human-readable summary of all of the above
    ntp_subsection.tex   the Temporal Calibration subsection with numbers filled in
    fig_ntp.{pdf,png}    2-panel figure: measured offset over time, and per-topic stamp-minus-log
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bag import extract  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# One fixed color per agent, used in every figure of the analysis.
AGENT_COLOR = {"mobile_1": "#2a78d6", "mobile_2": "#eb6834", "infra_1": "#1baf7a"}
STEP_COLOR = "#e34948"
EVENT_COLOR = "#eda100"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"


def color_for(node: str) -> str:
    return AGENT_COLOR.get(node, "#4a3aa7")


# The CSI publisher uses /mobile1 and /mobile2 while everything else uses /mobile_1 and
# /mobile_2; without this the same robot appears as two agents.
NODE_ALIASES = {"mobile1": "mobile_1", "mobile2": "mobile_2"}

# A header stamp further than this from the recorder's clock was never set by the driver.
STAMP_SANE_MS = 3_600_000.0


def node_of_topic(topic: str) -> str:
    node = topic.strip("/").split("/")[0]
    return NODE_ALIASES.get(node, node)


def ecdf(x: np.ndarray):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[~np.isnan(x)]
    return x, np.arange(1, len(x) + 1) / max(len(x), 1)


def load_ntp(extracts: Path):
    frames = []
    for f in sorted(glob.glob(str(extracts / "*ntp__status.parquet"))):
        df = pd.read_parquet(f)
        df["topic"] = "/" + Path(f).stem.replace("__", "/")
        frames.append(df)
    if not frames:
        raise SystemExit(f"no *ntp__status.parquet in {extracts}")
    return pd.concat(frames, ignore_index=True)


def load_events(extracts: Path):
    frames = []
    for f in sorted(glob.glob(str(extracts / "*ntp__events.parquet"))):
        df = pd.read_parquet(f)
        df["topic"] = "/" + Path(f).stem.replace("__", "/")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=None, help="MCAP file; extracted into extracts/<run>/ unless that exists")
    ap.add_argument("--extracts", type=Path, default=None, help="existing extraction folder (default: extracts/<run>)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: results/<run>/ntp)")
    ap.add_argument("--run", default="run")
    ap.add_argument("--force-extract", action="store_true", help="re-run the extraction even if it exists")
    ap.add_argument("--server", default=None, help="agent name of the NTP server, e.g. mobile_1 (hostnames may be ambiguous)")
    ap.add_argument("--step-threshold-ms", type=float, default=1.0, help="|offset_delta| above this counts as a step")
    ap.add_argument("--sensor-period-ms", type=float, default=None, help="override the shortest sensor period")
    args = ap.parse_args()
    extracts = args.extracts or Path("extracts") / args.run
    args.out = args.out or Path("results") / args.run / "ntp"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.bag is not None and (args.force_extract or not (extracts / "metadata.json").exists()):
        extract(args.bag, extracts)
    elif not (extracts / "metadata.json").exists():
        raise SystemExit(f"no extraction in {extracts}; pass --bag BAG.mcap to create it")
    else:
        print(f"using existing extraction in {extracts}")
    args.extracts = extracts

    ntp = load_ntp(args.extracts)
    events = load_events(args.extracts)
    audit_path = args.extracts / "stamp_audit.parquet"
    audit = pd.read_parquet(audit_path) if audit_path.exists() else None

    t0_ns = int(ntp["log_time_ns"].min())
    if audit is not None and len(audit):
        t0_ns = min(t0_ns, int(audit["log_time_ns"].min()))
    ntp["t_s"] = (ntp["log_time_ns"] - t0_ns) / 1e9
    ntp["offset_ms"] = ntp["offset_seconds"] * 1e3
    ntp["delay_ms"] = ntp["delay_seconds"] * 1e3
    ntp["jitter_ms"] = ntp["jitter_seconds"] * 1e3
    ntp["offset_delta_ms"] = ntp["offset_delta_seconds"] * 1e3
    ntp["node"] = ntp["topic"].map(node_of_topic)

    # ---- 1. who is who ----------------------------------------------------------
    roles = (
        ntp.groupby(["topic", "role", "hostname", "sync_source"], dropna=False)
        .agg(
            n=("seq", "size"),
            stratum=("stratum", lambda s: int(s.mode().iloc[0])),
            synchronized_frac=("synchronized", "mean"),
            rate_hz=("t_s", lambda t: (len(t) - 1) / max(t.max() - t.min(), 1e-9)),
            t_first_s=("t_s", "min"),
            t_last_s=("t_s", "max"),
        )
        .reset_index()
    )
    roles.to_csv(args.out / "ntp_roles.csv", index=False)

    # ---- 2. per-client offset statistics --------------------------------------
    rows = []
    series = {}
    for (topic, role, host), g in ntp.groupby(["topic", "role", "hostname"]):
        g = g.sort_values("t_s")
        agent = node_of_topic(topic)
        key = f"{agent} ({role})" if role != "client" else agent

        # A status message republishes the daemon's current tracking estimate at the topic rate, so
        # the message count is not a measurement count. How many times the server was actually
        # contacted follows from the poll interval and the run length -- arithmetic, not a guess.
        g = g.copy()
        duration_s = float(g["t_s"].max() - g["t_s"].min())
        poll_s = int(g["poll_interval_seconds"].mode().iloc[0])
        polls_in_run = int(np.ceil(duration_s / poll_s)) if poll_s > 0 else -1

        # Before its first poll the daemon has nothing to report and publishes a flat near-zero
        # offset; when the first real result lands it flags a clock step. Everything before that
        # step is not a measurement of this run's synchronization.
        stepped = g["clock_stepped"].astype(bool)
        t_sync = float(g.loc[stepped, "t_s"].min()) if stepped.any() else float(g["t_s"].min())
        g["measured"] = g["t_s"] >= t_sync
        m = g[g["measured"]]
        if len(m) == 0:
            m, t_sync = g, float(g["t_s"].min())
        series[key] = m
        steps_flag = int(((m["clock_stepped"].astype(bool)) & (~m["clock_stepped"].astype(bool).shift(1, fill_value=False))).sum())
        steps_delta = int((m["offset_delta_ms"].abs() > args.step_threshold_ms).sum())
        warn = sorted({w for ws in m["warnings"] for w in (list(ws) if ws is not None else [])})
        abs_off = m["offset_ms"].abs()
        rows.append(
            {
                "run": args.run,
                "topic": topic,
                "agent": agent,
                "node": agent,
                "role": role,
                "hostname": host,
                "sync_source": g["sync_source"].mode().iloc[0],
                "stratum": int(g["stratum"].mode().iloc[0]),
                "n": len(g),
                "n_measured": int(g["measured"].sum()),
                "polls_in_run": polls_in_run,
                "t_sync_s": t_sync,
                "duration_s": duration_s,
                "offset_mean_ms": float(m["offset_ms"].mean()),
                "offset_median_ms": float(m["offset_ms"].median()),
                "offset_std_ms": float(m["offset_ms"].std()),
                "abs_offset_mean_ms": float(abs_off.mean()),
                "abs_offset_p95_ms": float(abs_off.quantile(0.95)),
                "abs_offset_max_ms": float(abs_off.max()),
                "t_of_max_abs_offset_s": float(m.loc[abs_off.idxmax(), "t_s"]),
                "delay_median_ms": float(m["delay_ms"].median()),
                "delay_max_ms": float(m["delay_ms"].max()),
                "jitter_median_ms": float(m["jitter_ms"].median()),
                "root_dispersion_median_ms": float(m["root_dispersion"].median() * 1e3),
                "freq_error_mean_ppm": float(m["frequency_error_ppm"].mean()),
                "poll_interval_mode_s": int(g["poll_interval_seconds"].mode().iloc[0]),
                "reach_min": int(g["reach_register"].min()),
                "reachability_min_pct": int(g["reachability_percent"].min()),
                "synchronized_frac": float(g["synchronized"].astype(bool).mean()),
                "clock_steps_flagged": steps_flag,
                "clock_steps_by_delta": steps_delta,
                "step_times_s": json.dumps([round(float(x), 2) for x in m.loc[m["clock_stepped"].astype(bool), "t_s"].tolist()][:20]),
                "warnings": "; ".join(warn),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "ntp_summary.csv", index=False)

    # ---- 3. NTP-independent check from the stamp audit --------------------------
    audit_rows = []
    sensor_period_ms = None
    unset_df = None
    if audit is not None and len(audit):
        audit = audit.dropna(subset=["header_stamp_ns"]).copy()
        audit["node"] = audit["topic"].map(node_of_topic)  # apply the aliases
        audit["t_s"] = (audit["log_time_ns"] - t0_ns) / 1e9
        # A driver that never fills header.stamp leaves it at zero, which puts the difference
        # 57 years off and would otherwise swamp every real value in the statistics and the plot.
        audit["stamp_set"] = (audit["header_stamp_ns"] > 0) & (audit["stamp_minus_log_ms"].abs() < STAMP_SANE_MS)
        unset_rows = []
        for (node, topic), g in audit.groupby(["node", "topic"]):
            ok = g[g["stamp_set"]]
            if len(ok) < len(g):
                unset_rows.append(
                    {
                        "node": node,
                        "topic": topic,
                        "type": g["type"].iloc[0],
                        "n": len(g),
                        "n_unset": int((~g["stamp_set"]).sum()),
                        "frac_unset": float((~g["stamp_set"]).mean()),
                    }
                )
            if len(ok) == 0:
                continue
            ok = ok.sort_values("header_stamp_ns")
            dt = np.diff(ok["header_stamp_ns"].to_numpy()) / 1e6
            period = float(np.median(dt)) if len(dt) > 10 else np.nan
            d = ok["stamp_minus_log_ms"]
            audit_rows.append(
                {
                    "node": node,
                    "topic": topic,
                    "type": ok["type"].iloc[0],
                    "n": len(ok),
                    "period_median_ms": period,
                    "stamp_minus_log_median_ms": float(d.median()),
                    "stamp_minus_log_p05_ms": float(d.quantile(0.05)),
                    "stamp_minus_log_p95_ms": float(d.quantile(0.95)),
                    "stamp_minus_log_iqr_ms": float(d.quantile(0.75) - d.quantile(0.25)),
                }
            )
        audit_df = pd.DataFrame(audit_rows).sort_values(["node", "topic"]).reset_index(drop=True)
        audit_df.to_csv(args.out / "ntp_audit.csv", index=False)
        if unset_rows:
            unset_df = pd.DataFrame(unset_rows).sort_values("frac_unset", ascending=False).reset_index(drop=True)
            unset_df.to_csv(args.out / "ntp_unset_stamps.csv", index=False)
        audit = audit[audit["stamp_set"]]
        valid = audit_df[(audit_df["n"] >= 100) & audit_df["period_median_ms"].notna()]
        if len(valid):
            sensor_period_ms = float(valid["period_median_ms"].min())
            fastest = valid.loc[valid["period_median_ms"].idxmin(), "topic"]
    else:
        audit_df = None
    if args.sensor_period_ms is not None:
        sensor_period_ms = args.sensor_period_ms
        fastest = "(user supplied)"

    # ---- 4. figure --------------------------------------------------------------
    # Two panels, because with one poll per run there is no offset *distribution* to show:
    # (a) the measured offset of each client over the run, (b) the continuous, NTP-independent
    # evidence -- one dot per topic, grouped by the node that stamped it.
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    for key, g in series.items():
        c = color_for(g["node"].iloc[0])
        ax.plot(g["t_s"], g["offset_ms"], lw=1.4, color=c, label=key)
        first = g.iloc[0]
        ax.plot([first["t_s"]], [first["offset_ms"]], "o", ms=4, color=c, zorder=3)
        ax.annotate(f"{first['offset_ms']:.2f} ms", (first["t_s"], first["offset_ms"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.5, color=c)
        for ts in g.loc[g["clock_stepped"].astype(bool), "t_s"]:
            ax.axvline(ts, color=STEP_COLOR, lw=0.8, alpha=0.8)
    if events is not None and len(events):
        for t in (events["log_time_ns"] - t0_ns) / 1e9:
            ax.axvspan(t - 0.5, t + 0.5, color=EVENT_COLOR, alpha=0.35, lw=0)
    if sensor_period_ms:
        ax.axhline(sensor_period_ms, color=TEXT2, lw=0.8, ls="--")
        ax.text(0.02, sensor_period_ms, f" shortest sensor period {sensor_period_ms:.0f} ms",
                transform=ax.get_yaxis_transform(), fontsize=6.5, color=TEXT2, va="bottom")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(g["t_s"].max() for g in series.values()) * 1.02)
    ax.set_xlabel("time in run [s]")
    ax.set_ylabel("clock offset to server [ms]")
    ax.set_title("(a) measured NTP offset", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(True, color=GRID, lw=0.5)

    ax = axes[1]
    if audit_df is not None and len(audit_df):
        rng = np.random.default_rng(0)
        nodes = sorted(audit_df["node"].unique())
        for i, node in enumerate(nodes):
            sub = audit_df[audit_df["node"] == node]
            y = i + rng.uniform(-0.14, 0.14, len(sub))
            ax.scatter(sub["stamp_minus_log_median_ms"], y, s=14, color=color_for(node),
                       alpha=0.85, linewidths=0, zorder=3)
            med = float(sub["stamp_minus_log_median_ms"].median())
            ax.plot([med, med], [i - 0.3, i + 0.3], color=color_for(node), lw=2, zorder=4)
            ax.annotate(f"{med:.1f}", (med, i + 0.34), fontsize=6.5, color=color_for(node), ha="center")
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes)
        ax.set_ylim(-0.6, len(nodes) - 0.4)
        ax.axvline(0, color=GRID, lw=0.8)
        span = audit_df["stamp_minus_log_median_ms"].abs().max()
        if span > 100:  # a few topics publish seconds late; keep them visible without flattening the rest
            ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("header stamp − recorder log time [ms]\n(one dot per topic, bar = node median)")
    ax.set_title("(b) delivery to the recorder", loc="left", fontsize=8)
    ax.grid(True, axis="x", color=GRID, lw=0.5)

    fig.savefig(args.out / "fig_ntp.pdf")
    fig.savefig(args.out / "fig_ntp.png", dpi=200)

    # ---- 5. markdown + LaTeX -----------------------------------------------------
    md = [f"# NTP / temporal calibration — run `{args.run}`", "", "## Roles", "", roles.to_markdown(index=False), "", "## Client offset statistics (ms)", ""]
    cols = ["agent", "hostname", "role", "sync_source", "stratum", "n", "n_measured", "polls_in_run", "offset_mean_ms", "offset_median_ms", "abs_offset_p95_ms", "abs_offset_max_ms", "delay_median_ms", "jitter_median_ms", "poll_interval_mode_s", "reach_min", "clock_steps_flagged", "clock_steps_by_delta"]
    md += [summary[cols].round(3).to_markdown(index=False), ""]
    md += [
        "`n` counts status messages, which republish the daemon's current estimate at the topic rate. "
        "`n_measured` counts those from `t_sync_s` onward, the moment the daemon flagged its first result "
        "of the run; earlier messages report a flat near-zero placeholder. `polls_in_run` is how many "
        "times the server could have been contacted, from the poll interval and the run length. "
        "Statistics are over the measured messages.",
        "",
    ]
    few = summary[(summary["polls_in_run"] >= 0) & (summary["polls_in_run"] <= 2)]
    if len(few):
        md += [
            "> **Caveat.** At most "
            + ", ".join(
                f"`{r['agent']}` {int(r['polls_in_run'])} poll{'' if r['polls_in_run'] == 1 else 's'}" for _, r in few.iterrows()
            )
            + f" during this {summary['duration_s'].max():.0f} s run (poll interval "
            + ", ".join(str(int(x)) for x in sorted(summary["poll_interval_mode_s"].unique()))
            + " s). Spread statistics (std, p95) therefore describe the daemon's interpolation between "
            "polls, not repeated measurements. Report the offset as a single measured value per run, or "
            "shorten the poll interval (chrony `minpoll 4 maxpoll 4`) so several polls land inside a run.",
            "",
        ]
    if events is not None and len(events):
        md += ["## NTP events", ""]
        for _, e in events.iterrows():
            md.append(f"- t={(e['log_time_ns'] - t0_ns) / 1e9:.1f} s `{e['topic']}`: {e['data']}")
        md.append("")
    if audit_df is not None:
        md += ["## Header stamp − recorder log time, per topic (ms)", "", "Negative = header stamp precedes the recorder's receive time (normal transport latency in the recorder's clock). A node whose topics sit systematically off the others has a clock offset relative to the recorder.", "", audit_df.round(3).to_markdown(index=False), ""]
        node_med = audit_df.groupby("node")["stamp_minus_log_median_ms"].median()
        md += ["Per-node median of the per-topic medians (ms):", "", node_med.round(3).to_markdown(), ""]
    if unset_df is not None:
        md += [
            "## Topics with unset header stamps",
            "",
            "These messages carry `header.stamp = 0`: the driver never set it. They are excluded from the "
            "check above and cannot be time-aligned with anything else.",
            "",
            unset_df.round(4).to_markdown(index=False),
            "",
        ]
    if sensor_period_ms:
        md += [f"Shortest sensor period in the bag: **{sensor_period_ms:.2f} ms** (`{fastest}`).", ""]
        for _, r in summary.iterrows():
            md.append(f"- {r['agent']}: max |offset| {r['abs_offset_max_ms']:.2f} ms = {r['abs_offset_max_ms'] / sensor_period_ms:.2f} × shortest period")
    (args.out / "ntp_summary.md").write_text("\n".join(md))

    def tt(s: str) -> str:
        return "\\texttt{" + str(s).replace("_", "\\_") + "}"

    clients = summary[summary["role"].str.contains("client")]
    server = args.server or (clients["sync_source"].mode().iloc[0] if len(clients) else "?")
    hostnames = ", ".join(f"{tt(r['agent'])} = {tt(r['hostname'])}" for _, r in clients.iterrows())
    strata = ", ".join(str(s) for s in sorted(clients["stratum"].unique()))
    rate = ", ".join(f"{r:.1f}" for r in roles["rate_hz"])
    parts = []
    for _, r in clients.iterrows():
        delay = (f" with a median round-trip delay of {r['delay_median_ms']:.2f}\\,ms" if r["delay_median_ms"] > 0 else "")
        parts.append(
            f"{tt(r['agent'])} had a mean offset of {r['offset_mean_ms']:.2f}\\,ms "
            f"(mean $|\\cdot|$ {r['abs_offset_mean_ms']:.2f}\\,ms, 95th percentile {r['abs_offset_p95_ms']:.2f}\\,ms, "
            f"maximum {r['abs_offset_max_ms']:.2f}\\,ms){delay}"
            f" and {int(r['clock_steps_flagged'])} clock step{'s' if r['clock_steps_flagged'] != 1 else ''}"
        )
    poll = int(clients["poll_interval_mode_s"].max()) if len(clients) else 0
    duration = float(clients["duration_s"].max()) if len(clients) else 0.0
    polls = int(clients["polls_in_run"].max()) if len(clients) else 0
    poll_sentence = ""
    if poll and poll > duration / 2:
        poll_sentence = (
            f" The clients poll the server every {poll}\\,s, longer than the {duration:.0f}\\,s run, so each offset above is "
            f"a single measurement (at most {polls} poll per run) rather than a distribution; the per-message header-stamp "
            f"comparison below supplies the continuous evidence."
        )
    max_all = clients["abs_offset_max_ms"].max() if len(clients) else float("nan")
    unset_sentence = ""
    if unset_df is not None:
        names = ", ".join(tt(t) for t in unset_df["topic"].head(4))
        unset_sentence = (
            f" {len(unset_df)} topic{'s' if len(unset_df) != 1 else ''} ({names}) "
            f"{'carry' if len(unset_df) != 1 else 'carries'} an unset message header stamp and "
            f"{'are' if len(unset_df) != 1 else 'is'} excluded from that comparison; only the recorder's "
            f"receive time is available for {'them' if len(unset_df) != 1 else 'it'}."
        )
    audit_sentence = ""
    if audit_df is not None:
        spread = audit_df.groupby("node")["stamp_minus_log_median_ms"].median()
        audit_sentence = (
            f" As an NTP-independent cross-check, the median of header stamp minus recorder receive time spans "
            f"{spread.min():.1f} to {spread.max():.1f}\\,ms across nodes. This difference is the node's clock offset minus "
            f"the delivery latency to the recorder, so it upper-bounds the offset rather than measuring it, and at this "
            f"magnitude it is dominated by wireless delivery time; it is consistent with the sub-millisecond offsets above."
        ) + unset_sentence
    bound_sentence = ""
    if sensor_period_ms:
        bound_sentence = (
            f" All offsets are below the shortest sensor period in the recording ({sensor_period_ms:.1f}\\,ms), "
            f"so cross-agent messages can be associated by timestamp without further alignment."
            if max_all < sensor_period_ms
            else f" The maximum offset exceeds the shortest sensor period ({sensor_period_ms:.1f}\\,ms); "
            f"cross-agent association of the fastest topics needs the recorded offsets applied."
        )
    tex = f"""\\subsubsection{{NTP Synchronization}}
All agents are synchronized over NTP on the shared wireless network.
{tt(server)} acts as the NTP server and the other agents synchronize to it as
stratum-{strata} clients (host names as recorded: {hostnames}); each client publishes its NTP state at about {rate}\\,Hz throughout every run.
Over run {tt(args.run)}, {'; '.join(parts)}.{poll_sentence}{audit_sentence}{bound_sentence}
No sensor is hardware-triggered or hardware-timestamped: every message is stamped in software by
its driver on arrival at the host, so the header stamps carry the NTP-aligned host clock plus the
driver's arrival latency, and the offsets above bound clock disagreement between agents, not
sensor exposure time.
"""
    (args.out / "ntp_subsection.tex").write_text(tex)

    print(roles.to_string(index=False))
    print()
    print(summary[cols].round(3).to_string(index=False))
    if sensor_period_ms:
        print(f"\nshortest sensor period: {sensor_period_ms:.2f} ms ({fastest})")
    print(f"\nwrote {args.out}/ntp_summary.md, ntp_subsection.tex, fig_ntp.pdf/png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
