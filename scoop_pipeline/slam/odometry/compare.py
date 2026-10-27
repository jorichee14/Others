#!/usr/bin/env python3
"""One table of every run given, from the files the runs already wrote:
run_odometry.sh runs (summary.json with its accuracy), and, when a folder
holds one, a robot-to-robot table from relative_pose.json (Swarm-SLAM,
analysis/slam/relative_pose.py) or relative_pose_loc.json (two localizer runs,
analysis/slam/relative_pose_loc.py).

    python scoop_pipeline/slam/odometry/compare.py <run or folder of runs> [...] [--md OUT.md]

A folder is searched two levels down. Per run: method, robot, span, ATE
(after a rigid fit; rmse cm / deg), in map (no fit: localizer runs only),
RPE over 1 s, drift per 10 m (median %), coverage, CPU (mean / p95, 100 =
one core), peak memory, latency p95 and whether it kept up.
"""
import argparse
import glob
import json
import os


def runs(paths):
    """Run folders (with run.json) and relative-pose files, in the order given."""
    odo, rel = [], []
    for p in paths:
        p = os.path.abspath(p)
        cands = [p] + sorted(glob.glob(os.path.join(p, "*"))) + sorted(glob.glob(os.path.join(p, "*", "*")))
        for c in cands:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "run.json")) and c not in odo:
                odo.append(c)
            for f in ("relative_pose.json", "relative_pose_loc.json"):
                j = os.path.join(c, f)
                if os.path.isdir(c) and os.path.exists(j) and j not in rel:
                    rel.append(j)
    return odo, rel


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def num(v, fmt="%.2f"):
    return "-" if v is None else fmt % v


def odo_row(run):
    rj = json.load(open(os.path.join(run, "run.json")))
    s = os.path.join(run, "summary.json")
    s = json.load(open(s)) if os.path.exists(s) else {}
    a = s.get("accuracy")
    if a is None and os.path.exists(os.path.join(run, "accuracy.json")):
        a = json.load(open(os.path.join(run, "accuracy.json")))
    a = a or {}
    st = json.load(open(os.path.join(run, "status.json"))) if os.path.exists(os.path.join(run, "status.json")) else {}
    t, r = s.get("timing") or {}, s.get("resources") or {}
    ok = st.get("method") == "ran" and st.get("play") == "done"
    return [rj.get("method"), rj.get("machine"), num(a.get("span_s"), "%.0f"),
            num(g(a, "ate", "position_cm", "rmse")), num(g(a, "ate", "rotation_deg", "rmse")),
            num(g(a, "in_map", "position_cm", "rmse")) if rj.get("world") == "map" else "",
            num(g(a, "rpe_1s", "translation_cm", "rmse")),
            num(g(a, "drift_10m_pct", "median")),
            num(g(a, "coverage", "pct"), "%.0f"),
            "%s / %s" % (num(g(r, "method_cpu_pct", "median"), "%.0f"), num(g(r, "method_cpu_pct", "p95"), "%.0f")),
            num(r.get("method_rss_mb_peak"), "%.0f"),
            num(g(t, "latency_ms", "p95"), "%.0f"),
            "yes" if t.get("kept_up") else ("no" if t else "-"),
            "" if ok else "%s / %s" % (st.get("method", "?"), st.get("play", "?")),
            os.path.basename(run)]


ODO_HEAD = ["method", "robot", "s", "ATE cm", "ATE deg", "in map cm", "RPE 1 s cm", "drift / 10 m %",
            "coverage %", "CPU % mean / p95", "RSS MB", "latency p95 ms", "kept up", "status", "run"]


def rel_row(path):
    r = json.load(open(path))
    kind = "localized in map" if path.endswith("_loc.json") else "Swarm-SLAM"
    return [os.path.basename(os.path.dirname(path)), kind, num(r.get("available_pct"), "%.0f"),
            num(r.get("available_from_s"), "%.1f"),
            num(g(r, "translation_m", "median"), "%.3f"), num(g(r, "translation_m", "p95"), "%.3f"),
            num(g(r, "rotation_deg", "median")), num(g(r, "rotation_deg", "p95")),
            num(g(r, "usable", "pct_of_run"), "%.0f")]


REL_HEAD = ["run", "kind", "available %", "from s", "position median m", "p95 m", "rotation median deg",
            "p95 deg", "usable %"]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--md", default=None, help="also write the tables to this markdown file")
    a = ap.parse_args()
    odo, rel = runs(a.paths)
    parts = []
    if odo:
        parts.append("Odometry and localization runs\n\n" + table(ODO_HEAD, [odo_row(r) for r in odo]))
    if rel:
        parts.append("Robot to robot\n\n" + table(REL_HEAD, [rel_row(p) for p in rel]))
    if not parts:
        raise SystemExit("no runs (run.json) or relative_pose*.json under %s" % ", ".join(a.paths))
    text = "\n\n".join(parts)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
