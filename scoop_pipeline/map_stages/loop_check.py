#!/usr/bin/env python3
"""A trajectory's loop closure on the boards: stage 05 in its read-only check
mode (05_anchor.py --traj --check), summed up and set against GLIM's.

    python3 map_stages/loop_check.py <pipeline config> [--traj FILE] [--out REPORT]

--traj  the trajectory to check (default: dataset.traj, e.g. 02's
        traj_lidar_ba.txt in odometry/<machine>/)
--out   the report (default: <trajectory>_loop_closure.json next to it)

For every board seen at the start and again later -- start and end on the
same board -- where the trajectory puts it the first and the last time, in
mm / deg against the detections' own noise (DRIFT or not significant), and
the co-visible cross-checks. When GLIM's own check is there
(odometry/<machine>/glim_loop_closure.json, written after GLIM) the two are
printed side by side. Stages 02 and 02b run this on their results by themselves; the GLIM
step of process_recording.py runs it on GLIM's.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(config, traj, report, log=print):
    """05 --check on ``traj`` -> ``report``; the summary (:func:`summarize`),
    or None when the check failed."""
    code = subprocess.call([sys.executable, os.path.join(HERE, "05_anchor.py"), str(config),
                            "--traj", str(traj), "--check", str(report)])
    if code != 0 or not os.path.isfile(report):
        log("    loop closure: the check failed (exit %s)" % code)
        return None
    return summarize(report)


def summarize(report):
    """The report's per-board loop closures and cross-checks, compact."""
    with open(report) as fh:
        r = json.load(fh)
    boards = {}
    for name, b in (r.get("boards") or {}).items():
        lc = b.get("loop_closure")
        boards[name] = ({"sections": b["sections"], "from_section": lc["from_section"],
                         "to_section": lc["to_section"], "over_s": lc["dt_s"],
                         "mm": lc["mm"], "deg": lc["deg"], "noise_mm": lc["noise_mm"],
                         "significant": lc["significant"]} if lc else
                        {"sections": b["sections"], "mm": None})
    return {"report": str(report), "trajectory": r.get("trajectory"),
            "anchor_board": r.get("anchor_board"), "boards": boards,
            "cross_checks": [{k: c[k] for k in ("a", "b", "disagreement_mm",
                                                "disagreement_deg", "significant")}
                             for c in r.get("cross_checks") or []]}


def lines(summary, label="loop closure"):
    out = []
    for name, b in summary["boards"].items():
        if b.get("mm") is None:
            out.append("%s %s: seen in %d section(s), not revisited" % (label, name, b["sections"]))
        else:
            out.append("%s %s: s%d->s%d over %.0f s: %.1f mm / %.3f deg (noise %.1f mm) -> %s"
                       % (label, name, b["from_section"], b["to_section"], b["over_s"],
                          b["mm"], b["deg"], b["noise_mm"],
                          "DRIFT" if b["significant"] else "not significant"))
    return out


def compare(this, glim, label="refined"):
    """One line per board seen by both: GLIM's closure -> this one's."""
    out = []
    for name, b in this["boards"].items():
        g = glim["boards"].get(name) or {}
        if b.get("mm") is None or g.get("mm") is None:
            continue
        out.append("%-12s GLIM %6.1f mm / %.3f deg  ->  %s %6.1f mm / %.3f deg"
                   % (name, g["mm"], g["deg"], label, b["mm"], b["deg"]))
    return out


def check_trajectory(config, traj, report=None, log=print, label="refined"):
    """Run the check on ``traj``, print it, and set it against GLIM's when
    that is there (glim_loop_closure.json in the same folder)."""
    report = report or os.path.splitext(traj)[0] + "_loop_closure.json"
    log("\n=== loop closure of %s on the boards ===" % os.path.basename(traj))
    summ = run(config, traj, report, log)
    if summ is None:
        return None
    for ln in lines(summ, "loop closure"):
        log("  " + ln)
    glim = os.path.join(os.path.dirname(traj), "glim_loop_closure.json")
    if os.path.isfile(glim) and os.path.abspath(glim) != os.path.abspath(report):
        cmp_ = compare(summ, summarize(glim), label)
        if cmp_:
            log("  against GLIM:")
            for ln in cmp_:
                log("    " + ln)
    log("  report: %s" % report)
    return summ


def main():
    args = sys.argv[1:]
    opt = {}
    for flag in ("--traj", "--out"):
        if flag in args:
            i = args.index(flag)
            opt[flag] = args[i + 1]
            del args[i:i + 2]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    sys.path.insert(0, HERE)
    from pipeline_common import load_pipeline
    P = load_pipeline(args[0])
    traj = os.path.expanduser(opt.get("--traj") or P.outp(P.dataset["traj"]))
    if not os.path.isfile(traj):
        sys.exit("no trajectory at %s" % traj)
    if check_trajectory(args[0], traj, opt.get("--out")) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
