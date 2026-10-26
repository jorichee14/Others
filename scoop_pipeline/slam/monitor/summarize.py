#!/usr/bin/env python3
"""One run's cost and timing, from what probe.py and resources.py logged.

    python3 summarize.py RUN_DIR [--key TOPIC] [--out TOPIC] [--tol-s 0]

RUN_DIR holds messages.csv (probe.py), resources.csv and host.json
(resources.py), and run.json when the run was started by
slam/odometry/run_odometry.sh, which names the key input (the scan or left
image each output answers), the output and the stamp tolerance; the options
override it. Writes RUN_DIR/summary.json and prints it as a few lines.

  topics      per topic: messages, rate (Hz), mean size (KB), MB/s, over the
              span it was seen in
  latency     per output, from the wall time its input arrived to the wall time
              it did. Its input is the newest key input that had arrived by
              then with a stamp at or before the output's stamp + tol (a
              method may stamp its output later than the scan: FAST-LIO2 at the
              scan's last point, within the scan period, so tol stays 0; one
              that calibrates a camera-IMU time offset online, OpenVINS, needs
              a few ms). Outputs after the first for the same input count as
              repeats, not latencies
  drops       key inputs no output answered
  real time   kept up: drops <= 1 % and p95 latency <= the key input's period
  resources   over the play (first key input to last output): the method's CPU
              (100 = one core) mean / p95 / max, its peak RSS, the machine's
              CPU and memory, GPU utilisation / memory / power where measured
"""
import argparse
import bisect
import csv
import json
import os
import sys


def pct(v, q):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * q / 100.0
    i = int(k)
    return s[i] if i + 1 >= len(s) else s[i] + (s[i + 1] - s[i]) * (k - i)


def stats(v, nd=1):
    if not v:
        return None
    return {"mean": round(sum(v) / len(v), nd), "median": round(pct(v, 50), nd),
            "p95": round(pct(v, 95), nd), "max": round(max(v), nd)}


def load_messages(path):
    by = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by.setdefault(r["topic"], {"role": r["role"], "m": []})["m"].append(
                (int(r["wall_ns"]), int(r["stamp_ns"]), int(r["bytes"])))
    for t in by.values():
        t["m"].sort()
    return by


def topic_stats(by):
    out = {}
    for t, d in sorted(by.items()):
        m = d["m"]
        span = (m[-1][0] - m[0][0]) * 1e-9
        total = sum(b for _, _, b in m)
        out[t] = {"role": d["role"], "messages": len(m),
                  "hz": round((len(m) - 1) / span, 2) if span > 0 else None,
                  "mean_kb": round(total / len(m) / 1024.0, 2),
                  "mb_per_s": round(total / 2**20 / span, 3) if span > 0 else None}
    return out


def match(inputs, outputs, tol_ns):
    """[(input index or None, latency ns)] per output (wall-sorted), and the
    set of input indices answered. inputs / outputs: (wall, stamp, bytes),
    wall-sorted; a bag replays the key topic in stamp order."""
    walls = [w for w, _, _ in inputs]
    res, answered = [], set()
    stamps = []                        # stamps of the inputs arrived so far, in arrival order
    j = 0
    for w, s, _ in outputs:
        while j < len(inputs) and walls[j] <= w:
            stamps.append(inputs[j][1])
            j += 1
        k = bisect.bisect_right(stamps, s + tol_ns) - 1
        if k < 0:
            res.append((None, None))
            continue
        res.append((k, w - inputs[k][0]))
        answered.add(k)
    return res, answered


def latency(by, key, out, tol_s):
    if key not in by or out not in by:
        return {"error": "no messages on %s" % (key if key not in by else out)}
    ins, outs = by[key]["m"], by[out]["m"]
    res, answered = match(ins, outs, int(round(tol_s * 1e9)))
    first, lat = set(), []
    repeats = unmatched = 0
    for k, dt in res:
        if k is None:
            unmatched += 1
        elif k in first:
            repeats += 1
        else:
            first.add(k)
            lat.append(dt * 1e-6)
    st = sorted(s for _, s, _ in ins)
    gaps = [(b - a) * 1e-6 for a, b in zip(st, st[1:]) if b > a]
    period = pct(gaps, 50)
    # inputs after the last output may still have been in flight when the run stopped
    last_out = outs[-1][0] if outs else 0
    tail = sum(1 for w, _, _ in ins if w > last_out)
    drops = len(ins) - len(answered)
    drop_pct = 100.0 * drops / len(ins) if ins else None
    p95 = pct(lat, 95)
    return {"key": key, "out": out, "stamp_tol_s": tol_s,
            "inputs": len(ins), "outputs": len(outs), "answered": len(answered),
            "dropped": drops, "dropped_pct": round(drop_pct, 2) if drop_pct is not None else None,
            "dropped_after_last_output": tail, "repeats": repeats, "unmatched_outputs": unmatched,
            "input_period_ms": round(period, 2) if period else None,
            "latency_ms": stats(lat),
            "kept_up": (drop_pct is not None and p95 is not None and period is not None
                        and drop_pct <= 1.0 and p95 <= period)}


def resources(path, t0, t1):
    if not os.path.exists(path):
        return None
    cols = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            w = int(r["wall_ns"])
            if t0 is not None and (w < t0 or w > t1):
                continue
            for k, v in r.items():
                if k != "wall_ns" and v != "":
                    cols.setdefault(k, []).append(float(v))
    if not cols:
        return {"samples": 0}
    g = lambda k: cols.get(k, [])                                          # noqa: E731
    return {"samples": len(g("n_procs")),
            "method_processes_max": int(max(g("n_procs"))) if g("n_procs") else 0,
            "method_cpu_pct": stats(g("method_cpu_pct")),
            "method_rss_mb_peak": round(max(g("method_rss_mb")), 1) if g("method_rss_mb") else None,
            "system_cpu_pct": stats(g("system_cpu_pct")),
            "mem_used_mb_peak": round(max(g("mem_used_mb"))) if g("mem_used_mb") else None,
            "gpu_util_pct": stats(g("gpu_util_pct")),
            "gpu_mem_mb_peak": round(max(g("gpu_mem_mb"))) if g("gpu_mem_mb") else None,
            "power_mw": stats(g("power_mw"), 0)}


def summarize(run, key=None, out=None, tol_s=None):
    rj = os.path.join(run, "run.json")
    cfg = json.load(open(rj)) if os.path.exists(rj) else {}
    key = key or cfg.get("key_topic")
    out = out or cfg.get("out_topic")
    tol_s = cfg.get("stamp_tol_s", 0.0) if tol_s is None else tol_s
    by = load_messages(os.path.join(run, "messages.csv"))
    s = {"method": cfg.get("method"), "machine": cfg.get("machine"),
         "topics": topic_stats(by)}
    if key and out:
        s["timing"] = latency(by, key, out, tol_s)
    t0 = by[key]["m"][0][0] if key in by else None
    t1 = by[out]["m"][-1][0] if out in by else None
    s["resources"] = resources(os.path.join(run, "resources.csv"), t0, t1)
    hj = os.path.join(run, "host.json")
    if os.path.exists(hj):
        s["host"] = json.load(open(hj))
    tum = os.path.join(run, "trajectory.tum")
    s["trajectory"] = {"file": tum if os.path.exists(tum) else None,
                       "poses": sum(1 for _ in open(tum)) if os.path.exists(tum) else 0,
                       "frame": cfg.get("tum_frame")}
    json.dump(s, open(os.path.join(run, "summary.json"), "w"), indent=1)
    return s


def report(s):
    lines = ["%s on %s" % (s.get("method") or "?", s.get("machine") or "?")]
    t = s.get("timing") or {}
    if "error" in t:
        lines.append("  timing: %s" % t["error"])
    elif t:
        lat = t["latency_ms"] or {}
        lines.append("  %d inputs, %d answered, %d dropped (%.1f %%); latency median %s / p95 %s / "
                     "max %s ms against a %s ms period: %s"
                     % (t["inputs"], t["answered"], t["dropped"], t["dropped_pct"] or 0.0,
                        lat.get("median"), lat.get("p95"), lat.get("max"), t["input_period_ms"],
                        "kept up" if t["kept_up"] else "did NOT keep up"))
    r = s.get("resources") or {}
    if r.get("samples"):
        c = r["method_cpu_pct"] or {}
        lines.append("  CPU %s %% mean / %s %% p95 (100 = one core), RSS %s MB peak; machine CPU %s %% mean"
                     % (c.get("mean"), c.get("p95"), r["method_rss_mb_peak"],
                        (r["system_cpu_pct"] or {}).get("mean")))
        if r.get("gpu_util_pct"):
            lines.append("  GPU %s %% mean, %s MB peak; power %s mW mean"
                         % (r["gpu_util_pct"]["mean"], r["gpu_mem_mb_peak"],
                            (r.get("power_mw") or {}).get("mean")))
    for name, v in s["topics"].items():
        lines.append("  %-3s %-42s %6d msgs %8s Hz %9s KB %8s MB/s"
                     % (v["role"], name, v["messages"], v["hz"], v["mean_kb"], v["mb_per_s"]))
    tr = s["trajectory"]
    lines.append("  trajectory: %d poses (%s)" % (tr["poses"], tr["frame"] or "frame not recorded"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--key", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol-s", type=float, default=None)
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.run, "messages.csv")):
        sys.exit("no %s" % os.path.join(a.run, "messages.csv"))
    print(report(summarize(a.run, a.key, a.out, a.tol_s)))


if __name__ == "__main__":
    main()
