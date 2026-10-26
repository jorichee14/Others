#!/usr/bin/env python3
"""What a method costs while it runs: its processes' CPU and memory, the
machine's, and the GPU's, sampled from /proc (no psutil) every --period s.

    python3 resources.py --dir OUT --match 'fastlio_mapping' [--period 0.5]

Writes into OUT:
  resources.csv  wall_ns, n_procs, threads, method_cpu_pct (100 = one core),
                 method_rss_mb, system_cpu_pct (100 = every core), mem_used_mb,
                 gpu_util_pct, gpu_mem_mb, power_mw (empty where not measured)
  host.json      CPU model and cores, memory, GPU name, Jetson model

The method's processes are those whose command line matches --match and not
--exclude (the probe, this sampler and the bag player by default), found
again at every sample so restarts and late starts are followed. A process's
CPU counts from the first sample that sees it. GPU: nvidia-smi on a desktop
GPU, tegrastats on a Jetson (GR3D_FREQ as utilisation, VDD_IN as power),
each kept running as one process rather than started per sample. Stops on
SIGINT / SIGTERM.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time

EXCLUDE = r"probe\.py|resources\.py|summarize\.py|ros2 bag|ros2-bag|rosbag2|bag play"
TICK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def processes(match, exclude):
    """{pid: (cpu ticks, rss bytes, threads)} of the matching processes."""
    out = {}
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        cmd = read("/proc/%s/cmdline" % d)
        if not cmd:
            continue
        cmd = cmd.replace(b"\0", b" ").decode(errors="replace")
        if not match.search(cmd) or exclude.search(cmd):
            continue
        stat = read("/proc/%s/stat" % d)
        statm = read("/proc/%s/statm" % d)
        if stat is None or statm is None:
            continue
        f = stat.decode(errors="replace").rsplit(")", 1)[1].split()
        # after "pid (comm)": state is f[0]; utime, stime are fields 14, 15 -> f[11], f[12]
        out[int(d)] = (int(f[11]) + int(f[12]), int(statm.split()[1]) * PAGE, int(f[17]))
    return out


def system_cpu():
    """(busy, total) jiffies over every core."""
    line = read("/proc/stat").split(b"\n", 1)[0].split()[1:]
    v = [int(x) for x in line]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v) - idle, sum(v)


def meminfo():
    m = {}
    for line in read("/proc/meminfo").decode().splitlines():
        k, _, rest = line.partition(":")
        m[k] = int(rest.split()[0]) * 1024
    return m


class Gpu:
    """The latest GPU reading, from nvidia-smi or tegrastats running alongside."""

    def __init__(self, period_ms):
        self.util = self.mem = self.power = None
        self.kind, self.proc = None, None
        if shutil.which("nvidia-smi"):
            cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                   "--format=csv,noheader,nounits", "-lms", str(period_ms)]
            self.kind, parse = "nvidia-smi", self.parse_smi
        elif shutil.which("tegrastats"):
            cmd = ["tegrastats", "--interval", str(period_ms)]
            self.kind, parse = "tegrastats", self.parse_tegra
        else:
            return
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.kind = None
            return
        threading.Thread(target=self.reader, args=(parse,), daemon=True).start()

    def reader(self, parse):
        for line in self.proc.stdout:
            try:
                parse(line)
            except (ValueError, IndexError):
                pass

    def parse_smi(self, line):
        v = [x.strip() for x in line.split(",")]
        self.util = float(v[0])
        self.mem = float(v[1])
        self.power = float(v[2]) * 1000.0 if len(v) > 2 and v[2] not in ("", "[N/A]") else None

    def parse_tegra(self, line):
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        self.util = float(g.group(1)) if g else None
        p = re.search(r"(?:VDD_IN|POM_5V_IN) (\d+)mW", line)
        self.power = float(p.group(1)) if p else None
        self.mem = None              # unified memory: the RAM figure is the machine's

    def stop(self):
        if self.proc:
            self.proc.terminate()


def host_info(gpu):
    info = {"cores": os.cpu_count(), "mem_total_mb": round(meminfo()["MemTotal"] / 2**20)}
    for line in (read("/proc/cpuinfo") or b"").decode(errors="replace").splitlines():
        if line.startswith(("model name", "Model")):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    model = read("/proc/device-tree/model")
    if model:
        info["jetson_or_board"] = model.decode(errors="replace").strip("\0 \n")
    if gpu.kind == "nvidia-smi":
        try:
            info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                         capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    info["gpu_source"] = gpu.kind
    return info


def fmt(v, f="%.1f"):
    return "" if v is None else f % v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--match", required=True, help="regex on the method's process command lines")
    ap.add_argument("--exclude", default=EXCLUDE)
    ap.add_argument("--period", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    match, exclude = re.compile(a.match), re.compile(a.exclude)
    gpu = Gpu(int(a.period * 1000))
    json.dump(host_info(gpu), open(os.path.join(a.dir, "host.json"), "w"), indent=1)
    fh = open(os.path.join(a.dir, "resources.csv"), "w")
    fh.write("wall_ns,n_procs,threads,method_cpu_pct,method_rss_mb,system_cpu_pct,mem_used_mb,"
             "gpu_util_pct,gpu_mem_mb,power_mw\n")
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    seen, (busy0, tot0), t0 = {}, system_cpu(), time.monotonic()
    while not stop:
        time.sleep(a.period)
        now, t1 = time.time_ns(), time.monotonic()
        procs = processes(match, exclude)
        dticks = sum(c - seen[p] for p, (c, _, _) in procs.items() if p in seen)
        seen = {p: c for p, (c, _, _) in procs.items()}
        busy1, tot1 = system_cpu()
        m = meminfo()
        dt = t1 - t0
        fh.write("%d,%d,%d,%s,%s,%s,%s,%s,%s,%s\n" % (
            now, len(procs), sum(t for _, _, t in procs.values()),
            fmt(100.0 * dticks / TICK / dt if dt > 0 else None),
            fmt(sum(r for _, r, _ in procs.values()) / 2**20),
            fmt(100.0 * (busy1 - busy0) / (tot1 - tot0) if tot1 > tot0 else None),
            fmt((m["MemTotal"] - m.get("MemAvailable", m["MemFree"])) / 2**20, "%.0f"),
            fmt(gpu.util), fmt(gpu.mem, "%.0f"), fmt(gpu.power, "%.0f")))
        fh.flush()
        busy0, tot0, t0 = busy1, tot1, t1
    gpu.stop()
    fh.close()


if __name__ == "__main__":
    main()
