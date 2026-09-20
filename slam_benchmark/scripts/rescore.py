#!/usr/bin/env python3
"""Re-score every run under a runs root, from each run's own manifest.

    python3 scripts/rescore.py --runs runs/coop2_20260828 --config configs/coop2.yaml \
        --reference-dir runs/reference

Scoring reads files that are already on disk, so it is seconds per run and
should be re-done whenever the evaluator changes. Doing it with a hand-written
list of (method, stream) pairs went wrong three times — a missing cell, a
wrong reference, errors swallowed by a redirect — so the list comes from
`run.json`, which records the method, agent and stream that produced each
directory, and every failure is printed rather than hidden.

Also scores `odometry.tum` beside `trajectory.tum` wherever a run wrote one:
that is the front-end alone, it needs no interpolation at the anchors, and it
is what separated a real board residual from an interpolation artefact.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plan(runs_root: Path, reference_dir: Path, methods_dir: Path) -> list[dict]:
    """One entry per (run directory, trajectory file), from the manifests."""
    jobs = []
    for manifest in sorted(runs_root.rglob("run.json")):
        d = manifest.parent
        # An interrupted or killed run can leave a truncated manifest. That is
        # one directory's problem; crashing the planner on it hides every
        # other run, which is the whole failure this script exists to avoid.
        try:
            m = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError) as e:
            jobs.append({"run": d, "skip": f"unreadable run.json ({e})"})
            continue
        if not isinstance(m, dict):
            jobs.append({"run": d, "skip": f"run.json is {type(m).__name__}, not an object"})
            continue
        method, agent, stream = m.get("method"), m.get("agent"), m.get("stream")
        if not (method and agent and stream):
            jobs.append({"run": d, "skip": f"run.json lacks method/agent/stream"})
            continue
        cfg = methods_dir / f"{method}.yaml"
        if not cfg.exists():
            jobs.append({"run": d, "skip": f"no method config {cfg}"})
            continue
        ref = reference_dir / f"{agent}.tum"
        for traj in ("trajectory.tum", "odometry.tum"):
            f = d / traj
            if not f.exists():
                continue
            if f.stat().st_size == 0:
                jobs.append({"run": d, "traj": traj, "skip": "empty file"})
                continue
            jobs.append({"run": d, "traj": traj, "method": method, "stream": stream,
                         "config": cfg, "reference": ref})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference-dir", default="runs/reference")
    ap.add_argument("--methods-dir", default=str(ROOT / "configs" / "methods"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = plan(Path(args.runs), Path(args.reference_dir), Path(args.methods_dir))
    ok, failed, skipped = 0, [], []
    for j in jobs:
        if j.get("skip"):
            skipped.append(f"{j['run']}/{j.get('traj', '')}: {j['skip']}")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "eval_run.py"),
               "--config", args.config, "--method", str(j["config"]),
               "--stream", j["stream"], "--run", str(j["run"]),
               "--reference", str(j["reference"]), "--skip-map",
               "--trajectory-file", j["traj"]]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            failed.append(f"{j['run']}/{j['traj']}: "
                          f"{(r.stderr.strip().splitlines() or ['?'])[-1]}")
        else:
            ok += 1
            print(f"ok   {j['method']:<18} {j['stream']:<24} {j['run'].name}/{j['traj']}")
    if args.dry_run:
        return 0
    for s in skipped:
        print(f"skip {s}")
    for f in failed:
        print(f"FAIL {f}", file=sys.stderr)
    print(f"\n{ok} scored, {len(failed)} failed, {len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
