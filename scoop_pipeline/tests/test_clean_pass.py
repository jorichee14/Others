#!/usr/bin/env python
"""Self-test for clean_pass.py on a made-up mapping pass: GLIM's export,
02's reference with its reports, an up-to-date map, a seed map older than
GLIM's export (stale), a v1 trajectory listed under clean.keep, and side
results (an old map folder, a cache, a patches file) that must go to _old/.
A dry run moves nothing; --apply moves exactly the stale and old ones.

    python tests/test_clean_pass.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def touch(path, t):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("x\n")
    os.utime(path, (t, t))


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="clean_")
    try:
        proc = os.path.join(d, "data", "processed", "20260101", "mapping_X")
        odo, mp = os.path.join(proc, "odometry", "mobile_2"), os.path.join(proc, "mapping")
        tag = "mapping_X_20260101"
        t0 = 1.8e9
        keep = {odo: ["traj_lidar.txt", "glim.yaml", "glim_loop_closure.json",
                      "traj_lidar_ba_glim.txt", "traj_lidar_ba_glim_report.json",
                      "traj_lidar_ba_glim_loop_closure.json", "pose_time_check.json",
                      "traj_lidar_ba_glim_v1.txt"],
                mp: [f"denoised_{tag}.pcd", f"map_final_{tag}.pcd", f"denoised_union_{tag}.pcd"]}
        go = {odo: ["traj_lidar_ba_glim_keyframes.npz", "traj_compare.json"],
              mp: [f"denoised_seed_{tag}.pcd", "patches_ref.ply", "union_ba/denoised.pcd"]}
        touch(os.path.join(odo, "traj_lidar.txt"), t0 + 10)        # GLIM's export, re-done
        for f, rel in [(f, odo) for f in keep[odo][1:]] + [(f, mp) for f in keep[mp]]:
            touch(os.path.join(rel, f), t0 + 20)
        touch(os.path.join(mp, f"denoised_seed_{tag}.pcd"), t0)     # older than the export
        for f in go[odo][:2] + go[mp][1:]:
            touch(os.path.join(odo if f in go[odo] else mp, f), t0 + 20)
        os.makedirs(os.path.join(d, "data", "work", "20260101", "mapping_X", "mobile_2", "glim_imu11"))
        cfg = os.path.join(d, "cfg.json")
        json.dump({"dataset": {"bag": os.path.join(d, "data", "work", "20260101", "mapping_X",
                                                    "mobile_2", "bag"),
                               "traj": "traj_lidar_ba_glim.txt", "machine": "mobile_2"},
                   "02_bundle_adjust": {"output": "traj_lidar_ba_glim.txt"},
                   "clean": {"keep": ["traj_lidar_ba_glim_v1.txt"]}}, open(cfg, "w"))
        run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "clean_pass.py"),  # noqa: E731
                                         cfg, *a], capture_output=True, text=True)
        r = run()
        print(r.stdout[-2500:])
        if r.returncode or "dry run" not in r.stdout or os.path.isdir(os.path.join(proc, "_old")):
            fails.append("dry run: " + r.stdout[-800:] + r.stderr[-800:])
        if f"STALE" not in r.stdout or "denoised_seed_" not in r.stdout:
            fails.append("the seed map older than GLIM's export is not STALE")
        r = run("--apply", "--work")
        old = os.path.join(proc, "_old")
        for folder, names in keep.items():
            for f in names:
                if not os.path.exists(os.path.join(folder, f)):
                    fails.append(f"{f} moved, should stay")
        for folder, names in go.items():
            rel = os.path.relpath(folder, proc)
            for f in names:
                top = f.split("/")[0]
                if os.path.exists(os.path.join(folder, top)) or \
                        not os.path.exists(os.path.join(old, rel, top)):
                    fails.append(f"{f} not moved to _old/{rel}")
        wold = os.path.join(d, "data", "work", "20260101", "mapping_X", "_old", "mobile_2", "glim_imu11")
        if not os.path.isdir(wold):
            fails.append("--work: glim_imu11 not moved")
        if r.returncode:
            fails.append("--apply: " + r.stderr[-800:])
        if not fails:
            print("  ok    dry run moved nothing; --apply kept 11, moved the stale seed map, "
                  "the cache, the old map folder and GLIM's side run")
        # --redo: all but GLIM's export (and clean.keep) aside, the rest left in place
        for f in ("frames/publish_tfs_x.sh", "bags/x_best_poses/metadata.yaml", "comms/ntp.csv",
                  f"mapping_report_{tag}.md", "notes.txt"):
            touch(os.path.join(proc, f), t0 + 30)
        r = run("--redo")
        if r.returncode or "dry run" not in r.stdout or any(
                n.startswith("redo_") for n in os.listdir(old)):
            fails.append("--redo dry run: " + r.stdout[-800:] + r.stderr[-800:])
        r = run("--redo", "--apply")
        redo = [n for n in os.listdir(old) if n.startswith("redo_")]
        if r.returncode or len(redo) != 1:
            fails.append("--redo --apply: " + r.stdout[-800:] + r.stderr[-800:])
        else:
            rd = os.path.join(old, redo[0])
            stay = ["odometry/mobile_2/" + f for f in ("traj_lidar.txt", "glim.yaml",
                    "glim_loop_closure.json", "traj_lidar_ba_glim_v1.txt")] + ["notes.txt"]
            gone = ["mapping", "frames", "bags", "comms", f"mapping_report_{tag}.md",
                    "odometry/mobile_2/traj_lidar_ba_glim.txt",
                    "odometry/mobile_2/pose_time_check.json"]
            for f in stay:
                if not os.path.exists(os.path.join(proc, f)):
                    fails.append(f"--redo moved {f}")
            for f in gone:
                if os.path.exists(os.path.join(proc, f)) or not os.path.exists(os.path.join(rd, f)):
                    fails.append(f"--redo did not move {f} to _old/{redo[0]}/")
            if "notes.txt" not in r.stdout:
                fails.append("--redo: notes.txt not listed as left")
        if not fails:
            print("  ok    --redo moved the maps, frames, bags, comms, report and 02's outputs; "
                  "kept GLIM's export and clean.keep; left notes.txt")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("clean pass ok" if not fails else "clean pass FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
