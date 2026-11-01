#!/usr/bin/env python3
"""Put RTSP recordings of several cameras on one clock with a light flash seen by all of them.

Record every camera with ffmpeg (`-c copy` into .mkv keeps each frame's stream timestamp), switch
the room lights off and on while recording, then run this in the folder with the cam_N.mkv files:

  python flash_sync.py              # every cam_*.mkv here
  python flash_sync.py rec/         # another folder

For every camera: the frame where the lights go off (largest brightness drop) is time 0. Writes
cam_N_times.csv (frame, pts_s = the file's own timestamp, t_s = seconds from the lights going off,
the same clock for every camera) and prints per camera: frames, dropped frames, how long it recorded
before the flash, and the off -> on interval (should agree across cameras; a camera that differs by
more than a frame is suspect). Accuracy is about one frame (33 ms at 30 fps): the flash shows up on
whichever frame each free-running camera happens to take next.

Needs only ffprobe (comes with ffmpeg) on the PATH; no OpenCV.
"""
import csv
import glob
import os
import re
import subprocess
import sys


def frames(path):
    """[(pts seconds, mean brightness 0-255)] of every decoded frame, in order."""
    folder, name = os.path.split(os.path.abspath(path))
    out = subprocess.run(["ffprobe", "-v", "error", "-f", "lavfi", "-i", "movie=%s,signalstats" % name,
                          "-show_entries", "frame=best_effort_timestamp_time:frame_tags=lavfi.signalstats.YAVG",
                          "-of", "csv=p=0"], cwd=folder, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.split():
        t, y = line.split(",")[:2]
        if t not in ("", "N/A"):
            rows.append((float(t), float(y)))
    return rows


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(folder, "cam_*.mkv")), key=lambda s: int(re.findall(r"\d+", os.path.basename(s))[0]))
    if not files:
        sys.exit("no cam_*.mkv in %s" % os.path.abspath(folder))
    print("%-8s %7s %7s %10s %10s" % ("camera", "frames", "dropped", "before_s", "off->on_s"))
    for f in files:
        fr = frames(f)
        if len(fr) < 3:
            print("%-8s too few frames" % os.path.basename(f)[:-4])
            continue
        d = [fr[i + 1][1] - fr[i][1] for i in range(len(fr) - 1)]
        off = min(range(len(d)), key=d.__getitem__) + 1           # first dark frame
        on = max(range(off - 1, len(d)), key=d.__getitem__) + 1   # first bright frame after it
        t0 = fr[off][0]
        dt = sorted(fr[i + 1][0] - fr[i][0] for i in range(len(fr) - 1))[len(fr) // 2]   # frame period
        dropped = sum(round((fr[i + 1][0] - fr[i][0]) / dt) - 1 for i in range(len(fr) - 1))
        with open(f[:-4] + "_times.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "pts_s", "t_s"])
            for i, (t, _) in enumerate(fr):
                w.writerow([i, "%.4f" % t, "%.4f" % (t - t0)])
        print("%-8s %7d %7d %10.3f %10.3f" % (os.path.basename(f)[:-4], len(fr), dropped, t0 - fr[0][0], fr[on][0] - t0))
    print("\nt_s in cam_N_times.csv: seconds from the lights going off, the same clock for every camera (+- 1 frame).")


if __name__ == "__main__":
    main()
