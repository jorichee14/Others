#!/usr/bin/env python3
"""Sample frames from the room cameras' recordings, for labelling.

    python extract_frames.py rec/cam_*.mp4 --out data/overhead
    python extract_frames.py rec/*.mp4 --out data/overhead --every 2 --max-per-video 300

One frame per --every seconds, and only if enough of the picture changed since
the last kept frame (--min-change), so an hour of a still room does not turn
into 3600 copies of the same image. The camera name comes from the file name
(cam_7_....mp4 -> cam_7). Run it again with more videos: videos already in
frames.csv are skipped.

Writes to --out:
  images/<video stem>_t<ms>.jpg
  frames.csv   image, camera, video, t: what make_dataset.py splits on
"""
import argparse
from pathlib import Path

import cv2

from ed305.yolo_data import append_frames_csv, camera_of, read_frames_csv, sample_video


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between samples (default 1)")
    ap.add_argument("--min-change", type=float, default=0.005,
                    help="fraction of the image that must change to keep a frame (default 0.005; 0 keeps all)")
    ap.add_argument("--max-per-video", type=int, help="stop a video after this many frames")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    return ap.parse_args()


def main():
    a = parse()
    img_dir = a.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    done = {r["video"] for r in read_frames_csv(a.out)}
    stems = {Path(v).stem: v for v in done}
    total = 0
    for v in a.videos:
        if v.name in done:
            print(f"{v.name}: already in frames.csv, skipped")
            continue
        if v.stem in stems:
            raise SystemExit(f"{v}: another video named {v.stem} is already in the dataset; rename one")
        stems[v.stem] = v.name
        cam, rows = camera_of(v), []
        for t, frame in sample_video(v, a.every, a.min_change, a.max_per_video):
            name = f"{v.stem}_t{round(t * 1000):08d}.jpg"
            cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append({"image": name, "camera": cam, "video": v.name, "t": f"{t:.3f}"})
        append_frames_csv(a.out, rows)
        total += len(rows)
        print(f"{v.name} ({cam}): {len(rows)} frames")
    print(f"{total} new frames in {img_dir}")


if __name__ == "__main__":
    main()
