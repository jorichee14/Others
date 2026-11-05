#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The two robots' cameras side by side, slowed down, with cooperative
perception made visible.

    python3 scripts/coop_camera_movie.py --root <split> --out coop_cameras.mp4
        [--start 30] [--seconds 40] [--speed 0.5] [--cloud mapping/cloud_in_ref_<tag>.pcd]
        [--names mobile_1,mobile_2] [--max-range 6] [--hold 1.5] [--theme light]
    python3 scripts/coop_camera_movie.py --root <split> --scan --seconds 9999   # where the hand-overs are

Every frame, each labelled object near either robot (--max-range) is sorted by
whose own LiDAR sweep actually has returns inside it (bev_movie.seen_by, the
same test --coop-gain uses):

  on each robot's camera
    thin solid box      the robot's own LiDAR finds the object: it sees it itself
    highlighted box     the robot's own sweep does not reach it (behind something,
                        or out of its range) but the other robot's does: the
                        object cooperation hands it. Filled in the OTHER robot's
                        colour, pulsing, tagged "from <other robot>"
  on the map underneath
    each robot as an arrow with its camera's field of view; every object
    coloured by who sees it; a line from the robot that sees an object to it,
    for each object it hands to the other

The first time an object is handed over, the video holds on that frame for
--hold seconds with a caption saying what happened, so the moment can be read.
Only frames where the robot's image is fresh are used for that pause: a held
ZED image (1.3 Hz) is of where the robot was before.

--speed is seconds of the recording per second of video: 0.5 plays it at half
speed. The boxes go through the camera block of the frame each shown image came
from, exactly as bev_movie.py does.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                               # noqa: E402
import bev_movie as BM                                           # noqa: E402  (sets the Agg backend)
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Polygon                           # noqa: E402


def hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (Andrew's monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def class_name(entry) -> str:
    return str(entry.get("class_name") or entry.get("obj_type") or "object").replace("_", " ")


def fov_wedge(camera, reach: float):
    """The camera's horizontal field of view on the map: apex, left and right ends."""
    if not camera or "cords" not in camera or "intrinsic" not in camera:
        return None
    M = BM.x_to_world(camera["cords"])
    K = np.asarray(camera["intrinsic"], dtype=np.float64)
    width = 2.0 * K[0, 2]
    half = np.arctan2(width / 2.0, K[0, 0])
    fwd = M[:3, 2][:2]
    if np.linalg.norm(fwd) < 1e-6:
        return None
    fwd = fwd / np.linalg.norm(fwd)
    ang = np.arctan2(fwd[1], fwd[0])
    apex = M[:2, 3]
    ends = [apex + reach * np.array([np.cos(ang + s * half), np.sin(ang + s * half)]) for s in (1, -1)]
    return np.array([apex, ends[0], ends[1]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="split dir holding case_*/")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--out", default="coop_cameras.mp4")
    ap.add_argument("--names", default=None, help="the robots' names, in agent-folder order (e.g. mobile_1,mobile_2)")
    ap.add_argument("--start", type=float, default=0.0, help="where in the recording to begin (s)")
    ap.add_argument("--seconds", type=float, default=40.0, help="how much of the recording to show (s)")
    ap.add_argument("--speed", type=float, default=0.5, help="seconds of recording per second of video")
    ap.add_argument("--rate", type=float, default=10.0, help="the frame grid's rate (Hz)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--min-points", type=int, default=3, help="LiDAR returns inside a box to count it seen")
    ap.add_argument("--max-range", type=float, default=6.0, help="only objects this near either robot (m)")
    ap.add_argument("--hold", type=float, default=1.5, help="pause (s) the first time each object is handed over")
    ap.add_argument("--cloud", default=None, help="a map cloud for the floor plan under the inset")
    ap.add_argument("--cloud-stride", type=int, default=12)
    ap.add_argument("--theme", choices=sorted(BM.THEMES), default="light")
    ap.add_argument("--title", default="Cooperative perception: what each robot sees, and what the other hands it")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--scan", action="store_true",
                    help="list, once a second, where one robot hands objects to the other; no video")
    a = ap.parse_args()

    theme = BM.THEMES[a.theme]
    root = os.path.expanduser(a.root)
    scenarios = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    scenario = os.path.join(root, a.scenario or scenarios[0])
    agents = BM.agent_dirs(scenario)[:2]
    if len(agents) < 2:
        print(f"{scenario}: needs two agents", file=sys.stderr)
        return 2
    names = [s.strip() for s in a.names.split(",")] if a.names else [f"robot {x}" for x in agents]
    colour = dict(zip(agents, theme["agents"]))
    name = dict(zip(agents, names))
    other = {agents[0]: agents[1], agents[1]: agents[0]}

    tracks, boxes = {}, {}
    for agent in agents:
        track, found, _, _ = BM.read_track(os.path.join(scenario, agent), [], want_camera=True)
        tracks[agent] = track
        boxes.update(found)
    keys = [k for k, *_ in tracks[agents[0]]]
    n_frames = min(len(tracks[x]) for x in agents)
    first = int(round(a.start * a.rate))
    last = min(n_frames - 1, first + int(round(a.seconds * a.rate)))
    if last <= first:
        print(f"nothing to show: {n_frames} frames, start {first}", file=sys.stderr)
        return 2
    print(f"{len(boxes)} labelled objects, frames {first}..{last} of {n_frames}", flush=True)

    clouds = {}

    def returns_of(agent, frame):
        if (agent, frame) not in clouds:
            pts = BM.read_pcd(os.path.join(scenario, agent, f"{keys[frame]}.pcd"))[:, :3]
            pose = BM.x_to_world(tracks[agent][frame][5])
            clouds[(agent, frame)] = pts.astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
            while len(clouds) > 6:
                clouds.pop(next(iter(clouds)))
        return clouds[(agent, frame)]

    def near(frame):
        where = [tracks[x][frame][1:3] for x in agents]
        return {k for k, e in boxes.items()
                if min(np.hypot(e["location"][0] - hx, e["location"][1] - hy) for hx, hy in where) <= a.max_range}

    def who_sees(frame):
        close = near(frame)
        return {x: BM.seen_by(returns_of(x, frame), boxes, a.min_points) & close for x in agents}

    if a.scan:
        step = max(1, int(round(a.rate)))
        print("   t (s)   " + "   ".join(f"{name[x]} receives" for x in agents))
        busy = []
        for frame in range(first, last + 1, step):
            sees = who_sees(frame)
            got = {x: sees[other[x]] - sees[x] for x in agents}
            if any(got.values()):
                busy.append(frame / a.rate)
                print(f"{frame / a.rate:8.1f}   " + "   ".join(
                    f"{len(got[x]):3d} ({', '.join(sorted(class_name(boxes[k]) for k in got[x]))})" for x in agents))
        if busy:
            print(f"hand-overs at {len(busy)} of {len(range(first, last + 1, step))} sampled seconds; "
                  f"pick --start a few seconds before a run of them")
        return 0

    # ------------------------------------------------------------------ figure
    W, H = (int(v) for v in a.size.lower().split("x"))
    W, H = W - W % 2, H - H % 2
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=theme["bg"])
    aspect = BM.camera_aspect(scenario, agents)
    cam_w = 0.47
    cam_h = min(0.56, cam_w * (W / H) / aspect)
    cams = {agents[0]: fig.add_axes([0.02, 0.97 - 0.07 - cam_h, cam_w, cam_h]),
            agents[1]: fig.add_axes([0.51, 0.97 - 0.07 - cam_h, cam_w, cam_h])}
    inset_top = 0.97 - 0.07 - cam_h - 0.04
    axm = fig.add_axes([0.02, 0.03, 0.47, inset_top - 0.03])
    fig.text(0.02, 0.965, a.title, fontsize=19, fontweight="bold", color=theme["ink"], va="top")
    clock = fig.text(0.98, 0.965, "", fontsize=14, color=theme["muted"], va="top", ha="right")
    caption = fig.text(0.53, inset_top - 0.02, "", fontsize=15, color=theme["ink"], va="top", linespacing=1.5)
    banner = fig.text(0.745, inset_top - 0.075, "", fontsize=22, fontweight="bold", ha="center", va="center", color="white",
                      bbox=dict(boxstyle="round,pad=0.6", fc="#1F2933", ec="none", alpha=0.0), zorder=50)
    legend_y = 0.17
    fig.text(0.53, legend_y, "How to read it", fontsize=14, fontweight="bold", color=theme["ink"])
    fig.text(0.53, legend_y - 0.035, "Thin outline: the robot's own LiDAR sees the object.", fontsize=13,
             color=theme["ink"])
    fig.text(0.53, legend_y - 0.07, "Filled, pulsing box tagged \"from …\": the robot cannot see it itself;", fontsize=13,
             color=theme["ink"])
    fig.text(0.53, legend_y - 0.098, "the other robot does, and shares it. That is the cooperative gain.", fontsize=13,
             color=theme["ink"])
    fig.text(0.53, legend_y - 0.133, f"Objects within {a.max_range:g} m of either robot; seen = "
             f"at least {a.min_points} LiDAR returns inside the box.", fontsize=11, color=theme["muted"])

    # map inset: fixed extent over the shown stretch
    xy = np.array([[tracks[x][f][1], tracks[x][f][2]] for x in agents for f in range(first, last + 1)])
    xy = np.vstack([xy] + [[e["location"][0], e["location"][1]] for e in boxes.values()
                           if np.min(np.hypot(xy[:, 0] - e["location"][0], xy[:, 1] - e["location"][1])) <= a.max_range * 1.5]
                   if boxes else [xy])
    lo, hi = xy.min(0) - 3.0, xy.max(0) + 3.0
    if a.cloud:
        bxy = BM.read_backdrop_xy(a.cloud)[::max(1, a.cloud_stride)]
        m = (bxy[:, 0] > lo[0]) & (bxy[:, 0] < hi[0]) & (bxy[:, 1] > lo[1]) & (bxy[:, 1] < hi[1])
        axm.scatter(bxy[m, 0], bxy[m, 1], s=0.3, c=theme["map"], linewidths=0, zorder=0, rasterized=True)

    # ------------------------------------------------------------------ frames
    out = os.path.expanduser(a.out)
    if not shutil.which("ffmpeg"):
        print("ffmpeg is needed", file=sys.stderr)
        return 1
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264",
                             "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                            stdin=subprocess.PIPE)

    def emit():
        fig.canvas.draw()
        proc.stdin.write(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).tobytes())

    held = {x: (None, None, None) for x in agents}       # image, camera block, frame it came from
    handed_before = set()
    per_frame_video = max(1, int(round(a.fps / (a.rate * a.speed))))
    video_t = 0.0
    total = (last - first + 1) * per_frame_video
    print(f"{total} frames before pauses, {per_frame_video} video frame(s) per recording frame", flush=True)

    for frame in range(first, last + 1):
        sees_now = who_sees(frame)
        shown = {}
        for x in agents:
            image, fresh = BM.load_image(os.path.join(scenario, x), keys[frame], held[x][0])
            camera = tracks[x][frame][4] if fresh else held[x][1]
            source = frame if fresh else held[x][2]
            if fresh:
                held[x] = (image, camera, frame)
            if source is None:
                shown[x] = (image, camera, None, set(), set(), fresh)
                continue
            s = sees_now if source == frame else who_sees(source)
            own = s[x]
            gift = s[other[x]] - own
            shown[x] = (image, camera, source, own, gift, fresh)

        # first hand-overs on a fresh image: the moments to pause on
        new = [(x, k) for x in agents for k in shown[x][4]
               if shown[x][5] and (x, k) not in handed_before]
        for x, k in new:
            handed_before.add((x, k))

        repeats = per_frame_video + (int(round(a.hold * a.fps)) if new else 0)
        for rep in range(repeats):
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.2 * video_t)
            for x in agents:
                image, camera, source, own, gift, fresh = shown[x]
                ax = cams[x]
                ax.clear(); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(colour[x]); sp.set_linewidth(3)
                if image is None:
                    ax.set_facecolor(theme["panel"])
                    ax.text(0.5, 0.5, "no image yet", ha="center", va="center", transform=ax.transAxes,
                            color=theme["muted"])
                    continue
                ax.imshow(image)
                h_, w_ = image.shape[:2]
                outside = 0
                for k in sorted(own | gift):
                    e = boxes[k]
                    uvz = BM.project(BM.box_corners_3d(e), camera["cords"], camera["intrinsic"])
                    front = uvz[:, 2] > 0.05
                    inside = front & (uvz[:, 0] >= 0) & (uvz[:, 0] < w_) & (uvz[:, 1] >= 0) & (uvz[:, 1] < h_)
                    if not inside.any():
                        if k in gift:
                            outside += 1
                        continue
                    if k in own:
                        for i0, i1 in BM.BOX_EDGES:
                            if front[i0] and front[i1]:
                                ax.plot(uvz[[i0, i1], 0], uvz[[i0, i1], 1], color=BM.class_colour(e, theme), lw=1.6)
                    elif front.all():
                        c = colour[other[x]]
                        ax.add_patch(Polygon(hull(uvz[:, :2]), closed=True, fc=c, ec=c,
                                             alpha=0.25 + 0.35 * pulse, lw=0))
                        for i0, i1 in BM.BOX_EDGES:
                            ax.plot(uvz[[i0, i1], 0], uvz[[i0, i1], 1], color=c, lw=2.2, ls=(0, (4, 2)))
                        top = uvz[np.argmin(uvz[:, 1])]
                        ax.text(np.clip(top[0], 40, w_ - 40), max(14, top[1] - 8),
                                f"{class_name(e)} · from {name[other[x]]}", fontsize=12, color="white",
                                ha="center", va="bottom", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none", alpha=0.95))
                ax.set_xlim(0, w_); ax.set_ylim(h_, 0)
                gift_txt = f",  +{len(gift)} from {name[other[x]]}" if gift else ""
                ax.set_title(f"{name[x]}'s camera:  sees {len(own)} itself{gift_txt}"
                             + ("" if fresh else "   (last image held)"),
                             loc="left", fontsize=15, color=colour[x], fontweight="bold")

            # map inset
            for art in list(axm.lines) + list(axm.patches) + list(axm.texts):
                art.remove()
            for x in agents:
                f = shown[x][2] if shown[x][2] is not None else frame
                w = fov_wedge(shown[x][1], a.max_range)
                if w is not None:
                    axm.add_patch(Polygon(w, closed=True, fc=colour[x], ec=colour[x], alpha=0.10, lw=0.8, zorder=1))
                _, px, py, yaw, _, _ = tracks[x][frame]
                axm.add_patch(Polygon(BM.arrow(px, py, yaw, 0.9), closed=True, fc=colour[x], ec="white",
                                      lw=1.0, zorder=6))
                axm.text(px, py + 1.0, name[x], color=colour[x], fontsize=12, ha="center", fontweight="bold", zorder=7)
            close = near(frame)
            for k in close:
                e = boxes[k]
                corners = BM.box_corners(e)
                by = [x for x in agents if k in sees_now[x]]
                if len(by) == 2:
                    fc, ec, al = theme["muted"], theme["ink"], 0.5
                elif len(by) == 1:
                    fc, ec, al = colour[by[0]], colour[by[0]], 0.75
                else:
                    fc, ec, al = "none", theme["muted"], 0.6
                axm.add_patch(Polygon(corners, closed=True, fc=fc, ec=ec, alpha=al, lw=1.0, zorder=3))
            for x in agents:
                for k in shown[x][4]:
                    e = boxes[k]
                    src = tracks[other[x]][frame]
                    axm.plot([src[1], e["location"][0]], [src[2], e["location"][1]], color=colour[other[x]],
                             lw=1.6, ls=(0, (2, 2)), alpha=0.6 + 0.4 * pulse, zorder=4)
            axm.set_xlim(lo[0], hi[0]); axm.set_ylim(lo[1], hi[1]); axm.set_aspect("equal")
            axm.set_xticks([]); axm.set_yticks([])
            for sp in axm.spines.values():
                sp.set_color(theme["grid"])
            axm.set_title("From above: fields of view, objects coloured by who sees them, "
                          "dashed lines = shared", loc="left", fontsize=12, color=theme["muted"])

            lines = []
            for x in agents:
                own, gift = shown[x][3], shown[x][4]
                lines.append(f"{name[x]}: sees {len(own)} object(s) itself"
                             + (f", receives {len(gift)} from {name[other[x]]}" if gift else ""))
            both = set().union(*(sees_now[x] for x in agents))
            alone = max(len(sees_now[x]) for x in agents)
            lines.append(f"Together: {len(both)} object(s) known  (best single robot: {alone})")
            caption.set_text("\n".join(lines))
            clock.set_text(f"t = {frame / a.rate:6.1f} s    {a.speed:g}× speed")
            if new and rep >= per_frame_video:
                x, k = new[0]
                more = f"  (+{len(new) - 1} more)" if len(new) > 1 else ""
                banner.set_text(f"{name[other[x]]} shares a {class_name(boxes[k])}\nthat {name[x]} cannot see{more}")
                banner.get_bbox_patch().set_alpha(0.85)
                caption.set_alpha(0.0)
            else:
                banner.set_text("")
                banner.get_bbox_patch().set_alpha(0.0)
                caption.set_alpha(1.0)
            emit()
            video_t += 1.0 / a.fps
        if (frame - first) % 50 == 0:
            print(f"  recording frame {frame - first} of {last - first}", flush=True)

    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    print(f"wrote {out}: {video_t:.1f} s of video, {len(handed_before)} hand-over(s) paused on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
