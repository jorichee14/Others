#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One showcase video cut from a storyboard: title cards, then the clips the
other scripts make, each above a caption strip saying what the viewer is looking at.

    python3 scripts/showcase_video.py configs/showcase_capabilities.yaml \
        --clips ~/showcase --out showcase.mp4
        [--draft]            # 2 s per segment, low resolution: check the cut first
        [--keep-work]        # keep the per-segment files

The storyboard is a list of segments, played in order:

    - card:  {title: ..., subtitle: ..., bullets: [...], seconds: 6}
    - clip:  path/to/clip.mp4
      caption: what this shows          # a strip across the bottom
      start: 12.0                       # seconds into the clip (default 0)
      seconds: 20                       # how much of it (default: to the end)
      speed: 2.0                        # >1 plays faster (default 1)
      make: python3 ...                 # how the clip was made; for the reader only

Every clip is scaled into the frame keeping its shape (into the space above
its caption strip, the rest filled with the background), resampled to one rate and re-encoded, so clips from different
tools with different sizes and rates join cleanly. A clip whose file is not
there yet gets a placeholder card naming it and its `make` line, so the cut
can be timed and reviewed before every recording exists.

Text with Chinese characters needs a CJK font: set `style.font` in the
storyboard (e.g. "Noto Sans CJK TC").
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import yaml                                                     # noqa: E402

STYLE = dict(width=1920, height=1080, fps=30, bg="#FFFFFF", ink="#1F2933",
             muted="#67737D", accent="#1B6FA8", accent2="#C2410C",
             strip="#1F2933", strip_ink="#FFFFFF", font=None, card_seconds=6.0,
             crf=18)


def run(command):
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        tail = result.stderr.decode("utf-8", "replace")[-800:]
        raise SystemExit(f"ffmpeg failed:\n  {' '.join(command)}\n{tail}")


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def figure(style):
    dpi = 100
    fig = plt.figure(figsize=(style["width"] / dpi, style["height"] / dpi), dpi=dpi)
    fig.patch.set_facecolor(style["bg"])
    return fig, dpi


def render_card(spec, style, path):
    """A title card: a rule in the accent colour, the title, a subtitle, bullets."""
    fig, dpi = figure(style)
    x0, y = 0.08, 0.72
    fig.add_artist(plt.Line2D([x0, x0 + 0.06], [y + 0.07, y + 0.07],
                              color=style["accent"], lw=6, solid_capstyle="butt"))
    if spec.get("kicker"):
        fig.text(x0, y + 0.10, spec["kicker"], color=style["accent"], fontsize=20,
                 fontweight="bold", va="bottom")
    title = textwrap.fill(spec.get("title", ""), 48)
    fig.text(x0, y, title, color=style["ink"], fontsize=44, fontweight="bold", va="top",
             linespacing=1.15)
    y -= 0.075 * (title.count("\n") + 1) + 0.04
    if spec.get("subtitle"):
        sub = textwrap.fill(spec["subtitle"], 80)
        fig.text(x0, y, sub, color=style["muted"], fontsize=24, va="top", linespacing=1.3)
        y -= 0.045 * (sub.count("\n") + 1) + 0.05
    for bullet in spec.get("bullets") or []:
        body = textwrap.fill(bullet, 88, subsequent_indent="   ")
        fig.text(x0, y, "•  " + body, color=style["ink"], fontsize=22, va="top",
                 linespacing=1.3)
        y -= 0.042 * (body.count("\n") + 1) + 0.022
    if spec.get("footer"):
        fig.text(x0, 0.06, spec["footer"], color=style["muted"], fontsize=16, va="bottom")
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_caption(text, style, path):
    """The frame a captioned clip sits on: background, caption strip along the bottom.

    Returns the strip's height in pixels. The clip is scaled into the space
    above the strip, not under it: a strip laid over the clip would hide
    whatever that clip keeps along its bottom edge -- bev_movie's traces.
    """
    fig, dpi = figure(style)
    lines = textwrap.fill(text, 110)
    rows = lines.count("\n") + 1
    h = (0.055 + 0.04 * rows)
    fig.add_artist(plt.Rectangle((0, 0), 1, h, transform=fig.transFigure,
                                 facecolor=style["strip"], alpha=0.82, lw=0))
    fig.add_artist(plt.Rectangle((0, 0), 0.006, h, transform=fig.transFigure,
                                 facecolor=style["accent"], lw=0))
    fig.text(0.03, h / 2, lines, color=style["strip_ink"], fontsize=24, va="center",
             linespacing=1.3)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return int(round(h * style["height"] / 2.0)) * 2


def encode_still(image, seconds, style, out):
    run(["ffmpeg", "-y", "-loop", "1", "-t", f"{seconds:.3f}", "-i", image,
         "-f", "lavfi", "-t", f"{seconds:.3f}",
         "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-vf", f"scale={style['width']}:{style['height']},fps={style['fps']},format=yuv420p",
         "-c:v", "libx264", "-crf", str(style["crf"]), "-c:a", "aac", "-shortest", out])


def encode_clip(seg, style, caption_png, strip, out, draft):
    w, h, bg = style["width"], style["height"], style["bg"].lstrip("#")
    speed = float(seg.get("speed", 1.0))
    start = float(seg.get("start", 0.0))
    seconds = seg.get("seconds")
    if draft:
        seconds = 2.0 * speed
    room = h - strip                       # the height above the caption strip
    chain = (f"[0:v]setpts=PTS/{speed},"
             f"scale={w}:{room}:force_original_aspect_ratio=decrease,"
             f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={style['fps']},setsar=1")
    inputs = ["-ss", f"{start:.3f}"]
    if seconds:
        inputs += ["-t", f"{float(seconds):.3f}"]
    inputs += ["-i", seg["clip"]]
    if caption_png:
        inputs += ["-loop", "1", "-i", caption_png]
        chain += (f"[v];[1:v]fps={style['fps']}[bg];"
                  f"[bg][v]overlay=(W-w)/2:({room}-h)/2:shortest=1,format=yuv420p[o]")
    else:
        chain += f",pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},format=yuv420p[o]"
    label = "[o]"
    # Silent audio on every segment: some players refuse a concat whose parts
    # disagree about having a sound track at all.
    total = (float(seconds) / speed) if seconds else \
        max(0.1, ((probe_duration(seg["clip"]) or 1.0) - start) / speed)
    run(["ffmpeg", "-y", *inputs, "-f", "lavfi", "-t", f"{total:.3f}",
         "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-filter_complex", chain, "-map", label,
         "-map", f"{2 if caption_png else 1}:a", "-t", f"{total:.3f}",
         "-c:v", "libx264", "-crf", str(style["crf"]), "-c:a", "aac", out])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("storyboard")
    parser.add_argument("--out", default="showcase.mp4")
    parser.add_argument("--draft", action="store_true",
                        help="2 s per segment at 960x540: review the cut quickly")
    parser.add_argument("--clips", default=None, metavar="DIR",
                        help="where relative clip paths are found (default: "
                             "beside the storyboard)")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg is needed", file=sys.stderr)
        return 2
    with open(args.storyboard, encoding="utf-8") as handle:
        board = yaml.safe_load(handle)
    style = dict(STYLE, **(board.get("style") or {}))
    if args.draft:
        style.update(width=960, height=540, crf=28)
    if style["font"]:
        plt.rcParams["font.family"] = style["font"]
    # Clip paths are relative to the storyboard, so it can sit beside them.
    base = os.path.abspath(os.path.expanduser(args.clips)) if args.clips else \
        os.path.dirname(os.path.abspath(args.storyboard))

    work = os.path.splitext(os.path.abspath(args.out))[0] + "_work"
    os.makedirs(work, exist_ok=True)
    parts, missing = [], []
    for number, seg in enumerate(board["segments"]):
        stem = os.path.join(work, f"{number:03d}")
        if "card" in seg:
            spec = seg["card"]
            seconds = 2.0 if args.draft else float(spec.get("seconds",
                                                            style["card_seconds"]))
            render_card(spec, style, stem + ".png")
            encode_still(stem + ".png", seconds, style, stem + ".mp4")
            print(f"[{number:2d}] card  {seconds:5.1f} s  {spec.get('title', '')[:60]}")
        else:
            clip = os.path.expanduser(seg["clip"])
            if not os.path.isabs(clip):
                clip = os.path.join(base, clip)
            if not os.path.exists(clip):
                missing.append(seg["clip"])
                spec = dict(kicker="CLIP TO COME", title=seg["clip"],
                            subtitle=seg.get("caption", ""),
                            bullets=[f"make: {seg['make']}"] if seg.get("make") else [])
                seconds = 2.0 if args.draft else float(seg.get("seconds") or 5.0)
                render_card(spec, style, stem + ".png")
                encode_still(stem + ".png", seconds, style, stem + ".mp4")
                print(f"[{number:2d}] MISSING {seg['clip']} (placeholder card)")
            else:
                caption, strip = None, 0
                if seg.get("caption"):
                    caption = stem + "_caption.png"
                    strip = render_caption(seg["caption"], style, caption)
                encode_clip(dict(seg, clip=clip), style, caption, strip,
                            stem + ".mp4", args.draft)
                print(f"[{number:2d}] clip  {os.path.basename(clip)}")
        parts.append(stem + ".mp4")

    listing = os.path.join(work, "parts.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        handle.writelines(f"file '{p}'\n" for p in parts)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing,
         "-c", "copy", "-movflags", "+faststart", os.path.abspath(args.out)])
    total = probe_duration(os.path.abspath(args.out))
    print(f"wrote {args.out}" + (f"  ({total:.1f} s)" if total else ""))
    if missing:
        print(f"{len(missing)} clip(s) still to make, shown as placeholder cards:")
        for name in missing:
            print(f"  {name}")
    if not args.keep_work:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
