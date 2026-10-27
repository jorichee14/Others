#!/usr/bin/env python3
"""The plan figure: from real data to a twin checked against held-out real
passes, then train, test on held-out recordings, deploy on the testbed.

    python3 scoop_gym/docs/make_twin_plan.py      # writes twin_plan.svg/.png/.pdf here

Status per part: validated (filled), implemented (half), planned (hollow).
Each setting is tagged M (measured from the recordings) or A (added).
Edit STATUS below as parts move on, and rerun.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W = 780
FONT = "DejaVu Sans, Arial, sans-serif"
INK, MUTED = "#222", "#666"
ORANGE = "#c45a1c"

# (fill, stroke) per stage
C = {"data": ("#ececec", "#9a9a9a"), "twin": ("#fbf0dc", "#e0b46a"),
     "gate": ("#efe6f6", "#a582c4"), "train": ("#e3ecf8", "#7d9ccc"),
     "test": ("#e4f3e4", "#7cb37c"), "deploy": ("#dff1f1", "#62a9a9")}

out = []


def t(x, y, s, size=13, weight="normal", fill=INK, anchor="start", style=""):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
               f'fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>')


def box(x, y, w, h, kind, r=10, sw=1.6):
    f, s = C[kind]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{f}" stroke="{s}" stroke-width="{sw}"/>')


def card(x, y, w, h):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fff" stroke="#bbb" stroke-width="1"/>')


def status(x, y, st, r=6):
    if st == "validated":
        out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{ORANGE}"/>')
    elif st == "implemented":
        out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{ORANGE}" stroke-width="1.6"/>')
        out.append(f'<path d="M{x},{y - r} A{r},{r} 0 0,0 {x},{y + r} Z" fill="{ORANGE}"/>')
    else:
        out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{ORANGE}" stroke-width="1.6"/>')


def tag(x, y, kind):
    fill, txt = ("#3b3b3b", "#fff") if kind == "M" else ("#fff", "#3b3b3b")
    out.append(f'<rect x="{x}" y="{y - 10}" width="16" height="14" rx="3" fill="{fill}" stroke="#3b3b3b" stroke-width="1"/>')
    t(x + 8, y + 1, kind, 10, "bold", txt, "middle")


def item(x, y, w, st, title, sub, tags=()):
    """One part: status dot, title, subtitle, M/A tags at the right."""
    card(x, y, w, 46)
    status(x + 14, y + 16, st)
    t(x + 28, y + 21, title, 14, "bold")
    t(x + 28, y + 38, sub, 11.5, fill=MUTED)
    for k, g in enumerate(tags):
        tag(x + w - 22 - 20 * k, y + 20, g)


def arrow(x1, y1, x2, y2, dash=False, color="#333"):
    d = ' stroke-dasharray="5,4"' if dash else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d} marker-end="url(#ah)"/>')


def path(d, dash=False, color="#333"):
    dd = ' stroke-dasharray="5,4"' if dash else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dd} marker-end="url(#ah)"/>')


def stage(y, h, kind, num, title, note):
    box(60, y, 660, h, kind)
    t(78, y + 28, num, 21, "bold")
    t(102, y + 28, title, 19, "bold")
    if note:
        t(704, y + 26, note, 12, fill=MUTED, anchor="end", style='font-style="italic"')


# ---------------------------------------------------------------- title, legend
t(60, 30, "SCooP-iT Gym: plan from real data to a validated twin", 18, "bold")
lx = 60
for st, lab in (("validated", "validated"), ("implemented", "implemented, not yet run"), ("planned", "planned")):
    status(lx + 6, 52, st)
    t(lx + 18, 56, lab, 12, fill=MUTED)
    lx += 30 + len(lab) * 7.0
tag(lx + 12, 52, "M"); t(lx + 34, 56, "measured from recordings", 12, fill=MUTED)
tag(lx + 214, 52, "A"); t(lx + 236, 56, "added", 12, fill=MUTED)

# ---------------------------------------------------------------- 1 real data
y = 72
stage(y, 182, "data", "1", "Real data", "two kinds of pass, never mixed")
t(78, y + 54, "Build passes (site 1)", 13, "bold", MUTED)
t(400, y + 54, "Held-out passes (another day)", 13, "bold", MUTED)
item(78, y + 62, 300, "validated", "Reference map", "SLAM-built, cm-level, checked")
item(78, y + 114, 300, "implemented", "Robots, sensors, links", "calibration, paths, RTT/goodput")
item(400, y + 62, 302, "planned", "Survey + coop passes", "used for nothing but checks")
item(400, y + 114, 302, "planned", "Labels on held-out frames", "boxes for scoring step 5")

# ---------------------------------------------------------------- 2 twin
y = 284
stage(y, 314, "twin", "2", "Digital twin  ·  Isaac Sim", "built from build passes only")
cw, gx = 300, 400
item(78, y + 44, cw, "implemented", "Building", "USD from the reference map", ("M",))
item(gx, y + 44, 302, "implemented", "Sensor models", "LiDAR beams, cameras, real mounts", ("M",))
item(78, y + 96, cw, "implemented", "Paths: recorded", "robots replay where they drove", ("M",))
item(gx, y + 96, 302, "planned", "Paths: new routes", "robot control in Isaac", ("A",))
item(78, y + 148, cw, "implemented", "Link: logged / fitted", "on recorded paths (scoop_gym)", ("M",))
item(gx, y + 148, 302, "planned", "Link: indexed by place", "survey cells + contention mode", ("M",))
item(78, y + 200, cw, "planned", "Dropouts, pose error", "fitted from the recordings", ("M",))
item(gx, y + 200, 302, "implemented", "People and objects", "boxes now; assets + semantics next", ("A",))
# scenario strip
sy = y + 264
for k in (2, 1):
    out.append(f'<rect x="{78 + 4 * k}" y="{sy - 4 * k}" width="624" height="38" rx="6" fill="#fff" stroke="#d8c39a" stroke-width="1"/>')
card(78, sy, 624, 38)
t(92, sy + 16, "Scenario = a setting per part", 12.5, "bold")
t(92, sy + 31, "link: logged | fitted | place  ·  dropouts: fitted  ·  pose: reference | SLAM  ·  people: +N", 11.5, fill=MUTED)
t(690, sy + 25, "× N", 17, "bold", ORANGE, "end")

# ---------------------------------------------------------------- 3 gate
y = 628
stage(y, 112, "gate", "3", "Check the twin", "against held-out passes; train only on what passes")
item(78, y + 44, 200, "planned", "Scans", "≤ 3 cm from real, same pose")
item(290, y + 44, 200, "planned", "Conditions", "fitted dropouts match")
item(502, y + 44, 200, "planned", "Link", "in-frame share within 10 pt")

# ---------------------------------------------------------------- 4 train
y = 770
stage(y, 96, "train", "4", "Train in the twin", "")
item(78, y + 40, 300, "planned", "Perception and comms", "detector, fusion, what to send")
item(400, y + 40, 302, "planned", "Motion policies", "e.g. OcclusionHandoff")

# ---------------------------------------------------------------- 5 test, 6 deploy
y = 896
stage(y, 96, "test", "5", "Test on held-out recordings", "real data only")
item(78, y + 40, 624, "planned", "Perception and comms models", "scored on real frames and the logged link; motion cannot be tested on a recording")
y = 1022
stage(y, 96, "deploy", "6", "Deploy on the testbed", "")
item(78, y + 40, 624, "planned", "Motion policies on the real robots", "real outcome vs. twin prediction, across several policies (sim-vs-real correlation)")

# ---------------------------------------------------------------- arrows
for y1, y2 in ((254, 284), (598, 628), (740, 770), (866, 896), (992, 1022)):
    arrow(390, y1 + 1, 390, y2 - 3)
# held-out passes -> gate and test (dashed, right margin)
path("M721,160 L750,160 L750,684 L724,684", dash=True, color="#777")
path("M750,684 L750,944 L724,944", dash=True, color="#777")
t(764, 560, "held-out passes", 12, fill="#777", anchor="middle", style='transform="rotate(90 764 560)"')
# deploy -> new recordings (feedback, left margin)
path("M59,1070 L30,1070 L30,150 L56,150", color=ORANGE)
t(18, 610, "new recordings refit the twin", 12, fill=ORANGE, anchor="middle", style='transform="rotate(-90 18 610)"')

H = 1138
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       '<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
       '<path d="M0,0 L8,4.5 L0,9 Z" fill="context-stroke"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>")
p = os.path.join(HERE, "twin_plan.svg")
open(p, "w").write(svg)
import cairosvg                                                     # noqa: E402
cairosvg.svg2png(url=p, write_to=p[:-4] + ".png", output_width=W * 2)
cairosvg.svg2pdf(url=p, write_to=p[:-4] + ".pdf")
print("wrote", p, "+ .png, .pdf")
