#!/usr/bin/env python3
"""The framework schematics as standalone vector files, for papers and slides.

    python make_schematics.py                  # docs/figures/framework.svg, roadmap.svg (+ .pdf, .png)
    python make_schematics.py --out paper/figs

The same drawings as in the framework document (system overview, PhD roadmap), as
plain SVG. With cairosvg installed (pip install cairosvg) also PDF (vector, for
LaTeX) and PNG (300 dpi). Edit the labels below and run again; or open the SVG in
Inkscape / Illustrator / draw.io and edit it there.

LaTeX:  \\includegraphics[width=\\linewidth]{figs/framework.pdf}
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, INK2, EDGE, TINT = "#0b0b0b", "#52514e", "#8a8984", "#f1f0ec"
ACCENT = "#2a78d6"
FONT = "Helvetica, Arial, sans-serif"


def svg(w, h, body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' "
            f"font-family='{FONT}' font-size='11.5'><title>{title}</title>"
            f"<rect width='{w}' height='{h}' fill='white'/>"
            "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' "
            f"orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='{EDGE}'/></marker></defs>"
            f"{body}</svg>")


def text(x, y, s, size=11.5, weight=400, anchor="start", fill=INK):
    return f"<text x='{x}' y='{y}' font-size='{size}' font-weight='{weight}' text-anchor='{anchor}' fill='{fill}'>{s}</text>"


def rect(x, y, w, h, fill="none", stroke=EDGE, sw=1.25, rx=8, opacity=1):
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' fill-opacity='{opacity}' "
            f"stroke='{stroke}' stroke-width='{sw}'/>")


def framework():
    b = [text(24, 30, "Any sensor in, one calibrated belief out, any task on top", 15, 600)]
    y0, hh, mid = 56, 244, 178
    # sensors
    b += [rect(24, y0, 140, hh), text(94, 78, "Sensors (any)", 13, 600, "middle")]
    for i, s in enumerate(["Room cameras (16)", "mmWave radar", "Robot ZED, LiDAR", "Robot own pose"]):
        y = 92 + 44 * i
        b += [rect(34, y, 120, 36, sw=1, rx=6), text(94, y + 22, s, anchor="middle")]
    b += [text(94, 284, "measurement + model", 11, anchor="middle", fill=INK2)]
    # perception
    b += [rect(188, y0, 196, hh), text(286, 78, "Perception node", 13, 600, "middle")]
    for i, s in enumerate(["1  Multi-view association", "2  Tracklets, online", "3  Global linking, exact",
                           "4  Merge and relabel"]):
        y = 92 + 44 * i
        b += [rect(198, y, 176, 36, sw=1, rx=6), text(208, y + 22, s, 12)]
    b += [text(286, 284, "sensor-agnostic", 11, anchor="middle", fill=INK2)]
    # trust (the interface)
    b += [rect(408, y0, 150, hh, fill=ACCENT, stroke=ACCENT, sw=2, opacity=0.12),
          text(483, 78, "Belief + trust", 13, 600, "middle")]
    for i, s in enumerate(["position, covariance", "existence", "identity confidence", "freshness, sources",
                           "visibility map"]):
        b += [text(420, 110 + 26 * i, s)]
    b += [text(483, 284, "task-agnostic", 11, anchor="middle", fill=INK2)]
    # coordination
    b += [rect(582, y0, 154, hh), text(659, 78, "Coordination", 13, 600, "middle")]
    for i, (s1, s2) in enumerate([("Task allocation", "identity-gated"), ("Chance-constrained", "motion (MPC)"),
                                  ("VoI messages", "within bandwidth B")]):
        y = 92 + 68 * i
        b += [rect(592, y, 134, 56, sw=1, rx=6), text(659, y + 22, s1, anchor="middle"),
              text(659, y + 40, s2, anchor="middle", fill=INK2)]
    # other agents
    b += [rect(300, 332, 258, 52), text(429, 354, "Other agents (robots, room)", 13, 600, "middle"),
          text(429, 372, "same belief format, fused", anchor="middle", fill=INK2)]
    # connectors
    line = lambda d, both=False: (f"<path d='{d}' fill='none' stroke='{EDGE}' stroke-width='1.25' "
                                  + ("marker-start='url(#arrow)' " if both else "") + "marker-end='url(#arrow)'/>")
    b += [line(f"M164 {mid}H186"), line(f"M384 {mid}H406"), line(f"M558 {mid}H580"),
          line("M483 302V330", both=True), line("M659 300V412H244V302"),
          text(491, 321, "beliefs: delay, loss", 11, fill=INK2),
          text(450, 404, "re-identify tasks, new viewpoints", 11, anchor="middle", fill=INK2)]
    return svg(760, 430, "".join(b), "Framework: sensors, perception, trust interface, coordination")


def roadmap():
    b = [text(24, 30, "Three years, one paper per layer, each gated by real-data evidence", 15, 600)]
    phases = [(24, "Year 1 \u00b7 Perception, trust", ["Calibrated identity and", "existence trust (marginals)",
                                                         "ED305 dataset with identity", "ground truth"],
               "Paper: tracker + dataset"),
              (272, "Year 2 \u00b7 Multi-agent", ["Belief-level fusion of robots", "and room; agent reliability",
                                                 "What to share per bit of", "task value"],
               "Paper: sharing under bandwidth"),
              (520, "Year 3 \u00b7 Coordination", ["Identity-gated allocation,", "chance-constrained motion,",
                                                  "value-of-information messages", "on ED305 robots"],
               "Paper + thesis")]
    for x, name, lines, out in phases:
        b += [rect(x, 56, 216, 32, fill=TINT), text(x + 108, 77, name, 13, 600, "middle"), rect(x, 96, 216, 140)]
        for i, s in enumerate(lines):
            b += [text(x + 12, [120, 138, 162, 180][i], s)]
        b += [text(x + 12, 220, out, weight=600)]
    for cx, label in ((256, "Gate: trust calibrated on real data"), (504, "Gate: robots recorded, synced")):
        b += [f"<polygon points='{cx - 11},166 {cx},155 {cx + 11},166 {cx},177' fill='{ACCENT}' fill-opacity='0.25' "
              f"stroke='{ACCENT}' stroke-width='2'/>", text(cx, 262, label, anchor="middle", fill=INK2)]
    return svg(760, 280, "".join(b), "PhD roadmap")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "docs/figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        import cairosvg
    except ImportError:
        cairosvg = None
        print("cairosvg not installed: SVG only (pip install cairosvg for PDF and PNG)")
    for name, code in (("framework", framework()), ("roadmap", roadmap())):
        p = a.out / f"{name}.svg"
        p.write_text(code)
        made = [p.name]
        if cairosvg:
            cairosvg.svg2pdf(bytestring=code.encode(), write_to=str(p.with_suffix(".pdf")))
            cairosvg.svg2png(bytestring=code.encode(), write_to=str(p.with_suffix(".png")), scale=300 / 96)
            made += [p.with_suffix(".pdf").name, p.with_suffix(".png").name]
        print(f"{a.out}: {', '.join(made)}")


if __name__ == "__main__":
    main()
