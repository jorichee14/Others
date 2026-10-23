#!/usr/bin/env python3
"""Block diagram of what is made from the reference map (datasets/lod3dgs.py),
each block with the test that covers it. Writes docs/lod3dgs_pipeline.svg.

    python3 scoop_pipeline/analysis/lod3dgs_diagram.py [out.svg]
"""
import os
import sys

W, H = 1500, 800
FONT = "Helvetica, Arial, sans-serif"
# kind: map (the reference map), code (lod3dgs.py), ext (outside this repo), out (a result)
FILL = {"map": "#fde7b4", "code": "#dbe9f6", "ext": "#e6e6e6", "out": "#d9efd3", "in": "#f3f3f3"}
EDGE = {"map": "#b8860b", "code": "#3a6ea5", "ext": "#777777", "out": "#4f8a3c", "in": "#999999"}

BOXES = {
    # name: (x, y, w, h, kind, title, [lines], test)
    "bag":    (30, 40, 210, 70, "in", "Mapping pass bag",
               ["RealSense colour, Ouster LiDAR, IMU"], ""),
    "slam":   (30, 150, 210, 80, "ext", "Stages 01-02",
               ["TSDF map, bundle-adjusted poses"], "test_tsdf, test_bundle_adjust"),
    "map":    (30, 290, 210, 120, "map", "REFERENCE MAP",
               ["map_final_<tag>.pcd", "traj_lidar_ba_glim.txt",
                "never edited: every copy below", "is made from it"], ""),
    "map2":   (30, 500, 210, 90, "in", "Second session's map",
               ["map_final_20260924.pcd", "its own anchor, 24 Sept"], ""),
    "export2": (30, 640, 210, 80, "in", "Second session's export",
               ["replica_map_mapping_20260924", "960x540, rectified"], ""),

    "view":   (300, 40, 240, 120, "code", "view",
               ["strays, mirror ghosts, loose pieces out", "flat cells onto their plane",
                "floor holes filled, ceiling cut", "view_<tag>.pcd"],
               "check_view, _ghosts, _body, _sky, _scans, _ceilings"),
    "mesh":   (600, 40, 240, 120, "code", "mesh",
               ["Poisson on the view copy", "faces > 0.1 m from the map trimmed",
                "vertex colour: the map's nearest point", "mesh_view_<tag>.ply (+ _noceil)"],
               "check_mesh, _ghosts, _hanging"),
    "sionna": (900, 40, 260, 120, "code", "sionna",
               ["faces -> floor, walls, ceiling,", "objects, glass; openings filled",
                "walls flat, holes closed, colours", "scene.xml + meshes/*.ply"],
               "check_sionna, check_hanging"),
    "rt":     (1220, 40, 250, 120, "out", "Sionna RT scene",
               ["sionna_full: coloured, 38 M faces", "sionna_light: 200k faces, no colours",
                "ITU materials per part"], "check_sionna: loaded and traced"),
    "fly":    (900, 200, 260, 90, "code", "fly / walk",
               ["a loop inside the mesh, 4 m blocks", "near the eye; H.264 video"],
               "check_sionna: walk_start, loop_path"),
    "video":  (1220, 200, 250, 90, "out", "Fly-through video",
               ["sionna_full_fly.mp4"], ""),

    "export": (300, 330, 240, 120, "code", "mcap_convert replica",
               ["--depth-source map --map", "colour frames + depth rendered",
                "from the map at the BA poses", "replica_map_mapping_<tag>"],
               "test_replica: depth_source map"),
    "scene":  (600, 330, 240, 140, "code", "scene",
               ["COLMAP layout, pinhole, centred", "depth as LOD-3DGS reads it",
                "points every 4 cm on the mesh,", "coloured from the frames",
                "6 nested LOD levels (lod/)"],
               "main: 12-frame room; two exports, two cameras"),
    "train":  (900, 330, 260, 110, "ext", "LOD-3DGS train",
               ["known poses, depth loss, --use_lod", "level_0..5.ply (5 finest)",
                "lod3dgs_patch.py: reads lod/"], ""),
    "splat":  (1220, 330, 250, 110, "code", "splat_walk.py",
               ["viewer, LOD level by distance", "splat fly video",
                "--render-views: rgb + depth per camera"], ""),
    "clean":  (900, 490, 260, 100, "code", "splats-clean",
               ["Gaussians > 0.15 m from the mesh,", "faint or oversized removed",
                "level_5_clean.ply"], "check_splats"),
    "recol":  (1220, 490, 250, 100, "code", "recolour --renders",
               ["each mesh vertex: the mean of its", "rendered pixels, depth-checked"],
               "check_splats"),
    "mesh2":  (1220, 640, 250, 80, "out", "Mesh in the splats' colours",
               ["mesh_view_<tag>_recoloured.ply"], ""),
    "align":  (300, 500, 240, 100, "code", "align",
               ["FPFH RANSAC + ICP, 5 cm voxels", "map_<B>_to_map_<A>.json (T)",
                "--transform moves the export's poses"],
               "main: yaw 25 deg, (3, -2, 0.1) m recovered"),
}

ARROWS = [
    # (from, to, side_from, side_to, label, dashed)
    ("bag", "slam", "b", "t", "", False),
    ("slam", "map", "b", "t", "", False),
    ("map", "view", "r", "l", "", False),
    ("view", "mesh", "r", "l", "", False),
    ("mesh", "sionna", "r", "l", "", False),
    ("sionna", "rt", "r", "l", "", False),
    ("mesh", "fly", "b", "l", "", False),
    ("fly", "video", "r", "l", "", False),
    ("map", "export", "r", "l", "depth", False),
    ("export", "scene", "r", "l", "", False),
    ("mesh", "scene", "b", "t", "--mesh (else meshes --map itself)", True, None, (720, 215)),
    ("scene", "train", "r", "l", "", False),
    ("train", "splat", "r", "l", "", False),
    ("train", "clean", "b", "t", "", False),
    ("mesh", "clean", "b", "l", "reference surface", True, [(720, 300), (870, 300), (870, 540)], (795, 300)),
    ("splat", "recol", "b", "t", "renders", False),
    ("recol", "mesh2", "b", "t", "", False),
    ("map2", "align", "r", "l", "src", False),
    ("map", "align", "b", "t", "dst", True, [(135, 472), (420, 472)], (135, 440)),
    ("align", "scene", "r", "l", "--transform", False, [(570, 550), (570, 400)]),
    ("export2", "scene", "r", "l", "second export", False, [(570, 680), (570, 400)]),
]


def side(b, s):
    x, y, w, h = b[:4]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[s]


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box_svg(name, b):
    x, y, w, h, kind, title, lines, test = b
    o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{FILL[kind]}" '
         f'stroke="{EDGE[kind]}" stroke-width="{2 if kind == "map" else 1.3}"/>']
    o.append(f'<text x="{x + w / 2}" y="{y + 19}" text-anchor="middle" font-family="{FONT}" '
             f'font-size="14" font-weight="bold" fill="#222">{esc(title)}</text>')
    for i, t in enumerate(lines):
        o.append(f'<text x="{x + w / 2}" y="{y + 37 + 15 * i}" text-anchor="middle" font-family="{FONT}" '
                 f'font-size="11.5" fill="#333">{esc(t)}</text>')
    if test:
        o.append(f'<text x="{x + w / 2}" y="{y + h + 13}" text-anchor="middle" font-family="{FONT}" '
                 f'font-size="10" font-style="italic" fill="#6a6a6a">test: {esc(test)}</text>')
    return "\n".join(o)


def route(a, b, sa, sb, via=None):
    """Orthogonal path from side sa of a to side sb of b (via: its corners)."""
    (x0, y0), (x1, y1) = side(a, sa), side(b, sb)
    if via:
        pts = [(x0, y0)] + list(via) + [(x1, y1)]
    elif sa in "lr" and sb in "lr":
        xm = (x0 + x1) / 2
        pts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]
    elif sa in "tb" and sb in "tb":
        ym = (y0 + y1) / 2
        pts = [(x0, y0), (x0, ym), (x1, ym), (x1, y1)]
    elif sa in "tb":                                    # down, then across
        pts = [(x0, y0), (x0, y1), (x1, y1)]
    else:                                               # across, then down
        pts = [(x0, y0), (x1, y0), (x1, y1)]
    return pts


def arrow_svg(fr, to, sa, sb, label, dashed, via=None, at=None):
    pts = route(BOXES[fr], BOXES[to], sa, sb, via)
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    o = [f'<path d="{d}" fill="none" stroke="#444" stroke-width="1.6"{dash} marker-end="url(#ah)"/>']
    if label:
        # on the longest segment
        segs = list(zip(pts, pts[1:]))
        best = segs[0] if via and abs(segs[0][1][0] - segs[0][0][0]) + abs(segs[0][1][1] - segs[0][0][1]) > 60 \
            else max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
        lx, ly = at or ((best[0][0] + best[1][0]) / 2, (best[0][1] + best[1][1]) / 2)
        tw = 6.2 * len(label) + 8
        o.append(f'<rect x="{lx - tw / 2}" y="{ly - 8}" width="{tw}" height="15" fill="white" opacity="0.9"/>')
        o.append(f'<text x="{lx}" y="{ly + 4}" text-anchor="middle" font-family="{FONT}" font-size="10.5" '
                 f'fill="#444">{esc(label)}</text>')
    return "\n".join(o)


def legend():
    items = [("map", "the reference map"), ("code", "datasets/lod3dgs.py (this repo)"),
             ("ext", "other stage / external code"), ("out", "result")]
    o = []
    x = 300
    for k, t in items:
        o.append(f'<rect x="{x}" y="{H - 40}" width="22" height="14" rx="3" fill="{FILL[k]}" stroke="{EDGE[k]}"/>')
        o.append(f'<text x="{x + 28}" y="{H - 29}" font-family="{FONT}" font-size="11.5" fill="#333">{esc(t)}</text>')
        x += 40 + 6.5 * len(t)
    o.append(f'<text x="{x + 10}" y="{H - 29}" font-family="{FONT}" font-size="11" font-style="italic" '
             f'fill="#6a6a6a">test: check_* are in tests/test_lod3dgs.py; dashed = a second input</text>')
    return "\n".join(o)


def main(out=None):
    out = out or os.path.join(os.path.dirname(__file__), "..", "docs", "lod3dgs_pipeline.svg")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
             '<path d="M0,0 L9,4.5 L0,9 z" fill="#444"/></marker></defs>',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W / 2}" y="24" text-anchor="middle" font-family="{FONT}" font-size="17" font-weight="bold" '
             f'fill="#222">From the reference map: radio scene (top) and Gaussian splats (bottom)</text>']
    parts += [arrow_svg(*a) for a in ARROWS]
    parts += [box_svg(n, b) for n, b in BOXES.items()]
    parts.append(legend())
    parts.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
