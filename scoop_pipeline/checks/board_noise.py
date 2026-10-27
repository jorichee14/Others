#!/usr/bin/env python3
"""The LiDAR's noise on a ChArUco board, by distance: a flat target of known
size, placed where you choose, found by the camera and measured by the LiDAR.

    python3 check.py board <bag> --board board_4 [--config pipeline_config.json]
        [--squares 9x7 --square-len 0.0245 --marker-len 0.0175 --dict DICT_7X7_50]
        [--robot mobile_1] [--image TOPIC --info TOPIC --lidar TOPIC] [--raw]
        [--static-tf configs/static_tf.yaml] [--T x,y,z,qx,qy,qz,qw]
        [--period 0.5] [--tol 0.05] [--span FROM TO]
        [--margin 0.05] [--slab 0.25] [--min-points 12] [--min-range 0.3]
        [--runs] [--station-gap 0.5] [--bin 0] [--min-sightings 3] [--out DIR]

`check.py noise` reads the noise floor off whatever flat patches a scan
happens to hold: walls and floors at whatever range and incidence the room
gives. This reads it off ONE target you place: its size is known, its surface
is flat and matte, nothing stands within the slab behind it, and the camera
says exactly where it is. So the noise comes out per distance, with the
incidence measured rather than assumed, and with a bias against an
independent estimate of the same plane.

Per sampled image the board is detected (the pipeline's own detector, so the
same gates: corners, reprojection, ambiguity), which gives T_cam_board. The
camera -> LiDAR transform from the bag's /tf_static (or --static-tf / --T)
puts the board's rectangle in the LiDAR's frame; the scan within --tol of the
image is read, and the points inside the rectangle (less a --margin all round,
so no edge or mixed returns) and within --slab of the board's plane are the
board's returns. A plane is fitted to those points alone (one 3-sigma
rejection), and per sighting:

  sigma         rms of the points about THEIR OWN fitted plane: the sensor's
                scatter, free of any calibration, measured across the board
  along beam    that over cos(incidence): the range noise itself, since a
                range error moves a point along its beam, not along the normal
  thickness     their 5-95 % spread (3.29 sigma for a normal), the number
                `check.py noise`, `map` and `walls` print
  bias          where that plane sits against the CAMERA's plane, along the
                beam: + means the LiDAR ranges long. This one is only as good
                as the camera -> LiDAR translation and the board's own PnP
                depth, so read it as "within a cm", not as a range offset
  tilt          between the two planes' normals, the same caveat
  spacing       the measured point spacing on the board: how many beams the
                target is worth at that range
  repeat        per station: the scan-to-scan spread of that plane's own
                distance from the sensor, with board and robot standing
                still. No camera in it, and the random part of sigma is
                already averaged over the whole board, so this is what is
                left: the floor under any movement read off a surface
  step          per station: how far that plane sits from the station before

Sightings are grouped into stations (the places the board stood): by default
single-linkage on range with a --station-gap m gap, or fixed --bin m bins when
the board was carried about rather than placed. Per station the medians, and
the scan-to-scan spread of the bias, which is the sensor's frame-to-frame
range stability with everything standing still. Over the stations: a line
fitted to sigma along the beam against range, the sensor's noise model.

FINE MOVEMENT, 0.5 to 1.5 m
  To ask how small a movement the cloud can carry, `repeat` is the number,
  not `sigma`. A plane fitted to N points averages the random part down by
  sqrt(N) -- 2000 points on a board at 1 m turn 6 mm of range noise into
  well under a millimetre -- so what limits a small movement is what does not
  average: the sensor's own drift between scans, the deskew when anything
  moves, and the near-field behaviour of the first metre, which is the
  LiDAR's worst regime (an OS1 is specified from 0.3 m, and its precision is
  poorest below about 2 m). `repeat` holds all of it.

  Give the movement a truth: stand the board at the distance you work at,
  record a minute, then shim it forward by a known few mm (feeler gauges, a
  slip gauge, a micrometer stage) and record again, a few steps. Read it with
  --runs --station-gap 0.003, which keeps stations millimetres apart apart,
  and compare the `step` column with the shim. The smallest step that comes
  back cleanly, against the `repeat` beside it, is the answer for that
  surface at that range. A board of about 0.5 x 0.4 m (5 x 4 squares of
  0.1 m) is the one to use there: a 22 cm board is under 10 px of marker for
  the robot's cameras at 1.5 m, and a metre board overfills the image at
  0.5 m. Mind --min-range as well: the default 0.3 m drops returns nearer
  than that.

A station whose board is worth few beams is marked "!": a plane fitted to a
handful of points reads low whatever the sensor does (and the fit eats the
noise of 3 of them). Keep 25 points or so, which at 20 m takes a board of
about a metre.

It does not need a recording of its own: run it on a mapping pass with
--config <pipeline config> --board <name> and it reads the noise off the
boards that pass already saw, at whatever distances the robot drove past
them. A 22 cm board is only worth a few beams beyond 3 m, so that reads the
near end; the distances below are for the curve.

HOW TO RECORD IT
  Stand the board on a stand, face it at the sensor, turned 10-20 deg off
  square (a frontal board is a PnP mirror ambiguity, and the detector is right
  to refuse it). Nothing within a metre behind it. Park the robot, hold still
  10 s, then move the board (not the robot) out to the next distance: 2, 4, 6,
  8, 12, 16, 20 m, as far as the camera still reads the markers. One pass is
  enough; a second one at 45 deg gives the incidence curve too. Record the
  LiDAR, the camera and its camera_info, with /tf_static in the bag.

Writes board_noise.{json,md,png} to --out (default: next to the bag).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

from scoop import bag as B                                            # noqa: E402
from pipeline_boards import (Board, iter_images, pick_intrinsics,     # noqa: E402
                             read_tf_and_info)

CHI = 3.29                        # 5-95 % spread of a normal, in sigma
THIN = 25                         # points on the board below which a fit is not worth much


# ------------------------------------------------------------------- inputs --
def load_board(a):
    """The board design, from a config's registry (--board) or the geometry
    given on the command line."""
    if a.config:
        cfg = json.load(open(a.config))
        reg = cfg.get("boards") or {}
        if a.board not in reg:
            raise SystemExit("board %r not in %s's \"boards\" registry (have: %s)"
                             % (a.board, a.config, ", ".join(sorted(reg)) or "none"))
        return Board(a.board, reg[a.board])
    if not (a.squares and a.square_len and a.marker_len):
        raise SystemExit("give the board: --config <pipeline config> --board <name>, or "
                         "--squares 9x7 --square-len 0.0245 --marker-len 0.0175 "
                         "[--dict DICT_7X7_50]")
    try:
        sx, sy = (int(v) for v in a.squares.lower().split("x"))
    except ValueError:
        raise SystemExit("--squares is COLUMNSxROWS, e.g. 9x7")
    return Board(a.board or "board", {
        "squares_x": sx, "squares_y": sy, "square_len": a.square_len,
        "marker_len": a.marker_len, "dictionary": a.dict, "id_offset": a.id_offset,
        "min_corners": a.min_corners, "max_reproj": a.max_reproj})


def pick_topics(reader, a):
    """(image, camera_info, lidar) topics: what was asked for, else the
    robot's own RGB camera and Ouster."""
    topics = reader.topics()
    lidar = a.lidar or B.detect_points_topic(reader, "", log=lambda *_: None)
    image, info = a.image, a.info
    if not image:
        cands = [t for t, i in sorted(topics.items())
                 if i.msgtype.endswith(("msg/Image", "msg/CompressedImage"))
                 and "depth" not in t.lower() and "infra" not in t.lower()
                 and (not a.robot or t.strip("/").split("/")[0] == a.robot)]
        rect = [t for t in cands if "rect" in t]
        if not cands:
            raise SystemExit("no colour image topic in the bag; give --image / --info")
        image = (rect or cands)[0]
    if not info:
        guess = os.path.dirname(image) + "/camera_info"
        if guess not in topics:
            raise SystemExit("no %s in the bag; give --info" % guess)
        info = guess
    for t in (image, info, lidar):
        if t not in topics:
            raise SystemExit("topic %s not in the bag" % t)
    return image, info, lidar


def frame_of(reader, topic):
    """The frame_id the topic's first message carries."""
    for _, _, payload, decode in reader.iter_raw([topic]):
        m = decode(payload)
        return str(m.header.frame_id)
    return ""


def yaml_extrinsic(path, parent, child):
    """T_parent_child from a static_tf.yaml, either way round, or None."""
    import yaml
    from pipeline_common import make_T
    doc = yaml.safe_load(open(path)) or {}
    for tr in doc.get("transforms", []) or []:
        p, c = tr.get("parent"), tr.get("child")
        T = make_T(tr["translation"], tr["rotation_xyzw"])
        if (p, c) == (parent, child):
            return T
        if (p, c) == (child, parent):
            return np.linalg.inv(T)
    return None


def extrinsic(a, tree, lidar_frame, cam_frame, stamp):
    """T_lidar_cam: the camera optical frame in the LiDAR's."""
    if a.T:
        from pipeline_common import make_T
        v = [float(x) for x in a.T.replace(",", " ").split()]
        if len(v) != 7:
            raise SystemExit("--T is x,y,z,qx,qy,qz,qw")
        return make_T(v[:3], v[3:]), "--T"
    if tree is not None and lidar_frame and cam_frame:
        T = tree.lookup(lidar_frame, cam_frame, stamp)
        if T is not None:
            return T, "the bag's TF (%s -> %s)" % (lidar_frame, cam_frame)
        T = tree.lookup(cam_frame, lidar_frame, stamp)
        if T is not None:
            return np.linalg.inv(T), "the bag's TF (%s -> %s, inverted)" % (cam_frame, lidar_frame)
    if a.static_tf:
        T = yaml_extrinsic(a.static_tf, lidar_frame, cam_frame)
        if T is not None:
            return T, a.static_tf
    raise SystemExit(
        "no camera -> LiDAR transform: %s and %s are not connected in the bag's TF%s.\n"
        "Give --static-tf configs/static_tf.yaml or --T x,y,z,qx,qy,qz,qw."
        % (cam_frame or "the camera frame", lidar_frame or "the LiDAR frame",
           " and not in " + a.static_tf if a.static_tf else ""))


# -------------------------------------------------------------- measurement --
def fit_plane(P):
    """(unit normal, centroid) of the best plane through P."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[2], c


def robust_plane(P, k=3.0, min_points=8):
    """Plane through P with one k-sigma rejection: (normal, centroid, keep)."""
    keep = np.ones(len(P), bool)
    for _ in range(2):
        n, c = fit_plane(P[keep])
        r = (P - c) @ n
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        s = max(s, 1e-4)
        new = np.abs(r - np.median(r[keep])) <= k * s
        if new.sum() < min_points or (new == keep).all():
            break
        keep = new
    n, c = fit_plane(P[keep])
    return n, c, keep


def spacing_m(P):
    """Median nearest-neighbour distance of the points within the board's
    plane: the beam spacing at that range."""
    if len(P) < 4:
        return float("nan")
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P[:, :2]).query(P[:, :2], k=2)
    return float(np.median(d[:, 1]))


def measure(xyz, T_lb, size, a):
    """One sighting: the scan's returns on the board, in board coordinates
    (x, y across the printed face, z into it, away from the camera).

    Returns the statistics, or a string saying why there are none."""
    w, h = size
    R, t = T_lb[:3, :3], T_lb[:3, 3]
    Q = (np.asarray(xyz, np.float64) - t) @ R                 # LiDAR points, board frame
    m = ((Q[:, 0] > a.margin) & (Q[:, 0] < w - a.margin) &
         (Q[:, 1] > a.margin) & (Q[:, 1] < h - a.margin) &
         (np.abs(Q[:, 2]) <= a.slab))
    if m.sum() < a.min_points:
        return "%d points on the board (need %d)" % (m.sum(), a.min_points)
    P = Q[m]
    # anything else inside the slab (a wall close behind, a hand on the stand)
    z0 = np.median(P[:, 2])
    s = max(1.4826 * np.median(np.abs(P[:, 2] - z0)), 0.01)
    P = P[np.abs(P[:, 2] - z0) <= 4 * s]
    if len(P) < a.min_points:
        return "%d points left after the slab (need %d)" % (len(P), a.min_points)

    nrm, c, keep = robust_plane(P, min_points=max(8, a.min_points // 2))
    sensor = -t @ R                                            # LiDAR origin, board frame
    beam = c - sensor
    rng = float(np.linalg.norm(beam))
    beam /= max(rng, 1e-9)
    if nrm @ (sensor - c) < 0:                                 # normal toward the sensor
        nrm = -nrm
    cos_inc = float(min(abs(beam @ nrm), 1.0))
    cos_cam = float(min(abs(beam[2]), 1.0))                    # the camera plane's normal is z
    res = (P[keep] - c) @ nrm
    plane = float(abs((c - sensor) @ nrm))        # sensor to the fitted plane, no camera in it
    sigma = float(np.std(res, ddof=1))
    p5, p95 = np.percentile(res, [5, 95])
    board_z = float(np.median(P[keep, 2]))
    return dict(
        range_m=rng, plane_m=plane, n=int(keep.sum()), rejected=int((~keep).sum()),
        inc_deg=float(np.degrees(np.arccos(cos_inc))),
        sigma_mm=sigma * 1e3, beam_mm=sigma * 1e3 / max(cos_inc, 1e-3),
        thick_mm=float(p95 - p5) * 1e3,
        bias_mm=board_z * 1e3 / max(cos_cam, 1e-3),
        tilt_deg=float(np.degrees(np.arccos(min(abs(nrm[2]), 1.0)))),
        spacing_m=spacing_m(P[keep]),
        u_m=float(c[0]), v_m=float(c[1]))


# ----------------------------------------------------------------- stations --
def group(rows, bin_m, gap, min_n, by_runs=False):
    """Sightings -> stations: time-ordered runs (--runs), fixed bins
    (--bin) or single-linkage on range. [(label, [row, ...])], the small
    ones apart."""
    d = np.array([r["range_m"] for r in rows])
    order = np.argsort(d)
    lab = np.zeros(len(rows), int)
    if by_runs:
        # the board stands, is moved, stands again: a run ends where the range
        # leaves `gap` of the run's own median. Millimetres apart still part.
        k, cur = 0, []
        for i in sorted(range(len(rows)), key=lambda i: rows[i]["t"]):
            if cur and abs(d[i] - np.median(d[cur])) > gap:
                k, cur = k + 1, []
            cur.append(i)
            lab[i] = k
    elif bin_m > 0:
        lab = np.floor(d / bin_m).astype(int)
    else:
        k = 0
        for j, i in enumerate(order):
            if j and d[i] - d[order[j - 1]] > gap:
                k += 1
            lab[i] = k
    out = []
    keys = sorted(set(lab.tolist())) if by_runs else \
        sorted(set(lab.tolist()), key=lambda k: np.median(d[lab == k]))
    for k in keys:
        sel = [r for r, kk in zip(rows, lab) if kk == k]
        out.append((len(out) + 1, sel))
    return [(i, s) for i, s in out if len(s) >= min_n], \
           [(i, s) for i, s in out if len(s) < min_n]


def station_stats(sel, before=None):
    """One station's medians. `repeat_mm` is the scan-to-scan spread of the
    fitted plane's own distance from the sensor, with the board and the robot
    standing still: the floor under any movement read off this surface, and
    the one number here with no camera in it. `step_mm` is how far that plane
    sits from the station before, which a shimmed board gives a truth for."""
    med = lambda key: float(np.median([r[key] for r in sel]))             # noqa: E731
    bias = np.array([r["bias_mm"] for r in sel])
    plane = np.array([r["plane_m"] for r in sel]) * 1e3
    st = dict(
        sightings=len(sel), range_m=med("range_m"), points=med("n"),
        spacing_cm=med("spacing_m") * 100, inc_deg=med("inc_deg"),
        sigma_mm=med("sigma_mm"), beam_mm=med("beam_mm"), thick_mm=med("thick_mm"),
        plane_mm=float(np.median(plane)),
        repeat_mm=float(np.std(plane, ddof=1)) if len(plane) > 1 else None,
        bias_mm=float(np.median(bias)),
        bias_sd_mm=float(np.std(bias, ddof=1)) if len(bias) > 1 else None,
        tilt_deg=med("tilt_deg"), thin=med("n") < THIN)
    st["step_mm"] = None if before is None else st["plane_mm"] - before["plane_mm"]
    return st


def fit_line(x, y, span=0.5):
    """y = a + b x by least squares, with r^2. None under three points, or
    when they stand too close in x to say anything about the slope (stations
    shimmed millimetres apart)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < span:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss = np.sum((y - y.mean()) ** 2)
    return dict(a_mm=float(a), b_mm_per_m=float(b),
                r2=float(1 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan"))


# ------------------------------------------------------------------- output --
def table(stations, dropped):
    rows = ["  station  range m  views  pts/scan  spacing cm  inc deg  sigma mm  along beam"
            "  thick mm  repeat mm  step mm   bias mm  tilt deg",
            "  " + "-" * 128]
    for st in stations:
        rows.append("  %5d %c %7.3f %6d %9.0f %11.1f %8.1f %9.1f %11.1f %9.1f %10s %8s  %+8.1f %8.2f"
                    % (st["station"], "!" if st["thin"] else " ", st["range_m"],
                       st["sightings"], st["points"], st["spacing_cm"], st["inc_deg"],
                       st["sigma_mm"], st["beam_mm"], st["thick_mm"],
                       "-" if st["repeat_mm"] is None else "%.2f" % st["repeat_mm"],
                       "-" if st["step_mm"] is None else "%+.1f" % st["step_mm"],
                       st["bias_mm"], st["tilt_deg"]))
    for i, sel in dropped:
        rows.append("  %5d   %7.3f %6d   too few sightings, left out"
                    % (i, float(np.median([r["range_m"] for r in sel])), len(sel)))
    return "\n".join(rows)


def markdown(rep):
    md = ["# LiDAR noise on a ChArUco board", "",
          "`%s`" % rep["bag"], "",
          "%s, %.2f x %.2f m; camera `%s` -> LiDAR `%s` (%s)."
          % (rep["board"]["name"], rep["board"]["width_m"], rep["board"]["height_m"],
             rep["topics"]["image"], rep["topics"]["lidar"], rep["extrinsic"]["source"]),
          "%d sightings in %d stations." % (rep["sightings"], len(rep["stations"])), "",
          "| station | range m | views | pts/scan | spacing cm | incidence deg | sigma mm "
          "| along beam mm | thickness mm | repeat mm | step mm | bias mm | tilt deg |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for st in rep["stations"]:
        md.append("| %d%s | %.3f | %d | %.0f | %.1f | %.1f | %.1f | %.1f | %.1f | %s | %s | %+.1f | %.2f |"
                  % (st["station"], " !" if st["thin"] else "", st["range_m"], st["sightings"],
                     st["points"], st["spacing_cm"], st["inc_deg"], st["sigma_mm"],
                     st["beam_mm"], st["thick_mm"],
                     "-" if st["repeat_mm"] is None else "%.2f" % st["repeat_mm"],
                     "-" if st["step_mm"] is None else "%+.1f" % st["step_mm"],
                     st["bias_mm"], st["tilt_deg"]))
    f = rep.get("fit")
    if f:
        md += ["", "Range noise: **sigma = %.1f + %.2f d mm** (d in m, r2 %.2f), "
                   "%.1f x that as 5-95 %% thickness."
               % (f["a_mm"], f["b_mm_per_m"], f["r2"], CHI)]
    md += ["", "Repeat: the scan-to-scan spread of the fitted plane's own distance from the "
               "sensor, standing still -- the floor under any movement read off this surface, "
               "and the one column with no camera in it. Step: how far that plane sits from the "
               "station before.",
           "", "sigma: the points' rms about their own plane. Along beam: that over "
               "cos(incidence), the range noise. Thickness: their 5-95 %% spread, the number "
               "`check.py noise` prints. Bias: against the camera's plane, along the beam "
               "(+ = the LiDAR ranges long), only as good as the camera -> LiDAR translation. "
               "! : under %d points, too thin to fit." % THIN]
    return "\n".join(md)


def draw(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, st = rep["rows"], rep["stations"]
    d = [r["range_m"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(d, [r["beam_mm"] for r in rows], ".", ms=3, color="0.7", label="sightings")
    ax[0].plot([s["range_m"] for s in st], [s["beam_mm"] for s in st], "o-", color="C0",
               label="stations")
    f = rep.get("fit")
    if f:
        x = np.linspace(min(d), max(d), 50)
        ax[0].plot(x, f["a_mm"] + f["b_mm_per_m"] * x, "--", color="C3",
                   label="%.1f + %.2f d mm" % (f["a_mm"], f["b_mm_per_m"]))
    ax[0].set_ylabel("range noise, mm (1 sigma along the beam)")
    ax[0].legend(fontsize=8)
    ax[1].plot([s["range_m"] for s in st], [s["repeat_mm"] or np.nan for s in st], "o-",
               color="C2", label="plane, scan to scan")
    ax[1].plot([s["range_m"] for s in st], [abs(s["bias_mm"]) for s in st], "s--", color="0.6",
               label="|bias| vs the camera")
    ax[1].set_ylabel("repeatability standing still, mm")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[2].plot([s["range_m"] for s in st], [s["points"] for s in st], "o-", color="C1")
    ax[2].axhline(THIN, color="C3", ls="--", lw=1)
    ax[2].set_ylabel("points on the board per scan")
    ax[2].set_yscale("log")
    for x in ax:
        x.set_xlabel("range, m")
        x.grid(alpha=0.3)
    fig.suptitle("%s on %s, %.2f x %.2f m" % (rep["topics"]["lidar"], rep["board"]["name"],
                                              rep["board"]["width_m"], rep["board"]["height_m"]))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- main --
def run(a):
    board = load_board(a)
    reader = B.open_bag(a.bag)
    image_topic, info_topic, lidar_topic = pick_topics(reader, a)
    rectified = a.rectified or ("rect" in image_topic and not a.raw)
    infos, tree, _ = read_tf_and_info(a.bag, [info_topic], verbose=False)
    K, D, from_bag = pick_intrinsics(infos, info_topic, rectified)
    cam_frame = a.cam_frame or frame_of(reader, info_topic)
    lidar_frame = a.lidar_frame or frame_of(reader, lidar_topic)

    print("%s\n%s" % (a.bag, board.describe()))
    print("  camera %s (%s, %s)" % (image_topic, cam_frame,
                                    "rectified" if rectified else "raw, distortion applied"))
    print("  LiDAR  %s (%s)" % (lidar_topic, lidar_frame))
    if not from_bag:
        print("  ! no CameraInfo on %s" % info_topic)

    # the board in every sampled image
    t0 = None
    dets = []
    for stamp, gray in iter_images(a.bag, image_topic, detect_period=a.period):
        t0 = stamp if t0 is None else t0
        if a.span and not (a.span[0] <= stamp - t0 <= a.span[1]):
            continue
        d = board.detect(gray, K, D)
        if d is not None:
            dets.append((stamp, d))
    print("\n%d images sampled every %.2g s: %d sightings (%s)"
          % (sum(board.reject.values()), a.period, len(dets), board.reject_str()))
    if not dets:
        raise SystemExit("the board was not read in any frame: check --board's geometry, "
                         "--image, and board_scan.py for what the camera saw")

    T_lc, src = extrinsic(a, tree, lidar_frame, cam_frame, dets[0][0])
    print("  camera -> LiDAR from %s: |t| = %.3f m" % (src, np.linalg.norm(T_lc[:3, 3])))

    # the scan beside each sighting
    times = np.array([s for s, _ in dets])
    rows, why = [], {}
    for scan in B.iter_scans(reader, lidar_topic, select=B.nearest_pose(times, a.tol),
                             min_range=a.min_range, max_range=a.max_range, min_points=100):
        stamp, det = dets[scan.key]
        r = measure(scan.xyz, T_lc @ det.T, board.size_m, a)
        if isinstance(r, str):
            why[r.split("(")[0].strip()] = why.get(r.split("(")[0].strip(), 0) + 1
            continue
        r.update(t=float(scan.stamp - times[0]), dt_ms=(scan.stamp - stamp) * 1e3,
                 reproj_px=det.reproj, corners=det.n)
        rows.append(r)
    if not rows:
        raise SystemExit("no scan held the board: %s\nCheck the camera -> LiDAR transform, "
                         "--tol (the scan beside each image), and --slab / --margin."
                         % ("; ".join("%s x%d" % (k, v) for k, v in why.items()) or "no scan "
                            "within --tol of a sighting"))
    print("  %d sightings with a scan within %.0f ms%s" %
          (len(rows), a.tol * 1e3,
           ", %s" % "; ".join("%s x%d" % (k, v) for k, v in why.items()) if why else ""))

    stations, dropped = group(rows, a.bin, a.station_gap, a.min_sightings, a.runs)
    if not stations:
        raise SystemExit("no station with %d sightings; lower --min-sightings or raise "
                         "--station-gap" % a.min_sightings)
    st = []
    for i, sel in stations:
        st.append(dict(station=i, **station_stats(sel, st[-1] if st else None)))
    fit = fit_line([s["range_m"] for s in st], [s["beam_mm"] for s in st])

    how = ("runs, the board still within %.3g m" % a.station_gap if a.runs else
           "%.2g m bins" % a.bin if a.bin > 0 else "linked within %.2g m" % a.station_gap)
    print("\n%d sightings, %d stations (%s)\n" % (len(rows), len(st), how))
    print(table(st, dropped))
    allbeam = np.median([r["beam_mm"] for r in rows])
    print("\n  all stations: %.1f mm along the beam, %.1f mm (%.1f cm) as 5-95 %% thickness"
          % (allbeam, np.median([r["thick_mm"] for r in rows]),
             np.median([r["thick_mm"] for r in rows]) / 10))
    if fit:
        print("  range noise:  sigma = %.1f + %.2f d mm   (d in m, r2 %.2f)"
              % (fit["a_mm"], fit["b_mm_per_m"], fit["r2"]))
    rep_mm = [s["repeat_mm"] for s in st if s["repeat_mm"] is not None]
    if rep_mm:
        print("  standing still: the fitted plane repeats to %.2f mm scan to scan (%.2f at "
              "the worst station); a movement of that surface is read above it, not above "
              "sigma" % (float(np.median(rep_mm)), max(rep_mm)))
    if any(s["thin"] for s in st):
        print("  ! stations under %d points on the board: the plane fit eats the noise there; "
              "a bigger board or a shorter range" % THIN)

    rep = dict(bag=os.path.abspath(a.bag),
               topics=dict(image=image_topic, info=info_topic, lidar=lidar_topic),
               board=dict(name=board.name, squares=[board.sx, board.sy],
                          square_len=board.square_len, width_m=board.size_m[0],
                          height_m=board.size_m[1], dictionary=board.dict_name),
               extrinsic=dict(source=src, xyz=T_lc[:3, 3].round(6).tolist(),
                              lidar_frame=lidar_frame, camera_frame=cam_frame),
               settings=dict(period=a.period, tol=a.tol, margin=a.margin, slab=a.slab,
                             min_points=a.min_points, bin=a.bin, station_gap=a.station_gap,
                             min_sightings=a.min_sightings),
               sightings=len(rows), stations=st, fit=fit, runs=bool(a.runs),
               median=dict(beam_mm=float(allbeam),
                           thick_mm=float(np.median([r["thick_mm"] for r in rows]))),
               rows=rows)

    out = a.out or os.path.dirname(os.path.abspath(a.bag))
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "board_noise")
    with open(base + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(base + ".md", "w") as fh:
        fh.write(markdown(rep) + "\n")
    wrote = [base + ".json", base + ".md"]
    try:
        draw(rep, base + ".png")
        wrote.append(base + ".png")
    except ImportError:
        pass
    print("\nwritten: " + ", ".join(wrote))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--config", help="a pipeline config: --board names an entry of its "
                                     "\"boards\" registry")
    ap.add_argument("--board", default="", help="the board's name in that registry, or a "
                                                "label for the geometry given below")
    ap.add_argument("--squares", help="COLUMNSxROWS of the printed board, e.g. 9x7")
    ap.add_argument("--square-len", type=float, help="m, the chessboard square")
    ap.add_argument("--marker-len", type=float, help="m, the marker inside it")
    ap.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary")
    ap.add_argument("--id-offset", type=int, default=0)
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--max-reproj", type=float, default=1.0, help="px RMS")
    ap.add_argument("--robot", help="namespace to take the camera from (mobile_1, ...)")
    ap.add_argument("--image", help="image topic (default: the robot's rectified colour)")
    ap.add_argument("--info", help="its camera_info (default: beside the image topic)")
    ap.add_argument("--lidar", help="PointCloud2 topic (default: the bag's LiDAR)")
    ap.add_argument("--rectified", action="store_true", help="ignore the distortion "
                                                             "coefficients (default when "
                                                             "the topic is a rect one)")
    ap.add_argument("--raw", action="store_true", help="apply them even on a *_rect topic")
    ap.add_argument("--cam-frame", help="override the camera_info's frame_id")
    ap.add_argument("--lidar-frame", help="override the cloud's frame_id")
    ap.add_argument("--static-tf", help="configs/static_tf.yaml, when the bag's TF does not "
                                        "join the two")
    ap.add_argument("--T", help="the camera in the LiDAR as x,y,z,qx,qy,qz,qw")
    ap.add_argument("--period", type=float, default=0.5, help="s between sampled images")
    ap.add_argument("--tol", type=float, default=0.05, help="s, image to scan")
    ap.add_argument("--span", nargs=2, type=float, metavar=("FROM", "TO"),
                    help="s after the first image")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="m of the board's edge dropped (edge and mixed returns)")
    ap.add_argument("--slab", type=float, default=0.25,
                    help="m about the board's plane a return may be")
    ap.add_argument("--min-points", type=int, default=12, help="points for a sighting to count")
    ap.add_argument("--min-range", type=float, default=0.3,
                    help="m: returns nearer than this are dropped (the robot's own body). "
                         "Lower it for a board under 0.5 m away")
    ap.add_argument("--max-range", type=float, default=100.0)
    ap.add_argument("--bin", type=float, default=0.0,
                    help="m: fixed range bins instead of linked stations")
    ap.add_argument("--runs", action="store_true",
                    help="stations are time-ordered runs with the board standing still "
                         "(within --station-gap), not ranges linked: the way to separate "
                         "stations a few mm apart")
    ap.add_argument("--station-gap", type=float, default=0.5,
                    help="m: how far a station's sightings may spread in range "
                         "(with --runs, how still the board must stand)")
    ap.add_argument("--min-sightings", type=int, default=3, help="per station")
    ap.add_argument("--out", help="folder for board_noise.{json,md,png} (default: by the bag)")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
