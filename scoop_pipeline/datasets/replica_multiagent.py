#!/usr/bin/env python3
"""A multi-robot pass as a ReplicaMultiagent scene for MAGiC-SLAM
(github.com/VladimirYugay/MAGiC-SLAM; the CP-SLAM layout).

    python scoop_pipeline/datasets/replica_multiagent.py <finalized merged bag> [<out scene dir>]
        [--agents mobile_1 mobile_2] [--rate 10] [--width 640] [--pose-bag BAG]
        [--depth-max 10] [--min-valid 0.05] [--max-frames 0]

Layout (MAGiC-SLAM's Agent takes sorted(<scene>/*)[agent_id], so the scene
folder holds the agent folders and nothing else):

    <scene>/agent_0_mobile_1/results/frame000000.jpg   colour
                             results/depth000000.png   uint16 depth, metres = px / 6553.5
                             traj.txt                  camera-to-world 4x4 per frame, row-major,
                                                       OpenCV axes, in the shared map frame
                             timestamps.txt            the colour image's stamp per frame (s)
    <scene>/agent_1_mobile_2/...
    <scene>.yaml           MAGiC-SLAM config: inherits configs/ReplicaMultiagent/
                           replica_multiagent.yaml, sets input_path, agent_ids and cam
    <scene>_report.txt

Default <scene>: data/processed/<date>/<pass>/datasets/magic_slam/<pass>_<date>.

What it takes from the bag, per agent (AGENTS): the colour image and its
CameraInfo, the depth image and its CameraInfo, and the camera pose
(/<robot>/global_pose: the colour camera's optical frame in map, written by
10_poses and carried by the finalized bag; or --pose-bag, 10's own bag).

Two things MAGiC-SLAM assumes that the robots do not give:
  one camera   the config has a single cam block for every agent, and the
               ZED and the RealSense differ. Every agent's colour is
               resampled (undistorted with its CameraInfo) into one common
               pinhole camera: --width pixels wide, centred, with the field
               of view all the cameras cover (the narrowest extent of each
               side). Depth already in the colour camera (the ZED's
               depth_registered) is resampled the same way, nearest pixel;
               depth from another camera (the RealSense's depth imager) is
               back-projected with its own intrinsics, moved into the colour
               camera through /tf_static, projected into the common camera
               (nearest point per pixel) and its one-pixel gaps filled with
               the nearest neighbour.
  lockstep     agents run frame index against frame index. Frames are taken
               on one --rate Hz clock over the time every agent has poses:
               at each tick each agent's colour/depth pair nearest it; a
               tick is kept only when every agent has a pair with a pose and
               --min-valid depth, so frame i is the same moment for all.
"""
import argparse
import logging
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcap_convert as mc                                           # noqa: E402

log = logging.getLogger("replica_multiagent")

AGENTS = {
    "mobile_1": {"color": "/mobile_1/zed/left/image_rect_color",
                 "info": "/mobile_1/zed/left/camera_info",
                 "depth": "/mobile_1/zed/depth/depth_registered",
                 "depth_info": "/mobile_1/zed/depth/camera_info",
                 "pose": "/mobile_1/global_pose"},
    "mobile_2": {"color": "/mobile_2/color/image_raw",
                 "info": "/mobile_2/color/camera_info",
                 "depth": "/mobile_2/depth/image_rect_raw",
                 "depth_info": "/mobile_2/depth/camera_info",
                 "pose": "/mobile_2/global_pose"},
}
DEPTH_SCALE = 6553.5                 # Replica: uint16 0..65535 = 0..10 m
MAGIC_BASE = "configs/ReplicaMultiagent/replica_multiagent.yaml"


def k_d(info):
    """(K 3x3, distortion) of a CameraInfo."""
    return np.asarray(info.k, float).reshape(3, 3), np.asarray(info.d, float)


def common_camera(infos, width):
    """The centred pinhole camera every camera covers: per side the narrowest
    extent (tan of the half angle) over the cameras, --width pixels wide,
    square pixels."""
    tl = tr = tt = tb = np.inf
    for ci in infos:
        K, _ = k_d(ci)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        tl, tr = min(tl, cx / fx), min(tr, (ci.width - 1 - cx) / fx)
        tt, tb = min(tt, cy / fy), min(tb, (ci.height - 1 - cy) / fy)
    tx, ty = min(tl, tr), min(tt, tb)
    W = int(width)
    cx = (W - 1) / 2.0
    f = cx / tx
    H = int(2 * np.floor(f * ty))
    return mc.Intrinsics(f, f, cx, (H - 1) / 2.0, W, H, [0.0] * 4, None)


def colour_map(info, tgt):
    """cv2.remap maps from the camera's image (undistorted) into tgt."""
    K, D = k_d(info)
    Kt = np.array([[tgt.fx, 0, tgt.cx], [0, tgt.fy, tgt.cy], [0, 0, 1.0]])
    return cv2.initUndistortRectifyMap(K, D if np.any(D) else None, None, Kt,
                                       (int(tgt.width), int(tgt.height)), cv2.CV_32FC1)


def depth_points(d, K):
    """Depth image (metres, 0 = none) -> (N, 3) points in its optical frame."""
    v, u = np.nonzero(d > 0)
    z = d[v, u].astype(np.float64)
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def fill_gaps(D):
    """Empty pixels with a neighbour (3 x 3) get the nearest neighbour's depth."""
    m = cv2.erode(np.where(D > 0, D, np.float32(np.inf)), np.ones((3, 3), np.uint8))
    return np.where((D == 0) & np.isfinite(m), m, D).astype(np.float32)


def register_depth(d, dinfo, T_cd, tgt, dmin, dmax, dmap=None):
    """Depth (metres) of the depth camera -> depth in tgt (the colour camera's
    optical frame, common intrinsics). dmap: remap maps when the depth is in
    the colour camera already."""
    if dmap is not None:
        return cv2.remap(d, dmap[0], dmap[1], cv2.INTER_NEAREST, borderValue=0)
    p = depth_points(d, k_d(dinfo)[0])
    p = p @ T_cd[:3, :3].T + T_cd[:3, 3]
    return fill_gaps(mc.project_depth(p, tgt, dmin, dmax, occlusion_px=0))


class Agent:
    def __init__(self, i, machine, spec, paths, pose_paths, edges):
        self.i, self.machine, self.spec = i, machine, spec
        self.name = "agent_%d_%s" % (i, machine)
        self.info = mc.first_msg(paths, spec["info"])
        self.dinfo = mc.first_msg(paths, spec["depth_info"]) or self.info
        if self.info is None:
            raise mc.ConvertError(f"{machine}: no messages on {spec['info']}")
        self.poses = mc.PoseInterp.from_topic(pose_paths, spec["pose"])
        cm, dm = mc.first_msg(paths, spec["color"]), mc.first_msg(paths, spec["depth"])
        if cm is None or dm is None:
            raise mc.ConvertError(f"{machine}: no messages on {spec['color']} or {spec['depth']}")
        cf, df = cm.header.frame_id.lstrip("/"), dm.header.frame_id.lstrip("/")
        self.T_cd = np.eye(4) if cf == df else mc.lookup_static(edges, cf, df)
        if self.T_cd is None:
            raise mc.ConvertError(f"{machine}: depth frame {df} and colour frame {cf} are not "
                                  "connected in /tf_static")
        self.frames = (cf, df)
        if (dm.width, dm.height) != (self.dinfo.width, self.dinfo.height):
            if self.dinfo is not self.info:
                raise mc.ConvertError(f"{machine}: {spec['depth']} is {dm.width}x{dm.height}, "
                                      f"its CameraInfo {self.dinfo.width}x{self.dinfo.height}")
            # no depth CameraInfo: the colour camera's, scaled to the depth image
            K, D = k_d(self.info)
            sx, sy = dm.width / self.info.width, dm.height / self.info.height
            K = K * np.array([[sx], [sy], [1.0]])
            from types import SimpleNamespace
            self.dinfo = SimpleNamespace(k=K.reshape(-1), d=D, width=dm.width, height=dm.height)
            log.info(f"{machine}: no {spec['depth_info']}: colour intrinsics scaled to the "
                     f"{dm.width}x{dm.height} depth")


def agent_frames(paths, a, tgt, t0, period, n, out, sync_tol_ns, dmin, dmax, min_valid):
    """Write agent a's frame at every tick it has (frame<k>.jpg / depth<k>.png,
    k the tick); {k: (stamp_ns, T_map_cam, valid share)}, and drop counts."""
    res = os.path.join(out, a.name, "results")
    os.makedirs(res, exist_ok=True)
    mx, my = colour_map(a.info, tgt)
    dmap = colour_map(a.dinfo, tgt) if np.allclose(a.T_cd, np.eye(4)) else None
    pend, got = {}, {}
    drop = {"no pose": 0, "little depth": 0}

    def finish(k):
        _, ct, cm, dm = pend.pop(k)
        T = a.poses.at(ct)
        if T is None:
            drop["no pose"] += 1
            return
        d, _ = mc.depth_to_metres(mc.decode_image(dm), dm.encoding, 0.001, dmin, dmax)
        D = register_depth(d, a.dinfo, a.T_cd, tgt, dmin, dmax, dmap)
        valid = float((D > 0).mean())
        if valid < min_valid:
            drop["little depth"] += 1
            return
        bgr = mc.to_bgr(mc.decode_image(cm), cm.encoding)
        cv2.imwrite(os.path.join(res, "frame%06d.jpg" % k),
                    cv2.remap(bgr, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(res, "depth%06d.png" % k),
                    np.clip(np.round(D * DEPTH_SCALE), 0, 65535).astype(np.uint16))
        got[k] = (ct, T, valid)

    for ct, cm, _, dm in mc.iter_synced(paths, a.spec["color"], a.spec["depth"], sync_tol_ns):
        k = int(round((ct - t0) / period))
        if 0 <= k < n:
            off = abs(ct - (t0 + k * period))
            if k not in pend or off < pend[k][0]:
                pend[k] = (off, ct, cm, dm)
        for j in [j for j in pend if t0 + (j + 1) * period < ct]:
            finish(j)
    for j in sorted(pend):
        finish(j)
    return got, drop


def default_out(bag):
    dp = mc.pass_of(bag)
    if dp is None:
        raise mc.ConvertError(f"{bag} is not below data/{{raw,work,processed}}/<date>/<pass>/: "
                              "give the output folder")
    date, pas = dp
    p = os.path.abspath(os.path.expanduser(bag))
    root = p[:p.rindex(os.sep + date + os.sep + pas)]
    root = root[:root.rindex(os.sep)]
    return os.path.join(root, "processed", date, pas, "datasets", "magic_slam", "%s_%s" % (pas, date))


def convert(bag, out=None, *, agents=("mobile_1", "mobile_2"), specs=None, pose_bag=None,
            rate=10.0, width=640, depth_min=0.2, depth_max=10.0, min_valid=0.05,
            sync_tol_ms=20.0, max_frames=0):
    specs = specs or AGENTS
    paths = mc._paths(bag)
    out = os.path.abspath(os.path.expanduser(out or default_out(bag)))
    if os.path.isdir(out) and os.listdir(out):
        raise mc.ConvertError(f"{out} is not empty: remove it first (MAGiC-SLAM reads every "
                              "entry in it as an agent)")
    if depth_max * DEPTH_SCALE > 65535.5:
        raise mc.ConvertError(f"depth_max {depth_max} m does not fit Replica's uint16 at {DEPTH_SCALE}/m")
    edges = mc.build_tf_static(paths)
    pose_paths = mc._paths(pose_bag) if pose_bag else paths
    ags = [Agent(i, m, specs[m], paths, pose_paths, edges) for i, m in enumerate(agents)]
    tgt = common_camera([a.info for a in ags], width)
    log.info("common camera %dx%d f %.2f (hfov %.1f deg, vfov %.1f deg)" % (
        tgt.width, tgt.height, tgt.fx, 2 * np.degrees(np.arctan(tgt.cx / tgt.fx)),
        2 * np.degrees(np.arctan(tgt.cy / tgt.fy))))
    t0 = max(int(a.poses.ts[0]) for a in ags)
    t1 = min(int(a.poses.ts[-1]) for a in ags)
    if t1 <= t0:
        raise mc.ConvertError("the agents' pose streams do not overlap in time")
    period = int(round(1e9 / rate))
    n = int((t1 - t0) // period) + 1
    log.info("%d ticks at %g Hz over %.1f s (every agent has poses)" % (n, rate, (t1 - t0) * 1e-9))
    os.makedirs(out, exist_ok=True)
    per, drops = [], []
    for a in ags:
        log.info("%s: %s + %s, depth in %s%s" % (a.machine, a.spec["color"], a.spec["depth"],
                                                 a.frames[0], "" if a.frames[0] == a.frames[1]
                                                 else " (from %s)" % a.frames[1]))
        g, d = agent_frames(paths, a, tgt, t0, period, n, out, int(sync_tol_ms * 1e6),
                            depth_min, depth_max, min_valid)
        log.info("  %d ticks with a frame; dropped: %s" % (len(g), d))
        per.append(g)
        drops.append(d)
    keep = sorted(set.intersection(*[set(g) for g in per]))
    if max_frames:
        keep = keep[:max_frames]
    if not keep:
        raise mc.ConvertError("no tick where every agent has a frame with a pose and depth")
    for a, g in zip(ags, per):
        res = os.path.join(out, a.name, "results")
        for i, k in enumerate(keep):
            for kind, ext in (("frame", "jpg"), ("depth", "png")):
                os.replace(os.path.join(res, "%s%06d.%s" % (kind, k, ext)),
                           os.path.join(res, "%s%06d.%s.tmp" % (kind, i, ext)))
        names = os.listdir(res)
        for f in names:                                   # ticks not every agent has
            if not f.endswith(".tmp"):
                os.remove(os.path.join(res, f))
        for f in names:
            if f.endswith(".tmp"):
                os.replace(os.path.join(res, f), os.path.join(res, f[:-4]))
        with open(os.path.join(out, a.name, "traj.txt"), "w") as fh:
            fh.writelines(" ".join("%.9f" % v for v in g[k][1].reshape(-1)) + "\n" for k in keep)
        with open(os.path.join(out, a.name, "timestamps.txt"), "w") as fh:
            fh.writelines("%.9f\n" % (g[k][0] * 1e-9) for k in keep)

    # report
    st = np.array([[g[k][0] for g in per] for k in keep], float) * 1e-9
    skew = (st.max(1) - st.min(1)) * 1e3
    lines = ["scene               : %s" % out,
             "agents              : %s" % ", ".join(a.name for a in ags),
             "frames per agent    : %d of %d ticks at %g Hz (%.1f s)" % (len(keep), n, rate,
                                                                         (keep[-1] - keep[0]) / rate),
             "common camera       : %dx%d fx=fy=%.3f cx=%.1f cy=%.1f, depth_scale %g" % (
                 tgt.width, tgt.height, tgt.fx, tgt.cx, tgt.cy, DEPTH_SCALE),
             "agents' stamps apart: median %.1f ms, max %.1f ms" % (np.median(skew), skew.max())]
    for a, g, d in zip(ags, per, drops):
        xyz = np.array([g[k][1][:3, 3] for k in keep])
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.zeros(1)
        lines.append("%-20s: depth valid mean %.0f %% (min %.0f %%), path %.1f m, step mean %.1f cm "
                     "max %.1f cm; ticks dropped: %s" % (
                         a.name, 100 * np.mean([g[k][2] for k in keep]),
                         100 * np.min([g[k][2] for k in keep]), step.sum(), 100 * step.mean(),
                         100 * step.max(), d))
    report = "\n".join(lines) + "\n"
    open(out + "_report.txt", "w").write(report)
    name = os.path.basename(out)
    open(out + ".yaml", "w").write(
        "# MAGiC-SLAM config for %s: copy it into MAGiC-SLAM's configs/ (or run_slam.py it here)\n"
        "inherit_from: %s\n"
        "dataset_name: 'replica'\n"
        "multi_gpu: False  # True: one GPU per agent plus one for the server\n"
        "data:\n"
        "  frame_limit: -1\n"
        "  scene_name: %s\n"
        "  input_path: %s\n"
        "  output_path: output/scoop/%s/\n"
        "  agent_ids: [%s]\n"
        "cam:\n"
        "  H: %d\n  W: %d\n  fx: %.6f\n  fy: %.6f\n  cx: %.6f\n  cy: %.6f\n  depth_scale: %g\n"
        % (name, MAGIC_BASE, name, out, name, ", ".join(str(a.i) for a in ags),
           tgt.height, tgt.width, tgt.fx, tgt.fy, tgt.cx, tgt.cy, DEPTH_SCALE))
    log.info("\n" + report)
    return {"out": out, "frames": len(keep), "camera": tgt, "skew_ms": skew, "drops": drops}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="the finalized merged bag (has /<robot>/global_pose)")
    ap.add_argument("out", nargs="?", default=None, help="scene folder (default: next to the pass)")
    ap.add_argument("--agents", nargs="+", default=["mobile_1", "mobile_2"], choices=sorted(AGENTS))
    ap.add_argument("--pose-bag", default=None, help="poses from this bag (10's) instead")
    ap.add_argument("--rate", type=float, default=10.0, help="frames per second, every agent")
    ap.add_argument("--width", type=int, default=640, help="common camera width (px)")
    ap.add_argument("--depth-min", type=float, default=0.2)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--min-valid", type=float, default=0.05, help="least share of pixels with depth")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(a.bag, a.out, agents=a.agents, pose_bag=a.pose_bag, rate=a.rate, width=a.width,
                depth_min=a.depth_min, depth_max=a.depth_max, min_valid=a.min_valid,
                max_frames=a.max_frames)
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
