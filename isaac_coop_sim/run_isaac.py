#!/usr/bin/env python3
"""Run a scenario in Isaac Sim and record what the robots' sensors see as a
rosbag2 (MCAP) laid out like the real recordings, plus the ground-truth
boxes per frame.

    # in Isaac Sim's own python (4.5 or 5.x), with mcap, mcap-ros2-support, rosbags, pyyaml,
    # scipy and numpy installed into it (./python.sh -m pip install ...):
    ./python.sh <repo>/isaac_coop_sim/run_isaac.py <out>/scenario.yaml --out <out>/sim_bag
        [--headless] [--max-s 30] [--probe]
    # anywhere, without Isaac: the plan, the schedule and the labels only
    python3 isaac_coop_sim/run_isaac.py <out>/scenario.yaml --out <out>/sim_bag --dry-run

What it does, per scenario.yaml (scenario.py):
  scene      the building USD (export_usd.py) referenced under /World
  robots     an Xform per robot with a box body (so the other robot's lidar
             sees it), an RTX lidar (the Ouster config named) at the lidar
             mount, a pinhole camera per camera with the real intrinsics at
             its optical mount; moved kinematically along the trajectory
  actors     a referenced USD (asset) or a coloured box of the given size,
             moved along its path
  recording  every lidar period (10 Hz) the lidar's full scan as
             sensor_msgs/PointCloud2 (x y z intensity, the lidar frame), the
             robot's pose as /<robot>/global_pose (its lidar frame in map),
             /tf map -> <lidar frame>; every camera period (15 Hz) the image
             (rgb8) and its CameraInfo; /tf_static with each camera's mount,
             once. labels_dynamic.json beside the bag: the boxes (robots and
             actors) at every lidar stamp, in map, for the converter's
             output.labels_file. Stamps start at 1 800 000 000 s.
--probe prints the keys and shapes the lidar and camera annotators return
on the first frame and stops: the RTX lidar's output names differ between
Isaac Sim versions, and the runner reads the first of several it knows
(POINT_KEYS, INTENSITY_KEYS); add yours there if none matches.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scoop_pipeline"))

T0_NS = 1_800_000_000 * 10 ** 9
POINT_KEYS = ("point_cloud_data", "point_cloud", "data", "xyz")
INTENSITY_KEYS = ("intensities_data", "intensity", "intensities")
# optical (x right, y down, z forward) -> body (x forward, y left, z up): the columns
# are the optical axes in body coordinates
R_BODY_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


# ------------------------------------------------------------- the schedule --
def schedule(scen, max_s=None):
    """[(t_s, kind)] in order: 'lidar' at the lidar period, 'camera' at the camera's."""
    end = scen.duration if max_s is None else min(scen.duration, max_s)
    ev = [(float(t), "lidar") for t in scen.lidar_times() if t < end]
    ev += [(float(t), "camera") for t in scen.camera_times() if t < end]
    ev.sort(key=lambda e: (e[0], e[1] != "lidar"))
    return ev


def plan(scen, log=print):
    log(f"scene {scen.scene.get('usd')} (floor z {scen.floor_z} m); {scen.duration:.0f} s at "
        f"{scen.sim_hz:.0f} Hz, lidar {scen.lidar_hz:g} Hz, cameras {scen.camera_hz:g} Hz")
    for rb in scen.robots:
        tr = scen.tracks[rb["name"]]
        log(f"  {rb['name']}: {tr.kind} trajectory, {tr.duration:.0f} s; lidar {rb['lidar']['config']} on "
            f"{rb['lidar']['frame']}; cameras " + ", ".join(
                f"{c['name']} {c['width']}x{c['height']} fx {c['fx']:.0f} on {c['frame']}" for c in rb["cameras"]))
    for ac in scen.actors:
        tr = scen.tracks[ac["name"]]
        log(f"  {ac['name']} ({ac.get('kind')}): box {ac.get('box')}, {tr.kind} path "
            f"{'loop ' if tr.loop else ''}{tr.duration:.0f} s, asset {ac.get('asset') or 'box'}")


# ------------------------------------------------------------ the messages --
def _types():
    from scoop import rosmsg
    return rosmsg, rosmsg.typestore().types


def camera_info_msg(rosmsg, T, stamp_ns, cam):
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float)
    roi = T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
    return T["sensor_msgs/msg/CameraInfo"](
        header=rosmsg.header(stamp_ns, cam["frame"]), height=cam["height"], width=cam["width"],
        distortion_model="plumb_bob", d=np.zeros(5), k=K.ravel(), r=np.eye(3).ravel(),
        p=np.hstack([K, np.zeros((3, 1))]).ravel(), binning_x=0, binning_y=0, roi=roi)


def image_msg(rosmsg, T, stamp_ns, frame, rgb):
    h, w = rgb.shape[:2]
    return T["sensor_msgs/msg/Image"](header=rosmsg.header(stamp_ns, frame), height=h, width=w,
                                      encoding="rgb8", is_bigendian=0, step=w * 3,
                                      data=np.ascontiguousarray(rgb[:, :, :3]).ravel())


def cloud_msg(rosmsg, stamp_ns, frame, xyz, intensity):
    pts = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    pts["intensity"] = intensity if intensity is not None else 0.5
    return rosmsg.pointcloud2(pts, stamp_ns, frame)


# ----------------------------------------------------------------- the run --
class Recorder:
    """Writes the bag and the labels; the sensors' data comes from `sim`
    (IsaacSim below, or None for a dry run that records nothing)."""

    def __init__(self, scen, out, log=print):
        self.scen, self.out, self.log = scen, os.path.expanduser(out), log
        self.rosmsg, self.T = _types()
        from scoop.bagwrite import BagWriter
        os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
        self.w = BagWriter(self.out, library="isaac_coop_sim")
        self.n = {"cloud": 0, "image": 0}

    def static_tf(self):
        tfs = []
        for rb in self.scen.robots:
            for c in rb["cameras"]:
                tfs.append((rb["lidar"]["frame"], c["frame"], c["mount"]["xyz"], c["mount"]["qxyzw"]))
        if tfs:
            self.w.set_qos("/tf_static", "- history: 3\n  depth: 1\n  reliability: 1\n  durability: 1\n")
            self.w.write("/tf_static", self.rosmsg.tf_message(tfs, T0_NS), T0_NS)

    def lidar(self, t, rb, T_map_lidar, xyz, intensity):
        from scenario import R_to_quat
        st = T0_NS + int(round(t * 1e9))
        self.w.write(rb["lidar"]["topic"], cloud_msg(self.rosmsg, st, rb["lidar"]["frame"], xyz, intensity), st)
        q = R_to_quat(T_map_lidar[:3, :3])
        self.w.write(f"/{rb['name']}/global_pose", self.rosmsg.pose_stamped(st, "map", T_map_lidar[:3, 3], q), st)
        self.w.write("/tf", self.rosmsg.tf_message([("map", rb["lidar"]["frame"], T_map_lidar[:3, 3], q)], st), st)
        self.n["cloud"] += 1

    def camera(self, t, cam, rgb):
        st = T0_NS + int(round(t * 1e9))
        self.w.write(cam["info_topic"], camera_info_msg(self.rosmsg, self.T, st, cam), st)
        self.w.write(cam["topic"], image_msg(self.rosmsg, self.T, st, cam["frame"], rgb), st)
        self.n["image"] += 1

    def close(self):
        self.w.close()
        lp = os.path.join(os.path.dirname(os.path.abspath(self.out)), "labels_dynamic.json")
        self.scen.write_labels(lp, T0_NS)
        self.log(f"wrote {self.out}: {self.n['cloud']} clouds, {self.n['image']} images; labels {lp}")
        return lp


class IsaacSim:
    """The stage, the robots' sensors, the actors; moved by poses from the scenario."""

    def __init__(self, scen, headless=True, log=print):
        self.scen, self.log = scen, log
        from isaacsim import SimulationApp                                   # 4.x and 5.x
        self.app = SimulationApp({"headless": headless, "renderer": "RayTracedLighting"})
        import omni.usd
        from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
        try:
            from isaacsim.core.api import World                              # 4.5+
        except ImportError:
            from omni.isaac.core import World                                # 4.2
        try:
            from isaacsim.sensors.rtx import LidarRtx                        # 4.5+
        except ImportError:
            from omni.isaac.sensor import LidarRtx                           # 4.2
        try:
            from isaacsim.sensors.camera import Camera                      # 4.5+
        except ImportError:
            from omni.isaac.sensor import Camera                             # 4.2
        self.Usd, self.UsdGeom, self.Gf, self.Sdf = Usd, UsdGeom, Gf, Sdf
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / scen.sim_hz, rendering_dt=1.0 / scen.sim_hz)
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        # the building
        bp = self.stage.DefinePrim("/World/building_ref", "Xform")
        bp.GetReferences().AddReference(os.path.abspath(os.path.expanduser(scen.scene["usd"])))
        light = UsdLux.DomeLight.Define(self.stage, "/World/light")
        light.CreateIntensityAttr(1000.0)
        # the robots: a body box, the lidar, the cameras
        self.xforms, self.lidars, self.cameras = {}, {}, {}
        for rb in scen.robots:
            base = f"/World/{rb['name']}"
            self.xforms[rb["name"]] = self._xform(base)
            L, W, H = rb.get("box", [0.6, 0.5, 1.0])
            self._box(f"{base}/body", [L, W, H], (0.0, 0.0, -float(rb.get("box_z", 0.0))), (0.3, 0.3, 0.35))
            m = rb["lidar"]["mount"]
            self.lidars[rb["name"]] = LidarRtx(prim_path=f"{base}/lidar", config_file_name=rb["lidar"]["config"],
                                               translation=np.array(m["xyz"], float),
                                               orientation=self._wxyz(m["qxyzw"]))
            for c in rb["cameras"]:
                Tbo = np.eye(4)
                Tbo[:3, :3] = R_BODY_OPTICAL
                T_mount = self._T(c["mount"]["xyz"], c["mount"]["qxyzw"]) @ np.linalg.inv(Tbo)   # the camera body
                cam = Camera(prim_path=f"{base}/{c['name']}", resolution=(c["width"], c["height"]),
                             translation=T_mount[:3, 3], orientation=self._wxyz(self._quat(T_mount[:3, :3])))
                aperture = 20.955
                cam.set_horizontal_aperture(aperture)
                cam.set_vertical_aperture(aperture * c["height"] / c["width"] * (c["fx"] / c["fy"]))
                cam.set_focal_length(c["fx"] * aperture / c["width"])
                cam.set_clipping_range(0.05, 200.0)
                self.cameras[(rb["name"], c["name"])] = cam
        # the actors
        for k, ac in enumerate(scen.actors):
            p = f"/World/{ac['name']}"
            self.xforms[ac["name"]] = self._xform(p)
            if ac.get("asset"):
                self.stage.DefinePrim(f"{p}/asset", "Xform").GetReferences().AddReference(ac["asset"])
            else:
                col = [(0.8, 0.2, 0.2), (0.2, 0.5, 0.9), (0.9, 0.7, 0.1), (0.3, 0.8, 0.3)][k % 4]
                self._box(f"{p}/body", ac.get("box", [0.5, 0.5, 1.75]), (0, 0, 0), col)
        self.world.reset()
        for ld in self.lidars.values():
            ld.initialize()
            for fn in ("add_point_cloud_data_to_frame", "add_intensities_data_to_frame"):
                if hasattr(ld, fn):
                    getattr(ld, fn)()
        for cam in self.cameras.values():
            cam.initialize()
        self.probe_done = False

    # ---- USD helpers
    def _xform(self, path):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.ClearXformOpOrder()
        self._ops = getattr(self, "_ops", {})
        self._ops[path] = (x.AddTranslateOp(), x.AddOrientOp())
        return x

    def _box(self, path, size, centre, colour):
        from pxr import Vt
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.ClearXformOpOrder()
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(self.Gf.Vec3f(*[float(s) for s in size]))
        cube.CreateDisplayColorAttr(Vt.Vec3fArray([self.Gf.Vec3f(*colour)]))
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube

    def set_pose(self, name, T):
        tr, orient = self._ops[f"/World/{name}"]
        tr.Set(self.Gf.Vec3d(*[float(v) for v in T[:3, 3]]))
        q = self._quat(T[:3, :3])
        orient.Set(self.Gf.Quatd(float(q[3]), self.Gf.Vec3d(float(q[0]), float(q[1]), float(q[2]))))

    @staticmethod
    def _T(xyz, q):
        from scenario import T_of
        return T_of(xyz, q)

    @staticmethod
    def _quat(R):
        from scenario import R_to_quat
        return R_to_quat(R)

    @staticmethod
    def _wxyz(qxyzw):
        return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], float)

    # ---- stepping and reading
    def step(self):
        self.world.step(render=True)

    def lidar_frame(self, name, probe=False):
        fr = self.lidars[name].get_current_frame()
        if probe:
            self.log(f"  lidar {name} frame keys: " + ", ".join(
                f"{k}: {getattr(v, 'shape', type(v).__name__)}" for k, v in fr.items()))
        pts = next((fr[k] for k in POINT_KEYS if k in fr and fr[k] is not None), None)
        inten = next((fr[k] for k in INTENSITY_KEYS if k in fr and fr[k] is not None), None)
        if pts is None:
            return None, None
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        inten = np.asarray(inten, dtype=np.float32).reshape(-1) if inten is not None else None
        if inten is not None and len(inten) != len(pts):
            inten = None
        ok = np.isfinite(pts).all(1) & (np.linalg.norm(pts, axis=1) > 0.05)
        return pts[ok], (inten[ok] if inten is not None else None)

    def camera_frame(self, key, probe=False):
        rgba = self.cameras[key].get_rgba()
        if probe:
            self.log(f"  camera {key}: rgba {getattr(rgba, 'shape', None)}")
        if rgba is None or np.size(rgba) == 0:
            return None
        return np.asarray(rgba)[:, :, :3].astype(np.uint8)

    def close(self):
        self.app.close()


def run(scen, out, headless=True, max_s=None, probe=False, dry=False, log=print):
    plan(scen, log)
    ev = schedule(scen, max_s)
    n_l = sum(1 for _, k in ev if k == "lidar")
    n_c = sum(1 for _, k in ev if k == "camera")
    log(f"schedule: {n_l} lidar frames per robot, {n_c} camera frames per camera, "
        f"{(ev[-1][0] if ev else 0):.1f} s")
    rec = Recorder(scen, out, log)
    rec.static_tf()
    sim = None if dry else IsaacSim(scen, headless, log)
    dt = 1.0 / scen.sim_hz
    t, i = 0.0, 0
    t_wall = time.time()
    try:
        for t_ev, kind in ev:
            while t < t_ev - 1e-9:               # advance the world to the event's time
                t += dt
                poses = scen.at(t)
                if sim:
                    for name, T in poses.items():
                        if T is not None:
                            sim.set_pose(name, T)
                    sim.step()
            poses = scen.at(t_ev)
            if kind == "lidar":
                for rb in scen.robots:
                    T = poses.get(rb["name"])
                    if T is None:
                        continue
                    T_lidar = T @ scen_T(rb["lidar"]["mount"])
                    if sim:
                        xyz, inten = sim.lidar_frame(rb["name"], probe and not sim.probe_done)
                        if xyz is None:
                            continue
                    else:
                        xyz, inten = np.zeros((0, 3), np.float32), None
                    rec.lidar(t_ev, rb, T_lidar, xyz, inten)
            else:
                for rb in scen.robots:
                    if poses.get(rb["name"]) is None:
                        continue
                    for c in rb["cameras"]:
                        if sim:
                            rgb = sim.camera_frame((rb["name"], c["name"]), probe and not sim.probe_done)
                            if rgb is None:
                                continue
                        else:
                            rgb = np.zeros((c["height"], c["width"], 3), np.uint8)
                        rec.camera(t_ev, c, rgb)
            if sim and probe and not sim.probe_done and kind == "lidar" and rec.n["cloud"]:
                sim.probe_done = True
                log("probe done")
                break
            i += 1
            if i % 100 == 0:
                log(f"  {t_ev:.1f} s of {ev[-1][0]:.1f} ({time.time() - t_wall:.0f} s wall)")
    finally:
        lp = rec.close()
        if sim:
            sim.close()
    return rec.out, lp


def scen_T(mount):
    from scenario import T_of
    return T_of(mount["xyz"], mount["qxyzw"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario")
    ap.add_argument("--out", required=True, help="the bag folder to write")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-s", type=float, default=None, help="stop after this many seconds of the run")
    ap.add_argument("--probe", action="store_true", help="print the sensors' output keys on the first frame, stop")
    ap.add_argument("--dry-run", action="store_true", help="no Isaac: the plan, an empty-sensor bag, the labels")
    a = ap.parse_args()
    from scenario import Scenario
    scen = Scenario.load(a.scenario)
    run(scen, a.out, a.headless, a.max_s, a.probe, a.dry_run)


if __name__ == "__main__":
    main()
