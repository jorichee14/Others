"""A synthetic coop run: the robots with their sensors, where they drive,
the actors that move, and the ground truth that follows from all of it.

A scenario is a yaml (make_scenario.py writes one from a real coop pass):

    duration_s: 120
    seed: 0
    rates: {lidar_hz: 10, camera_hz: 15, sim_hz: 60}
    scene: {usd: scene.usd, floor_z: -0.72}
    robots:
      - name: mobile_1
        base_frame: os_lidar_mobile1          # the trajectory's frame
        trajectory: {tum: traj_mobile_1_lidar.tum}          # replay a real run, or
        # trajectory: {waypoints: [[x, y, yaw_deg], ...], speed: 0.8}
        box: [0.6, 0.5, 1.2]                  # L W H of the robot as an object (m), centred
        box_z: 0.3                            # the box centre below the base frame (m)
        lidar: {config: OS1_REV6_64ch10hz1024res, frame: os_lidar_mobile1,
                topic: /mobile_1/ouster/points, mount: {xyz: [0,0,0], qxyzw: [0,0,0,1]}}
        cameras:
          - {name: zed, frame: zed_left_camera_optical_frame, topic: /mobile_1/zed/left/image_rect_color,
             info_topic: /mobile_1/zed/left/camera_info, width: 960, height: 540,
             fx: 544.98, fy: 544.98, cx: 489.4, cy: 266.66,
             mount: {xyz: [-0.0749, -0.0670, -0.0916], qxyzw: [-0.4978, -0.4980, 0.5018, 0.5023]}}
    actors:
      - name: person_1
        kind: walker                          # walker | robot | chair (what the labeller calls it)
        asset: null                           # a USD to reference (Isaac's people), or null: a box
        box: [0.5, 0.5, 1.75]
        path: {waypoints: [[x, y], ...], speed: 1.2, loop: true, start_s: 0}
        yaw_follows_path: true

Poses are 4 x 4 in the map frame, z up. `at(t)` gives every robot's base
pose and every actor's pose (centre of its box on the floor plus half its
height) at time t into the run; `labels_at(t)` the boxes the converter's
external labels take ({id, location, extent (half sizes), angle (roll,
yaw, pitch deg)}), the robots included so each is an object for the other.
"""
import json
import math
import os

import numpy as np

ROBOT_ID0, ACTOR_ID0 = 100, 1          # label ids (the converter reserves 10000+)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat().tolist()


def T_of(xyz, qxyzw=(0, 0, 0, 1)):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(qxyzw)
    T[:3, 3] = xyz
    return T


def yaw_T(x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def load_tum(path):
    d = np.loadtxt(os.path.expanduser(path))
    t = d[:, 0]
    T = np.array([T_of(r[1:4], r[4:8]) for r in d])
    return t, T


class Track:
    """A pose over time: a TUM trajectory replayed from its first stamp, or
    waypoints driven at a constant speed, yaw along the path."""

    def __init__(self, spec, z=0.0, loop=False, start_s=0.0, yaw_follows=True):
        self.loop = bool(spec.get("loop", loop))
        self.start = float(spec.get("start_s", start_s))
        if "tum" in spec:
            self.t, self.T = load_tum(spec["tum"])
            self.t = self.t - self.t[0]
            self.kind = "tum"
            return
        self.kind = "waypoints"
        W = np.array(spec["waypoints"], float)
        if W.shape[1] == 2:
            W = np.c_[W, np.zeros(len(W))]
        self.W = W
        self.z = float(spec.get("z", z))
        self.speed = float(spec.get("speed", 1.0))
        self.yaw_follows = yaw_follows
        seg = np.linalg.norm(np.diff(W[:, :2], axis=0), axis=1)
        self.cum = np.r_[0, np.cumsum(seg)]
        self.length = float(self.cum[-1])
        self.t = None
        self.T = None

    @property
    def duration(self):
        if self.kind == "tum":
            return float(self.t[-1])
        return self.length / max(self.speed, 1e-9)

    def pose(self, t):
        """4 x 4 at time t into the run (None before start, or after the end without loop)."""
        tt = t - self.start
        if tt < 0:
            return None
        if self.kind == "tum":
            if tt > self.t[-1]:
                if not self.loop:
                    return None
                tt = tt % self.t[-1]
            i = int(np.searchsorted(self.t, tt, side="right")) - 1
            i = min(max(i, 0), len(self.t) - 2)
            a = (tt - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)
            from scipy.spatial.transform import Rotation, Slerp
            T = np.eye(4)
            T[:3, 3] = (1 - a) * self.T[i, :3, 3] + a * self.T[i + 1, :3, 3]
            sl = Slerp([0, 1], Rotation.from_matrix(self.T[i:i + 2, :3, :3]))
            T[:3, :3] = sl([a]).as_matrix()[0]
            return T
        if self.length <= 0:
            x, y, yaw = self.W[0, 0], self.W[0, 1], math.radians(self.W[0, 2])
            return yaw_T(x, y, self.z, yaw)
        s = tt * self.speed
        if s > self.length:
            if not self.loop:
                return None
            s = s % self.length
        i = int(np.searchsorted(self.cum, s, side="right")) - 1
        i = min(max(i, 0), len(self.cum) - 2)
        a = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        p = (1 - a) * self.W[i, :2] + a * self.W[i + 1, :2]
        d = self.W[i + 1, :2] - self.W[i, :2]
        yaw = math.atan2(d[1], d[0]) if (self.yaw_follows and np.linalg.norm(d) > 1e-9) \
            else math.radians((1 - a) * self.W[i, 2] + a * self.W[i + 1, 2])
        return yaw_T(p[0], p[1], self.z, yaw)


class Scenario:
    def __init__(self, spec, base_dir="."):
        self.spec = spec
        self.base_dir = base_dir
        self.duration = float(spec.get("duration_s", 60))
        self.seed = int(spec.get("seed", 0))
        r = spec.get("rates") or {}
        self.lidar_hz = float(r.get("lidar_hz", 10))
        self.camera_hz = float(r.get("camera_hz", 15))
        self.sim_hz = float(r.get("sim_hz", 60))
        self.scene = dict(spec.get("scene") or {})
        self.floor_z = float(self.scene.get("floor_z", 0.0))
        self.robots = [dict(x) for x in spec.get("robots") or []]
        self.actors = [dict(x) for x in spec.get("actors") or []]
        self.tracks = {}
        for rb in self.robots:
            tr = dict(rb["trajectory"])
            if "tum" in tr:
                tr["tum"] = self._path(tr["tum"])
            self.tracks[rb["name"]] = Track(tr, z=self.floor_z + float(rb.get("base_height", 0.0)))
        for ac in self.actors:
            p = dict(ac.get("path") or {"waypoints": [ac.get("at", [0, 0])], "speed": 0})
            self.tracks[ac["name"]] = Track(p, z=self.floor_z, loop=p.get("loop", True),
                                            start_s=p.get("start_s", 0.0),
                                            yaw_follows=ac.get("yaw_follows_path", True))

    def _path(self, p):
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.join(self.base_dir, p)

    @classmethod
    def load(cls, path):
        import yaml
        path = os.path.expanduser(path)
        return cls(yaml.safe_load(open(path)), os.path.dirname(os.path.abspath(path)))

    def at(self, t):
        """{name: 4 x 4 pose in map or None}: the robots' base frames and the actors'
        box centres (on the floor plus half their height)."""
        out = {}
        for rb in self.robots:
            out[rb["name"]] = self.tracks[rb["name"]].pose(t)
        for ac in self.actors:
            T = self.tracks[ac["name"]].pose(t)
            if T is not None:
                T = T.copy()
                T[2, 3] = self.floor_z + float(ac.get("box", [1, 1, 1])[2]) / 2 + float(ac.get("lift", 0.0))
            out[ac["name"]] = T
        return out

    def labels_at(self, t, poses=None):
        """The boxes at t: [{id, name, kind, location, extent, angle}], the
        robots included (box centred box_z below the base frame)."""
        from scipy.spatial.transform import Rotation
        poses = poses if poses is not None else self.at(t)
        out = []
        for k, rb in enumerate(self.robots):
            T = poses.get(rb["name"])
            if T is None:
                continue
            L, W, H = rb.get("box", [0.6, 0.5, 1.0])
            c = T[:3, 3] + T[:3, :3] @ np.array([0, 0, -float(rb.get("box_z", 0.0))])
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            out.append({"id": ROBOT_ID0 + k, "name": rb["name"], "kind": "robot",
                        "location": c.round(4).tolist(), "extent": [L / 2, W / 2, H / 2],
                        "angle": [0.0, round(yaw, 3), 0.0]})
        for k, ac in enumerate(self.actors):
            T = poses.get(ac["name"])
            if T is None:
                continue
            L, W, H = ac.get("box", [0.5, 0.5, 1.75])
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            out.append({"id": ACTOR_ID0 + k, "name": ac["name"], "kind": ac.get("kind", "actor"),
                        "location": T[:3, 3].round(4).tolist(), "extent": [L / 2, W / 2, H / 2],
                        "angle": [0.0, round(yaw, 3), 0.0]})
        return out

    def lidar_times(self):
        return np.arange(0.0, self.duration, 1.0 / self.lidar_hz)

    def camera_times(self):
        return np.arange(0.0, self.duration, 1.0 / self.camera_hz)

    def write_labels(self, out, t0_ns=0):
        """labels_dynamic.json: {"stamps_ns": [...], "frames": {stamp_ns: [boxes]}}
        at every lidar time, the stamps on the bag's clock (t0_ns + t)."""
        frames = {}
        for t in self.lidar_times():
            frames[str(int(t0_ns + round(t * 1e9)))] = self.labels_at(float(t))
        json.dump({"stamps_ns": [int(k) for k in frames], "frames": frames}, open(out, "w"))
        return out
