#!/usr/bin/env python3
"""
project_to_camera_live.py — live ROS 2 version of project_to_camera.py.

List your point-cloud topics; each one is projected into the live camera
image and republished as an overlay on its own output topic:

    <cloud topic>          ->   <overlay_ns>/<sanitised cloud topic>
    /radar1/radar/points_all -> /overlay/radar1_radar_points_all
    /ouster/points           -> /overlay/ouster_points

Radar-style (big ringed dots, optional range labels) vs lidar-style
(dense 1-px dots) rendering is chosen per topic from its name
('radar' / 'lidar' / 'ouster' / 'velodyne' / 'livox'), falling back to
cloud density (>1500 pts = lidar), overridable with <key>.type.

EXTRINSICS — resolved per topic, in priority order:
  1. parameter  <key>.extrinsic := "x y z qx qy qz qw"
     mapping sensor points INTO the camera optical frame (p_cam = R p + t).
     If your value is the inverse (e.g. direct_visual_lidar_calibration's
     T_lidar_camera), set <key>.invert_extrinsic := true.
  2. built-in deployed defaults for the known radar1 / radar2 / ouster topics
     (same numbers as project_to_camera.py, range scales included).
  3. TF: camera optical frame <- cloud.header.frame_id, i.e. whatever the
     calibration tool / static_transform_publisher is broadcasting.

<key> is the cloud topic with slashes turned into underscores
(leading slash dropped): /radar1/radar/points_all -> radar1_radar_points_all.

Per-topic parameters (all optional):
    <key>.type              'auto' | 'radar' | 'lidar'        [auto]
    <key>.extrinsic         'x y z qx qy qz qw' ('' = default/TF)
    <key>.invert_extrinsic  bool                              [false]
    <key>.scale             range scale about the sensor origin, applied
                            BEFORE the extrinsic                [1.0]
    <key>.axes              signed axis remap of the raw cloud, e.g. 'x,-y,z'
                            (fixes a mirrored driver convention) ['x,y,z']

Usage
-----
    python3 project_to_camera_live.py --ros-args \
      -p image_topic:=/zed/zed_node/left/image_rect_color \
      -p cam_info_topic:=/zed/zed_node/left/camera_info \
      -p "cloud_topics:=['/radar1/radar/points_all','/radar2/radar/points_all','/ouster/points']"

    # view:
    ros2 run rqt_image_view rqt_image_view /overlay/ouster_points

    # explicit extrinsic + mirrored radar example:
    ... -p "radar2_radar_points_all.extrinsic:=-0.1194 -0.0096 -0.0157 0.7572 0.0539 0.6506 -0.0217" \
        -p radar2_radar_points_all.axes:=x,-y,z

Requires: ROS 2 (rclpy, tf2_ros), numpy, opencv-python, scipy.
"""

import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python")
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import tf2_ros


_PF_DTYPE = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
             5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}

# deployed extrinsics, same as project_to_camera.py:
#   radars: parent zed_left_camera_optical_frame -> child radarN_link (direct)
#   lidar:  T_lidar_camera from direct_visual_lidar_calibration (inverted here)
BUILTIN = {
    "radar1": dict(T=[0.2368, 0.0190, -0.0542, -0.4995, 0.6007, -0.4224, -0.4596],
                   invert=False, scale=0.958),
    "radar2": dict(T=[-0.1194, -0.0096, -0.0157, 0.7572, 0.0539, 0.6506, -0.0217],
                   invert=False, scale=0.967),
    "ouster": dict(T=[-0.07492821535373663, -0.06697097901204006,
                      -0.09162651926397122, -0.4978291081882739,
                      -0.4980354235251849, 0.501788779666838, 0.502329490031218],
                   invert=True, scale=1.0),
}

LIDAR_HINTS = ("lidar", "ouster", "velodyne", "livox", "hesai")
RADAR_HINTS = ("radar",)


# ── geometry / rendering helpers (from project_to_camera.py) ──────────────

def rigid(v):
    T = np.eye(4)
    q = np.asarray(v[3:], float)
    T[:3, :3] = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
    T[:3, 3] = v[:3]
    return T


def axes_matrix(spec):
    """'x,-y,z' -> signed permutation applied to the raw cloud. A left-right
    mirror cannot come from the extrinsic (det must be +1); it is a driver
    axis-convention difference and is fixed here, before scale+extrinsic."""
    A = np.zeros((3, 3))
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"axes needs 3 comma-separated terms, got '{spec}'")
    for row, p in enumerate(parts):
        sgn = -1.0 if p.startswith("-") else 1.0
        ax = p.lstrip("+-")
        if ax not in "xyz":
            raise ValueError(f"bad axis '{p}' in '{spec}'")
        A[row, "xyz".index(ax)] = sgn
    return A


def cloud_xyz(msg):
    f = {x.name: x for x in msg.fields}
    names, formats, offsets = [], [], []
    for n in ("x", "y", "z"):
        if n not in f:
            return np.zeros((0, 3))
        dt = np.dtype(_PF_DTYPE[f[n].datatype])
        if msg.is_bigendian:
            dt = dt.newbyteorder(">")
        names.append(n); formats.append(dt); offsets.append(f[n].offset)
    dt = np.dtype(dict(names=names, formats=formats, offsets=offsets,
                       itemsize=msg.point_step))
    a = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    p = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    return p[np.abs(p).sum(axis=1) > 1e-6]


def image_to_bgr(msg):
    enc = msg.encoding.lower()
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1,
          "8uc1": 1, "8uc3": 3, "8uc4": 4}.get(enc)
    if ch is None:
        raise ValueError(f"unhandled image encoding '{msg.encoding}'")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = buf.reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = rows.reshape(msg.height, msg.width, ch)
    if enc in ("bgra8", "8uc4"):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ch == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def project(P, T_cam_sensor, K, w, h, z_min=0.3, z_max=None):
    if len(P) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    Pc = P @ T_cam_sensor[:3, :3].T + T_cam_sensor[:3, 3]
    z = Pc[:, 2]
    keep = z > z_min
    if z_max is not None:
        keep &= z < z_max
    Pc, z = Pc[keep], z[keep]
    if len(Pc) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.stack([u[inb], v[inb]], axis=1), z[inb]


def depth_colors(z, lo, hi, mode="linear", boost=1.0):
    z = np.asarray(z, float)
    if z.size == 0:
        return np.zeros((0, 3), np.uint8)
    z = np.maximum(z, 1e-6)
    lo = max(lo, 1e-3)
    hi = max(hi, lo * 1.01)
    if mode == "log":
        t = (np.log(np.clip(z, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    elif mode == "inverse":
        a, b = 1.0 / hi, 1.0 / lo
        t = 1.0 - np.clip((1.0 / z - a) / max(b - a, 1e-9), 0, 1)
    else:
        t = np.clip((z - lo) / (hi - lo), 0, 1)
    idx = (np.clip(t, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    c = cv2.applyColorMap(idx, cv2.COLORMAP_TURBO).reshape(-1, 3)
    if boost > 1.0:
        hsv = cv2.cvtColor(c.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[..., 1] = np.clip(hsv[..., 1] * boost, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * min(boost, 1.25), 0, 255)
        c = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return c


def fit_depth_limits(z, qlo=2.0, qhi=98.0):
    if len(z) < 20:
        return None
    lo, hi = float(np.percentile(z, qlo)), float(np.percentile(z, qhi))
    return max(lo, 0.2), max(hi, lo * 1.5)


def ramp_value(frac, lo, hi, mode):
    frac = np.asarray(frac, float)
    if mode == "log":
        return lo * (hi / lo) ** frac
    if mode == "inverse":
        return 1.0 / (1.0 / lo + frac * (1.0 / hi - 1.0 / lo))
    return lo + frac * (hi - lo)


def colorbar(img, lo, hi, mode, x, y, w=200, h=10, label="", nticks=5):
    ramp = np.linspace(0, 1, w)
    zz = ramp_value(ramp, lo, hi, mode)
    strip = depth_colors(zz, lo, hi, mode).reshape(1, w, 3)
    img[y:y + h, x:x + w] = np.repeat(strip, h, axis=0)
    cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)
    for frac in np.linspace(0, 1, nticks):
        val = float(ramp_value(frac, lo, hi, mode))
        tx = int(x + frac * w) - (2 if frac == 0 else (16 if frac == 1 else 9))
        cv2.line(img, (int(x + frac * w), y + h), (int(x + frac * w), y + h + 3),
                 (255, 255, 255), 1)
        txt = f"{val:.1f}" if hi - lo < 6 else f"{val:.0f}"
        cv2.putText(img, txt, (tx, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (tx, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_points(img, uv, cols, radius, ring=False):
    if len(uv) == 0:
        return
    u = uv[:, 0].astype(np.int32)
    v = uv[:, 1].astype(np.int32)
    if radius <= 0:
        img[v, u] = cols
        return
    for uu, vv, c in zip(u, v, cols):
        pt = (int(uu), int(vv))
        if ring:
            cv2.circle(img, pt, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, pt, radius, tuple(int(x) for x in c), -1, cv2.LINE_AA)


def banner(img, lines, org=(8, 18), color=(255, 255, 255)):
    for i, s in enumerate(lines):
        y = org[1] + i * 16
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)


def stamp_to_sec(st):
    return st.sec + st.nanosec * 1e-9


def tf_to_matrix(tr):
    """geometry_msgs/TransformStamped -> 4x4. lookup_transform(cam, sensor)
    returns the transform that maps sensor-frame points into cam."""
    q = tr.transform.rotation
    t = tr.transform.translation
    return rigid([t.x, t.y, t.z, q.x, q.y, q.z, q.w])


def sanitise(topic):
    return topic.strip("/").replace("/", "_")


# ── the node ──────────────────────────────────────────────────────────────

class LiveProjector(Node):
    def __init__(self):
        super().__init__("project_to_camera_live")
        d = self.declare_parameter
        self.image_topic = d("image_topic",
                             "/zed/zed_node/left/image_rect_color").value
        self.info_topic = d("cam_info_topic",
                            "/zed/zed_node/left/camera_info").value
        cloud_topics = d("cloud_topics",
                         ["/radar1/radar/points_all",
                          "/radar2/radar/points_all",
                          "/ouster/points"]).value
        self.overlay_ns = d("overlay_ns", "/overlay").value.rstrip("/")
        self.camera_frame = d("camera_frame", "").value  # '' = camera_info's
        self.max_dt = d("max_dt", 0.25).value        # cloud staleness [s]
        self.range_min = d("range_min", 0.5).value
        self.range_max = d("range_max", 20.0).value
        self.dim = d("dim", 0.55).value
        self.depth_scale = d("depth_scale", "linear").value
        self.color_min = d("color_min", 0.0).value   # 0 = fit to data (EMA)
        self.color_max = d("color_max", 0.0).value
        self.saturate = d("saturate", 1.4).value
        self.point_size = d("point_size", 4).value   # radar dot radius
        self.lidar_size = d("lidar_size", 1).value   # lidar dot radius (0 = px)
        self.label_range = d("label_range", False).value
        self.publish_always = d("publish_always", False).value

        self.K = None
        self.ci_wh = None
        self.ci_frame = None
        self.lim = {}          # per-sensor EMA'd colour limits
        self.warned = set()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # one entry per listed topic; everything else hangs off it
        self.sensors = []
        for topic in cloud_topics:
            key = sanitise(topic)
            hint = next((BUILTIN[k] for k in BUILTIN if k in topic), None)
            ext = d(f"{key}.extrinsic", "").value
            s = dict(
                topic=topic, key=key,
                type=d(f"{key}.type", "auto").value,
                invert=d(f"{key}.invert_extrinsic",
                         hint["invert"] if hint and not ext else False).value,
                scale=d(f"{key}.scale",
                        hint["scale"] if hint else 1.0).value,
                axes=axes_matrix(d(f"{key}.axes", "x,y,z").value),
                T=None, last=None,
            )
            if ext:
                v = [float(x) for x in ext.replace(",", " ").split()]
                if len(v) != 7:
                    raise ValueError(f"{key}.extrinsic needs 7 numbers, got {len(v)}")
                s["T"] = np.linalg.inv(rigid(v)) if s["invert"] else rigid(v)
                src = "parameter"
            elif hint:
                s["T"] = (np.linalg.inv(rigid(hint["T"])) if hint["invert"]
                          else rigid(hint["T"]))
                src = "built-in default"
            else:
                src = "TF (pending lookup)"
            det = float(np.linalg.det(s["axes"]))
            if abs(det + 1) < 1e-6:
                self.get_logger().warning(
                    f"{key}: axes remap is a REFLECTION (det=-1) — intentional "
                    f"only if the driver's convention is mirrored")
            s["pub"] = self.create_publisher(Image,
                                             f"{self.overlay_ns}/{key}", 2)
            self.create_subscription(
                PointCloud2, topic,
                lambda m, ss=s: self._on_cloud(ss, m),
                qos_profile_sensor_data)
            self.sensors.append(s)
            self.get_logger().info(
                f"{topic}  ->  {self.overlay_ns}/{key}   "
                f"(extrinsic: {src}, scale {s['scale']})")

        self.create_subscription(CameraInfo, self.info_topic,
                                 self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic,
                                 self._on_image, qos_profile_sensor_data)
        self.get_logger().info(
            f"listening: image {self.image_topic}, info {self.info_topic}; "
            f"{len(self.sensors)} cloud topic(s)")

    # ── callbacks ────────────────────────────────────────────────────────

    def _on_info(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.ci_wh = (msg.width, msg.height)
        self.ci_frame = msg.header.frame_id

    def _on_cloud(self, s, msg):
        s["last"] = (stamp_to_sec(msg.header.stamp), msg)
        if s["type"] == "auto":
            n = msg.width * msg.height
            name = s["topic"].lower()
            if any(h in name for h in RADAR_HINTS):
                s["type"] = "radar"
            elif any(h in name for h in LIDAR_HINTS) or n > 1500:
                s["type"] = "lidar"
            elif n > 0:
                s["type"] = "radar"
            if s["type"] != "auto":
                self.get_logger().info(f"{s['key']}: rendering as {s['type']}")

    def _on_image(self, msg):
        if self.K is None:
            self._warn_once("noinfo", f"no camera_info yet on {self.info_topic}")
            return
        active = [s for s in self.sensors
                  if self.publish_always
                  or s["pub"].get_subscription_count() > 0]
        if not active:
            return
        try:
            bgr = image_to_bgr(msg)
        except ValueError as e:
            self._warn_once("enc", str(e))
            return
        t_img = stamp_to_sec(msg.header.stamp)
        h, w = bgr.shape[:2]
        Ks = self.K.copy()
        if self.ci_wh and (w, h) != tuple(self.ci_wh):
            Ks[0, :] *= w / self.ci_wh[0]
            Ks[1, :] *= h / self.ci_wh[1]
        base = ((bgr.astype(np.float32) * self.dim).astype(np.uint8)
                if self.dim < 1.0 else bgr)

        for s in active:
            self._render(s, msg, base, Ks, t_img, w, h)

    # ── per-sensor render ────────────────────────────────────────────────

    def _resolve_tf(self, s, frame_id):
        cam = self.camera_frame or self.ci_frame
        if not cam or not frame_id:
            return None
        try:
            tr = self.tf_buffer.lookup_transform(cam, frame_id, Time())
        except Exception:
            self._warn_once(
                f"tf.{s['key']}",
                f"{s['key']}: no extrinsic param/default and no TF "
                f"{cam} <- {frame_id} yet; will keep trying")
            return None
        self.get_logger().info(f"{s['key']}: extrinsic from TF {cam} <- {frame_id}")
        return tf_to_matrix(tr)

    def _render(self, s, img_msg, base, Ks, t_img, w, h):
        if s["last"] is None:
            return
        t_cloud, cloud_msg = s["last"]
        age = t_img - t_cloud
        if abs(age) > self.max_dt:
            self._warn_once(
                f"stale.{s['key']}",
                f"{s['key']}: cloud is {age*1e3:.0f} ms from the image "
                f"(max_dt {self.max_dt*1e3:.0f} ms) — not overlaying. "
                f"Raise max_dt for slow sensors.")
            return
        self.warned.discard(f"stale.{s['key']}")
        if s["T"] is None:
            s["T"] = self._resolve_tf(s, cloud_msg.header.frame_id)
            if s["T"] is None:
                return
        pts = (cloud_xyz(cloud_msg) @ s["axes"].T) * s["scale"]
        uv, z = project(pts, s["T"], Ks, w, h,
                        z_min=self.range_min, z_max=self.range_max)

        if self.color_min > 0 and self.color_max > 0:
            lo, hi = self.color_min, self.color_max
        else:
            fit = fit_depth_limits(z)
            old = self.lim.get(s["key"])
            if fit and old:                       # EMA so colours don't flicker
                self.lim[s["key"]] = (0.9 * old[0] + 0.1 * fit[0],
                                      0.9 * old[1] + 0.1 * fit[1])
            elif fit:
                self.lim[s["key"]] = fit
            lo, hi = self.lim.get(s["key"], (1.0, 10.0))

        im = base.copy()
        cols = depth_colors(z, lo, hi, self.depth_scale, self.saturate)
        radar = s["type"] != "lidar"
        draw_points(im, uv, cols,
                    self.point_size if radar else self.lidar_size, ring=radar)
        if radar and self.label_range:
            for (uu, vv), zz, cc in zip(uv, z, cols):
                org = (int(uu) + 7, int(vv) - 5)
                cv2.putText(im, f"{zz:.1f}", org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(im, f"{zz:.1f}", org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, tuple(int(x) for x in cc), 1, cv2.LINE_AA)
        colorbar(im, lo, hi, self.depth_scale, w - 220, h - 36,
                 label="range [m]" if radar else "depth [m]")
        banner(im, [f"{s['topic']}  ({s['type']})",
                    f"{len(uv)}/{len(pts)} pts in view   scale={s['scale']}",
                    f"cloud-image dt {age*1e3:+.0f} ms"])

        out = Image()
        out.header = img_msg.header      # keep the camera stamp for sync
        out.height, out.width = im.shape[:2]
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = im.shape[1] * 3
        out.data = im.tobytes()
        s["pub"].publish(out)

    def _warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warning(text)


def main():
    rclpy.init()
    node = LiveProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
