# -*- coding: utf-8 -*-
"""
Ouster ``lidar_packets`` -> point clouds, offline, without a ROS replay.

Replaces this chain, which runs in real time and needs a second bag:

    ros2 launch ouster_ros replay.launch.xml bag_file:=<packets bag> ...  (sensor time)
    ros2 bag record <ns>/points
    retime_bag.py <packets bag> <sensor-time bag> <out bag>

with one pass over the ORIGINAL packets bag:

1. :func:`fit_clock` -- ``retime_bag.fit_clock_map``, ported line for line:
   pair every lidar packet's first-column sensor timestamp with its rosbag
   receive time, take the per-window minimum of (host - sensor), fit a line
   through that lower envelope twice (dropping the upper 20% of residuals).
   Same packets, same windows, same fit, so the same ``host = f(sensor)``.

2. :func:`iter_ouster_scans` -- batch packets into frames with the Ouster SDK,
   the way ouster_ros does in its default sensor-time mode, then map the frame
   time through the clock fit:

   * frame stamp  = first NON-ZERO column timestamp (sensor clock) -> host ns
   * per-point t  = that point's column timestamp - frame stamp (as ``t``)
   * xyz          = the SDK lookup table on the staggered range image, then
                    destaggered, i.e. the cloud's row-major order
   * ``driver_min_range`` / ``driver_max_range`` drop points by the RANGE field,
     as the driver's min_range / max_range do (min_range:=1.30 in the replays)
   * ``frame``    = 'lidar' (os_lidar, ouster_ros's default point_cloud_frame)
                    or 'sensor' (os_sensor: 180 deg about z, +36 mm up)

   It yields the same :class:`scoop.bag.Scan` as :func:`scoop.bag.iter_scans`,
   so a stage does not care whether its bag holds packets or clouds.

Which of these conventions the replayed clouds actually followed is checked,
not assumed: ``scripts/ouster_check.py`` decodes a packets bag both ways and
compares scan by scan against a replayed (and retimed) points bag.

Requires ``ouster-sdk``. Written against 1.0 (LidarFrame / FrameBatcher);
the 0.x names (LidarScan / ScanBatcher, ouster.sdk.client) are mapped too.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import numpy as np

from .bag import BagError, BagReader, Scan, Select, _Cdr, _Unsupported

__all__ = ["ClockMap", "OusterFrame", "ImuSample", "RawMessage", "read_metadata",
           "sensor_info", "fit_clock", "iter_ouster", "iter_ouster_scans", "sdk",
           "ROS_POINT_DTYPE", "ros_points"]

WINDOW_S = 2.0                      # retime_bag.WINDOW_S


# --------------------------------------------------------------------------- #
# SDK compatibility
# --------------------------------------------------------------------------- #
_SDK = None


def sdk():
    """The Ouster SDK's names, whichever version is installed."""
    global _SDK
    if _SDK is not None:
        return _SDK
    import importlib
    mod = None
    for name in ("ouster.sdk.core", "ouster.sdk.client", "ouster.client"):
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "SensorInfo") and hasattr(m, "PacketFormat"):
            mod = m
            break
    if mod is None:
        raise BagError("ouster-sdk not found (pip install ouster-sdk)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # 1.0 deprecates the 0.x names
        _SDK = SimpleNamespace(
            name=mod.__name__, mod=mod,
            SensorInfo=mod.SensorInfo, PacketFormat=mod.PacketFormat,
            Frame=getattr(mod, "LidarFrame", None) or mod.LidarScan,
            Batcher=getattr(mod, "FrameBatcher", None) or mod.ScanBatcher,
            XYZLut=getattr(mod, "XYZLutFloat", None) or mod.XYZLut,
            destagger=mod.destagger, ChanField=mod.ChanField,
            ColHeader=mod.ColHeader, LidarPacket=mod.LidarPacket)
    return _SDK


def _packet_format(info):
    S = sdk()
    try:
        return S.PacketFormat(info)
    except TypeError:
        return S.PacketFormat.from_info(info)


def _new_frame(info):
    S = sdk()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return S.Frame(info)
        except TypeError:                                   # 0.x
            return S.Frame(info.format.columns_per_frame,
                           info.format.pixels_per_column,
                           info.format.udp_profile_lidar)


def _make_packet(buf, info):
    S = sdk()
    try:
        p = S.LidarPacket(len(buf))
        p.buf[:] = buf
        return p
    except TypeError:                                       # 0.x: (data, info)
        return S.LidarPacket(buf, info)


def _col_ts0(pf, buf) -> int:
    """First-column sensor timestamp of a packet (retime_bag's rule)."""
    S = sdk()
    try:
        return int(pf.packet_header(S.ColHeader.TIMESTAMP, buf)[0])
    except Exception:
        return int(pf.col_timestamp(0, buf))


# --------------------------------------------------------------------------- #
# CDR of the two ouster_ros message types in the packets bag
# --------------------------------------------------------------------------- #
def _packet_buf(payload) -> np.ndarray:
    """ouster_sensor_msgs/PacketMsg (``uint8[] buf``) -> read-only view."""
    c = _Cdr(payload)
    start, n = c.blob()
    return np.frombuffer(payload, np.uint8, n, start)


def _string(payload) -> str:
    """std_msgs/String -> str."""
    return _Cdr(payload).string()


def sensor_info(meta):
    """SensorInfo from the metadata JSON (or pass one through). Refuses an
    info whose lidar packet profile parsed as OFF: every packet would then be
    misread, silently."""
    S = sdk()
    info = meta if isinstance(meta, S.SensorInfo) else S.SensorInfo(meta)
    prof = str(getattr(info.format, "udp_profile_lidar", ""))
    if prof.endswith("OFF"):
        raise BagError(f"ouster-sdk {S.name} parsed the metadata with udp_profile_lidar "
                       f"OFF; this SDK version cannot read this sensor's metadata")
    return info


def read_metadata(reader: BagReader, ns: str = "/ouster") -> str:
    """The sensor metadata JSON the driver published on ``<ns>/metadata``."""
    topic = f"{ns}/metadata"
    for _, _, payload, _ in reader.iter_raw([topic]):
        s = _string(payload)
        json.loads(s)                                    # validate
        return s
    raise BagError(f"no {topic} in the bag; topics: {sorted(reader.topics())}")


# --------------------------------------------------------------------------- #
# clock
# --------------------------------------------------------------------------- #
@dataclass
class ClockMap:
    """``host = sensor + offset_s + slope * (sensor - s0)``, seconds."""
    offset_s: float
    slope: float
    s0: float
    n_packets: int = 0
    resid_ms: float = float("nan")

    @property
    def drift_ppm(self) -> float:
        return self.slope * 1e6

    def host_ns(self, sensor_ns: int) -> int:
        """retime_bag's sensor_to_host_ns. The sensor time goes through the
        same float as there -- header sec + nanosec * 1e-9 -- so a stamp maps
        to the same nanosecond whichever of the two computes it."""
        sec, nsec = divmod(int(sensor_ns), 1_000_000_000)
        t = sec + nsec * 1e-9
        return int(round((t + self.offset_s + self.slope * (t - self.s0)) * 1e9))

    @classmethod
    def identity(cls) -> "ClockMap":
        """Keep sensor time (what a replay without retiming produces)."""
        return cls(0.0, 0.0, 0.0)

    def save(self, path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path) -> "ClockMap":
        return cls(**json.loads(Path(path).read_text()))


def fit_clock(reader: BagReader, ns: str = "/ouster", window_s: float = WINDOW_S,
              metadata=None, log=print) -> ClockMap:
    """Port of ``retime_bag.fit_clock_map``: same inputs, same arithmetic.
    ``metadata``: JSON string or SensorInfo; default read from the bag."""
    info = sensor_info(metadata or read_metadata(reader, ns))
    pf = _packet_format(info)
    sensor_t, host_t = [], []
    for _, log_ns, payload, _ in reader.iter_raw([f"{ns}/lidar_packets"]):
        try:
            ts = _col_ts0(pf, _packet_buf(payload))
        except Exception:
            continue
        if ts == 0:
            continue
        sensor_t.append(ts * 1e-9)
        host_t.append(log_ns * 1e-9)
    s = np.asarray(sensor_t)
    h = np.asarray(host_t)
    if len(s) < 1000:
        raise BagError(f"only {len(s)} usable lidar packets; cannot fit")

    order = np.argsort(s)
    s, h = s[order], h[order]
    d = h - s
    s0 = s[0]
    bins = ((s - s0) // window_s).astype(int)
    xs, ys = [], []
    for b in np.unique(bins):
        sel = bins == b
        i = np.argmin(d[sel])
        xs.append(s[sel][i] - s0)
        ys.append(d[sel][i])
    xs, ys = np.asarray(xs), np.asarray(ys)
    coef = np.polyfit(xs, ys, 1)
    res = ys - np.polyval(coef, xs)
    keep = res < np.percentile(res, 80)
    if len(xs) < 4 or keep.sum() < 2:
        # With 3 windows the top-20% cut leaves 1 or 2 points depending on
        # float rounding, i.e. the fit is decided by noise. retime_bag has the
        # same weakness on such short bags; a mapping bag is minutes long.
        raise BagError(f"only {len(xs)} windows of {window_s:g} s "
                       f"({s[-1] - s[0]:.1f} s of packets); the clock fit needs "
                       f"at least 4 windows (> {int(np.ceil(3 * window_s))} s)")
    coef = np.polyfit(xs[keep], ys[keep], 1)
    resid = ys[keep] - np.polyval(coef, xs[keep])

    cm = ClockMap(offset_s=float(coef[1]), slope=float(coef[0]), s0=float(s0),
                  n_packets=int(len(s)), resid_ms=float(np.std(resid) * 1e3))
    log(f"    [clock] {cm.n_packets} packets, sensor {s[0]:.3f}..{s[-1]:.3f} "
        f"({s[-1] - s[0]:.1f} s), offset {cm.offset_s:+.6f} s, drift "
        f"{cm.drift_ppm:+.2f} ppm, envelope residual {cm.resid_ms:.3f} ms rms")
    return cm


# --------------------------------------------------------------------------- #
# scans
# --------------------------------------------------------------------------- #
STANDARD_G = 9.80665                 # ouster_ros: accel is reported in g


@dataclass
class OusterFrame:
    """One scan as the driver publishes it: organized H x W, destaggered."""
    stamp_ns: int                  # frame time, mapped through the clock
    sensor_ns: int                 # frame time on the sensor clock
    log_ns: int                    # receive time of the packet that closed it
    frame_id: int
    key: object                    # what select() returned
    xyz: np.ndarray                # (H,W,3) float32, NaN where not valid
    valid: np.ndarray              # (H,W) bool: a return inside the driver range
    t_ns: np.ndarray               # (H,W) uint32, point time - frame time
    fields: dict                   # (H,W) channel images: range, signal, ...


@dataclass
class ImuSample:
    stamp_ns: int                  # gyro timestamp, mapped through the clock
    sensor_ns: int
    log_ns: int
    accel: np.ndarray              # (3,) m/s^2
    gyro: np.ndarray               # (3,) rad/s


@dataclass
class RawMessage:
    """A message of a pass-through topic, untouched (for copying/remapping)."""
    topic: str
    log_ns: int
    payload: bytes


def _lut_info(S, src, info, frame):
    if frame == "sensor":
        return info
    # a second SensorInfo from the sensor's own JSON, so the intrinsics are
    # exactly the bag's; only the lidar->sensor transform is dropped
    li = S.SensorInfo(src if isinstance(src, str) else info.to_json_string())
    li.format.udp_profile_lidar = info.format.udp_profile_lidar
    li.lidar_to_sensor_transform = np.eye(4)
    return li


def iter_ouster(reader: BagReader, ns: str = "/ouster",
                clock: Optional[ClockMap] = None, select: Optional[Select] = None,
                frame: str = "lidar", driver_min_range: float = 0.0,
                driver_max_range: float = np.inf, imu: bool = False,
                fields: bool = False, metadata=None, passthrough=(),
                stats: Optional[dict] = None) -> Iterator[object]:
    """Everything the driver would publish, in recording order:
    :class:`OusterFrame` per lidar frame and, with ``imu=True``,
    :class:`ImuSample` per IMU packet.

    Frame rules (ouster_ros, sensor-time mode):
      * stamp = first NON-ZERO column timestamp, mapped by ``clock``
        (None keeps sensor time, as a replay without timestamp_mode does);
      * point time = its column timestamp - stamp (0 for dead columns);
      * xyz from the SDK lookup table on the staggered range, destaggered;
      * outside [driver_min_range, driver_max_range] (RANGE field) or no
        return -> not valid, xyz NaN;
      * a frame is emitted when the next frame's first packet arrives, so the
        bag's last frame is never produced.
    IMU rules: stamp = gyro timestamp mapped by ``clock``; accel in g * 9.80665,
    gyro in deg/s * pi/180; LEGACY IMU profile (one sample per packet).

    ``select(stamp_ns)`` runs before a frame is projected; ``fields=True`` also
    carries signal / reflectivity / near_ir images (for writing a bag).
    Topics in ``passthrough`` come out as :class:`RawMessage`, interleaved in
    recording order (a packet topic listed there is still decoded too)."""
    if frame not in ("lidar", "sensor"):
        raise ValueError("frame must be 'lidar' or 'sensor'")
    S = sdk()
    clock = clock or ClockMap.identity()
    src = metadata or read_metadata(reader, ns)
    info = sensor_info(src)
    pf = _packet_format(info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lut = S.XYZLut(_lut_info(S, src, info, frame))
        batcher = S.Batcher(info)
    CF = S.ChanField
    extra = [("signal", "SIGNAL"), ("reflectivity", "REFLECTIVITY"), ("near_ir", "NEAR_IR")]
    st = stats if stats is not None else {}
    for k in ("packets", "frames", "selected", "imu"):
        st.setdefault(k, 0)
    lo_mm = float(driver_min_range) * 1000.0
    hi_mm = float(driver_max_range) * 1000.0
    lidar_topic, imu_topic = f"{ns}/lidar_packets", f"{ns}/imu_packets"
    passthrough = set(passthrough)
    topics = sorted(set([lidar_topic] + ([imu_topic] if imu else [])) | passthrough)
    missing = [t for t in passthrough if t not in reader.topics()]
    if missing:
        raise BagError(f"pass-through topics not in the bag: {missing}")
    if imu and imu_topic not in reader.topics():
        raise BagError(f"{imu_topic} not in the bag")
    if imu and int(getattr(pf, "imu_measurements_per_packet", 0) or 0) > 1:
        raise BagError("IMU packets carry several measurements (non-LEGACY IMU "
                       "profile); only the LEGACY layout is decoded")

    cur = _new_frame(info)
    for topic, log_ns, payload, _ in reader.iter_raw(topics):
        if topic in passthrough:
            yield RawMessage(topic, log_ns, payload)
            if topic not in (lidar_topic, imu_topic) or (topic == imu_topic and not imu):
                continue
        try:
            buf = _packet_buf(payload)
        except _Unsupported:
            continue
        if topic == imu_topic:
            g_ns = int(pf.imu_gyro_ts(buf))
            st["imu"] += 1
            yield ImuSample(
                clock.host_ns(g_ns), g_ns, log_ns,
                np.array([pf.imu_la_x(buf), pf.imu_la_y(buf), pf.imu_la_z(buf)]) * STANDARD_G,
                np.radians([pf.imu_av_x(buf), pf.imu_av_y(buf), pf.imu_av_z(buf)]))
            continue

        st["packets"] += 1
        if not batcher(_make_packet(buf, info), cur):
            continue
        done, cur = cur, _new_frame(info)
        st["frames"] += 1
        ts = np.asarray(done.timestamp, np.uint64)
        nz = np.flatnonzero(ts)
        if nz.size == 0:
            continue
        frame_ts = int(ts[nz[0]])
        stamp = clock.host_ns(frame_ts)
        key = select(stamp) if select is not None else st["frames"] - 1
        if key is None:
            continue
        st["selected"] += 1

        rng = np.asarray(done.field(CF.RANGE))
        r = S.destagger(info, np.ascontiguousarray(rng))
        valid = (r > 0) & (r >= lo_mm) & (r <= hi_mm)
        xyz = np.asarray(S.destagger(info, np.ascontiguousarray(lut(rng))), np.float32)
        xyz[~valid] = np.nan
        col = np.where(ts > frame_ts, ts - np.uint64(frame_ts), np.uint64(0))
        t = S.destagger(info, np.ascontiguousarray(np.broadcast_to(col, rng.shape)))
        chans = {"range": r}
        if fields:
            have = set(str(f) for f in done.fields)
            for name, cf in extra:
                if cf in have:
                    chans[name] = S.destagger(info, np.ascontiguousarray(
                        np.asarray(done.field(getattr(CF, cf)))))
        yield OusterFrame(stamp, frame_ts, log_ns, int(done.frame_id), key, xyz,
                          valid, t.astype(np.uint32), chans)


def iter_ouster_scans(reader: BagReader, ns: str = "/ouster",
                      clock: Optional[ClockMap] = None,
                      select: Optional[Select] = None, frame: str = "lidar",
                      driver_min_range: float = 0.0,
                      driver_max_range: float = np.inf,
                      min_range: float = 0.0, max_range: float = np.inf,
                      with_time: bool = True, min_points: int = 1,
                      metadata=None,
                      stats: Optional[dict] = None) -> Iterator[Scan]:
    """:func:`iter_ouster` as :class:`scoop.bag.Scan` objects -- the valid
    points only, in the cloud's row-major order -- so a stage reads a packets
    bag exactly like a points bag. ``min_range``/``max_range`` gate the xyz
    norm like :func:`scoop.bag.iter_scans`."""
    st = stats if stats is not None else {}
    st.setdefault("empty", 0)
    lo2 = float(min_range) ** 2
    hi2 = float(max_range) ** 2 if np.isfinite(max_range) else np.inf
    for f in iter_ouster(reader, ns, clock=clock, select=select, frame=frame,
                         driver_min_range=driver_min_range,
                         driver_max_range=driver_max_range,
                         metadata=metadata, stats=st):
        xyz = f.xyz.reshape(-1, 3)
        keep = f.valid.reshape(-1).copy()
        d2 = np.einsum("ij,ij->i", np.nan_to_num(xyz), np.nan_to_num(xyz))
        keep &= (d2 > lo2) & (d2 < hi2)
        if int(keep.sum()) < max(1, int(min_points)):
            st["empty"] += 1
            continue
        t = f.t_ns.reshape(-1)[keep] * 1e-9 if with_time else None
        yield Scan(f.stamp_ns, f.key, xyz[keep], t)


# --------------------------------------------------------------------------- #
# ouster_ros message conventions (what its replay publishes)
# --------------------------------------------------------------------------- #
# point_type "original" (ouster_ros::Point): PCL point4d padding after z,
# 16-byte aligned to 48 bytes.
ROS_POINT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "ambient", "range"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "<u2", "<u2", "<u4"],
    "offsets": [0, 4, 8, 16, 20, 24, 26, 28, 32],
    "itemsize": 48})
ROS_FRAME_ID = {"lidar": "os_lidar", "sensor": "os_sensor"}
ROS_IMU_FRAME_ID = "os_imu"
ROS_IMU_ACCEL_COV = 0.01            # ouster_ros's fixed IMU covariances
ROS_IMU_GYRO_COV = 6e-4


def ros_points(f: OusterFrame) -> np.ndarray:
    """An :class:`OusterFrame` as ouster_ros's organized cloud: H*W records
    of :data:`ROS_POINT_DTYPE`, row-major, invalid points NaN. intensity is
    SIGNAL, ambient is NEAR_IR, ring is the beam (row) index."""
    H, W = f.valid.shape
    a = np.zeros(H * W, ROS_POINT_DTYPE)
    xyz = f.xyz.reshape(-1, 3)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["t"] = f.t_ns.reshape(-1)
    a["range"] = f.fields["range"].reshape(-1)
    for dst, src in (("intensity", "signal"), ("reflectivity", "reflectivity"),
                     ("ambient", "near_ir")):
        if src in f.fields:
            a[dst] = f.fields[src].reshape(-1)
    a["ring"] = np.repeat(np.arange(H, dtype=np.uint16), W)
    return a
