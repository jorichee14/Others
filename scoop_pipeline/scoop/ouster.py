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

__all__ = ["ClockMap", "read_metadata", "sensor_info", "fit_clock",
           "iter_ouster_scans", "sdk"]

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
        """retime_bag's sensor_to_host_ns, on an integer ns sensor stamp."""
        t = sensor_ns * 1e-9
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
def iter_ouster_scans(reader: BagReader, ns: str = "/ouster",
                      clock: Optional[ClockMap] = None,
                      select: Optional[Select] = None, frame: str = "lidar",
                      driver_min_range: float = 0.0,
                      driver_max_range: float = np.inf,
                      min_range: float = 0.0, max_range: float = np.inf,
                      with_time: bool = True, min_points: int = 1,
                      metadata=None,
                      stats: Optional[dict] = None) -> Iterator[Scan]:
    """Decode ``<ns>/lidar_packets`` into :class:`Scan` objects.

    ``clock`` maps sensor time to host capture time (:func:`fit_clock`); None
    keeps sensor time. ``select(stamp_ns)`` runs on the mapped frame stamp
    before the frame is projected to xyz. ``driver_*_range`` gate the RANGE
    field (the driver's own filter); ``min_range``/``max_range`` then gate the
    xyz norm exactly like :func:`scoop.bag.iter_scans`.

    Like the driver, a frame is emitted when the next frame's first packet
    arrives, so the bag's last frame is never produced."""
    if frame not in ("lidar", "sensor"):
        raise ValueError("frame must be 'lidar' or 'sensor'")
    S = sdk()
    clock = clock or ClockMap.identity()
    src = metadata or read_metadata(reader, ns)
    info = sensor_info(src)
    lut_info = info
    if frame == "lidar":
        # a second SensorInfo from the sensor's own JSON, so the intrinsics are
        # exactly the bag's; only the lidar->sensor transform is dropped
        lut_info = S.SensorInfo(src if isinstance(src, str) else info.to_json_string())
        lut_info.format.udp_profile_lidar = info.format.udp_profile_lidar
        lut_info.lidar_to_sensor_transform = np.eye(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lut = S.XYZLut(lut_info)
        batcher = S.Batcher(info)
    rng_field = S.ChanField.RANGE
    st = stats if stats is not None else {}
    for k in ("packets", "frames", "selected", "empty"):
        st.setdefault(k, 0)
    lo_mm = float(driver_min_range) * 1000.0
    hi_mm = float(driver_max_range) * 1000.0
    lo2 = float(min_range) ** 2
    hi2 = float(max_range) ** 2 if np.isfinite(max_range) else np.inf

    cur = _new_frame(info)
    for _, _, payload, _ in reader.iter_raw([f"{ns}/lidar_packets"]):
        st["packets"] += 1
        try:
            buf = _packet_buf(payload)
        except _Unsupported:
            continue
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

        rng = np.asarray(done.field(rng_field))
        xyz = S.destagger(info, np.ascontiguousarray(lut(rng)))
        r = S.destagger(info, np.ascontiguousarray(rng))
        keep = (r > 0) & (r >= lo_mm) & (r <= hi_mm)
        xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
        keep = keep.reshape(-1)
        d2 = np.einsum("ij,ij->i", xyz, xyz)
        keep &= (d2 > lo2) & (d2 < hi2)
        if int(keep.sum()) < max(1, int(min_points)):
            st["empty"] += 1
            continue
        t = None
        if with_time:
            col = np.where(ts > frame_ts, ts - np.uint64(frame_ts), np.uint64(0))
            tt = np.broadcast_to(col, rng.shape)
            t = S.destagger(info, np.ascontiguousarray(tt)).reshape(-1)[keep] * 1e-9
        yield Scan(stamp, key, xyz[keep], t)
