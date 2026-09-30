# -*- coding: utf-8 -*-
"""
Bag decoding for the SCooP pipeline. Every stage reads LiDAR scans and camera
frames through this module; no stage opens a bag itself.

Built on ``rosbag_to_opv2v``'s :class:`BagReader` (MCAP without a ROS install,
``.db3`` through rosbags) and adds what the mapping stages need:

1. **Select before decoding.** The header stamp is read straight out of the CDR
   payload and handed to a caller-supplied ``select(stamp_ns)``. Whatever it
   returns (a pose index, say) travels with the decoded message; ``None`` skips
   the message before a single byte of it is decoded. The stage scripts used to
   deserialise every scan and only then check the trajectory tolerance.

2. **Direct CDR parsing of PointCloud2 and Image.** Both are fixed-layout
   messages whose bulk is one ``uint8[]``. Reading the few header fields by hand
   and viewing the data with ``np.frombuffer`` skips the generic decoder and the
   ``bytes(msg.data)`` copies the scripts made per scan and per field. The field
   dtype is built once per topic, not once per message. Anything the fast path
   does not recognise (another CDR encapsulation, a truncated payload) goes to
   the generic decoder, so an unexpected message is slower, never wrong.

3. **Per-point time in seconds relative to the header stamp**, the unit detected
   once per topic from the sweep span. Ported from ``01_build_map.pc2_xyzt``;
   the state is per iterator now, not a module global.

Typical use::

    reader = open_bag(cfg["dataset"]["bag"])
    topic = detect_points_topic(reader, n_poses=len(traj_t))
    for scan in iter_scans(reader, topic, select=nearest_pose(traj_t, 0.05),
                           min_range=0.5, max_range=40.0, with_time=True):
        T = traj_T[scan.key]           # key = what select() returned
        ... scan.xyz (N,3) float32, scan.t (N,) float64 s or None
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np

try:
    from ros2opv2v.bagreader import BagError, BagReader, stamp_from_cdr
    from ros2opv2v.pointclouds import pointcloud2_to_array
except ImportError:                         # sibling checkout, no install
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rosbag_to_opv2v"))
    from ros2opv2v.bagreader import BagError, BagReader, stamp_from_cdr
    from ros2opv2v.pointclouds import pointcloud2_to_array

__all__ = ["BagError", "BagReader", "Scan", "Frame", "open_bag",
           "detect_points_topic", "nearest_pose", "iter_scans", "iter_images",
           "parse_pointcloud2", "parse_image"]

Select = Callable[[int], object]

# sensor_msgs/PointField datatype enum -> numpy
_PF_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
          5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
# per-point time field, in the order drivers commonly name it
_T_NAMES = ("t", "time", "timestamp", "time_offset", "point_time", "ts")


def open_bag(path) -> BagReader:
    """A reader on a rosbag2 directory, split set, or single .mcap/.db3."""
    return BagReader(str(Path(path).expanduser()), stamp_source="header")


# --------------------------------------------------------------------------- #
# CDR
# --------------------------------------------------------------------------- #
class _Unsupported(Exception):
    """The fast path cannot read this payload; use the generic decoder."""


class _Cdr:
    """Cursor over a classic (XCDR1) CDR payload.

    Only encapsulations 0x0000/0x0001 (CDR_BE/CDR_LE), which is what rmw
    writes for ROS 2 messages. XCDR2 puts a DHEADER in front of sequences of
    structs (PointField[]), so it is refused rather than guessed at.
    Alignment is relative to the end of the 4-byte encapsulation header."""
    __slots__ = ("buf", "pos", "end", "big")

    def __init__(self, buf):
        if len(buf) < 12 or buf[0] != 0 or buf[1] not in (0, 1):
            raise _Unsupported
        self.buf = buf
        self.pos = 4
        self.end = len(buf)
        self.big = buf[1] == 0

    def _take(self, size, fmt):
        r = (self.pos - 4) % size
        if r:
            self.pos += size - r
        if self.pos + size > self.end:
            raise _Unsupported
        v = struct.unpack_from((">" if self.big else "<") + fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def u32(self):
        return self._take(4, "I")

    def i32(self):
        return self._take(4, "i")

    def u8(self):
        if self.pos >= self.end:
            raise _Unsupported
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def string(self):
        n = self.u32()                      # length includes the trailing NUL
        if self.pos + n > self.end:
            raise _Unsupported
        s = bytes(self.buf[self.pos:self.pos + n]).split(b"\0", 1)[0]
        self.pos += n
        return s.decode("utf-8", "replace")

    def blob(self):
        """(offset, length) of a uint8[] without copying it."""
        n = self.u32()
        start = self.pos
        if start + n > self.end:
            raise _Unsupported
        self.pos += n
        return start, n

    def header_ns(self):
        sec = self.i32()
        nsec = self.u32()
        self.string()                       # frame_id
        return sec * 1_000_000_000 + nsec


# --------------------------------------------------------------------------- #
# PointCloud2
# --------------------------------------------------------------------------- #
_DTYPE_CACHE: Dict[tuple, np.dtype] = {}


def _cloud_dtype(fields, big, point_step):
    key = (fields, big, point_step)
    dt = _DTYPE_CACHE.get(key)
    if dt is None:
        names, formats, offsets = [], [], []
        for name, off, code, count in fields:
            t = _PF_DT.get(code)
            if t is None or name in names:
                continue
            names.append(name)
            formats.append((t, (count,)) if count > 1 else t)
            offsets.append(off)
        if not names:
            raise _Unsupported
        dt = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                       "itemsize": point_step})
        if big:
            dt = dt.newbyteorder(">")
        _DTYPE_CACHE[key] = dt
    return dt


def parse_pointcloud2(payload) -> Tuple[int, np.ndarray]:
    """CDR ``sensor_msgs/PointCloud2`` -> (header stamp ns, structured array).

    The array is a read-only view into ``payload`` whenever rows are unpadded
    (every Ouster/Velodyne/Livox driver); padded rows cost one copy. Raises
    ``_Unsupported`` when the payload is not something this parser reads."""
    c = _Cdr(payload)
    stamp = c.header_ns()
    height = c.u32()
    width = c.u32()
    fields = []
    for _ in range(c.u32()):
        name = c.string()
        off = c.u32()
        code = c.u8()
        count = c.u32()
        fields.append((name, off, code, count))
    big = c.u8() != 0
    point_step = c.u32()
    row_step = c.u32()
    start, n = c.blob()
    if point_step <= 0:
        raise _Unsupported
    dt = _cloud_dtype(tuple(fields), big, point_step)
    npts = height * width
    if height <= 1 or row_step == width * point_step:
        usable = min(npts, n // point_step)
        return stamp, np.frombuffer(payload, dt, count=usable, offset=start)
    if n < height * row_step or row_step < width * point_step:
        raise _Unsupported
    rows = np.frombuffer(payload, np.uint8, count=height * row_step, offset=start)
    rows = rows.reshape(height, row_step)[:, :width * point_step]
    return stamp, np.ascontiguousarray(rows).view(dt).reshape(-1)


class _PointTime:
    """Per-point time -> float64 seconds relative to the header stamp.

    Handles both conventions: relative offsets (Ouster ``t``, uint32 ns) and
    absolute per-point stamps (float64 s). The unit is picked from the first
    cloud's span, which a LiDAR sweep puts at 5 ms..1 s -- a wrong guess is
    silent, it just produces a differently-shaped smear, so it is detected from
    the data rather than trusted from config, and announced once."""

    def __init__(self, log=print):
        self.log = log
        self.ready = False
        self.field = None
        self.scale = 0.0
        self.absolute = False

    def setup(self, names, sample):
        self.ready = True
        self.field = next((n for n in _T_NAMES if n in names), None)
        if self.field is None:
            self.log(f"    [deskew] no per-point time field (fields: "
                     f"{sorted(names)}) -> deskew DISABLED")
            return
        tt = np.asarray(sample[self.field], np.float64)
        span = float(tt.max() - tt.min()) if tt.size else 0.0
        for cand in (1e-9, 1e-6, 1e-3, 1.0):
            if span > 0 and 0.005 < span * cand < 1.0:
                self.scale = cand
                break
        if not self.scale:
            self.log(f"    [deskew] time field '{self.field}' has an unusable "
                     f"span ({span:g}) -> deskew DISABLED")
            return
        self.absolute = bool(tt.min() * self.scale > 1e6)
        self.log(f"    [deskew] time field '{self.field}', scale {self.scale:g}, "
                 f"{'absolute' if self.absolute else 'relative'}, "
                 f"sweep {span * self.scale * 1e3:.1f} ms")

    def __call__(self, arr, stamp_ns):
        if not self.ready:
            self.setup(arr.dtype.names, arr)
        if not self.scale:
            return None
        tt = np.asarray(arr[self.field], np.float64) * self.scale
        if self.absolute:
            tt -= stamp_ns * 1e-9
        return tt


_F32 = np.dtype("=f4")


def _xyz(arr, topic):
    """(N,3) float32 xyz of a structured cloud.

    When x, y, z are three adjacent native float32 (every common LiDAR driver)
    this is a strided VIEW into the payload, so the only copy is the range-gated
    selection the caller makes; otherwise the three fields are gathered."""
    f = arr.dtype.fields or {}
    if not {"x", "y", "z"} <= set(f):
        raise BagError(f"{topic}: PointCloud2 has no x/y/z (fields {list(f)})")
    (tx, ox), (ty, oy), (tz, oz) = (f[k][:2] for k in "xyz")
    if (tx == ty == tz == _F32 and oy == ox + 4 and oz == ox + 8
            and arr.flags.c_contiguous):
        step = arr.dtype.itemsize
        return arr.view(np.uint8).reshape(-1, step)[:, ox:ox + 12].view(np.float32)
    xyz = np.empty((arr.shape[0], 3), np.float32)
    xyz[:, 0] = arr["x"]
    xyz[:, 1] = arr["y"]
    xyz[:, 2] = arr["z"]
    return xyz


@dataclass
class Scan:
    stamp_ns: int
    key: object                    # whatever select() returned
    xyz: np.ndarray                # (N,3) float32, sensor frame
    t: Optional[np.ndarray]        # (N,) float64 s rel. stamp, same mask as xyz

    @property
    def stamp(self) -> float:
        return self.stamp_ns * 1e-9


def iter_scans(reader: BagReader, topic: str, select: Optional[Select] = None,
               min_range: float = 0.0, max_range: float = np.inf,
               with_time: bool = False, min_points: int = 1,
               stats: Optional[dict] = None, log=print) -> Iterator[Scan]:
    """Yield the selected LiDAR scans of ``topic``, finite and range-gated.

    ``select(stamp_ns)`` runs on the header stamp BEFORE decoding; ``None``
    skips the message. ``stats`` (if given) is filled with counters: seen,
    selected, fast, fallback, empty."""
    st = stats if stats is not None else {}
    for k in ("seen", "selected", "fast", "fallback", "empty"):
        st.setdefault(k, 0)
    ptime = _PointTime(log) if with_time else None
    lo2 = float(min_range) ** 2
    hi2 = float(max_range) ** 2 if np.isfinite(max_range) else np.inf

    for _, log_ns, payload, decode in reader.iter_raw([topic]):
        st["seen"] += 1
        stamp = stamp_from_cdr(payload) or log_ns
        key = select(stamp) if select is not None else st["seen"] - 1
        if key is None:
            continue
        st["selected"] += 1
        try:
            stamp, arr = parse_pointcloud2(payload)
            st["fast"] += 1
        except _Unsupported:
            msg = decode(payload)
            arr = pointcloud2_to_array(msg)
            st["fallback"] += 1
        xyz = _xyz(arr, topic)
        d2 = np.einsum("ij,ij->i", xyz, xyz)          # NaN for non-finite points
        keep = (d2 > lo2) & (d2 < hi2)
        n = int(keep.sum())
        if n < max(1, int(min_points)):
            st["empty"] += 1
            continue
        t = ptime(arr, stamp) if ptime is not None else None
        if t is not None:
            keep &= np.isfinite(t)
            t = t[keep]
        yield Scan(stamp, key, xyz[keep], t)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
_PIX = {"mono8": (np.uint8, 1), "8uc1": (np.uint8, 1),
        "bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4),
        "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1),
        "32fc1": (np.float32, 1)}


def parse_image(payload, compressed: bool = False):
    """CDR ``sensor_msgs/Image`` (or ``CompressedImage``) -> (stamp ns,
    encoding, array). Raw images come back as a read-only view (H, W[, C]) in
    the message's own encoding; compressed ones decoded by OpenCV as BGR."""
    c = _Cdr(payload)
    stamp = c.header_ns()
    if compressed:
        c.string()                                     # format
        start, n = c.blob()
        import cv2
        img = cv2.imdecode(np.frombuffer(payload, np.uint8, n, start),
                           cv2.IMREAD_UNCHANGED)
        if img is None:
            raise _Unsupported
        return stamp, "bgr8" if img.ndim == 3 else "mono8", img
    h = c.u32()
    w = c.u32()
    enc = c.string().lower()
    big = c.u8() != 0
    step = c.u32()
    start, n = c.blob()
    if enc not in _PIX:
        raise _Unsupported
    t, ch = _PIX[enc]
    dt = np.dtype(t).newbyteorder(">" if big else "<")
    row = w * ch * dt.itemsize
    if step < row or n < h * step:
        raise _Unsupported
    raw = np.frombuffer(payload, np.uint8, h * step, start).reshape(h, step)
    if step != row:
        raw = np.ascontiguousarray(raw[:, :row])
    img = raw.view(dt).reshape((h, w, ch) if ch > 1 else (h, w))
    return stamp, enc, img


def _convert(enc, img, mode):
    """Raw image in `enc` -> requested mode ('bgr', 'gray' or 'raw')."""
    if mode == "raw":
        return img
    import cv2
    if enc in ("mono8", "8uc1"):
        return img.copy() if mode == "gray" else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc not in ("bgr8", "rgb8", "bgra8", "rgba8"):
        raise BagError(f"cannot convert {enc!r} to {mode}; use mode='raw'")
    code = {("bgr8", "gray"): cv2.COLOR_BGR2GRAY, ("rgb8", "gray"): cv2.COLOR_RGB2GRAY,
            ("bgra8", "gray"): cv2.COLOR_BGRA2GRAY, ("rgba8", "gray"): cv2.COLOR_RGBA2GRAY,
            ("rgb8", "bgr"): cv2.COLOR_RGB2BGR, ("bgra8", "bgr"): cv2.COLOR_BGRA2BGR,
            ("rgba8", "bgr"): cv2.COLOR_RGBA2BGR}.get((enc, mode))
    return img.copy() if code is None else cv2.cvtColor(img, code)


def _generic_image(msg, compressed):
    """Decoded message -> (stamp ns, encoding, array), for the fallback path."""
    s = msg.header.stamp
    stamp = int(s.sec) * 1_000_000_000 + int(s.nanosec)
    data = np.frombuffer(bytes(msg.data), np.uint8)
    if compressed:
        import cv2
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise BagError(f"cannot decode compressed image ({msg.format!r})")
        return stamp, "bgr8" if img.ndim == 3 else "mono8", img
    enc = str(msg.encoding).lower()
    if enc not in _PIX:
        raise BagError(f"unsupported image encoding {msg.encoding!r}")
    t, ch = _PIX[enc]
    dt = np.dtype(t).newbyteorder(">" if msg.is_bigendian else "<")
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    raw = data[:h * step].reshape(h, step)[:, :w * ch * dt.itemsize]
    img = np.ascontiguousarray(raw).view(dt).reshape((h, w, ch) if ch > 1 else (h, w))
    return stamp, enc, img


@dataclass
class Frame:
    stamp_ns: int
    key: object
    encoding: str                  # encoding in the bag
    image: np.ndarray              # converted to the requested mode

    @property
    def stamp(self) -> float:
        return self.stamp_ns * 1e-9


def iter_images(reader: BagReader, topic: str, select: Optional[Select] = None,
                mode: str = "bgr", period: float = 0.0,
                stats: Optional[dict] = None) -> Iterator[Frame]:
    """Yield the selected frames of an Image or CompressedImage topic.

    ``mode``: 'bgr' (uint8 HxWx3), 'gray' (uint8 HxW) or 'raw' (the bag's own
    encoding, read-only view -- depth images come out as uint16/float32).
    ``period`` > 0 keeps at most one frame per that many seconds of header
    time, applied before ``select`` and before decoding."""
    if mode not in ("bgr", "gray", "raw"):
        raise ValueError("mode must be 'bgr', 'gray' or 'raw'")
    info = reader.topics().get(topic)
    if info is None:
        raise BagError(f"topic {topic!r} not in bag; available: "
                       f"{sorted(reader.topics())}")
    compressed = "CompressedImage" in info.msgtype
    st = stats if stats is not None else {}
    for k in ("seen", "selected", "fast", "fallback"):
        st.setdefault(k, 0)
    gap = int(float(period) * 1e9)
    last = None
    for _, log_ns, payload, decode in reader.iter_raw([topic]):
        st["seen"] += 1
        stamp = stamp_from_cdr(payload) or log_ns
        if gap and last is not None and stamp - last < gap:
            continue
        key = select(stamp) if select is not None else st["seen"] - 1
        if key is None:
            continue
        last = stamp
        st["selected"] += 1
        try:
            stamp, enc, img = parse_image(payload, compressed)
            st["fast"] += 1
        except _Unsupported:
            stamp, enc, img = _generic_image(decode(payload), compressed)
            st["fallback"] += 1
        yield Frame(stamp, key, enc, _convert(enc, img, mode))


# --------------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------------- #
def nearest_pose(times_s, tol_s: float) -> Select:
    """Selector: index of the nearest trajectory time within ``tol_s``, else
    None. The same association rule the stage scripts used."""
    t = np.asarray(times_s, np.float64)
    if t.size == 0:
        raise ValueError("empty trajectory")
    tol = float(tol_s)
    n = len(t)

    def select(stamp_ns):
        s = stamp_ns * 1e-9
        i = int(np.searchsorted(t, s))
        if i <= 0:
            j = 0
        elif i >= n:
            j = n - 1
        else:
            j = i if (t[i] - s) < (s - t[i - 1]) else i - 1
        return j if abs(t[j] - s) <= tol else None
    return select


def detect_points_topic(reader: BagReader, override: str = "", n_poses: int = 0,
                        log=print) -> str:
    """Pick the mapping LiDAR's PointCloud2 topic.

    Message count alone is a bad discriminator -- a radar or an RGB-D camera can
    publish more clouds than the LiDAR. The trajectory was produced from the
    LiDAR, so the topic whose count is closest to the number of poses wins;
    sensors that are never the mapping LiDAR are demoted outright. Reads the
    file index only, nothing is decoded."""
    if override:
        return override
    demote = ("radar", "zed", "depth", "rgbd", "realsense", "camera")
    cands = [(t.name, t.count) for t in reader.topics().values()
             if "PointCloud2" in t.msgtype]
    if not cands:
        raise BagError("no PointCloud2 topic in the bag")
    if len(cands) == 1:
        return cands[0][0]

    def score(tc):
        name, count = tc
        bad = any(k in name.lower() for k in demote)
        near = abs(count - n_poses) / max(n_poses, 1) if n_poses else 0.0
        return (bad, near, -count)

    topic = sorted(cands, key=score)[0][0]
    log(f"    PointCloud2 topics in bag: {[f'{t} ({n})' for t, n in cands]}")
    log(f"    -> using {topic!r} ({dict(cands)[topic]} msgs vs {n_poses} poses); "
        f"set points_topic to override")
    return topic
