"""Camera placement within each mount's range of movement.

A camera's aim is (heading, tilt), roll kept at 0 (image upright):
  heading  direction of the optical axis in the floor plane, degrees,
           0 = +x, 90 = +y (counter-clockwise seen from above).
           For a camera looking straight down, the direction the top of the
           image faces.
  tilt     degrees below horizontal: 0 = level, 90 = straight down.

The mounts YAML says how far each camera can move (see configs/mounts_wide.yaml):
  pan:    [lo, hi]   heading change from `heading0` (default: where it points now)
  tilt:   [lo, hi]   absolute tilt range
  tilt_change: [lo, hi]   or: tilt range relative to the current tilt
                     (+ = further down), clipped to 0..90
  along:  [[x,y,z], [x,y,z]]   the camera can sit anywhere on this segment
                     (a rail, a stretch of wall); leave out = position fixed
  fixed:  true       the camera does not move at all
`defaults:` fills what a camera leaves out.
"""
from dataclasses import dataclass, replace

import numpy as np
import yaml

from .coverage import camera_visibility


def pose_from_aim(position, heading_deg, tilt_deg):
    """Camera -> room transform, roll 0."""
    h, t = np.radians(heading_deg), np.radians(tilt_deg)
    z = np.array([np.cos(t) * np.cos(h), np.cos(t) * np.sin(h), -np.sin(t)])
    x = np.array([np.sin(h), -np.cos(h), 0.0])        # right of the heading
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = position
    return T


def aim_from_pose(T):
    """(heading, tilt, roll) in degrees of a camera -> room transform."""
    R = T[:3, :3]
    z = R[:, 2]
    tilt = np.degrees(np.arcsin(np.clip(-z[2], -1, 1)))
    up = -R[:, 1]                                       # top of the image
    hz = z[:2] if np.hypot(*z[:2]) > 0.05 else up[:2]   # straight down: image top
    heading = np.degrees(np.arctan2(hz[1], hz[0]))
    # roll: angle of the image x axis out of the horizontal plane
    roll = np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    return heading, tilt, roll


def room_aim_point(T, room):
    """First floor or wall point on the optical axis: (surface name, xyz) or None.

    This is where to put a marker when aiming the camera by hand: turn it
    until the marker sits at the image centre.
    """
    c, d = T[:3, 3], T[:3, 2]
    planes = [("floor", 2, 0.0), ("ceiling", 2, room.height),
              ("left wall", 0, room.x[0]), ("right wall", 0, room.x[1]),
              ("bottom wall", 1, room.y[0]), ("top wall", 1, room.y[1])]
    best = None
    for name, ax, v in planes:
        if abs(d[ax]) < 1e-9:
            continue
        t = (v - c[ax]) / d[ax]
        if t <= 1e-6:
            continue
        p = c + t * d
        if (room.x[0] - 1e-6 <= p[0] <= room.x[1] + 1e-6 and room.y[0] - 1e-6 <= p[1] <= room.y[1] + 1e-6
                and -1e-6 <= p[2] <= room.height + 1e-6) and (best is None or t < best[0]):
            best = (t, name, p)
    return None if best is None else best[1:]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class Mount:
    name: str
    heading0: float
    pan: tuple
    tilt: tuple
    along: np.ndarray = None      # (2,3) segment, or None
    fixed: bool = False

    def position(self, s, default):
        if self.along is None:
            return default
        return self.along[0] + s * (self.along[1] - self.along[0])

    def s_of(self, p):
        """Segment parameter of the point nearest to p."""
        if self.along is None:
            return 0.0
        a, b = self.along
        d = b - a
        return float(np.clip((p - a) @ d / max(d @ d, 1e-12), 0, 1))

    def length(self):
        return 0.0 if self.along is None else float(np.linalg.norm(self.along[1] - self.along[0]))


def load_mounts(path, cameras):
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    defaults = doc.get("defaults", {}) or {}
    per = doc.get("mounts", {}) or {}
    unknown = set(per) - {c.name for c in cameras}
    if unknown:
        raise ValueError(f"mounts for unknown cameras: {', '.join(sorted(unknown))}")
    mounts = []
    for cam in cameras:
        own = per.get(cam.name) or {}
        m = {**defaults, **own}
        heading, tilt, _ = aim_from_pose(cam.T_world_cam)
        # the camera's own tilt / tilt_change wins over either from defaults
        if "tilt_change" in own or ("tilt_change" in m and "tilt" not in own):
            lo, hi = m["tilt_change"]
            m["tilt"] = [max(0.0, tilt + lo), min(90.0, tilt + hi)]
        along = m.get("along")
        mounts.append(Mount(
            name=cam.name,
            heading0=float(m.get("heading0", heading)),
            pan=tuple(m.get("pan", [0, 0])),
            tilt=tuple(m.get("tilt", [tilt, tilt])),
            along=None if along is None else np.asarray(along, float),
            fixed=bool(m.get("fixed", False)),
        ))
    return mounts


@dataclass
class Setting:
    s: float          # position on the mount segment, 0..1
    pan: float        # degrees from heading0
    tilt: float

    def pose(self, mount, default_position):
        return pose_from_aim(mount.position(self.s, default_position), mount.heading0 + self.pan, self.tilt)


def current_setting(mount, cam):
    heading, tilt, _ = aim_from_pose(cam.T_world_cam)
    pan = float(np.clip(wrap(heading - mount.heading0), *mount.pan))
    return Setting(mount.s_of(cam.center), pan, float(np.clip(tilt, *mount.tilt)))


def _axis(lo, hi, step, around=None, half=None):
    if around is not None:
        lo, hi = max(lo, around - half), min(hi, around + half)
    if hi - lo < 1e-9:
        return np.array([lo])
    n = max(1, int(np.ceil((hi - lo) / step)))
    return np.linspace(lo, hi, n + 1)


def candidates(mount, step_deg, step_m, near=None):
    """Settings on a grid over the mount's range; `near` = finer grid around a setting."""
    if mount.fixed:
        return []
    L = mount.length()
    ds = step_m / L if L > 0 else 1.0
    if near is None:
        S = _axis(0, 1, ds) if L > 0 else np.array([0.0])
        P = _axis(*mount.pan, step_deg)
        T = _axis(*mount.tilt, step_deg)
    else:
        S = _axis(0, 1, ds / 2, near.s, ds) if L > 0 else np.array([0.0])
        P = _axis(*mount.pan, step_deg / 4, near.pan, step_deg)
        T = _axis(*mount.tilt, step_deg / 4, near.tilt, step_deg)
    return [Setting(float(s), float(p), float(t)) for s in S for p in P for t in T]


def score(counts, k, valid):
    """Covered fraction at k, plus a small reward for views beyond k (redundancy)."""
    c = counts[valid]
    return float(np.mean(c >= k) + 0.05 * np.mean(np.minimum(c, k + 1)) / (k + 1))


def optimise(cams, mounts, room, pts, blocked, k, step_deg=10.0, step_m=0.5, max_moved=None,
             min_gain=1e-3, log=print):
    """Best-improvement search: each round, try every camera over its whole range
    (coarse grid, then finer around the best) with the others held, and apply the
    single change that helps most. Stops when the best change gains less than
    `min_gain` (score ~ fraction of the volume; every move means re-calibrating
    a camera) or `max_moved` cameras have changed. Returns new cameras, their
    settings and the indices moved."""
    valid = ~blocked
    cams = list(cams)
    start_pos = [c.center.copy() for c in cams]
    # cameras start where they are (calibrated roll kept); a moved one is rebuilt from its setting
    settings = [current_setting(m, c) for m, c in zip(mounts, cams)]
    vis = np.stack([camera_visibility(c, room, pts, blocked) for c in cams])
    counts = vis.sum(0)
    best = score(counts, k, valid)
    log(f"start: {100 * np.mean(counts[valid] >= k):.2f}% at k={k}")
    moved = set()

    def trial(i, st):
        cand = replace(cams[i], T_world_cam=st.pose(mounts[i], start_pos[i]))
        v = camera_visibility(cand, room, pts, blocked)
        return score(counts - vis[i] + v, k, valid), cand, v

    rnd = 0
    while True:
        rnd += 1
        top = None
        for i, m in enumerate(mounts):
            if max_moved is not None and len(moved) >= max_moved and i not in moved:
                continue
            local = None
            for st in candidates(m, step_deg, step_m):
                sc, cand, v = trial(i, st)
                if local is None or sc > local[0]:
                    local = (sc, st, cand, v)
            if local is None:
                continue
            for st in candidates(m, step_deg, step_m, near=local[1]):
                sc, cand, v = trial(i, st)
                if sc > local[0]:
                    local = (sc, st, cand, v)
            if top is None or local[0] > top[0]:
                top = (local[0], i, *local[1:])
        if top is None or top[0] < best + min_gain:
            break
        sc, i, st, cand, v = top
        counts = counts - vis[i] + v
        vis[i], cams[i], settings[i] = v, cand, st
        best = sc
        moved.add(i)
        log(f"round {rnd}: {cams[i].name} -> pan {st.pan:+.0f}, tilt {st.tilt:.0f}, s {st.s:.2f}: "
            f"{100 * np.mean(counts[valid] >= k):.2f}%")
    return cams, settings, sorted(moved)
