"""The contract between sensors and the tracker.

Measurement: what one sensor reported about one target at one time, already in
the room frame (metres, floor plane), with its uncertainty.

SensorModel: what the tracker needs to know about a sensor to weigh what it
reported and what it did NOT report:
  p_detect      chance it detects a person who is inside its coverage
  p_detect_static, min_speed
                a different chance for people moving slower than min_speed
                (a Doppler radar loses people who sit still)
  coverage      where it can see at all (polygons, a callable, or everywhere)
  clutter       false reports per square metre per scan
  ident         for a sensor that reports one known target (a robot's own pose, a
                tag): it detects only that target, so it says nothing about others
A miss counts against a track only as much as the sensor should have seen it.

Zones: where people may enter and leave (the door). Tracks that start or end
elsewhere cost more, so a person who "appears" in the middle of the room is
taken as someone seen before, not a new person.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml


@dataclass
class Measurement:
    t: float                              # seconds, common clock
    sensor: str                           # sensor name, a key of the sensor models
    xy: np.ndarray                        # (2,) floor position in the room frame, metres
    cov: np.ndarray                       # (2, 2) covariance of xy, m^2
    feat: Optional[np.ndarray] = None     # appearance feature (any length), compared by cosine
    ident: Optional[str] = None           # known identity (a robot reporting its own pose, a tag)
    score: float = 1.0                    # detector confidence, informational
    meta: dict = field(default_factory=dict)   # provenance, e.g. which camera boxes

    def __post_init__(self):
        self.xy = np.asarray(self.xy, float).reshape(2)
        self.cov = np.asarray(self.cov, float).reshape(2, 2)
        if self.feat is not None:
            f = np.asarray(self.feat, float).ravel()
            n = np.linalg.norm(f)
            self.feat = f / n if n > 0 else None


def points_in_polygon(xy, poly):
    """(N,) bool: points (N,2) inside polygon [(x,y), ...] (even-odd rule)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    p = np.asarray(poly, float)
    x, y = xy[:, 0:1], xy[:, 1:2]
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    return cross.sum(1) % 2 == 1


def _region(spec):
    """A region from config: {polygon: [[x,y],...]} or {circle: [x, y, r]} -> f(xy (N,2)) -> bool (N,)."""
    if callable(spec):
        return spec
    if "polygon" in spec:
        poly = spec["polygon"]
        return lambda xy: points_in_polygon(xy, poly)
    if "circle" in spec:
        cx, cy, r = spec["circle"]
        return lambda xy: np.hypot(np.atleast_2d(xy)[:, 0] - cx, np.atleast_2d(xy)[:, 1] - cy) <= r
    raise ValueError(f"region needs polygon or circle: {spec}")


_SAMPLES = np.random.default_rng(0).standard_normal((64, 2))   # fixed: the result is deterministic


@dataclass
class SensorModel:
    name: str
    p_detect: float = 0.9
    clutter: float = 1e-3                 # false reports per m^2 per scan
    coverage: Optional[Callable] = None   # f(xy (N,2)) -> (N,) bool or [0,1]; None: everywhere
    p_detect_static: Optional[float] = None
    min_speed: float = 0.0                # m/s; slower than this uses p_detect_static
    ident: Optional[str] = None           # reports only this target

    def pd_for(self, idents, xy, v=None, Pv=None):
        """pd() for tracks with these identities: a self-reporting sensor sees only its own target."""
        p = self.pd(xy, v, Pv)
        if self.ident is not None:
            p = np.where(np.array([i == self.ident for i in idents]), p, 0.0)
        return p

    def pd(self, xy, v=None, Pv=None):
        """Detection probability for people at xy (N,2) moving with velocity v (N,2).

        With Pv (N,2,2), the velocity's covariance, a speed-dependent sensor is
        averaged over how unsure the speed is: someone sitting still with a
        noisy speed estimate is not taken for someone walking."""
        xy = np.atleast_2d(np.asarray(xy, float))
        p = np.full(len(xy), float(self.p_detect))
        if self.p_detect_static is not None and v is not None:
            v = np.atleast_2d(np.asarray(v, float))
            if Pv is None:
                moving = (np.linalg.norm(v, axis=1) >= self.min_speed).astype(float)
            else:
                L = np.linalg.cholesky(np.asarray(Pv, float) + np.eye(2) * 1e-9)        # (N,2,2)
                vs = v[:, None, :] + np.einsum("nij,kj->nki", L, _SAMPLES)            # (N,K,2)
                moving = (np.linalg.norm(vs, axis=2) >= self.min_speed).mean(1)
            p = self.p_detect_static + (self.p_detect - self.p_detect_static) * moving
        if self.coverage is not None:
            p = p * np.asarray(self.coverage(xy), float)
        return np.clip(p, 0.0, 0.999)

    @classmethod
    def from_dict(cls, name, d):
        d = dict(d or {})
        cov = d.pop("coverage", None)
        regions = [] if cov is None else [_region(c) for c in (cov if isinstance(cov, list) else [cov])]
        coverage = (lambda xy: np.any([r(xy) for r in regions], axis=0)) if regions else None
        return cls(name=name, coverage=coverage, **{k: v for k, v in d.items()})


@dataclass
class Zones:
    """Where tracks may start and end. Without zones, anywhere at the same (moderate) cost."""
    entry: list = field(default_factory=list)     # region functions
    p_inside: float = 0.9                         # chance an entry/exit happens in a zone
    p_outside: float = 1e-3                       # ... and per track elsewhere (tracks cut by a missed stretch)

    def log_p(self, xy):
        """log probability that a track starts / ends at xy (N,2)."""
        xy = np.atleast_2d(xy)
        if not self.entry:
            return np.full(len(xy), np.log(0.05))
        inside = np.any([r(xy) for r in self.entry], axis=0)
        return np.where(inside, np.log(self.p_inside), np.log(self.p_outside))

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        entry = [_region(z) for z in d.pop("entry", [])]
        return cls(entry=entry, **d)


def load_config(path):
    """sensors YAML -> ({name: SensorModel}, Zones, tracker overrides dict).

    sensors:
      cams:  {p_detect: 0.95, clutter: 0.0005}
      radar: {p_detect: 0.9, p_detect_static: 0.05, min_speed: 0.2, clutter: 0.01,
              coverage: {polygon: [[-4, -5], [0, -5], [0, 5], [-4, 5]]}}
      robot_1: {p_detect: 0.99, clutter: 1.0e-6}
    zones:
      entry: [{circle: [-3.5, 4.8, 0.8]}]     # the door
    tracker: {max_gap: 60}                     # any TrackerConfig field
    """
    doc = yaml.safe_load(Path(path).read_text()) or {}
    sensors = {n: SensorModel.from_dict(n, d) for n, d in (doc.get("sensors") or {}).items()}
    return sensors, Zones.from_dict(doc.get("zones")), doc.get("tracker") or {}
