"""Sensors -> Measurements. One small function per kind of sensor.

cameras   people_to_measurements(): the 16-camera association (ed305/multiview.py)
          gives a floor point per person per time step, seen by n cameras;
          camera_sensor() is its SensorModel, with the coverage the cameras have.
radar     polar_measurement(): a range / azimuth point (after clustering) from a
          sensor at a known place in the room.
robots    pose_measurement(): a robot's own pose with its identity.
any       read_measurements_csv() / write_measurements_csv(): a file per sensor,
          so a new sensor only has to write this format.

CSV columns: t, sensor, x, y, sxx, sxy, syy [, ident] [, f0, f1, ...]
(sxx, sxy, syy: the position covariance in m^2; f*: an appearance feature).
"""
import csv
from pathlib import Path

import numpy as np

from .model import Measurement, SensorModel


def camera_sensor(grid, min_views=2, p_detect=0.93, clutter=1e-3, name="cams", shape=1):
    """SensorModel of the fused cameras: covered where >= min_views cameras see a person whole."""
    seen = grid.full[:, shape, :].sum(0) >= min_views
    xs, ys = np.unique(grid.xy[:, 0]), np.unique(grid.xy[:, 1])
    step = grid.step

    def coverage(xy):
        xy = np.atleast_2d(xy)
        ix = np.clip(np.round((xy[:, 0] - xs[0]) / step).astype(int), 0, len(xs) - 1)
        iy = np.clip(np.round((xy[:, 1] - ys[0]) / step).astype(int), 0, len(ys) - 1)
        inside = (xy[:, 0] >= xs[0] - step) & (xy[:, 0] <= xs[-1] + step) & \
                 (xy[:, 1] >= ys[0] - step) & (xy[:, 1] <= ys[-1] + step)
        return seen[iy * len(xs) + ix] & inside

    return SensorModel(name, p_detect=p_detect, clutter=clutter, coverage=coverage)


def people_to_measurements(t, people, cams, boxes, confs, feats=None, sensor="cams", sigma=0.15, floor=0.05):
    """multiview.associate() people at time t -> Measurements.

    Position noise shrinks with the number of cameras that agree:
    sigma / sqrt(views), at least `floor` metres. feats[c][d]: appearance feature of
    detection d in camera c (or None); a person's feature is their mean over views.
    meta["boxes"]: {camera name: (x0, y0, x1, y1)} to label the boxes with the track later."""
    out = []
    for p in people:
        s = max(floor, sigma / np.sqrt(p.views))
        fs = [feats[c][d] for c, d in p.dets.items() if feats is not None and feats[c] is not None
              and feats[c][d] is not None]
        f = None
        if fs:
            f = np.mean([v / (np.linalg.norm(v) + 1e-12) for v in fs], axis=0)
        out.append(Measurement(t, sensor, p.xy, np.eye(2) * s * s, feat=f,
                               score=float(np.mean([confs[c][d] for c, d in p.dets.items()])),
                               meta={"boxes": {cams[c].name: tuple(float(v) for v in boxes[c][d])
                                               for c, d in p.dets.items()}}))
    return out


def polar_measurement(t, sensor, origin, yaw, r, az, sigma_r=0.1, sigma_az=np.radians(3), **kw):
    """A point at range r (m), azimuth az (rad, CCW from the sensor's forward axis) from a sensor
    at origin (x, y) looking along yaw (rad) -> Measurement with the matching covariance."""
    a = yaw + az
    xy = np.asarray(origin, float) + r * np.array([np.cos(a), np.sin(a)])
    J = np.array([[np.cos(a), -r * np.sin(a)], [np.sin(a), r * np.cos(a)]])
    cov = J @ np.diag([sigma_r ** 2, sigma_az ** 2]) @ J.T
    return Measurement(t, sensor, xy, cov, **kw)


def pose_measurement(t, name, xy, sigma=0.05):
    """A robot's own position: its identity is known."""
    return Measurement(t, name, xy, np.eye(2) * sigma ** 2, ident=name)


def read_measurements_csv(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            fk = sorted((k for k in r if k.startswith("f") and k[1:].isdigit()), key=lambda k: int(k[1:]))
            feat = np.array([float(r[k]) for k in fk]) if fk and all(r[k] != "" for k in fk) else None
            cov = np.array([[float(r["sxx"]), float(r["sxy"])], [float(r["sxy"]), float(r["syy"])]])
            out.append(Measurement(float(r["t"]), r["sensor"], [float(r["x"]), float(r["y"])], cov,
                                   feat=feat, ident=r.get("ident") or None))
    return out


def write_measurements_csv(path, meas, owner=None):
    """owner: {measurement index: track id} adds a `track` column."""
    nf = max((len(m.feat) for m in meas if m.feat is not None), default=0)
    fields = ["t", "sensor", "x", "y", "sxx", "sxy", "syy", "ident"] + [f"f{i}" for i in range(nf)]
    if owner is not None:
        fields.append("track")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k, m in enumerate(meas):
            row = {"t": f"{m.t:.3f}", "sensor": m.sensor, "x": f"{m.xy[0]:.3f}", "y": f"{m.xy[1]:.3f}",
                   "sxx": f"{m.cov[0, 0]:.6f}", "sxy": f"{m.cov[0, 1]:.6f}", "syy": f"{m.cov[1, 1]:.6f}",
                   "ident": m.ident or ""}
            if m.feat is not None:
                row.update({f"f{i}": f"{v:.5f}" for i, v in enumerate(m.feat)})
            if owner is not None:
                row["track"] = owner.get(k, "")
            w.writerow(row)
