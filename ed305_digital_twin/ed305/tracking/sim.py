"""A simulated room to test the tracker against known identities.

People come in through the door, walk to random spots, sit for a while,
walk on, and some leave through the door again; one robot drives around and
reports its own pose. Two of the people wear almost the same clothes.

Sensors (each a SensorModel the tracker also gets):
  cams    the fused 16-camera detections: 10 Hz, 7 cm noise, misses, people close
          together sometimes merged into one detection, a few false ones,
          an appearance feature most of the time.
          "sparse": the cameras see only x < -1 and x > 1 (a blind band between).
  radar   ("fused"): left half of the room, 20 cm noise, loses people who stand
          still, many false points, no appearance.
  robot_1 the robot's own pose with its identity.
"""
import numpy as np

from .model import Measurement, SensorModel, Zones

ROOM = (-4.0, 4.0, -5.0, 5.0)
DOOR = (-3.5, 4.8)
DIM = 16


def _walk(rng, t_in, t_out, start, rate, speed, dwell, pause_p=0.6, leaves=False):
    """Positions sampled at `rate` between t_in and t_out: walk to random spots, sometimes sit;
    with leaves, head for the door in the last 25 s and stop there (gone)."""
    x0, x1, y0, y1 = ROOM
    pos = np.array(start, float)
    t = t_in
    ts, ps = [], []
    leaving = False
    dt = 1.0 / rate
    while t < t_out:
        if leaves and not leaving and t_out - t < 25:
            goal = np.array(DOOR)
            leaving = True
        else:
            goal = np.array([rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5)])
        v = speed * rng.uniform(0.8, 1.2)
        while np.linalg.norm(goal - pos) > v * dt and t < t_out:
            pos = pos + (goal - pos) / np.linalg.norm(goal - pos) * v * dt
            ts.append(t), ps.append(pos.copy())
            t += dt
        if leaving:
            break
        if rng.random() < pause_p:
            stay = rng.uniform(*dwell)
            for _ in range(int(stay * rate)):
                if t >= t_out:
                    break
                ts.append(t), ps.append(pos.copy())
                t += dt
    return np.round(np.array(ts), 6), np.array(ps)


def simulate(kind="dense", seed=0, duration=120.0, n_people=8, rate=10.0, twins=True):
    """-> (measurements, sensors, zones, ground truth {id: (times, xy)})."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = ROOM
    area = (x1 - x0) * (y1 - y0)
    gt, look = {}, {}
    for k in range(n_people):
        inside_at_start = k < n_people // 2
        t_in = 0.0 if inside_at_start else rng.uniform(5, duration / 2)
        t_out = rng.uniform(duration * 0.6, duration - 5) if k % 3 == 2 else 1e9
        start = (rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5)) if inside_at_start else DOOR
        t, p = _walk(rng, t_in, min(t_out, duration), start, rate, 1.0, (3, 20), leaves=t_out < duration)
        t_in_grid = np.round(np.ceil(t_in * rate) / rate, 6)
        t = np.round(t_in_grid + np.arange(len(t)) / rate, 6)
        keep = t < duration
        gt[f"person_{k}"] = (t[keep], p[keep])
        look[f"person_{k}"] = rng.normal(size=DIM)
    if twins:
        look["person_1"] = look["person_0"] + rng.normal(scale=0.35, size=DIM)    # similar clothes
    tr, pr = _walk(rng, 0.0, duration, (2.0, -3.0), rate, 0.6, (1, 3), pause_p=0.2)
    gt["robot_1"] = (np.round(np.arange(len(tr)) / rate, 6), pr)

    cams_cov = (lambda xy: (np.atleast_2d(xy)[:, 0] < -1) | (np.atleast_2d(xy)[:, 0] > 1)) if kind == "sparse" else None
    radar_cov = lambda xy: np.atleast_2d(xy)[:, 0] < 0
    sensors = {
        "cams": SensorModel("cams", p_detect=0.93, clutter=0.1 / area, coverage=cams_cov),
        "robot_1": SensorModel("robot_1", p_detect=0.99, clutter=1e-6, ident="robot_1"),
    }
    if kind == "fused":
        sensors["radar"] = SensorModel("radar", p_detect=0.9, p_detect_static=0.05, min_speed=0.2,
                                       clutter=1.0 / (area / 2), coverage=radar_cov)

    meas = []
    steps = np.round(np.arange(0, duration, 1.0 / rate), 6)
    people = [g for g in gt if g.startswith("person")]
    for t in steps:
        here = {}
        for g in people:
            tt, pp = gt[g]
            i = np.searchsorted(tt, t)
            if i < len(tt) and tt[i] == t:
                k = max(i - 1, 0)
                here[g] = (pp[i], (pp[i] - pp[k]) * rate)
        names = list(here)
        merged = set()
        for a in range(len(names)):                       # people close together: sometimes one detection
            for b in range(a + 1, len(names)):
                pa, pb = here[names[a]][0], here[names[b]][0]
                if np.linalg.norm(pa - pb) < 0.4 and rng.random() < 0.5 and names[b] not in merged:
                    merged.add(names[b])
        for g, (p, v) in here.items():
            if g in merged:
                continue
            if cams_cov is not None and not cams_cov(p[None])[0]:
                continue
            crowd = any(np.linalg.norm(p - here[h][0]) < 0.6 for h in here if h != g)
            if rng.random() < (0.75 if crowd else 0.95):
                f = look[g] + rng.normal(scale=0.6, size=DIM) if rng.random() < 0.8 else None
                meas.append(Measurement(t, "cams", p + rng.normal(scale=0.07, size=2), np.eye(2) * 0.07 ** 2,
                                        feat=f, meta={"truth": g}))
        for _ in range(rng.poisson(0.1)):
            q = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if cams_cov is None or cams_cov(q[None])[0]:
                meas.append(Measurement(t, "cams", q, np.eye(2) * 0.07 ** 2, feat=rng.normal(size=DIM),
                                        meta={"truth": None}))
        if kind == "fused":
            for g, (p, v) in here.items():
                if p[0] < 0 and rng.random() < (0.9 if np.linalg.norm(v) >= 0.2 else 0.05):
                    meas.append(Measurement(t, "radar", p + rng.normal(scale=0.2, size=2), np.eye(2) * 0.2 ** 2,
                                            meta={"truth": g}))
            for _ in range(rng.poisson(1.0)):
                q = np.array([rng.uniform(x0, 0), rng.uniform(y0, y1)])
                meas.append(Measurement(t, "radar", q, np.eye(2) * 0.2 ** 2, meta={"truth": None}))
        i = int(round(t * rate))
        if i < len(pr):
            meas.append(Measurement(t, "robot_1", pr[i] + rng.normal(scale=0.03, size=2), np.eye(2) * 0.03 ** 2,
                                    ident="robot_1", meta={"truth": "robot_1"}))
    zones = Zones.from_dict({"entry": [{"circle": [DOOR[0], DOOR[1], 1.0]}]})
    return meas, sensors, zones, gt
