"""Trajectory error: ATE, RPE, drift, and the absolute check.

Three tiers, and a run is only reported with all three, because on coop2 the
first tier alone cannot support an accuracy claim:

  tier 1  ATE / RPE against the reference trajectory. The reference is the
          offline mapping pipeline's own output (stage 09), so this is
          AGREEMENT WITH THAT PIPELINE, not accuracy. It is still the most
          discriminative number available, and it is the only one that exists
          at every frame.
  tier 2  the absolute check: the estimate's position at the surveyed-board
          dwell frames against the board survey, which is independent evidence
          with a stated uncertainty (3-15 mm on coop2). Sparse, but it is the
          only tier whose error bar does not come from the thing being tested.
  tier 3  self-consistency: revisit residual — how far apart the estimate puts
          two passes over the same place. Needs no reference at all, so it is
          the one tier that survives on a sequence with no ground truth.

`report_uncertainty_m` is carried through every tier-1 result. A method whose
ATE is below it has not been shown to be better than the reference; it has been
shown to be indistinguishable from it.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import se3
from .align import Alignment, fit
from .trajectory import Trajectory

__all__ = ["ErrorStats", "AteResult", "RpeResult", "ate", "rpe",
           "absolute_check", "revisit_residual"]


@dataclasses.dataclass
class ErrorStats:
    n: int
    rmse: float
    mean: float
    median: float
    p90: float
    max: float

    @staticmethod
    def of(e: np.ndarray) -> "ErrorStats":
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        if not len(e):
            raise ValueError("no errors to summarise")
        return ErrorStats(int(len(e)), float(np.sqrt(np.mean(e ** 2))), float(np.mean(e)),
                          float(np.median(e)), float(np.percentile(e, 90)), float(np.max(e)))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class AteResult:
    trans: ErrorStats            # metres
    rot: ErrorStats              # degrees
    alignment: Alignment
    n_ref: int
    n_est: int
    coverage: float              # fraction of estimate poses that found a reference
    path_length_m: float
    duration_s: float
    drift_percent: float         # ATE RMSE as a percentage of path length
    below_reference_uncertainty: bool | None = None
    # The reference's own duration, so a reader can see that an estimate covers
    # less of the run than the row beside it. `coverage` cannot show this: it
    # asks what fraction of the ESTIMATE found a reference, which is ~1.0 for an
    # estimate that simply stops early. On coop2 this is not hypothetical --
    # RTAB-Map's loop-closed graph adds a node on MOTION, so the 27 s the cart
    # spends parked at `rs_anchor` produces no nodes at all, and every graph row
    # is scored over a shorter run than the odometry row printed next to it.
    ref_duration_s: float | None = None

    def as_dict(self) -> dict:
        return {
            "ate_trans_m": self.trans.as_dict(),
            "ate_rot_deg": self.rot.as_dict(),
            "alignment": {"mode": self.alignment.mode, "scale": self.alignment.scale,
                          "scale_observed": self.alignment.scale_observed},
            "n_ref": self.n_ref, "n_est": self.n_est, "coverage": self.coverage,
            "path_length_m": self.path_length_m, "duration_s": self.duration_s,
            "drift_percent": self.drift_percent,
            "below_reference_uncertainty": self.below_reference_uncertainty,
            "ref_duration_s": self.ref_duration_s,
            "span_fraction": self.span_fraction,
        }

    @property
    def span_fraction(self) -> float | None:
        """How much of the reference's span the estimate covers, 0..1."""
        if not self.ref_duration_s:
            return None
        return float(self.duration_s / self.ref_duration_s)


@dataclasses.dataclass
class RpeResult:
    delta: float
    unit: str                    # "m" or "s"
    trans: ErrorStats            # metres per delta
    rot: ErrorStats              # degrees per delta
    n_pairs: int

    def as_dict(self) -> dict:
        return {"delta": self.delta, "unit": self.unit, "n_pairs": self.n_pairs,
                "rpe_trans_m": self.trans.as_dict(), "rpe_rot_deg": self.rot.as_dict()}


def ate(est: Trajectory, ref: Trajectory, mode: str = "se3",
        max_gap_s: float = 0.2, n_first: int | None = None,
        reference_uncertainty_m: float | None = None) -> AteResult:
    """Absolute trajectory error, reference interpolated onto the estimate's stamps.

    This direction is deliberate: the estimate's stamps are the sensor's own,
    and resampling the estimate would smooth exactly the high-frequency error a
    tracking failure shows up as.
    """
    if len(est) < 3:
        raise ValueError(f"{est.name or 'estimate'}: {len(est)} poses is not a trajectory")
    n_est = len(est)
    ref_i = ref.interpolate(est.stamps, max_gap_s=max_gap_s)
    est_i = est.subset(np.searchsorted(est.stamps, ref_i.stamps))
    if len(est_i) != len(ref_i):
        raise ValueError("interpolation returned stamps not present in the estimate")

    al = fit(est_i, ref_i, mode=mode, n_first=n_first)
    est_a = al.apply(est_i)

    e_t = np.linalg.norm(est_a.positions - ref_i.positions, axis=1)
    e_r = np.array([np.degrees(se3.rotation_angle(
        ref_i.poses[i][:3, :3].T @ est_a.poses[i][:3, :3])) for i in range(len(ref_i))])

    L = est_a.path_length()
    res = AteResult(
        trans=ErrorStats.of(e_t), rot=ErrorStats.of(e_r), alignment=al,
        n_ref=len(ref), n_est=n_est, coverage=len(ref_i) / n_est,
        path_length_m=L, duration_s=est_a.duration,
        drift_percent=float(100.0 * ErrorStats.of(e_t).rmse / L) if L > 1e-6 else float("nan"),
        ref_duration_s=float(ref.duration),
    )
    if reference_uncertainty_m is not None:
        res.below_reference_uncertainty = bool(res.trans.rmse < reference_uncertainty_m)
    return res


def rpe(est: Trajectory, ref: Trajectory, delta: float, unit: str = "m",
        max_gap_s: float = 0.2) -> RpeResult:
    """Relative pose error over a fixed sub-path length or time.

    Scored on *relative* motion, so it needs no alignment and is immune to the
    reference's own global anchoring error. On a sequence whose ATE sits near
    the reference's uncertainty, this is the metric that still separates
    methods: the reference is far more trustworthy over 1 m than over 16 m.
    """
    if unit not in ("m", "s"):
        raise ValueError(f"RPE unit {unit!r} must be 'm' or 's'")
    ref_i = ref.interpolate(est.stamps, max_gap_s=max_gap_s)
    est_i = est.subset(np.searchsorted(est.stamps, ref_i.stamps))

    axis = est_i.arc_length() if unit == "m" else est_i.stamps - est_i.stamps[0]
    j = np.searchsorted(axis, axis + delta, side="left")
    pairs = [(i, int(j[i])) for i in range(len(axis)) if j[i] < len(axis)]
    if not pairs:
        raise ValueError(f"no pose pair spans {delta} {unit} "
                         f"(the trajectory covers {axis[-1]:.2f} {unit})")

    e_t, e_r = [], []
    for a, b in pairs:
        de = np.linalg.inv(est_i.poses[a]) @ est_i.poses[b]
        dr = np.linalg.inv(ref_i.poses[a]) @ ref_i.poses[b]
        E = np.linalg.inv(dr) @ de
        e_t.append(np.linalg.norm(E[:3, 3]))
        e_r.append(np.degrees(se3.rotation_angle(E)))
    return RpeResult(delta, unit, ErrorStats.of(np.array(e_t)),
                     ErrorStats.of(np.array(e_r)), len(pairs))


def absolute_check(est: Trajectory, anchors: list[dict], radius_m: float = 0.5,
                   max_dt_s: float = 0.05, interpolate: bool = False,
                   ) -> list[dict]:
    """Tier 2: the estimate against surveyed board positions, in its own aligned
    frame.

    THE MEASUREMENT IS OF AN OBSERVATION, NOT OF PROXIMITY. The platform does
    not stand ON a board; it stands a metre or so away LOOKING at it. So the
    residual that means something is:

        board position according to the estimate  =  est_pose(t) @ p_board_cam(t)
        residual                                  =  || that  -  surveyed position ||

    where `p_board_cam` is the board's position in the camera frame at time t,
    which comes from detecting the board and solving PnP. That residual is a
    true absolute error and is comparable against the survey's 3-15 mm sigma.

    Each anchor is {name, position, uncertainty_m, window: [t0, t1]} plus ONE of:

      observed_poses: Trajectory
          camera poses in the MAP frame derived from board detection + PnP --
          coop2's mapping pipeline already emits these (zed_cam_in_map.tum,
          realsense_cam_in_map.tum: 112 and 92 per-frame rows at ~15 Hz, 1.98
          and 6.93 mm std). PREFERRED, because it is the pipeline's own
          detection rather than a reimplementation of its board conventions,
          and because it carries orientation too. Residual is the position
          difference at matched stamps, with the rotation reported beside it.
      observations: [{stamp, p_board_cam: [x, y, z]}, ...]
          board detections as a point in the camera frame. Equivalent, and what
          a fresh detector run would produce.
      standoff: [dx, dy, dz]
          the platform's surveyed offset from the board during the dwell, if the
          dwell was at a marked spot. A weaker substitute: it assumes the cart
          was parked exactly there.

    With NEITHER, this refuses. It used to compare the platform's mean position
    directly against the board's, which is the standoff distance (~0.7 m on
    coop2) wearing the units of an error, and it would have been declared
    "resolved above the survey's uncertainty" against a 7 mm sigma every single
    time -- a confident number that measures nothing. Rule 3: refuse rather
    than default.

    The estimate must already be in the reference world frame -- pass the output
    of `Alignment.apply`, not the raw method output.

    `max_dt_s` / `interpolate` exist for a KEYFRAME trajectory. MASt3R-SLAM
    emits ~38 poses over 156 s, so a board-derived pose at 15 Hz almost never
    lands within 50 ms of one and every anchor row came back empty -- a sparse
    method silently scoring nothing at the only independent tier. With
    `interpolate`, the estimate is evaluated BETWEEN its poses at the board's
    stamp, and each row reports `interp_gap_s`, the median distance to the
    nearest real pose, so the reader can judge how much of the residual is the
    interpolation. Straight-line between keyframes seconds apart is an
    approximation, and it is named in the row rather than hidden.
    """
    rows = []
    for a in anchors:
        name, u = a["name"], a.get("uncertainty_m")
        pos = np.asarray(a["position"], dtype=np.float64)
        window = a.get("window")
        if window is None:
            rows.append({"anchor": name, "n": 0, "residual_m": None,
                         "uncertainty_m": u,
                         "verdict": "no dwell window declared"})
            continue
        t0, t1 = window
        obs = a.get("observations")
        standoff = a.get("standoff")
        oposes = a.get("observed_poses")

        if a.get("observed_poses_error"):
            rows.append({"anchor": name, "n": 0, "residual_m": None,
                         "uncertainty_m": u,
                         "verdict": "observed_poses_by_agent names a file that "
                                    "could not be read -- " +
                                    str(a["observed_poses_error"])})
            continue

        if oposes is not None and len(oposes):
            m = (oposes.stamps >= t0) & (oposes.stamps <= t1)
            if not m.any():
                # Say BOTH spans. mobile_2's window missed its detections by
                # 100 ms and the bare message sent nobody to the cause, so the
                # only agent with no reference of its own scored nothing at the
                # only tier that could have spoken for it.
                rows.append({"anchor": name, "n": 0, "residual_m": None,
                             "uncertainty_m": u,
                             "observed_span": [float(oposes.stamps[0]),
                                               float(oposes.stamps[-1])],
                             "window": [float(t0), float(t1)],
                             "verdict": f"no board-derived pose inside the window: the "
                                        f"detections span {oposes.stamps[0]:.3f}.."
                                        f"{oposes.stamps[-1]:.3f} and the window is "
                                        f"{t0:.3f}..{t1:.3f} — they do not overlap, so "
                                        f"one of the two is wrong in the dataset config"})
                continue
            dp, dr, gaps = [], [], []
            for k in np.flatnonzero(m):
                t = float(oposes.stamps[k])
                j = int(np.argmin(np.abs(est.stamps - t)))
                gap = float(abs(est.stamps[j] - t))
                if gap <= max_dt_s:
                    P = est.poses[j]
                elif interpolate and est.stamps[0] <= t <= est.stamps[-1]:
                    got = est.interpolate(np.array([t]), max_gap_s=np.inf)
                    if not len(got):
                        continue
                    P = got.poses[0]
                else:
                    continue
                gaps.append(gap)
                dp.append(np.linalg.norm(P[:3, 3] - oposes.positions[k]))
                dr.append(np.degrees(se3.rotation_angle(
                    P[:3, :3].T @ oposes.poses[k][:3, :3])))
            if not dp:
                rows.append({"anchor": name, "n": 0, "residual_m": None,
                             "uncertainty_m": u,
                             "verdict": f"board-derived poses in the window matched "
                                        f"no estimate pose within {max_dt_s*1e3:.0f} ms"
                                        + ("" if interpolate else
                                           " (a sparse trajectory needs interpolate)")
                                        + ("; the window lies outside the estimate's span"
                                           if interpolate else "")})
                continue
            dp = np.asarray(dp)
            d = float(np.median(dp))
            spread = float(dp.std())
            row = {
                "anchor": name, "n": len(dp), "residual_m": d, "uncertainty_m": u,
                "spread_m": spread, "source": "observed_poses",
                "residual_deg": float(np.median(dr)),
                "verdict": ("indistinguishable from the survey" if u and d <= u
                            else "resolved above the survey's uncertainty" if u
                            else "no uncertainty declared for this anchor"),
            }
            if interpolate and max(gaps) > max_dt_s:
                row["interp_gap_s"] = float(np.median(gaps))
            rows.append(row)
            continue

        if obs:
            # est_pose(t) @ p_board_cam(t), one estimate of the board per detection
            est_pts = []
            for o in obs:
                t = float(o["stamp"])
                if not (t0 <= t <= t1):
                    continue
                k = int(np.argmin(np.abs(est.stamps - t)))
                if abs(est.stamps[k] - t) > 0.05:
                    continue
                T = est.poses[k]
                est_pts.append(T[:3, :3] @ np.asarray(o["p_board_cam"],
                                                      dtype=np.float64) + T[:3, 3])
            if not est_pts:
                rows.append({"anchor": name, "n": 0, "residual_m": None,
                             "uncertainty_m": u,
                             "verdict": "no board observation inside the window "
                                        "matched an estimate pose"})
                continue
            est_pts = np.asarray(est_pts)
            p = est_pts.mean(axis=0)
            d = float(np.linalg.norm(p - pos))
            spread = float(np.linalg.norm(est_pts.std(axis=0)))
            source = "observed"
        elif standoff is not None:
            m = (est.stamps >= t0) & (est.stamps <= t1)
            if not m.any():
                rows.append({"anchor": name, "n": 0, "residual_m": None,
                             "uncertainty_m": u,
                             "verdict": "no estimate pose in the dwell window"})
                continue
            p = est.positions[m].mean(axis=0)
            d = float(np.linalg.norm(p - (pos + np.asarray(standoff,
                                                           dtype=np.float64))))
            spread = float(np.linalg.norm(est.positions[m].std(axis=0)))
            source = "standoff"
        else:
            why = a.get("observations_impossible")
            rows.append({
                "anchor": name, "n": 0, "residual_m": None, "uncertainty_m": u,
                "verdict": (f"NO EVIDENCE POSSIBLE HERE: {why} This board cannot "
                            f"enter tier 2 for this agent; it is not waiting on a "
                            f"detector run." if why else
                            "REFUSED: a window alone cannot make an absolute check. "
                            "The platform stands off from the board, so comparing "
                            "its position against the board's measures the standoff, "
                            "not the error. Supply `observed_poses` (the pipeline's "
                            "*_cam_in_map.tum), `observations` (board detections + "
                            "PnP) or a surveyed `standoff`.")})
            continue

        n = len(est_pts) if obs else int(m.sum())
        rows.append({
            "anchor": name, "n": n, "residual_m": d, "uncertainty_m": u,
            "spread_m": spread, "source": source,
            "verdict": ("indistinguishable from the survey" if u and d <= u
                        else "resolved above the survey's uncertainty" if u
                        else "no uncertainty declared for this anchor"),
        })
    return rows


def revisit_residual(est: Trajectory, radius_m: float = 0.30,
                     min_separation_s: float = 20.0) -> dict:
    """Tier 3: how far the estimate drifts between two passes over one place.

    A revisit is a pose pair close in space and far apart in time. With no
    reference at all, the gap the estimate leaves between the two passes bounds
    accumulated drift from below.

    Two things this cannot do, and pretending otherwise turns a sampling
    artefact into a drift number:

      * The matched partner must be an INTERIOR closest approach — a strict
        local minimum of distance inside the far window. A partner clamped at
        the end of the trajectory is not a revisit; it is the trajectory running
        out, and its "gap" is however far the last pose fell short. On a
        single-lap route that alone produced tens of spurious pairs at 18 cm.
      * Even an interior match is a pair of SAMPLES, not the same point. The gap
        it reports has a floor at roughly the inter-pose spacing, which is
        returned as `sampling_floor_m`; a residual at that scale is the sampling
        grid, not drift.

    Confirming a revisit properly means registering the two passes' sensor data
    and reading the residual transform, which needs the clouds and not just the
    poses. This function finds the candidates and bounds them; it does not close
    the loop.
    """
    P, t = est.positions, est.stamps
    floor = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0
    pairs, clamped = [], 0
    for i in range(len(P)):
        far_idx = np.nonzero(t > t[i] + min_separation_s)[0]
        if len(far_idx) < 3:
            continue
        d = np.linalg.norm(P[far_idx] - P[i], axis=1)
        k = int(np.argmin(d))
        if k == 0 or k == len(d) - 1:
            clamped += 1                       # the window ran out, not a revisit
            continue
        if d[k] < radius_m:
            pairs.append((i, int(far_idx[k]), float(d[k])))
    if not pairs:
        return {"n_revisits": 0, "sampling_floor_m": floor, "n_clamped": clamped,
                "note": f"no interior closest approach within {radius_m} m and "
                        f"{min_separation_s} s apart. Either the route does not "
                        f"revisit, in which case this tier is silent and loop "
                        f"closure cannot be evaluated on this sequence at all, or "
                        f"{clamped} candidate(s) were rejected as end-of-trajectory "
                        f"clamping rather than counted as drift."}
    d = np.array([p[2] for p in pairs])
    stats = ErrorStats.of(d).as_dict()
    return {"n_revisits": len(pairs), "residual_m": stats, "sampling_floor_m": floor,
            "n_clamped": clamped,
            "informative": bool(stats["median"] > 3 * floor),
            "note": ("lower bound on drift, unconfirmed at scan level"
                     if stats["median"] > 3 * floor else
                     f"residual is within 3x the {floor*1e3:.0f} mm inter-pose "
                     f"spacing — this is the sampling grid, not measured drift")}
