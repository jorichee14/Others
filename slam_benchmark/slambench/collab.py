"""Collaborative SLAM and infrastructure-anchored localization.

The metric that matters for two agents is NOT each one's ATE. Two trajectories
can each be excellent and still be useless together if the transform between
them is wrong — and on coop2 the two agents are in one frame only because the
pseudo ground truth's anchoring put them there. A collaborative system has to
ESTIMATE that transform. So the primary quantity here is frame-free:

    T_ab(t) = inv(T_wa(t)) @ T_wb(t)          b's pose in a's own frame

computed from the estimate and from the reference and compared directly. No
alignment, no common world, nothing borrowed from the pseudo ground truth
except the answer it is being scored against. A method that puts both agents in
its own arbitrary frame is scored exactly as well as one that happens to land
in `map`.

That error splits, and the split is the result rather than a detail:

    constant part    the two agents' frames are mis-registered by a fixed
                     transform -> a map-merge / place-recognition failure
    varying part     the frames are right and the relative geometry drifts
                     -> a tracking failure, in one agent or both

A single number hides which one happened, and they have different fixes.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import se3
from .align import _umeyama
from .traj_metrics import ErrorStats
from .trajectory import Trajectory

__all__ = ["RelativeResult", "relative_trajectory_error", "agent_separation",
           "place_overlap", "completeness_gain", "predict_observation",
           "observation_residual"]


@dataclasses.dataclass
class RelativeResult:
    n: int
    trans: ErrorStats              # metres, error in T_ab
    rot: ErrorStats                # degrees
    trans_after_constant: ErrorStats   # residual once a fixed mis-registration
    rot_after_constant: ErrorStats     # is removed -> the drift component
    constant_trans_m: float        # size of that fixed part -> the merge error
    constant_rot_deg: float
    separation_m: dict             # how far apart the agents were, for context
    time_span_s: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "rel_trans_m": self.trans.as_dict(),
            "rel_rot_deg": self.rot.as_dict(),
            "rel_trans_after_constant_m": self.trans_after_constant.as_dict(),
            "rel_rot_after_constant_deg": self.rot_after_constant.as_dict(),
            "constant_trans_m": self.constant_trans_m,
            "constant_rot_deg": self.constant_rot_deg,
            "separation_m": self.separation_m,
            "time_span_s": self.time_span_s,
            "verdict": self.verdict(),
        }

    def verdict(self) -> str:
        """Which failure this is, in words, from the two components."""
        c, d = self.constant_trans_m, self.trans_after_constant.rmse
        if c < 0.05 and d < 0.05:
            return "frames registered and relative geometry holds"
        if c >= 0.05 and d < 0.5 * c:
            return ("a fixed mis-registration dominates: place recognition or "
                    "map merging put the two frames in the wrong relative pose, "
                    "and the tracking is comparatively fine")
        if d >= 0.05 and c < 0.5 * d:
            return ("the frames are approximately right and the relative "
                    "geometry drifts: a tracking failure, not a merge failure")
        return "both a mis-registration and relative drift"


def _common(a: Trajectory, b: Trajectory, stamps: np.ndarray | None,
            max_gap_s: float) -> tuple[Trajectory, Trajectory]:
    if stamps is None:
        lo = max(a.stamps[0], b.stamps[0])
        hi = min(a.stamps[-1], b.stamps[-1])
        if hi <= lo:
            raise ValueError(
                f"{a.name or 'a'} spans {a.stamps[0]:.3f}..{a.stamps[-1]:.3f} and "
                f"{b.name or 'b'} spans {b.stamps[0]:.3f}..{b.stamps[-1]:.3f} — "
                f"they do not overlap in time, so there is no relative pose to "
                f"score. Two agents on different clocks look exactly like this.")
        stamps = a.stamps[(a.stamps >= lo) & (a.stamps <= hi)]
    ai = a.interpolate(stamps, max_gap_s=max_gap_s)
    bi = b.interpolate(ai.stamps, max_gap_s=max_gap_s)
    ai = ai.interpolate(bi.stamps, max_gap_s=max_gap_s)
    return ai, bi


def _rel(a: Trajectory, b: Trajectory) -> np.ndarray:
    """T_ab(t) for paired trajectories."""
    return np.stack([se3.invert(a.poses[i]) @ b.poses[i] for i in range(len(a))])


def relative_trajectory_error(est_a: Trajectory, est_b: Trajectory,
                              ref_a: Trajectory, ref_b: Trajectory,
                              max_gap_s: float = 0.25) -> RelativeResult:
    """Error of the inter-agent transform, and what kind of error it is.

    Both estimates must be in ONE frame (whichever the collaborative system
    chose); both references in one frame. Nothing requires those two frames to
    be the same — that is the point, and it is why the headline number needs no
    alignment at all.

    Two quantities, and they answer different questions:

    `trans` / `rot` — gauge-free. The error in T_ab(t) directly. This is what a
    downstream consumer of the partner's pose actually suffers, and it borrows
    nothing from the reference but the answer.

    `constant_*` / `*_after_constant` — the DECOMPOSITION, which needs a gauge,
    so one is fixed: the estimate is aligned on agent A, and agent B's residual
    against its own reference is then split by a best-fit rigid transform. That
    split is meaningful because a map-merge failure IS a fixed transform between
    the two agents' frames, while drift is not.

    One property of the split to know before quoting it: a MONOTONIC drift has a
    non-zero mean, so a best-fit constant absorbs roughly half of it. A large
    constant term with a comparable residual is therefore consistent with either
    failure, and `verdict()` says so rather than picking one.
    """
    ea, eb = _common(est_a, est_b, None, max_gap_s)
    ra, rb = _common(ref_a, ref_b, ea.stamps, max_gap_s)
    ea, eb = _common(ea, eb, ra.stamps, max_gap_s)
    n = min(len(ea), len(eb), len(ra), len(rb))
    if n < 3:
        raise ValueError(f"only {n} shared stamps across both agents and the "
                         f"reference — not enough to score a relative trajectory")
    ea, eb, ra, rb = (t.subset(np.arange(n)) for t in (ea, eb, ra, rb))

    # ---- gauge-free: the error in b's pose as seen from a, at every instant
    Te, Tr = _rel(ea, eb), _rel(ra, rb)
    E = np.stack([se3.invert(Tr[i]) @ Te[i] for i in range(n)])
    e_t = np.linalg.norm(E[:, :3, 3], axis=1)
    e_r = np.array([np.degrees(se3.rotation_angle(E[i])) for i in range(n)])

    # ---- decomposition: fix the gauge on A, then split B's residual.
    # A fixed mis-registration of the two map frames is a constant transform in
    # the WORLD, and a world-constant transform is NOT constant in the moving
    # frame T_ab — conjugating it by a rotating agent A makes it look like
    # drift. So the split has to be done in a static frame, which is what
    # aligning on A provides.
    Ga, _, _ = _umeyama(ea.positions, ra.positions, with_scale=False)
    G = np.eye(4)
    G[:3, :3] = Ga
    G[:3, 3] = ra.positions.mean(0) - Ga @ ea.positions.mean(0)
    eb_w = eb.transform_left(G)

    Rd, td, _ = _umeyama(eb_w.positions, rb.positions, with_scale=False)
    D = np.eye(4)
    D[:3, :3] = Rd
    D[:3, 3] = td
    eb_d = eb_w.transform_left(D)
    d_t = np.linalg.norm(eb_d.positions - rb.positions, axis=1)
    d_r = np.array([np.degrees(se3.rotation_angle(
        rb.poses[i][:3, :3].T @ eb_d.poses[i][:3, :3])) for i in range(n)])

    sep = np.linalg.norm(Tr[:, :3, 3], axis=1)
    return RelativeResult(
        n=n, trans=ErrorStats.of(e_t), rot=ErrorStats.of(e_r),
        trans_after_constant=ErrorStats.of(d_t), rot_after_constant=ErrorStats.of(d_r),
        constant_trans_m=float(np.linalg.norm(D[:3, 3])),
        constant_rot_deg=float(np.degrees(se3.rotation_angle(D))),
        separation_m={"min": float(sep.min()), "median": float(np.median(sep)),
                      "max": float(sep.max())},
        time_span_s=float(ea.stamps[-1] - ea.stamps[0]))


def agent_separation(ref_a: Trajectory, ref_b: Trajectory,
                     max_gap_s: float = 0.25) -> dict:
    """How far apart the two agents were, over the time both were recording."""
    a, b = _common(ref_a, ref_b, None, max_gap_s)
    n = min(len(a), len(b))
    d = np.linalg.norm(a.positions[:n] - b.positions[:n], axis=1)
    return {"n": int(n), "min_m": float(d.min()), "median_m": float(np.median(d)),
            "max_m": float(d.max()),
            "time_overlap_s": float(a.stamps[n - 1] - a.stamps[0])}


def place_overlap(ref_a: Trajectory, ref_b: Trajectory, radius_m: float = 5.0,
                  ) -> dict:
    """Whether the two agents ever occupied the same PLACE, at any time.

    This, not the time-synchronised separation, is what decides whether an
    inter-robot loop closure is possible at all: two agents that never stand
    where the other stood share no observed surface for place recognition to
    match, and every collaborative method reduces to its single-agent behaviour
    plus a guess. Answerable from the reference trajectories alone, before a
    single SLAM system is run — and if the answer is no, that is a finding
    about the RECORDING, to fix in the next session rather than to discover
    from six flat result rows.

    `radius_m` should be the smaller agent's usable sensing range, not an
    arbitrary distance: overlap only counts where the weaker sensor could
    actually have seen the shared place.
    """
    A, B = ref_a.positions, ref_b.positions
    dmin = np.array([np.min(np.linalg.norm(A - p, axis=1)) for p in B])
    near = dmin <= radius_m
    out = {"radius_m": radius_m,
           "fraction_of_b_near_a": float(np.mean(near)),
           "closest_approach_m": float(dmin.min()),
           "n_b_poses": int(len(B))}
    if not near.any():
        out["verdict"] = (f"the agents never came within {radius_m} m of each "
                          f"other's path (closest {dmin.min():.1f} m). No shared "
                          f"geometry, so inter-robot place recognition cannot "
                          f"succeed and a collaborative result on this sequence "
                          f"would measure the fallback, not the collaboration.")
        return out
    j = np.flatnonzero(near)
    k = np.array([int(np.argmin(np.linalg.norm(A - B[i], axis=1))) for i in j])
    dt = np.abs(ref_b.stamps[j] - ref_a.stamps[k])
    out["time_offset_s"] = {"min": float(dt.min()), "median": float(np.median(dt)),
                            "max": float(dt.max())}
    out["verdict"] = (f"{100 * out['fraction_of_b_near_a']:.0f}% of b's path came "
                      f"within {radius_m} m of a's, closest {dmin.min():.2f} m — "
                      f"inter-robot loop closure is geometrically possible")
    return out


def completeness_gain(solo: dict, joint: dict) -> dict:
    """What the partner added to the map, from two `evaluate_map` results.

    Reported as the gain, not as two separate scores, because the single-agent
    number is the only thing that makes the collaborative one interpretable: a
    joint map that scores 0.8 where the ego alone scored 0.79 has not
    collaborated, whatever the absolute figure looks like.
    """
    out = {}
    for th in sorted(set(solo["fscore"]) & set(joint["fscore"])):
        s, j = solo["fscore"][th], joint["fscore"][th]
        out[th] = {"recall_solo": s["recall"], "recall_joint": j["recall"],
                   "recall_gain": j["recall"] - s["recall"],
                   "precision_solo": s["precision"], "precision_joint": j["precision"],
                   "precision_change": j["precision"] - s["precision"],
                   "f_solo": s["f"], "f_joint": j["f"], "f_gain": j["f"] - s["f"]}
    out["chamfer_solo_m"] = solo["chamfer_m"]
    out["chamfer_joint_m"] = joint["chamfer_m"]
    out["completeness_gain_m"] = (solo["completeness_m"]["mean"]
                                  - joint["completeness_m"]["mean"])
    return out


def predict_observation(traj: Trajectory, infra_pose: np.ndarray,
                        target_offset=(0.0, 0.0, 0.0), axes: str = "ros") -> dict:
    """Where a static surveyed node should see an agent, per pose.

    `infra_pose` is `map <- infra sensor frame`, 4x4. Returns range, azimuth
    and elevation of the agent IN THAT SENSOR'S FRAME, which is what a radar
    detection or an image bearing is directly comparable against.

    `axes` is not a detail and has no safe default for this rig:

        "ros"       x forward, y left, z up — a body frame such as a radar link
        "optical"   z forward, x right, y DOWN — every camera optical frame

    On coop2 the infrastructure node's `static_world_pose` describes the Arducam
    OPTICAL frame, so "ros" there puts forward along the optical x axis (which
    points right) and reads elevation off the axis that points down. The result
    is a full set of finite, plausible bearings that are wrong by 90 degrees —
    a node 1.8 m ABOVE the agents reports them at +70 degrees elevation instead
    of below it, and nothing downstream complains. Pass the frame the pose
    actually describes.

    This is the only continuous evidence on coop2 that comes from neither the
    pseudo ground truth nor the method: a surveyed node, on its own clock,
    watching the agents from outside. Its value is bounded by the node's own
    pose uncertainty, which coop2 does not state — so measure it before
    quoting any residual from this, or the tier inherits an unknown bias.
    """
    T = se3.invert(np.asarray(infra_pose, dtype=np.float64))
    p = traj.positions + np.asarray(target_offset, dtype=np.float64)
    q = p @ T[:3, :3].T + T[:3, 3]
    rng = np.linalg.norm(q, axis=1)
    if axes == "optical":
        fwd, right, upward = q[:, 2], q[:, 0], -q[:, 1]
    elif axes == "ros":
        fwd, right, upward = q[:, 0], -q[:, 1], q[:, 2]
    else:
        raise ValueError(f"axes {axes!r} must be 'ros' or 'optical'")
    return {"stamps": traj.stamps, "range_m": rng, "axes": axes,
            "azimuth_deg": np.degrees(np.arctan2(-right, fwd)),
            "elevation_deg": np.degrees(np.arctan2(upward, np.hypot(fwd, right))),
            "xyz_infra": q}


def observation_residual(predicted: dict, observed_stamps: np.ndarray,
                         observed_azimuth_deg: np.ndarray,
                         observed_range_m: np.ndarray | None = None,
                         max_dt_s: float = 0.1) -> dict:
    """Predicted-minus-observed bearing (and range) at matched stamps.

    Bearing is reported separately from range because they fail differently: a
    bearing residual with a consistent sign is a pose or extrinsic error in the
    infrastructure node, while a range residual that grows with distance is the
    agent's trajectory drifting away along the line of sight — the direction a
    single static observer constrains worst.
    """
    ps, pa = np.asarray(predicted["stamps"]), np.asarray(predicted["azimuth_deg"])
    os_ = np.asarray(observed_stamps, dtype=np.float64)
    j = np.clip(np.searchsorted(ps, os_), 1, len(ps) - 1)
    pick = np.where(np.abs(ps[j - 1] - os_) <= np.abs(ps[j] - os_), j - 1, j)
    ok = np.abs(ps[pick] - os_) <= max_dt_s
    if not ok.any():
        raise ValueError(f"no infrastructure detection within {max_dt_s * 1e3:.0f} ms "
                         f"of a predicted pose — check the node's clock offset first")
    d_az = (np.asarray(observed_azimuth_deg)[ok] - pa[pick[ok]] + 180.0) % 360.0 - 180.0
    out = {"n": int(ok.sum()), "azimuth_deg": ErrorStats.of(np.abs(d_az)).as_dict(),
           "azimuth_bias_deg": float(np.mean(d_az))}
    if observed_range_m is not None:
        pr = np.asarray(predicted["range_m"])
        d_r = np.asarray(observed_range_m)[ok] - pr[pick[ok]]
        out["range_m"] = ErrorStats.of(np.abs(d_r)).as_dict()
        out["range_bias_m"] = float(np.mean(d_r))
    return out
