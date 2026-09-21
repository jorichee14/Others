"""The construction's pose graph. Design: docs/STAGE2_GRAPH.md.

One trajectory, N poses (R_k, t_k), two kinds of factor:

    between  k -> k+1   the backbone's own relative motion, in frame k
    prior    k          an absolute pose, from a surveyed board seen at k

and a Gauss-Newton / Levenberg-Marquardt solve over all of them at once.

What this is NOT: a general sparse pose-graph library. Every between factor
joins CONSECUTIVE poses and every prior touches one pose, so the normal
equations are block-tridiagonal with 6x6 blocks and are solved exactly by a
block Thomas sweep in O(N) -- pure numpy, no scipy, no GTSAM. `solve` asserts
that structure rather than silently handling a graph it was not written for;
an inter-agent edge (V2e) is a different solver, not a flag.

Residuals, with the pose perturbed as R <- R exp(phi), t <- t + rho:

    between   r_t = R_i^T (t_j - t_i) - m_t        m = measured (R, t), frame i
              r_R = log(m_R^T R_i^T R_j)
    prior     r_t = t_i - m_t                       m in the world frame
              r_R = log(m_R^T R_i)

i.e. SO(3) x R^3, not SE(3)'s coupled exponential. The optimum is defined by
the residual, not the chart, and this residual is the one whose covariance
means something measured: sigma_t is relative translation error per edge in
the same units tier-1 RPE already reports, sigma_R the relative rotation.

Jacobians are NUMERICAL (central differences), batched over all factors of a
kind. Twelve residual evaluations per between factor per iteration is nothing
at this size, and it removes the one place a hand-derived Jacobian can be
wrong while the cost still goes down. The synthetic tests recover a planted
truth through the whole pipeline; that is the check.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["so3_exp", "so3_log", "Factors", "SolveReport", "solve",
           "odometry_factors", "anchor_factors", "gauge_prior", "reset_edges",
           "anchor_coverage", "move_by_region", "sampling_regularity"]


# ----------------------------------------------------------------- SO(3), batched
def so3_exp(phi: np.ndarray) -> np.ndarray:
    """(K,3) rotation vectors -> (K,3,3). Rodrigues, series below 1e-6 rad."""
    phi = np.asarray(phi, dtype=np.float64).reshape(-1, 3)
    th = np.linalg.norm(phi, axis=1)
    K = np.zeros((len(phi), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -phi[:, 2], phi[:, 1]
    K[:, 1, 0], K[:, 1, 2] = phi[:, 2], -phi[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -phi[:, 1], phi[:, 0]
    small = th < 1e-6
    a = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(th) / np.where(small, 1.0, th))
    b = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(th)) / np.where(small, 1.0, th ** 2))
    return np.eye(3)[None] + a[:, None, None] * K + b[:, None, None] * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """(K,3,3) -> (K,3) rotation vectors, via the quaternion.

    Not arccos((tr-1)/2): that loses half its digits near identity, which is
    exactly where every residual in a converged graph sits. Shepperd's method
    picks the numerically largest quaternion component per row.
    """
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    tr = np.trace(R, axis1=1, axis2=2)
    d = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    pick = np.argmax(d, axis=1)
    q = np.zeros((len(R), 4))                       # x, y, z, w
    for c in range(4):
        m = pick == c
        if not m.any():
            continue
        r = R[m]
        if c == 0:
            s = np.sqrt(1.0 + tr[m]) * 2.0
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s, 0.25 * s], axis=1)
        elif c == 1:
            s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2.0
            q[m] = np.stack([0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 2, 1] - r[:, 1, 2]) / s], axis=1)
        elif c == 2:
            s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2.0
            q[m] = np.stack([(r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s], axis=1)
        else:
            s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2.0
            q[m] = np.stack([(r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s,
                             0.25 * s, (r[:, 1, 0] - r[:, 0, 1]) / s], axis=1)
    q[q[:, 3] < 0] *= -1.0
    v = np.linalg.norm(q[:, :3], axis=1)
    ang = 2.0 * np.arctan2(v, q[:, 3])
    scale = np.where(v < 1e-12, 2.0, ang / np.where(v < 1e-12, 1.0, v))
    return q[:, :3] * scale[:, None]


# --------------------------------------------------------------------- factors
@dataclasses.dataclass
class Factors:
    """Every factor of the graph, columnar. `between` rows join pose i to i+1.

    sigma_* are per-factor (K,6): three translation, three rotation. Diagonal
    covariance is all this construction declares -- a survey states a scatter,
    not a correlation -- so nothing here pretends to more.
    """
    # between factors: pose i -> i+1, measured relative pose in frame i
    b_i: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=int))
    b_R: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 3, 3)))
    b_t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 3)))
    b_sigma: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 6)))
    b_tag: list = dataclasses.field(default_factory=list)
    # Huber scale per factor, on the WHITENED residual norm. inf = plain
    # least squares, which is what a single trusted source wants. With several
    # absolute sources the outliers become real -- a flipped PnP solution, a
    # mis-associated radar return, an ICP basin error -- and an unkerneled
    # outlier pulls quadratically.
    b_huber: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    # prior factors: pose i, measured absolute pose in the world frame
    p_i: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=int))
    p_R: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 3, 3)))
    p_t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 3)))
    p_sigma: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 6)))
    p_tag: list = dataclasses.field(default_factory=list)
    # SOURCE KIND per prior, e.g. "board", "visloc", "infra", "peer", "map".
    # Not the same as the tag, which names the instance ("board/anchor"):
    # leave-one-SOURCE-out holds out a kind, because a bias shared by every
    # instance of one kind is exactly what leave-one-instance-out cannot see.
    p_src: list = dataclasses.field(default_factory=list)
    p_huber: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))

    def add_between(self, i, R, t, sigma, tag="", huber=np.inf):
        self.b_i = np.append(self.b_i, int(i))
        self.b_R = np.concatenate([self.b_R, np.asarray(R).reshape(1, 3, 3)])
        self.b_t = np.concatenate([self.b_t, np.asarray(t).reshape(1, 3)])
        self.b_sigma = np.concatenate([self.b_sigma, np.asarray(sigma).reshape(1, 6)])
        self.b_tag.append(tag)
        self.b_huber = np.append(self.b_huber, float(huber))

    def add_prior(self, i, R, t, sigma, tag="", src="", huber=np.inf):
        self.p_i = np.append(self.p_i, int(i))
        self.p_R = np.concatenate([self.p_R, np.asarray(R).reshape(1, 3, 3)])
        self.p_t = np.concatenate([self.p_t, np.asarray(t).reshape(1, 3)])
        self.p_sigma = np.concatenate([self.p_sigma, np.asarray(sigma).reshape(1, 6)])
        self.p_tag.append(tag)
        self.p_src.append(src or (tag.split("/")[0] if tag else ""))
        self.p_huber = np.append(self.p_huber, float(huber))

    def __post_init__(self):
        """Backfill the columns added after the first version of this class.

        Factors built positionally by older callers (and by `odometry_factors`,
        which assigns the arrays directly) arrive without `*_huber` or `p_src`.
        Defaulting them here rather than at every use site keeps one definition
        of "no kernel declared" instead of several."""
        if len(self.b_huber) != len(self.b_i):
            self.b_huber = np.full(len(self.b_i), np.inf)
        if len(self.p_huber) != len(self.p_i):
            self.p_huber = np.full(len(self.p_i), np.inf)
        if len(self.p_src) != len(self.p_i):
            self.p_src = [t.split("/")[0] if t else "" for t in self.p_tag] \
                if len(self.p_tag) == len(self.p_i) else [""] * len(self.p_i)

    def sources(self) -> list:
        """The distinct absolute source KINDS present, in first-seen order."""
        out = []
        for k in self.p_src:
            if k and k not in out:
                out.append(k)
        return out

    def without(self, src: str) -> "Factors":
        """A copy with every prior of source kind `src` removed.

        This is leave-one-source-out. The odometry is untouched: holding out a
        source must change what anchors the trajectory, not what shapes it."""
        keep = np.array([k != src for k in self.p_src], dtype=bool)
        return Factors(
            self.b_i, self.b_R, self.b_t, self.b_sigma, list(self.b_tag), self.b_huber,
            self.p_i[keep], self.p_R[keep], self.p_t[keep], self.p_sigma[keep],
            [t for t, k in zip(self.p_tag, keep) if k],
            [t for t, k in zip(self.p_src, keep) if k],
            self.p_huber[keep])

    def only(self, src: str) -> "Factors":
        """A copy with ONLY the priors of source kind `src`, and no between
        factors. Used to evaluate a held-out source's residuals at a solution
        it did not contribute to."""
        keep = np.array([k == src for k in self.p_src], dtype=bool)
        return Factors(
            p_i=self.p_i[keep], p_R=self.p_R[keep], p_t=self.p_t[keep],
            p_sigma=self.p_sigma[keep],
            p_tag=[t for t, k in zip(self.p_tag, keep) if k],
            p_src=[t for t, k in zip(self.p_src, keep) if k],
            p_huber=self.p_huber[keep])

    def extend(self, other: "Factors") -> "Factors":
        return Factors(
            np.concatenate([self.b_i, other.b_i]), np.concatenate([self.b_R, other.b_R]),
            np.concatenate([self.b_t, other.b_t]), np.concatenate([self.b_sigma, other.b_sigma]),
            self.b_tag + other.b_tag, np.concatenate([self.b_huber, other.b_huber]),
            np.concatenate([self.p_i, other.p_i]), np.concatenate([self.p_R, other.p_R]),
            np.concatenate([self.p_t, other.p_t]), np.concatenate([self.p_sigma, other.p_sigma]),
            self.p_tag + other.p_tag, self.p_src + other.p_src,
            np.concatenate([self.p_huber, other.p_huber]))

    def counts(self) -> dict:
        from collections import Counter
        return {"between": dict(Counter(self.b_tag)), "prior": dict(Counter(self.p_tag)),
                "by_source": dict(Counter(k for k in self.p_src if k))}


def _between_residual_g(Ri, ti, Rj, tj, mR, mt) -> np.ndarray:
    """Between residual on GATHERED endpoints (K,3,3)/(K,3) each."""
    Ri_T = np.transpose(Ri, (0, 2, 1))
    r_t = np.einsum("kij,kj->ki", Ri_T, tj - ti) - mt
    r_R = so3_log(np.transpose(mR, (0, 2, 1)) @ Ri_T @ Rj)
    return np.concatenate([r_t, r_R], axis=1)


def _prior_residual_g(Ri, ti, mR, mt) -> np.ndarray:
    r_t = ti - mt
    r_R = so3_log(np.transpose(mR, (0, 2, 1)) @ Ri)
    return np.concatenate([r_t, r_R], axis=1)


def _between_residual(R, t, f: Factors) -> np.ndarray:
    i, j = f.b_i, f.b_i + 1
    return _between_residual_g(R[i], t[i], R[j], t[j], f.b_R, f.b_t)


def _prior_residual(R, t, f: Factors) -> np.ndarray:
    return _prior_residual_g(R[f.p_i], t[f.p_i], f.p_R, f.p_t)


def _perturb(R, t, idx, delta):
    """Copy of (R, t) with poses[idx] moved by delta (K,6): rho then phi."""
    R2, t2 = R.copy(), t.copy()
    t2[idx] = t[idx] + delta[:, :3]
    R2[idx] = R[idx] @ so3_exp(delta[:, 3:])
    return R2, t2


def _jac_wrt(residual_fn, Rg, tg, eps=1e-6):
    """d residual / d delta for ONE gathered endpoint set, one 6x6 per factor.

    `residual_fn(Rg', tg')` must hold every other endpoint fixed. Gathering
    the endpoints first is what makes batching exact: in a chain, pose k is
    the i of factor k and the j of factor k-1, so perturbing "all the i's" in
    place would move both ends of every edge at once and fold the j-Jacobian
    into the i-Jacobian. That is not a small error; it makes the normal
    matrix singular.
    """
    K = len(Rg)
    J = np.zeros((K, 6, 6))
    for d in range(6):
        delta = np.zeros((K, 6)); delta[:, d] = eps
        Rp = Rg @ so3_exp(delta[:, 3:]); tp = tg + delta[:, :3]
        Rm = Rg @ so3_exp(-delta[:, 3:]); tm = tg - delta[:, :3]
        J[:, :, d] = (residual_fn(Rp, tp) - residual_fn(Rm, tm)) / (2.0 * eps)
    return J


# ---------------------------------------------------------------------- solver
@dataclasses.dataclass
class SolveReport:
    iterations: int
    converged: bool
    initial_cost: float
    final_cost: float
    initial_rms: dict            # whitened RMS per factor kind, before
    final_rms: dict              # ... and after
    max_step_m: float            # largest single-pose move over the solve
    max_step_deg: float
    n_poses: int
    n_between: int
    n_prior: int
    lam_final: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def huber_weight(rw: np.ndarray, c: np.ndarray) -> np.ndarray:
    """IRLS weight per factor for a Huber kernel on the WHITENED residual.

    `rw` is (K,6) whitened residuals, `c` the (K,) scale. Returns (K,) weights
    w = min(1, c/s) with s the residual norm. Applying sqrt(w) to both the
    whitened Jacobian rows and the whitened residual turns the next
    Gauss-Newton step into the Huber step exactly -- standard IRLS, and the
    reason the kernel costs no change to the solver below.

    c = inf gives w = 1 everywhere, i.e. plain least squares. That is the right
    default for a single trusted source; with several sources the outliers are
    real and an unkerneled one pulls quadratically.
    """
    s = np.linalg.norm(rw, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(s > c, np.divide(c, s, out=np.ones_like(s), where=s > 0), 1.0)
    return np.clip(np.nan_to_num(w, nan=1.0, posinf=1.0), 0.0, 1.0)


def _robust_cost(rw: np.ndarray, c: np.ndarray) -> float:
    """Huber loss summed over factors: s^2 below the scale, 2*c*s - c^2 above.

    The solver accepts a step only when THIS drops, so it must be the same
    objective the weights linearise -- using the plain sum of squares here
    while weighting the step would let an outlier veto a good step."""
    s = np.linalg.norm(rw, axis=1)
    quad = np.minimum(s, c)
    return float(np.sum(quad ** 2 + 2.0 * quad * np.maximum(s - c, 0.0)))


def _cost(R, t, f: Factors) -> tuple[float, dict]:
    parts = {}
    total = 0.0
    if len(f.b_i):
        rb = _between_residual(R, t, f) / f.b_sigma
        parts["between"] = float(np.sqrt(np.mean(rb ** 2)))
        total += _robust_cost(rb, f.b_huber)
    if len(f.p_i):
        rp = _prior_residual(R, t, f) / f.p_sigma
        parts["prior"] = float(np.sqrt(np.mean(rp ** 2)))
        total += _robust_cost(rp, f.p_huber)
        # per SOURCE KIND, because one source can be satisfied while another is
        # abused and a single pooled number hides exactly that.
        for k in f.sources():
            m = np.array([x == k for x in f.p_src], dtype=bool)
            parts[f"prior/{k}"] = float(np.sqrt(np.mean(rp[m] ** 2)))
    return total, parts


def _block_tridiag_solve(D: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve H x = b for H block-tridiagonal SPD with diagonal blocks D (N,6,6)
    and upper blocks U (N-1,6,6); the lower blocks are U^T. Block Thomas."""
    N = len(D)
    Dp = np.empty_like(D); bp = np.empty_like(b)
    Dp[0], bp[0] = D[0], b[0]
    for k in range(1, N):
        L = np.linalg.solve(Dp[k - 1].T, U[k - 1]).T        # U^T Dp^-1
        Dp[k] = D[k] - L @ U[k - 1]
        bp[k] = b[k] - L @ bp[k - 1]
    x = np.empty_like(b)
    x[N - 1] = np.linalg.solve(Dp[N - 1], bp[N - 1])
    for k in range(N - 2, -1, -1):
        x[k] = np.linalg.solve(Dp[k], bp[k] - U[k] @ x[k + 1])
    return x


def solve(poses0: np.ndarray, f: Factors, max_iter: int = 50, lam0: float = 1e-4,
          rel_tol: float = 1e-10, verbose: bool = False) -> tuple[np.ndarray, SolveReport]:
    """Levenberg-Marquardt over all poses. Returns (N,4,4) and a report.

    Refuses a graph it cannot solve exactly: between factors must join k and
    k+1 only, and at least one prior must exist or the gauge is free and the
    normal matrix singular. Both are the caller's design, not runtime state.
    """
    poses0 = np.asarray(poses0, dtype=np.float64).reshape(-1, 4, 4)
    N = len(poses0)
    if len(f.b_i) and (f.b_i.min() < 0 or f.b_i.max() + 1 >= N):
        raise ValueError("a between factor indexes a pose that does not exist")
    if len(f.p_i) == 0:
        raise ValueError("no prior factor: the gauge is free and the system is singular. "
                         "Add gauge_prior() for a backbone-only graph.")
    if len(f.p_i) and (f.p_i.min() < 0 or f.p_i.max() >= N):
        raise ValueError("a prior factor indexes a pose that does not exist")

    R = poses0[:, :3, :3].copy()
    t = poses0[:, :3, 3].copy()
    cost, rms0 = _cost(R, t, f)
    cost0 = cost
    lam = lam0
    it = 0
    converged = False
    max_step_m = max_step_deg = 0.0

    for it in range(1, max_iter + 1):
        D = np.zeros((N, 6, 6)); U = np.zeros((max(N - 1, 0), 6, 6)); b = np.zeros((N, 6))

        if len(f.b_i):
            rb = _between_residual(R, t, f)
            W = 1.0 / f.b_sigma                                   # (K,6)
            W = W * np.sqrt(huber_weight(rb * W, f.b_huber))[:, None]
            Ri, ti, Rj, tj = R[f.b_i], t[f.b_i], R[f.b_i + 1], t[f.b_i + 1]
            Ji = _jac_wrt(lambda Rg, tg: _between_residual_g(Rg, tg, Rj, tj, f.b_R, f.b_t), Ri, ti)
            Jj = _jac_wrt(lambda Rg, tg: _between_residual_g(Ri, ti, Rg, tg, f.b_R, f.b_t), Rj, tj)
            JiW = Ji * W[:, :, None]; JjW = Jj * W[:, :, None]   # rows whitened
            rw = rb * W
            np.add.at(D, f.b_i, np.einsum("kri,krj->kij", JiW, JiW))
            np.add.at(D, f.b_i + 1, np.einsum("kri,krj->kij", JjW, JjW))
            np.add.at(U, f.b_i, np.einsum("kri,krj->kij", JiW, JjW))
            np.add.at(b, f.b_i, -np.einsum("kri,kr->ki", JiW, rw))
            np.add.at(b, f.b_i + 1, -np.einsum("kri,kr->ki", JjW, rw))
        if len(f.p_i):
            rp = _prior_residual(R, t, f)
            W = 1.0 / f.p_sigma
            W = W * np.sqrt(huber_weight(rp * W, f.p_huber))[:, None]
            Rp_, tp_ = R[f.p_i], t[f.p_i]
            Jp = _jac_wrt(lambda Rg, tg: _prior_residual_g(Rg, tg, f.p_R, f.p_t), Rp_, tp_)
            JpW = Jp * W[:, :, None]
            rw = rp * W
            np.add.at(D, f.p_i, np.einsum("kri,krj->kij", JpW, JpW))
            np.add.at(b, f.p_i, -np.einsum("kri,kr->ki", JpW, rw))

        # LM: damp the diagonal, try the step, accept only if the cost drops.
        while True:
            Dd = D + lam * np.einsum("kii->ki", D)[:, :, None] * np.eye(6)[None]
            try:
                delta = _block_tridiag_solve(Dd, U, b)
            except np.linalg.LinAlgError:
                lam *= 10.0
                if lam > 1e12:
                    raise
                continue
            R_new, t_new = _perturb(R, t, np.arange(N), delta)
            cost_new, rms_new = _cost(R_new, t_new, f)
            if cost_new < cost:
                step_m = float(np.linalg.norm(delta[:, :3], axis=1).max())
                step_deg = float(np.degrees(np.linalg.norm(delta[:, 3:], axis=1).max()))
                max_step_m, max_step_deg = max(max_step_m, step_m), max(max_step_deg, step_deg)
                rel = (cost - cost_new) / max(cost, 1e-300)
                R, t, cost = R_new, t_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                if verbose:
                    print(f"  it {it:3d}  cost {cost:.6e}  rel {rel:.2e}  lam {lam:.1e}  "
                          f"step {step_m * 1e3:.2f} mm / {step_deg:.3f} deg")
                if rel < rel_tol or (step_m < 1e-9 and step_deg < 1e-9):
                    converged = True
                break
            lam *= 10.0
            if lam > 1e12:
                converged = True                              # cannot improve: at optimum
                break
        if converged:
            break

    _, rms1 = _cost(R, t, f)
    out = np.tile(np.eye(4), (N, 1, 1))
    out[:, :3, :3] = R; out[:, :3, 3] = t
    return out, SolveReport(it, converged, cost0, cost, rms0, rms1, max_step_m, max_step_deg,
                            N, int(len(f.b_i)), int(len(f.p_i)), lam)


# -------------------------------------------------------------------- builders
def anchor_coverage(stamps: np.ndarray, anchored_idx: np.ndarray,
                    positions: np.ndarray | None = None) -> dict:
    """Where the anchors are in TIME, which is what conditions this graph.

    Board COUNT is not the quantity that matters; the mobile_1 sweep settled
    that (2026-09-20). Two boards bracketing a run hold it at both ends and the
    middle is bridged. The same two boards both near one end leave a LEVER:
    the far stretch is held only through the odometry chain, a rotation inside
    the nearest board's own sigma becomes metres of displacement out there, and
    the solve reports convergence with near-zero residuals while the tail
    swings. Holding out `rs_anchor` on mobile_1 did exactly that -- 7.26 m.

    So the three numbers are: how long the run goes before its FIRST anchor,
    how long it goes after its LAST, and the widest interior gap. The first two
    are levers, the third is a bridge, and they are not the same risk.
    """
    stamps = np.asarray(stamps, dtype=np.float64)
    a = np.sort(np.unique(np.asarray(anchored_idx, dtype=int)))
    if not len(a) or len(stamps) < 2:
        return {"anchored_poses": 0, "leading_s": float(stamps[-1] - stamps[0])
                if len(stamps) > 1 else 0.0, "trailing_s": 0.0,
                "max_interior_gap_s": 0.0, "regions": {}}
    ta = stamps[a]
    gaps = np.diff(ta)
    out = {
        "anchored_poses": int(len(a)),
        "span_s": float(stamps[-1] - stamps[0]),
        "first_anchor_s": float(ta[0] - stamps[0]),
        "last_anchor_s": float(ta[-1] - stamps[0]),
        "leading_s": float(ta[0] - stamps[0]),
        "trailing_s": float(stamps[-1] - ta[-1]),
        "max_interior_gap_s": float(gaps.max()) if len(gaps) else 0.0,
    }
    # Seconds are the wrong unit on their own: a parked robot accumulates no
    # drift. Metres of unanchored PATH is what the odometry has to carry.
    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        step = np.concatenate([[0.0], np.linalg.norm(np.diff(pos, axis=0), axis=1)])
        arc = np.cumsum(step)
        out["path_m"] = float(arc[-1])
        out["leading_path_m"] = float(arc[a[0]])
        out["trailing_path_m"] = float(arc[-1] - arc[a[-1]])
    return out


def move_by_region(stamps: np.ndarray, anchored_idx: np.ndarray,
                   moved_m: np.ndarray) -> dict:
    """Median move in the leading / anchored / trailing stretches.

    A well-conditioned solve moves all three by a similar amount -- the whole
    trajectory shifts into the map frame together. A lever moves the
    unanchored end by a multiple of the anchored middle, and THAT ratio is the
    diagnostic, not the absolute number.
    """
    stamps = np.asarray(stamps, dtype=np.float64)
    moved_m = np.asarray(moved_m, dtype=np.float64)
    a = np.sort(np.unique(np.asarray(anchored_idx, dtype=int)))
    if not len(a):
        return {}
    lo, hi = stamps[a[0]], stamps[a[-1]]
    out = {}
    for name, m in (("leading", stamps < lo), ("anchored", (stamps >= lo) & (stamps <= hi)),
                    ("trailing", stamps > hi)):
        if m.any():
            out[name] = {"poses": int(m.sum()), "median_mm": float(np.median(moved_m[m]) * 1e3),
                         "max_mm": float(moved_m[m].max() * 1e3)}
    return out


def reset_edges(stamps: np.ndarray, gap_factor: float = 3.0,
                max_fraction: float = 0.10) -> np.ndarray:
    """Indices i where the edge i -> i+1 crosses a stamp gap wider than
    `gap_factor` x the median period.

    RTAB-Map's automatic reset publishes nothing for Odom/ResetCountdown frames
    and resumes from the last good pose, so the motion across the gap was never
    observed and the relative pose the file implies there is fiction. The log
    line that names the reset is not kept with the run; the hole in the stamps
    is, and it is the same event.

    THIS ONLY MEANS ANYTHING ON A REGULARLY SAMPLED STREAM, and the guard
    below is not a nicety. A keyframing method emits poses when the VIEW
    changes, so irregular spacing is its normal behaviour and carries no
    information about tracking loss. Run on `mast3r_slam`'s 151 keyframes the
    heuristic flagged 65 of 150 edges (2026-09-20); at x100 sigma that is 43%
    of the chain set nearly free, and the solve duly drove its residuals to
    zero because the system had stopped being determined by the data. A
    flagged fraction above `max_fraction` is not a run full of resets, it is
    the heuristic being applied where it does not belong, so none are returned
    and the caller is expected to say so.
    """
    dt = np.diff(np.asarray(stamps, dtype=np.float64))
    if len(dt) < 3:
        return np.zeros(0, dtype=int)
    idx = np.flatnonzero(dt > gap_factor * np.median(dt))
    return idx if len(idx) <= max_fraction * len(dt) else np.zeros(0, dtype=int)


def sampling_regularity(stamps: np.ndarray, gap_factor: float = 3.0) -> dict:
    """How uniform the pose stream is, and what the reset heuristic would say.

    Returned so a build can report that it DECLINED to treat gaps as resets,
    rather than silently either freeing half the chain or ignoring real ones.
    """
    dt = np.diff(np.asarray(stamps, dtype=np.float64))
    if len(dt) < 3:
        return {"n_edges": int(len(dt)), "applies": False,
                "why": "fewer than 4 poses"}
    med = float(np.median(dt))
    flagged = int((dt > gap_factor * med).sum())
    frac = flagged / len(dt)
    return {"n_edges": int(len(dt)), "median_period_s": med,
            "period_iqr_s": float(np.percentile(dt, 75) - np.percentile(dt, 25)),
            "flagged": flagged, "flagged_fraction": float(frac),
            "applies": bool(frac <= 0.10),
            "why": "" if frac <= 0.10 else
                   (f"{100 * frac:.0f}% of edges exceed {gap_factor}x the median period, "
                    f"so the stream is not regularly sampled and a wide gap says nothing "
                    f"about tracking loss -- no edge is treated as a reset")}


def odometry_factors(poses: np.ndarray, stamps: np.ndarray, sigma_t: float, sigma_r: float,
                     reset_gap_factor: float = 3.0, reset_inflate: float = 100.0) -> Factors:
    """One between factor per consecutive pose pair, from the backbone itself.

    sigma_t [m] and sigma_r [rad] are PER EDGE and declared by the caller with
    their provenance (build_pseudo_gt derives them from the run's own RPE).
    Edges that cross a reset gap get both sigmas multiplied by reset_inflate:
    the factor stays, so the chain stays connected, but it carries almost no
    weight, which is the truthful statement of what the backbone knows there.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    N = len(poses)
    f = Factors()
    if N < 2:
        return f
    R, t = poses[:, :3, :3], poses[:, :3, 3]
    i = np.arange(N - 1)
    Ri_T = np.transpose(R[i], (0, 2, 1))
    f.b_i = i
    f.b_R = Ri_T @ R[i + 1]
    f.b_t = np.einsum("kij,kj->ki", Ri_T, t[i + 1] - t[i])
    sig = np.tile([sigma_t] * 3 + [sigma_r] * 3, (N - 1, 1)).astype(np.float64)
    resets = reset_edges(stamps, reset_gap_factor)
    sig[resets] *= reset_inflate
    f.b_sigma = sig
    f.b_tag = ["odom"] * (N - 1)
    for k in resets:
        f.b_tag[k] = "odom/reset"
    return f


def gauge_prior(poses: np.ndarray, k: int = 0, sigma_t: float = 1e-4,
                sigma_r: float = 1e-5) -> Factors:
    """Pin pose k at its initial value. For a graph with no anchors this is the
    only thing that makes the normal matrix invertible; with anchors present it
    must NOT be added, or it fights them."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    f = Factors()
    f.add_prior(k, poses[k, :3, :3], poses[k, :3, 3], [sigma_t] * 3 + [sigma_r] * 3, "gauge")
    return f


def anchor_factors(observed_poses: np.ndarray, observed_stamps: np.ndarray,
                   keyframe_stamps: np.ndarray, sigma_t: float, sigma_r: float,
                   name: str, max_gap_s: float = 0.2, window=None) -> tuple[Factors, dict]:
    """Prior factors from board-derived camera poses, interpolated onto the
    keyframes they bracket.

    The OBSERVATION is resampled to the state's stamp, never the other way
    round -- same rule as `ate()`: the estimate's stamps are the sensor's own.
    Linear/SLERP here; docs/STAGE2_GRAPH.md names the cubic B-spline upgrade.
    Returns the factors and a small provenance dict for the report.
    """
    from .trajectory import Trajectory
    obs = Trajectory(np.asarray(observed_stamps, dtype=np.float64),
                     np.asarray(observed_poses, dtype=np.float64), name=name)
    ks = np.asarray(keyframe_stamps, dtype=np.float64)
    sel = np.ones(len(ks), dtype=bool)
    if window is not None:
        sel &= (ks >= window[0]) & (ks <= window[1])
    idx = np.flatnonzero(sel)
    f = Factors()
    info = {"anchor": name, "keyframes_in_window": int(len(idx)), "factors": 0,
            "sigma_t_m": sigma_t, "sigma_r_deg": float(np.degrees(sigma_r))}
    if not len(idx):
        return f, info
    try:
        at = obs.interpolate(ks[idx], max_gap_s=max_gap_s)
    except ValueError:
        return f, info
    # interpolate() returns exactly the query stamps it could serve, so a
    # searchsorted on the full keyframe axis recovers each one's global index
    f.p_i = np.searchsorted(ks, at.stamps)
    f.p_R = at.poses[:, :3, :3].copy()
    f.p_t = at.poses[:, :3, 3].copy()
    f.p_sigma = np.tile([sigma_t] * 3 + [sigma_r] * 3, (len(at), 1)).astype(np.float64)
    f.p_tag = [f"anchor/{name}"] * len(at)
    info["factors"] = int(len(at))
    return f, info
