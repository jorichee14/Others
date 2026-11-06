"""Identity scores on the floor plane (Ristani et al. 2016 for IDF1).

gt and pred: {id: (times (N,), xy (N, 2))}. Scored at the ground-truth times;
a predicted track is present at a time if it spans it (interpolated between
its samples). A ground-truth person and a track "match" at a time when they
are closer than thr metres.

  IDF1      share of person-time with the right identity (the one score to watch)
  IDP, IDR  its precision / recall parts
  switches  how often a person's matched track changes
  recall    share of person-time covered by some track (detection, not identity)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _at(times, t, xy):
    out = np.full((len(times), 2), np.nan)
    if len(t) == 0:
        return out
    inside = (times >= t[0]) & (times <= t[-1])
    out[inside, 0] = np.interp(times[inside], t, xy[:, 0])
    out[inside, 1] = np.interp(times[inside], t, xy[:, 1])
    return out


def id_metrics(gt, pred, thr=0.5):
    times = np.unique(np.concatenate([np.asarray(t, float) for t, _ in gt.values()]))
    gids, pids = list(gt), list(pred)
    G = np.stack([_exact(times, *gt[g]) for g in gids], 1) if gids else np.zeros((len(times), 0, 2))
    P = np.stack([_at(times, np.asarray(pred[p][0], float), np.asarray(pred[p][1], float)) for p in pids], 1) \
        if pids else np.zeros((len(times), 0, 2))
    gp = ~np.isnan(G[..., 0])
    pp = ~np.isnan(P[..., 0])
    co = np.zeros((len(gids), len(pids)))
    switches, matched, last = 0, 0, {}
    for k in range(len(times)):
        gi, pi = np.flatnonzero(gp[k]), np.flatnonzero(pp[k])
        if not len(gi) or not len(pi):
            continue
        D = np.linalg.norm(G[k, gi][:, None] - P[k, pi][None], axis=2)
        close = D < thr
        co[np.ix_(gi, pi)] += close
        r, c = linear_sum_assignment(np.where(close, D, 1e6))
        for a, b in zip(r, c):
            if close[a, b]:
                matched += 1
                g, p = gi[a], pi[b]
                if g in last and last[g] != p:
                    switches += 1
                last[g] = p
    n_gt, n_pred = int(gp.sum()), int(pp.sum())
    idtp = 0.0
    if co.size:
        r, c = linear_sum_assignment(-co)
        idtp = float(co[r, c].sum())
    return {
        "IDF1": 2 * idtp / max(n_gt + n_pred, 1),
        "IDP": idtp / max(n_pred, 1),
        "IDR": idtp / max(n_gt, 1),
        "switches": switches,
        "recall": matched / max(n_gt, 1),
        "gt_ids": len(gids),
        "pred_ids": len(pids),
    }


def _exact(times, t, xy):
    """Ground truth at its own sample times, NaN where it has none."""
    t = np.asarray(t, float)
    xy = np.asarray(xy, float)
    out = np.full((len(times), 2), np.nan)
    idx = np.searchsorted(times, t)
    ok = (idx < len(times)) & (times[np.clip(idx, 0, len(times) - 1)] == t)
    out[idx[ok]] = xy[ok]
    return out


def tracks_to_dict(tracks):
    out = {}
    for tk in tracks:
        t, x = tk.samples()
        out[tk.id] = (t, x[:, :2])
    return out
