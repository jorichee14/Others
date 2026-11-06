"""Time offsets between cameras from what they see, without a flash or shared clock.

When people move, every camera that sees them registers motion at the same
moment. Per camera, a motion signal: how much the picture changed between
consecutive frames (mean absolute change of small blurred grey thumbnails), on the file's own
timestamps. For
every pair of cameras the cross-correlation of their signals peaks at their
offset (sub-frame by a parabola through the peak). Pairs that do not see the
same motion give a flat correlation and are left out by their confidence. The
offsets of all cameras then come from one weighted least-squares fit over the
pairs; pairs that disagree with it by more than `tol` are dropped and the fit
repeated. The spread of the remaining pairs says how good the result is.

Assumes a constant offset per camera over the recording (free-running cameras
also drift a little: compare the offsets from the first and second half of a
long recording with `motion_sync.py --check-drift`).
"""
import cv2
import numpy as np


def motion_signal(path, seconds=None, start=0.0, size=(160, 90)):
    """Read a video once. -> (pts of every frame (N,), signal times (M,), motion (M,)).

    Motion only between start and start + seconds (of the file's own time); the
    timestamps of all frames are returned for writing the times file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 30.0
    pts, ts, ms, prev, i = [], [], [], None, 0
    while cap.grab():
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        t = pos if pos > 0 or i == 0 else i / fps
        pts.append(t)
        i += 1
        if t < start or (seconds is not None and t - start > seconds):
            prev = None
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        g = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA),
                             (3, 3), 0)
        if prev is not None:
            ts.append(t)
            ms.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    cap.release()
    pts = np.array(pts)
    if len(pts) > 1 and np.any(np.diff(pts) <= 0):             # no usable timestamps: count frames
        ts = list(np.interp(ts, pts, np.arange(len(pts)) / fps))
        pts = np.arange(len(pts)) / fps
    return pts, np.array(ts), np.array(ms)


def _uniform(t, x, rate, t0, t1):
    grid = np.arange(t0, t1, 1.0 / rate)
    y = np.interp(grid, t, x)
    w = max(1, int(rate * 2))                                  # remove slow changes (light, exposure): 2 s
    k = np.ones(w) / w
    y = y - np.convolve(y, k, mode="same")
    s = y.std()
    return y / s if s > 0 else y


def pair_offset(ta, xa, tb, xb, rate=100.0, max_lag=10.0):
    """Offset d with  clock_b = clock_a + d  (an event at time t in a's file is at t + d in b's),
    and a confidence (peak height over the spread of the correlation)."""
    t0, t1 = min(ta[0], tb[0]) - max_lag, max(ta[-1], tb[-1]) + max_lag
    a = _uniform(ta, xa, rate, t0, t1)
    b = _uniform(tb, xb, rate, t0, t1)
    a = np.where((np.arange(len(a)) / rate + t0 >= ta[0]) & (np.arange(len(a)) / rate + t0 <= ta[-1]), a, 0)
    b = np.where((np.arange(len(b)) / rate + t0 >= tb[0]) & (np.arange(len(b)) / rate + t0 <= tb[-1]), b, 0)
    n = 1 << int(np.ceil(np.log2(2 * len(a))))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)
    L = int(max_lag * rate)
    lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
    vals = np.r_[c[:L + 1], c[-L:]]
    k = int(np.argmax(vals))
    lag = float(lags[k])
    if 0 < k < len(vals) - 1 and lags[k - 1] == lags[k] - 1 and lags[(k + 1) % len(vals)] == lags[k] + 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[(k + 1) % len(vals)]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag += 0.5 * (y0 - y2) / den
    far = np.abs(lags - lags[k]) > rate * 0.5                  # the rest of the correlation, away from the peak
    conf = float((vals[k] - np.median(vals[far])) / (np.std(vals[far]) + 1e-12))
    return lag / rate, conf


def solve_offsets(names, pairs, min_conf=6.0, tol=0.05, ref=None):
    """pairs: {(a, b): (d, conf)} with clock_b = clock_a + d.
    -> ({name: offset}, {(a, b): residual}, used pairs). offset: add to a camera's own time to get the
    common clock (the reference camera's clock). Cameras not linked by any good pair get None."""
    names = list(names)
    ref = ref or names[0]
    use = {k: v for k, v in pairs.items() if v[1] >= min_conf}
    while True:
        # connected to ref?
        linked, frontier = {ref}, [ref]
        while frontier:
            n = frontier.pop()
            for a, b in use:
                for x, y in ((a, b), (b, a)):
                    if x == n and y not in linked:
                        linked.add(y)
                        frontier.append(y)
        idx = {n: i for i, n in enumerate(sorted(linked - {ref}))}
        if not idx:
            return {n: (0.0 if n == ref else None) for n in names}, {}, use
        rows, rhs, w = [], [], []
        for (a, b), (d, conf) in use.items():
            if a not in linked or b not in linked:
                continue
            # common = own + off  ->  off_a - off_b = d  (event: t_a + off_a = t_b + off_b, t_b = t_a + d)
            r = np.zeros(len(idx))
            if a in idx:
                r[idx[a]] += 1
            if b in idx:
                r[idx[b]] -= 1
            rows.append(r), rhs.append(d), w.append(conf)
        A, y, W = np.array(rows), np.array(rhs), np.sqrt(np.array(w))
        sol = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)[0]
        off = {n: (0.0 if n == ref else (float(sol[idx[n]]) if n in idx else None)) for n in names}
        res = {k: off[k[0]] - off[k[1]] - v[0] for k, v in use.items() if off[k[0]] is not None and off[k[1]] is not None}
        bad = [k for k, r in res.items() if abs(r) > tol]
        if not bad:
            return off, res, use
        worst = max(bad, key=lambda k: abs(res[k]))
        del use[worst]
