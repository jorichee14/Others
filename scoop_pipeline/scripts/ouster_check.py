#!/usr/bin/env python3
"""Check offline Ouster packet decoding against the replayed point clouds.

    python3 scripts/ouster_check.py <packets_bag> [--points-bag <bag>]
        [--packets-ns /ouster] [--points-ns /mobile_1/ouster]
        [--driver-min-range 1.30] [--n 50] [--save-clock clock.json]

<packets_bag>  the ORIGINAL recording (<ns>/lidar_packets, <ns>/metadata)
--points-bag   a bag the replay produced from it: the retimed one
               (retime_bag.py output) or a plain replay. Without it, only the
               clock fit and the decode speed are reported.

What it reports:
1. the clock fit, as retime_bag.py prints it -- the offset, drift and
   residual should be the numbers retime_bag.py gave for the same bag;
2. decode speed, as a multiple of real time (the replay runs at 1x);
3. per matched scan: stamp difference, point count, max xyz difference in
   the lidar AND sensor frames, max per-point time difference.

A retimed bag should agree to ~1 us in stamp and < 1 mm in xyz in exactly one
of the two frames. A plain TIME_FROM_ROS_TIME replay carries replay-time
stamps, so there the stamps are reported as a constant offset and only the
geometry is compared. Exits non-zero when neither frame matches.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, ouster                                       # noqa: E402

MM = 1e-3


def quiet(*_):
    pass


def take(it, n):
    out = []
    for x in it:
        out.append(x)
        if len(out) >= n:
            break
    return out


def best_shift(dec, pts, max_shift=30):
    """Index shift k (pts[i+k] ~ dec[i]) and the stamp offset (ns).

    Stamps alone cannot decide it: scans come every 100 ms, so every shift
    gives an equally constant difference. Among the shifts whose difference
    IS constant, take the one where the most scans have identical point
    counts, then the smallest offset."""
    d_st = np.array([s.stamp_ns for s in dec], np.int64)
    p_st = np.array([s.stamp_ns for s in pts], np.int64)
    d_n = np.array([len(s.xyz) for s in dec])
    p_n = np.array([len(s.xyz) for s in pts])
    cands = []
    for k in range(-max_shift, max_shift + 1):
        i = np.arange(len(d_st))
        j = i + k
        ok = (j >= 0) & (j < len(p_st))
        if ok.sum() < max(3, len(d_st) // 2):
            continue
        diff = p_st[j[ok]] - d_st[i[ok]]
        off = int(np.median(diff))
        spread = float(np.median(np.abs(diff - off)))
        same = float(np.mean(d_n[i[ok]] == p_n[j[ok]]))
        cands.append((spread, -same, abs(off), k, off))
    if not cands:
        return None
    floor = min(c[0] for c in cands) + 50_000              # within 50 us
    spread, neg_same, _, k, off = min((c for c in cands if c[0] <= floor),
                                      key=lambda c: (c[1], c[2]))
    return spread, k, off, -neg_same


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packets_bag")
    ap.add_argument("--points-bag", default="")
    ap.add_argument("--packets-ns", default="/ouster")
    ap.add_argument("--points-ns", default="/mobile_1/ouster")
    ap.add_argument("--driver-min-range", type=float, default=1.30)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--save-clock", default="")
    a = ap.parse_args()

    S = ouster.sdk()
    reader = bag.open_bag(a.packets_bag)
    print(f"ouster-sdk: {S.name}")
    topics = reader.topics()
    for t in (f"{a.packets_ns}/metadata", f"{a.packets_ns}/lidar_packets"):
        if t not in topics:
            sys.exit(f"{t} not in {a.packets_bag}; ouster topics: "
                     f"{sorted(x for x in topics if 'ouster' in x)}")
    meta = ouster.read_metadata(reader, a.packets_ns)
    info = ouster.sensor_info(meta)
    print(f"sensor: {info.prod_line} {info.w}x{info.h}, profile "
          f"{info.format.udp_profile_lidar}, {topics[a.packets_ns + '/lidar_packets'].count} packets")

    print("\n[1] clock fit (compare with retime_bag.py's output for this bag)")
    t0 = time.perf_counter()
    clock = ouster.fit_clock(reader, a.packets_ns, metadata=info)
    print(f"    {time.perf_counter() - t0:.1f} s")
    if a.save_clock:
        clock.save(a.save_clock)
        print(f"    saved {a.save_clock}")

    print(f"\n[2] decode {a.n} scans")
    st = {}
    t0 = time.perf_counter()
    dec = take(ouster.iter_ouster_scans(reader, a.packets_ns, clock=clock, metadata=info,
                                        driver_min_range=a.driver_min_range,
                                        min_range=0.1, stats=st), a.n)
    dt = time.perf_counter() - t0
    fps = float(getattr(info.format, "fps", 0) or 10)
    print(f"    {len(dec)} scans in {dt:.1f} s = {len(dec) / dt:.1f} scans/s "
          f"({len(dec) / dt / fps:.1f}x real time at {fps:g} Hz; stats {st})")

    if not a.points_bag:
        print("\nno --points-bag: nothing to compare against")
        return 0

    print(f"\n[3] compare with {a.points_ns}/points in {a.points_bag}")
    preader = bag.open_bag(a.points_bag)
    ptopic = f"{a.points_ns}/points"
    if ptopic not in preader.topics():
        sys.exit(f"{ptopic} not in points bag; PointCloud2 topics: "
                 f"{[t.name for t in preader.topics().values() if 'PointCloud2' in t.msgtype]}")
    pts = take(bag.iter_scans(preader, ptopic, min_range=0.1, with_time=True, log=quiet),
               a.n + 60)
    sh = best_shift(dec, pts)
    if sh is None:
        sys.exit("could not align the two scan sequences")
    spread, k, off, same = sh
    print(f"    scan index shift {k:+d}, stamp offset {off * 1e-6:+.3f} ms "
          f"(spread {spread * 1e-3:.1f} us, equal point counts {same:.0%})")
    if abs(off) > 1_000_000:
        print("    -> the points bag is NOT on the retimed capture clock (a plain "
              "replay?); geometry is still compared")

    T = np.asarray(info.lidar_to_sensor_transform, float)
    R, tr = T[:3, :3], T[:3, 3] * MM
    rows = []
    for i, d in enumerate(dec):
        j = i + k
        if not (0 <= j < len(pts)):
            continue
        p = pts[j]
        row = {"dstamp_us": (p.stamp_ns - d.stamp_ns - (off if abs(off) > 1_000_000 else 0)) * 1e-3,
               "n_dec": len(d.xyz), "n_pts": len(p.xyz)}
        if len(d.xyz) == len(p.xyz):
            ls = d.xyz.astype(np.float64)
            ss = ls @ R.T + tr
            row["lidar_mm"] = float(np.abs(ls - p.xyz).max() / MM)
            row["sensor_mm"] = float(np.abs(ss - p.xyz).max() / MM)
            if d.t is not None and p.t is not None:
                row["t_us"] = float(np.abs(d.t - p.t).max() * 1e6)
        rows.append(row)

    same = [r for r in rows if "lidar_mm" in r]
    print(f"    matched {len(rows)} scans; equal point count in {len(same)}")
    if len(same) < len(rows):
        bad = [r for r in rows if "lidar_mm" not in r][:3]
        print("    count mismatches (decoded vs replayed): "
              + ", ".join(f"{r['n_dec']} vs {r['n_pts']}" for r in bad)
              + "  -> check --driver-min-range")
    ds = np.array([r["dstamp_us"] for r in rows])
    print(f"    stamp difference: median {np.median(ds):+.2f} us, max |.| {np.abs(ds).max():.2f} us")
    if not same:
        return 1
    lm = max(r["lidar_mm"] for r in same)
    sm = max(r["sensor_mm"] for r in same)
    tu = max((r.get("t_us", np.nan) for r in same), default=np.nan)
    print(f"    max xyz difference: lidar frame {lm:.3f} mm | sensor frame {sm:.3f} mm")
    print(f"    max per-point time difference: {tu:.3f} us")
    frame = "lidar" if lm < 1.0 else "sensor" if sm < 1.0 else None
    if frame is None:
        print("\nNO MATCH: neither frame reproduces the replayed clouds")
        return 1
    ok_stamp = np.abs(ds).max() < 5.0
    if abs(off) > 1_000_000:
        stamps = f"on another clock ({off * 1e-9:+.3f} s); per-scan spacing matches" \
            if ok_stamp else "on another clock, and the spacing differs too"
    else:
        stamps = "identical (retimed clock reproduced)" if ok_stamp else "differ, see above"
    print(f"\nMATCH in the {frame} frame; stamps {stamps}"
          f"\n  -> iter_ouster_scans(..., frame={frame!r}, "
          f"driver_min_range={a.driver_min_range})")
    return 0 if ok_stamp else 1


if __name__ == "__main__":
    sys.exit(main())
