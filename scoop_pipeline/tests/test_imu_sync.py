#!/usr/bin/env python
"""Self-test for scoop/imu_sync.py and processing/imu_sync.py on made-up
bags: a robot turning at random for 60 s with two IMUs, the reference
(/mobile_1/ouster/imu, 100 Hz) and the ZED's (/mobile_1/zed/imu/data, 200 Hz,
mounted turned, its stamps 2.12 s late); /other carries a camera_info.

  measure      2.12 s found (2 ms), correlation high, scale 1; a robot that
               never turns: not sure
  --out copy   the ZED IMU's header stamps 2.12 s earlier, every other byte
               and every log time the same; measured again: 0
  retime       retime_bag(imu_sync=...) moves the ZED IMU's stamps by what
               the measurement says, and leaves them when it is not sure

    python scoop_pipeline/tests/test_imu_sync.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
from scoop import bag, imu_sync as IS, retime, rosmsg               # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402

T0 = 1_790_000_000_000_000_000
REF, ZED = "/mobile_1/ouster/imu", "/mobile_1/zed/imu/data"
LATE = 2.12
R = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])                 # the ZED turned against the Ouster


def make_bag(path, turning=True):
    rng = np.random.default_rng(5)
    t = np.arange(0, 60, 0.005)
    w = np.convolve(rng.normal(0, 1, (len(t), 3)).ravel(), np.ones(200) / 200, "same").reshape(-1, 3) * 4
    w[:, :2] *= 0.2                                                  # mostly yaw, as a ground robot
    if not turning:
        w *= 0
    recs = []
    for k in range(0, len(t), 2):                                    # reference, 100 Hz
        ts = T0 + int(round(t[k] * 1e9))
        recs.append((ts, REF, rosmsg.imu(ts, "os_imu", [0, 0, 9.8], w[k] + rng.normal(0, 0.003, 3))))
    for k in range(len(t)):                                          # the ZED, 200 Hz, stamped late
        ts = T0 + int(round((t[k] + LATE) * 1e9))
        recs.append((ts, ZED, rosmsg.imu(ts, "zed_imu", [0, 0, 9.8], R.T @ w[k] + rng.normal(0, 0.003, 3))))
    tgt = mc.Intrinsics(100.0, 100.0, 50, 50, 100, 100, [0.0] * 4, None)
    for k in range(0, len(t), 100):
        ts = T0 + int(round(t[k] * 1e9))
        recs.append((ts, "/other", SS.camera_info(tgt, "cam", ts)))
    w_ = BagWriter(path)
    for ts, topic, msg in sorted(recs, key=lambda r: (r[0], r[1])):
        w_.write(topic, msg, ts)
    w_.close()


def records(path):
    out = {}
    for _, _, ch, m in bag.iter_mcap_records(bag.open_bag(path)):
        out.setdefault(ch.topic, []).append((m.log_time, bytes(m.data)))
    return out


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        b = os.path.join(tmp, "late")
        make_bag(b)
        r = IS.measure(bag.open_bag(b), ZED, REF)
        if not (r.ok and abs(r.shift_s - LATE) <= 0.003 and r.correlation > 0.9 and abs(r.scale - 1) < 0.1):
            f.append("measure: %s" % r)
        b0 = os.path.join(tmp, "still")
        make_bag(b0, turning=False)
        r0 = IS.measure(bag.open_bag(b0), ZED, REF)
        if r0.ok:
            f.append("a robot that never turns measured as sure: %s" % r0)

        out = os.path.join(tmp, "fixed")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "processing", "imu_sync.py"), b, "--out", out],
                           capture_output=True, text=True)
        if p.returncode != 0 or "+2.120 s behind" not in p.stdout or not os.path.exists(
                os.path.join(out, "imu_sync.json")):
            f.append("processing/imu_sync.py: %s\n%s" % (p.stdout, p.stderr))
        else:
            a, c = records(b), records(out)
            if set(a) != set(c) or any(len(a[t]) != len(c[t]) for t in a):
                f.append("copy: topics or counts differ")
            for t in a:
                for (la, da), (lc, dc) in zip(a[t], c[t]):
                    if la != lc:
                        f.append("copy: log time of %s changed" % t)
                        break
                    if t == ZED:
                        if retime.header_stamp_ns(da) - retime.header_stamp_ns(dc) != int(LATE * 1e9) \
                                or da[12:] != dc[12:]:
                            f.append("copy: %s not moved back by exactly %.2f s" % (t, LATE))
                            break
                    elif da != dc:
                        f.append("copy: %s bytes changed" % t)
                        break
            r2 = IS.measure(bag.open_bag(out), ZED, REF)
            if abs(r2.shift_s) > 0.003:
                f.append("measured again after the copy: %s" % r2)

        # retime_bag's step 5, the measurement stubbed (the Ouster fixture is test_ouster's)
        src = records(b)
        for ok, want in ((True, 1_500_000_000), (False, 0)):
            orig = IS.measure
            IS.measure = lambda reader, topic, ref, **kw: IS.ImuSync(topic, ref, 1.5, 0.95 if ok else 0.3, 1.0,
                                                                    1000, 0.2, ok)
            try:
                from scoop import ouster
                dst = os.path.join(tmp, "retimed_%s" % ok)
                res = retime.retime_bag(b, b, dst, "/ouster", "/mobile_1/ouster", clock=ouster.ClockMap.identity(),
                                        imu_sync=[{"topic": ZED}], log=lambda *_: None)
            finally:
                IS.measure = orig
            got = records(dst)
            moved = {retime.header_stamp_ns(da) - retime.header_stamp_ns(dg)
                     for (_, da), (_, dg) in zip(src[ZED], got[ZED])}
            if moved != {want} or got["/other"] != [(lt, d) for lt, d in src["/other"]] or len(res.imu_sync) != 1:
                f.append("retime imu_sync ok=%s: stamps moved by %s (want %d)" % (ok, moved, want))
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "imu sync ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
