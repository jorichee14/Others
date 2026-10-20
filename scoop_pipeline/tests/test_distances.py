#!/usr/bin/env python
"""Self-test for check.py distances on a synthetic 8 x 6 x 3 m room (walls
and floor, 1 cm of noise): the 8 m and 6 m wall-to-wall distances from
picks a few cm off the walls (one near the floor, so the floor is in its
ball), one given on the csv's rows and one through a CloudCompare-style
picks file; a pick in the air is reported; a map scaled by 1.002 shows
+2000 ppm; picks alone (no csv) give d1, d2 with their map distances, and
with --measured their errors; --ask takes the readings typed in and
saves them with the picks (re-run from that csv: the same errors); the
floor plan of --click: clicks a few cm off the walls snap and fit them, a
click in an empty room is refused, u undoes, and the clicked pairs give the
same distances through --click.

    python tests/test_distances.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def room(rng, s=0.02, noise=0.01):
    g = lambda a, b: np.arange(a, b, s)                                  # noqa: E731
    pts = []
    for x in (0.0, 8.0):
        Y, Z = np.meshgrid(g(0, 6), g(0, 3))
        pts.append(np.c_[np.full(Y.size, x), Y.ravel(), Z.ravel()])
    for y in (0.0, 6.0):
        X, Z = np.meshgrid(g(0, 8), g(0, 3))
        pts.append(np.c_[X.ravel(), np.full(X.size, y), Z.ravel()])
    X, Y = np.meshgrid(g(0, 8), g(0, 6))
    pts.append(np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
    P = np.vstack(pts)
    n = np.zeros_like(P)                                                 # noise along each surface's normal
    n[P[:, 0] < 1e-6, 0] = n[np.abs(P[:, 0] - 8) < 1e-6, 0] = 1
    n[(P[:, 1] < 1e-6) | (np.abs(P[:, 1] - 6) < 1e-6), 1] = 1
    n[P[:, 2] < 1e-6, 2] = 1
    return P + n * rng.normal(0, noise, (len(P), 1))


def run(*a):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "distances", *a],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="dist_")
    try:
        P = room(np.random.default_rng(0))
        np.save(os.path.join(d, "map.npy"), P)
        np.save(os.path.join(d, "map_scaled.npy"), P * 1.002)
        with open(os.path.join(d, "dist.csv"), "w") as fh:
            fh.write("name,measured_m\n")
            fh.write("length,8.000,0.03,3.0,1.0,7.96,3.1,1.1\n")
            fh.write("width,6.000\n")
            fh.write("air,5.000,4,3,1.5,4,3,2.5\n")
        with open(os.path.join(d, "picks.txt"), "w") as fh:
            fh.write("Point#0,0.0,0.0,0.0\nPoint#1,0.0,0.0,0.0\n")       # length's (on its row)
            fh.write("Point#2,4.0,0.02,0.15\nPoint#3,4.2,5.97,0.18\n")    # width's, near the floor
            fh.write("Point#4,0,0,0\nPoint#5,0,0,0\n")
        rc, out = run(os.path.join(d, "map.npy"), os.path.join(d, "dist.csv"),
                      "--picks", os.path.join(d, "picks.txt"))
        print(out)
        rows = {ln.split()[0]: ln.split() for ln in out.splitlines()
                if ln.strip() and ln.split()[0] in ("length", "width", "air")}
        for name, want in (("length", 8.0), ("width", 6.0)):
            r = rows.get(name, [])
            if len(r) < 4 or abs(float(r[3])) > 3.0:
                fails.append(f"{name}: {r} (want {want} m within 3 mm)")
        if "no flat wall" not in " ".join(rows.get("air", [])):
            fails.append("a pick in the air is not reported")
        rc, out = run(os.path.join(d, "map_scaled.npy"), os.path.join(d, "dist.csv"),
                      "--picks", os.path.join(d, "picks.txt"))
        ppm = next((ln for ln in out.splitlines() if ln.startswith("scale")), "")
        try:
            v = float(ppm.split("(")[1].split()[0])
        except (IndexError, ValueError):
            v = float("nan")
        if not abs(v - 2000) < 300:
            fails.append(f"scaled map: {ppm!r} (want +2000 ppm)")
        else:
            print(f"  ok    scaled map: {ppm}")
        # picks alone: pairs numbered d1, d2; map distances, then with --measured
        with open(os.path.join(d, "pairs.txt"), "w") as fh:
            fh.write("1,0.03,3.0,1.0\n2,7.96,3.1,1.1\n3,4.0,0.02,0.15\n4,4.2,5.97,0.18\n")
        rc, out = run(os.path.join(d, "map.npy"), "--picks", os.path.join(d, "pairs.txt"))
        got = {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.split()[:1] in (["d1"], ["d2"])}
        ok = rc == 0 and all(abs(float(got.get(k, ["", "", "nan"])[2]) - v) < 0.003
                             for k, v in (("d1", 8.0), ("d2", 6.0)))
        (print if ok else fails.append)("  ok    picks alone: d1, d2 map distances" if ok
                                        else f"picks alone: {got} rc {rc}\n{out}")
        rc, out = run(os.path.join(d, "map.npy"), "--picks", os.path.join(d, "pairs.txt"),
                      "--measured", "8.0,6.0")
        got = {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.split()[:1] in (["d1"], ["d2"])}
        ok = rc == 0 and all(abs(float(got.get(k, ["", "", "", "nan"])[3])) < 3.0 for k in ("d1", "d2"))
        (print if ok else fails.append)("  ok    picks alone with --measured: errors within 3 mm" if ok
                                        else f"picks with --measured: {got} rc {rc}\n{out}")
        # --ask: readings typed in, saved with the picks; the csv runs again the same
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "distances",
                            os.path.join(d, "map.npy"), "--picks", os.path.join(d, "pairs.txt"), "--ask"],
                           input="8.0\n6.002\n", capture_output=True, text=True)
        saved = os.path.join(d, "map_distances.csv")
        ok = r.returncode == 0 and os.path.exists(saved) and "error mean" in r.stdout
        rc, again = run(os.path.join(d, "map.npy"), saved)
        same = [ln for ln in r.stdout.splitlines() if ln.startswith("scale")] == \
            [ln for ln in again.splitlines() if ln.startswith("scale")]
        (print if ok and same and rc == 0 else fails.append)(
            "  ok    --ask: readings saved; the saved csv gives the same result" if ok and same and rc == 0
            else f"--ask: rc {r.returncode}\n{r.stdout}{r.stderr}\nagain rc {rc}\n{again}")

        # --click: the floor plan's clicks (drawn on Agg, no window)
        sys.path.insert(0, ROOT)
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.spatial import cKDTree
        from analysis import maps
        S = P[(P[:, 2] >= 0.3) & (P[:, 2] <= 1.2)]
        fig, ax = plt.subplots()
        pk = maps.PlanPicker(ax, P, cKDTree(P), S)
        msgs = [pk.click(0.06, 3.0), pk.click(4.0, 3.0), pk.click(7.93, 3.1),
                pk.click(4.0, 0.05), pk.undo(), pk.click(4.0, 0.05), pk.click(4.2, 5.95)]
        plt.close(fig)
        want = ["wall A", "no map point", "d1 = 8.00", "wall A", "undone", "wall A", "d2 = 6.00"]
        ok = all(w in m for w, m in zip(want, msgs)) and len(pk.points()) == 4
        (print if ok else fails.append)("  ok    floor plan: clicks snap to the walls, empty space refused, undo"
                                        if ok else f"floor plan: {msgs}")
        real, real_in, real_out = maps.click_plan, sys.stdin, sys.stdout
        maps.click_plan = lambda *a, **k: pk.points()
        sys.stdin, sys.stdout = io.StringIO("8.0\n6.0\n"), io.StringIO()
        try:
            maps.main_distances([os.path.join(d, "map.npy"), "--click", "--ask"])
            out = sys.stdout.getvalue()
        except SystemExit as e:
            out = sys.stdout.getvalue() + f"exit {e}"
        finally:
            maps.click_plan, sys.stdin, sys.stdout = real, real_in, real_out
        rms = next((ln for ln in out.splitlines() if "error mean" in ln), "")
        ok = os.path.exists(os.path.join(d, "map_picks.txt")) and rms and \
            float(rms.split("RMS")[1].split()[0]) < 5.0
        (print if ok else fails.append)(f"  ok    --click: two pairs, {rms.strip()}" if ok
                                        else f"--click:\n{out}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("distances ok" if not fails else "distances FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
