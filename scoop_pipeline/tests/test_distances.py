#!/usr/bin/env python
"""Self-test for check.py distances on a synthetic 8 x 6 x 3 m room (walls
and floor, 1 cm of noise): the 8 m and 6 m wall-to-wall distances from
picks a few cm off the walls (one near the floor, so the floor is in its
ball), one given on the csv's rows and one through a CloudCompare-style
picks file; a pick in the air is reported; a map scaled by 1.002 shows
+2000 ppm; picks alone (no csv) give d1, d2 with their map distances, and
with --measured their errors; --ask takes the readings typed in and
saves them with the picks (re-run from that csv: the same errors); the
floor plan of --click: clicks up to 0.4 m off the walls find them, a
click in an empty room is refused, u undoes, and the clicked pairs give the
same distances through --click; the reading box: readings in m or mm,
junk refused, errors and RMSE at once, the json saved, run again and a
session resumed, --redo; a corner to a wall (key c); the window's events sent by hand (keys typed in the box
not taken as commands).

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
        msgs = [pk.click(0.35, 3.0), pk.click(4.0, 3.0), pk.click(7.6, 3.1),
                pk.click(4.0, 0.3), pk.undo(), pk.click(4.0, 0.3), pk.click(4.2, 5.62)]
        plt.close(fig)
        want = ["wall A", "no straight wall", "d1 = 8.00", "wall A", "undone", "wall A", "d2 = 6.00"]
        ok = all(w in m for w, m in zip(want, msgs)) and len(pk.points()) == 4
        (print if ok else fails.append)("  ok    floor plan: clicks 0.4 m off the walls find them, empty space refused, undo"
                                        if ok else f"floor plan: {msgs}")
        # the reading box: typed after each pair, errors and totals at once, saved to the json
        fig, ax = plt.subplots()
        pk = maps.PlanPicker(ax, P, cKDTree(P), S)
        msgs = [pk.click(0.35, 3.0), pk.click(7.6, 3.1), pk.enter("8.004"), pk.click(4.0, 0.3),
                pk.click(4.2, 5.62), pk.enter("abc"), pk.enter("5998")]
        plt.close(fig)
        sm = maps.distance_summary(pk.res)
        js = os.path.join(d, "session.json")
        maps.write_distances_json(js, os.path.join(d, "map.npy"), pk.res, 0.3)
        import json
        jd = json.load(open(js))
        ok = ("error -4" in msgs[2] or "error -3" in msgs[2] or "error -5" in msgs[2]) and \
            "not a distance" in msgs[5] and "RMSE" in msgs[6] and sm["n"] == 2 and \
            abs(pk.res[1][1] - 5.998) < 1e-9 and len(jd["pairs"]) == 2 and \
            abs(jd["summary"]["rmse_mm"] - sm["rmse_mm"]) < 1e-6 and jd["pairs"][0]["measured_m"] == 8.004
        (print if ok else fails.append)(
            f"  ok    reading box: m and mm read, junk refused, {maps.summary_line(sm)}; json saved"
            if ok else f"reading box: {msgs} {sm} {jd}")
        rc, again = run(os.path.join(d, "map.npy"), js)
        rms = next((ln for ln in again.splitlines() if "error mean" in ln), "")
        ok = rc == 0 and rms and abs(float(rms.split("RMS")[1].split()[0]) - sm["rmse_mm"]) < 0.05
        (print if ok else fails.append)("  ok    the json runs again: the same RMSE" if ok
                                        else f"json again: rc {rc}\n{again}")
        # the window itself, its events sent by hand: clicks, the reading typed
        # (u and q typed into the box neither undo nor close), a session resumed
        from matplotlib.backend_bases import KeyEvent, MouseEvent
        real_show = plt.show

        def drive(script):
            def show(*_a, **_k):
                f = plt.gcf()
                axm = f.axes[0]
                for kind, v in script:
                    if kind == "click":
                        px, py = axm.transData.transform(v)
                        f.canvas.callbacks.process("button_press_event",
                                                   MouseEvent("button_press_event", f.canvas, px, py, button=1))
                    else:
                        for key in v:
                            f.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", f.canvas,
                                                                                    key))
            return show
        js2 = os.path.join(d, "win.json")
        plt.show = drive([("click", (0.35, 3.0)), ("click", (7.6, 3.1)), ("keys", list("8.0uq") + ["backspace"] * 2
                                                                          + ["enter"])])
        try:
            got = maps.click_plan(P, cKDTree(P), (0.3, 1.2), 0.3, save=js2, map_path="m")
            plt.show = drive([("click", (4.0, 0.3)), ("click", (4.2, 5.62)), ("keys", list("6002") + ["enter"])])
            got2 = maps.click_plan(P, cKDTree(P), (0.3, 1.2), 0.3, save=js2,
                                   earlier=maps.read_json_distances(js2), map_path="m")
        finally:
            plt.show = real_show
            plt.close("all")
        # --redo d1: d1 dropped, the next pair clicked is d1 again; a gap in the names is not reused
        plt.show = drive([("click", (0.35, 3.0)), ("click", (7.6, 3.1)), ("keys", list("8.001") + ["enter"]),
                          ("click", (4.0, 0.3)), ("click", (4.2, 5.62)), ("keys", list("6.0") + ["enter"])])
        try:
            got3 = maps.click_plan(P, cKDTree(P), (0.3, 1.2), 0.3, save=os.path.join(d, "redo.json"),
                                   earlier=[e for e in maps.read_json_distances(js2) if e[0] != "d1"],
                                   map_path="m", redo=["d1"])
        finally:
            plt.show = real_show
            plt.close("all")
        names3 = sorted(r[0] for r in got3)
        ok = names3 == ["d1", "d2", "d3"] and [r[1] for r in got3 if r[0] == "d1"] == [8.001]
        (print if ok else fails.append)("  ok    --redo d1: picked again under its name, the next pair d3"
                                        if ok else f"--redo: {got3}")
        # a pair drawn again from the json: the meter's line along A's normal, as when clicked
        fig, ax = plt.subplots()
        pk2 = maps.PlanPicker(ax, P, cKDTree(P), S)
        pk2.add("d1", 8.0, np.array([0.3, 2.0, 0.75]), np.array([7.7, 4.5, 0.75]))
        ln = [a_ for a_ in ax.lines if a_.get_color() == "#eb6834"][0].get_xydata()
        plt.close(fig)
        ok = abs(ln[0, 1] - ln[1, 1]) < 0.02 and abs(abs(ln[1, 0] - ln[0, 0]) - 8.0) < 0.02
        (print if ok else fails.append)("  ok    a pair drawn again: its line square to wall A, not pick to pick"
                                        if ok else f"redrawn line: {ln}")
        # a corner to a wall: a room with a 45 deg wall cut across its corner, the tip
        # where it meets the wall x = 8 at (8, 5): to the wall y = 0, 5 m; key c
        rng2 = np.random.default_rng(1)
        g = lambda a, b: np.arange(a, b, 0.02)                           # noqa: E731
        Z = g(0, 3)
        segs = [((0, 0), (0, 6)), ((0, 0), (8, 0)), ((8, 0), (8, 5)), ((0, 6), (7, 6)), ((8, 5), (7, 6))]
        W = []
        for (x0, y0), (x1, y1) in segs:
            L_ = np.hypot(x1 - x0, y1 - y0)
            t_ = g(0, L_) / L_
            T_, Zz = np.meshgrid(t_, Z)
            nrm = np.array([y1 - y0, x0 - x1]) / L_
            off = rng2.normal(0, 0.01, T_.size)
            W.append(np.c_[x0 + T_.ravel() * (x1 - x0) + off * nrm[0], y0 + T_.ravel() * (y1 - y0) + off * nrm[1],
                           Zz.ravel()])
        Xf, Yf = np.meshgrid(g(0, 8), g(0, 6))
        Pc = np.vstack(W + [np.c_[Xf.ravel(), Yf.ravel(), rng2.normal(0, 0.01, Xf.size)]])
        Sc = Pc[(Pc[:, 2] >= 0.3) & (Pc[:, 2] <= 1.2)]
        fig, ax = plt.subplots()
        pkc = maps.PlanPicker(ax, Pc, cKDTree(Pc), Sc)
        pkc.corner = True
        mc = [pkc.click(7.8, 4.9), pkc.click(7.4, 0.3), pkc.enter("5.000")]       # B where the dot landed
        plt.close(fig)
        rowc = pkc.res[-1] if pkc.res else None
        tab = maps._distance_rows(cKDTree(Pc), Pc, pkc.rows(), 0.3)
        fig, ax = plt.subplots()
        pk3 = maps.PlanPicker(ax, Pc, cKDTree(Pc), Sc)
        pk3.add(*pkc.rows()[0])
        plt.close(fig)
        ok = rowc is not None and "corner found" in mc[0] and abs(rowc[6] - 5.0) < 0.01 and \
            np.linalg.norm(rowc[4][0][:2] - [8.0, 5.0]) < 0.03 and abs(tab[0][6] - 5.0) < 0.01 and \
            tab[0][8] == "corner" and abs(pk3.res[0][6] - 5.0) < 0.01 and \
            maps.distance_summary(pkc.res)["n"] == 1
        (print if ok else fails.append)(
            f"  ok    corner to wall (key c): the tip of a 45 deg wall at {rowc[4][0][:2].round(3)}, "
            f"{rowc[6]:.3f} m to the wall across (5 m), counted; the table and a reopened session the same"
            if ok else f"corner: {mc} {rowc and rowc[4:8]} {tab and tab[0][4:9]}")
        jd2 = json.load(open(js2))
        ok = len(got) == 1 and got[0][1] == 8.0 and len(got2) == 2 and got2[1][0] == "d2" and \
            abs(got2[1][1] - 6.002) < 1e-9 and jd2["summary"]["n"] == 2
        (print if ok else fails.append)("  ok    the window: the reading typed in the box (u, q in it typed, "
                                        "not obeyed), saved; a session resumed from its json" if ok
                                        else f"window: {got} {got2} {jd2}")

        real, real_in, real_out = maps.click_plan, sys.stdin, sys.stdout
        maps.click_plan = lambda *a, **k: [(n_, None, a_, b_, k_) for n_, _, a_, b_, k_ in pk.rows()]
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
