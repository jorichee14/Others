#!/usr/bin/env python3
"""Run Sionna RT on a scene the pipeline exported (datasets/lod3dgs.py sionna):
the paths between a transmitter and receivers, and a radio map of the floor.

    python3 scoop_pipeline/analysis/comms/sionna_run.py <scene.xml> --tx X Y Z
        [--rx X Y Z ...] [--freq 5.745e9] [--max-depth 3] [--power 20]
        [--radio-map [--cell 0.5] [--height 1.2]] [--samples 1000000] [--out DIR]

Positions are in the scene's frame, the mesh's: a scene made from
mesh_view_<tag>.ply is in the mapping pass's GLIM world; one made from the
anchored mesh (lod3dgs.py anchor <mesh> --frame frames/anchor_frame.json,
then sionna on <mesh>_anchored.ply) takes map coordinates, the ones the
survey's trajectories, boards and cameras are in.

For each receiver: the paths (PathSolver, line of sight, specular
reflections and refraction up to --max-depth bounces), their total path
gain (dB), the line-of-sight delay, the RMS delay spread, the strongest
paths listed, and the channel impulse response written as cir_<rx>.npz
(a: complex gains, tau: delays in s, per path). --radio-map: the path gain
over a plane --height m above the scene's floor in --cell m cells
(RadioMapSolver), written as radio_map.npy (dB, [ny, nx]), radio_map.csv
(cell centre, dB) and radio_map.png. Channel 149 is 5.745 GHz (--freq).
"""
import argparse
import csv
import os
import sys

import numpy as np

C0 = 299792458.0


def build(scene_path, freq, power_dbm, tx, rxs):
    import sionna.rt as rt
    sc = rt.load_scene(scene_path)
    sc.frequency = freq
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.add(rt.Transmitter("tx", position=[float(v) for v in tx], power_dbm=power_dbm))
    for k, p in enumerate(rxs):
        sc.add(rt.Receiver("rx%d" % k, position=[float(v) for v in p]))
    return sc


def paths_report(sc, max_depth, samples, log=print):
    """[{rx, n_paths, gain_db, los_delay_ns, rms_delay_ns, a, tau}] per receiver."""
    import sionna.rt as rt
    paths = rt.PathSolver()(sc, max_depth=max_depth, samples_per_src=samples)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a, tau = np.asarray(a), np.asarray(tau)
    # a: [num_rx, rx_ant, num_tx, tx_ant, num_paths, time]; tau: [num_rx, rx_ant?, ..., num_paths]
    a = a.reshape(a.shape[0], -1, a.shape[-2], a.shape[-1])[:, 0, :, 0]      # [rx, paths]
    tau = tau.reshape(tau.shape[0], -1, tau.shape[-1])[:, 0]                   # [rx, paths]
    out = []
    for r in range(a.shape[0]):
        ok = np.isfinite(tau[r]) & (tau[r] >= 0) & (np.abs(a[r]) > 0)
        ar, tr = a[r][ok], tau[r][ok]
        p = np.abs(ar) ** 2
        rec = {"rx": "rx%d" % r, "n_paths": int(ok.sum()), "a": ar, "tau": tr}
        if ok.any():
            rec["gain_db"] = float(10 * np.log10(p.sum()))
            rec["los_delay_ns"] = float(tr.min() * 1e9)
            w = p / p.sum()
            m1 = (w * tr).sum()
            rec["rms_delay_ns"] = float(np.sqrt(max((w * tr ** 2).sum() - m1 ** 2, 0)) * 1e9)
            strongest = np.argsort(-p)[:5]
            log("  %s: %d paths, path gain %.1f dB, first %.1f ns, RMS delay spread %.1f ns"
                % (rec["rx"], rec["n_paths"], rec["gain_db"], rec["los_delay_ns"], rec["rms_delay_ns"]))
            for i in strongest:
                log("      %7.1f ns  %6.1f dB" % (tr[i] * 1e9, 10 * np.log10(p[i])))
        else:
            rec.update(gain_db=-np.inf, los_delay_ns=np.nan, rms_delay_ns=np.nan)
            log("  %s: no path" % rec["rx"])
        out.append(rec)
    return out


def radio_map(sc, cell, height, max_depth, samples, log=print):
    """(gain_db [ny, nx], centres [ny, nx, 3]) over the scene's floor plane."""
    import sionna.rt as rt
    bb = sc.mi_scene.bbox()
    lo, hi = np.array(bb.min), np.array(bb.max)
    centre = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(lo[2] + height)]
    size = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    log("radio map: %.0f x %.0f m, %.2f m cells, %.2f m above the lowest point"
        % (size[0], size[1], cell, height))
    rm = rt.RadioMapSolver()(sc, center=centre, orientation=[0, 0, 0], size=size, cell_size=[cell, cell],
                             samples_per_tx=samples, max_depth=max_depth)
    g = np.asarray(rm.path_gain)[0]
    with np.errstate(divide="ignore"):
        g_db = np.where(g > 0, 10 * np.log10(np.maximum(g, 1e-30)), np.nan)
    return g_db, np.asarray(rm.cell_centers)


def write_map(out, g_db, centres, tx):
    np.save(os.path.join(out, "radio_map.npy"), g_db)
    with open(os.path.join(out, "radio_map.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "z", "path_gain_db"])
        for (j, i), v in np.ndenumerate(g_db):
            if np.isfinite(v):
                c = centres[j, i]
                w.writerow(["%.3f" % c[0], "%.3f" % c[1], "%.3f" % c[2], "%.2f" % v])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    x, y = centres[..., 0], centres[..., 1]
    im = ax.imshow(np.ma.masked_invalid(g_db), origin="lower", cmap="viridis",
                   extent=(x.min(), x.max(), y.min(), y.max()), vmin=np.nanpercentile(g_db, 2),
                   vmax=np.nanmax(g_db))
    ax.plot(tx[0], tx[1], "*", color="r", ms=12, mec="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="path gain (dB)")
    fig.savefig(os.path.join(out, "radio_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene.xml, or a built-in scene's name (box, ...)")
    ap.add_argument("--tx", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--rx", type=float, nargs=3, action="append", default=[], metavar=("X", "Y", "Z"))
    ap.add_argument("--freq", type=float, default=5.745e9, help="Hz (channel 149: 5.745 GHz)")
    ap.add_argument("--power", type=float, default=20.0, help="transmit power (dBm)")
    ap.add_argument("--max-depth", type=int, default=3, help="bounces")
    ap.add_argument("--samples", type=int, default=1000000, help="rays launched per source")
    ap.add_argument("--radio-map", action="store_true")
    ap.add_argument("--cell", type=float, default=0.5, help="radio map cell (m)")
    ap.add_argument("--height", type=float, default=1.2, help="radio map plane above the floor (m)")
    ap.add_argument("--out", default=None, help="default: next to the scene, sionna_run/")
    a = ap.parse_args()
    import sionna.rt as rt
    scene = getattr(rt.scene, a.scene) if not os.path.exists(a.scene) and hasattr(rt.scene, a.scene) \
        else os.path.expanduser(a.scene)
    out = os.path.expanduser(a.out) if a.out else os.path.join(
        os.path.dirname(os.path.abspath(a.scene)) if os.path.exists(a.scene) else ".", "sionna_run")
    os.makedirs(out, exist_ok=True)
    sc = build(scene, a.freq, a.power, a.tx, a.rx)
    print("scene %s: %d objects, %.3f GHz, tx at %s, %d receiver(s)"
          % (a.scene, len(sc.objects), a.freq / 1e9, a.tx, len(a.rx)))
    if a.rx:
        print("paths (up to %d bounces):" % a.max_depth)
        for rec in paths_report(sc, a.max_depth, a.samples):
            np.savez(os.path.join(out, "cir_%s.npz" % rec["rx"]), a=rec["a"], tau=rec["tau"],
                     gain_db=rec["gain_db"], rms_delay_ns=rec["rms_delay_ns"])
    if a.radio_map:
        g_db, centres = radio_map(sc, a.cell, a.height, a.max_depth, a.samples)
        write_map(out, g_db, centres, a.tx)
        print("  covered cells %d of %d, path gain median %.1f dB, best %.1f dB"
              % (np.isfinite(g_db).sum(), g_db.size, np.nanmedian(g_db), np.nanmax(g_db)))
    print("wrote %s/" % out)


if __name__ == "__main__":
    main()
