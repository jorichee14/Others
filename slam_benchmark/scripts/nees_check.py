#!/usr/bin/env python3
"""Is the error bar the construction reports actually right?

    python3 scripts/nees_check.py --config configs/coop2.yaml \
        --run runs/coop2_20260828/pseudo_gt_v2c/mobile_1/<UTC>

Every ground-truth pipeline reports an uncertainty and none of them tests it.
PALoc propagates a covariance through its factor graph. LaMAR inverts the
Hessian of its refinement and calibrates it by keypoint noise. The
Cramer-Rao work computes the bound analytically from the graph's Laplacian.
**Each of those is precision conditional on the noise model being right**, and
a covariance has no units to sanity-check: it cannot see a mis-declared
extrinsic, a wrong fiducial convention, or a biased anchor, and it will report
the same tidy ellipsoid either way.

This script runs the missing test. It takes the construction's own per-pose
marginal covariance -- the number the estimator claims about itself -- and
compares it against the error measured with a source that was HELD OUT of the
fit. On `mobile_1` that source is the LiDAR, withheld throughout. The
statistic is NEES, normalised so that

    1.0   the reported uncertainty is honest
    >1    OVERCONFIDENT: the error bar is too small, whatever the trajectory's
          own quality, and `inflate_sigma_by` says by what factor
    <1    conservative

WHY IT MATTERS BEYOND THIS DATASET. Ground truth is what everything else is
measured against. Brachmann et al. showed the reference's *ranking* bias;
nobody has tested whether its stated *uncertainty* survives contact with
held-out evidence. If the covariance is honest, then the same machinery on a
platform with no reference sensor produces an error bar there is finally reason
to believe. If it is not, that is a finding about how the field states
ground-truth accuracy.

ALIGNMENT IS TAKEN FROM THE RUN, never chosen here. A v2c construction is
scored `none` -- unaligned, in the surveyed map frame -- and the covariance is
in that same frame, so the two are commensurable. Fitting the estimate to the
reference first would compare a map-frame covariance against a post-alignment
error and the NEES would be meaningless.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import fit, load_dataset, load_tum, se3                     # noqa: E402
from slambench.graph import Factors, marginal_covariances, nees            # noqa: E402


def pose_errors(est, ref, mode: str, max_gap_s: float = 0.2):
    """(idx, e_t, e_r) -- per-pose translation and rotation error, and which
    estimate poses they belong to, so the covariance can be indexed to match."""
    ref_i = ref.interpolate(est.stamps, max_gap_s=max_gap_s)
    idx = np.searchsorted(est.stamps, ref_i.stamps)
    est_i = est.subset(idx)
    est_a = fit(est_i, ref_i, mode=mode).apply(est_i)
    e_t = est_a.positions - ref_i.positions
    e_r = np.stack([se3.so3_log(ref_i.poses[i][:3, :3].T @ est_a.poses[i][:3, :3])
                    for i in range(len(ref_i))]) if hasattr(se3, "so3_log") else None
    if e_r is None:
        from slambench.graph import so3_log
        e_r = so3_log(np.einsum("kij,kjl->kil",
                                np.transpose(ref_i.poses[:, :3, :3], (0, 2, 1)),
                                est_a.poses[:, :3, :3]))
    return idx, e_t, e_r


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="a pseudo_gt run dir (needs factors.npz)")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--include-reference-uncertainty", action="store_true",
                    help="add the reference's own declared sigma to the covariance before "
                         "the test. Correct when the reference is not perfect -- coop2's "
                         "is 15 mm against a 199 mm error, so it changes little here, but "
                         "on a tighter construction it would dominate.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    fz = run / "factors.npz"
    if not fz.exists():
        raise SystemExit(
            f"{fz} does not exist. The covariance must come from the factors that were "
            f"ACTUALLY solved, not from a rebuild. Re-run scripts/build_pseudo_gt.py for "
            f"this cell -- it now writes factors.npz beside graph.json.")
    f = Factors.load(fz)
    g = json.loads((run / "graph.json").read_text())
    rj = json.loads((run / "run.json").read_text())
    agent = rj.get("agent") or rj.get("stream", "").split(".")[0]
    cfg = load_dataset(args.config)

    m = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
    mode = (m.get("ate", {}).get("alignment", {}) or {}).get("mode")
    if mode is None:
        raise SystemExit("no recorded alignment in metrics.json; run scripts/rescore.py first.")
    if mode != "none":
        print(f"\n  WARNING: this run is scored `{mode}`, not `none`. The covariance is in "
              f"the\n  map frame and the error is measured after a fit TO the reference, so "
              f"the two\n  are not commensurable and the NEES below is only indicative. "
              f"v2c is the\n  run to test.\n")

    est_file = next((run / n for n in ("pseudo_gt.tum", "trajectory.tum") if (run / n).exists()), None)
    ref_file = Path(args.reference) if args.reference else \
        run.parents[2] / "reference" / f"{agent}.tum"
    if est_file is None or not ref_file.exists():
        raise SystemExit(f"missing trajectory ({est_file}) or reference ({ref_file})")
    est, ref = load_tum(est_file), load_tum(ref_file)

    print(f"\n=== NEES   {run.name}   {rj.get('method', '?')} x {agent}")
    print(f"  held out : {', '.join(g.get('anchors', {}).get('held_out', [])) or 'the LiDAR reference only'}")
    print(f"  consumed : " + ", ".join(c['anchor'] for c in g.get('anchors', {}).get('consumed', [])))
    print(f"  factors  : {len(f.b_i)} between, {len(f.p_i)} prior {f.counts()['by_source']}")

    C = marginal_covariances(est.poses, f)
    idx, e_t, e_r = pose_errors(est, ref, mode)
    C = C[idx]

    if args.include_reference_uncertainty:
        s = cfg.reference_uncertainty_m()
        C[:, :3, :3] += (s ** 2) * np.eye(3)
        print(f"  reference uncertainty {s * 1e3:.1f} mm added to the covariance")

    sd = np.sqrt(np.einsum("kii->ki", C))
    print(f"\n  CLAIMED  translation sd  median {np.median(np.linalg.norm(sd[:, :3], axis=1)) * 1e3:7.1f} mm"
          f"   max {np.linalg.norm(sd[:, :3], axis=1).max() * 1e3:7.1f} mm")
    print(f"  OBSERVED translation err median {np.median(np.linalg.norm(e_t, axis=1)) * 1e3:7.1f} mm"
          f"   max {np.linalg.norm(e_t, axis=1).max() * 1e3:7.1f} mm")

    out = {"run": str(run), "alignment": mode}
    for label, e, block in (("translation", e_t, np.s_[:3, :3]),
                            ("rotation", e_r, np.s_[3:, 3:])):
        r = nees(e, C[:, block[0], block[1]])
        per = r.pop("per_sample")
        out[label] = r
        print(f"\n  {label.upper():<12} NEES {r['nees_mean']:8.2f}  (median {r['nees_median']:6.2f}, "
              f"n={r['n']}, consistent band "
              f"[{r['ci95_of_consistent'][0]:.3f}, {r['ci95_of_consistent'][1]:.3f}])")
        print(f"  {'':<12} VERDICT {r['verdict']}"
              + (f"  -- inflate the declared sigmas by x{r['inflate_sigma_by']:.2f} "
                 f"to make the error bar honest" if not r["consistent"] else ""))
        worst = int(np.argmax(per))
        print(f"  {'':<12} worst pose at t+{est.stamps[idx][worst] - est.stamps[0]:.1f} s, "
              f"NEES {per[worst]:.1f}")

    print("\nHOW TO READ THIS. NEES near 1 means the construction's own covariance is an "
          "honest\nerror bar, so the same machinery on a platform with no reference sensor "
          "produces a\nnumber there is reason to believe. NEES >> 1 means it is "
          "overconfident by that\nfactor, which is a statement about how ground-truth "
          "accuracy is reported and not\njust about this run. The test is only as strong as "
          "the held-out source: here the\nLiDAR is withheld from the construction entirely, "
          "but both it and the boards'\npositions descend from one offline mapping solution, "
          "so this does not test that\nsolution -- it tests whether 2 anchors predict 1516 "
          "poses as well as claimed.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":                                          # pragma: no cover
    sys.exit(main())
