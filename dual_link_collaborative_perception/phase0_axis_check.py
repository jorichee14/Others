#!/usr/bin/env python3
"""
Phase 0 — Axis sanity check for Joint Selection-and-Link-Routing (Collaborative Perception).

Question this script answers (the project kill-switch):
    Are Importance (Imp) and Reconstructability (Rec) genuinely DISTINCT axes, and is
    Rec (a graded, content/side-info quantity) different from the cheap OVERLAP mask
    (HydraCollab's `1(c_i>=lambda) & 1(c_j>=lambda)`)?

If Imp and Rec collapse into one axis, or graded-Rec collapses into the overlap mask,
there is nothing for joint dual-link routing to exploit and the line of work should stop.

What it computes, per grid cell u of a two-agent scene:
    Imp(u)          task value of the cell            (synthetic: object coverage; real: detector ablation hook)
    Rec(u)          graded reconstructability of the collaborator's cell from the EGO's side-info:
                      - mutually-observed region: cross-agent predictability (ego already has a view)
                      - ego-occluded region:      spatial predictability via inpainting from ego context
    Rec_overlap(u)  the cheap binary proxy: 1 iff the ego directly sees the cell (HydraCollab-style)

Outputs (in --out dir):
    scatter_imp_rec.png     Imp vs Rec, quadrant lines, points colored by overlap mask
    panel_scene.png         the two agent views + Imp / Rec / overlap heatmaps
    phase0_report.txt       correlations, 2x2 quadrant table, G1/G1' verdict

Default run (zero setup):  python3 phase0_axis_check.py
Real image:                python3 phase0_axis_check.py --image path/to.jpg --occlude 0.33

This is an INDICATIVE single-scene sanity check. A pass here justifies moving to Phase 1 on
real paired-view data (DAIR-V2X / OPV2V / V2X-R); it is not itself evidence of the final claim.
"""

import argparse
import os
import numpy as np
import cv2
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------------------
def make_synthetic_scene(H=384, W=640, seed=0):
    """A controllable 'street' scene. Returns (image_bgr, object_mask float[0,1]).

    Background is smoothly-varying (highly predictable => high Rec, low Imp).
    Objects (cars/pedestrians) have unique texture (task-relevant => high Imp; and,
    in ego-occluded regions, hard to inpaint => low Rec)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((H, W, 3), np.uint8)

    # sky gradient (very predictable)
    for y in range(H):
        t = y / H
        img[y, :, :] = (np.array([200, 175, 150]) * (1 - t) + np.array([120, 120, 120]) * t)
    # road band (predictable, low texture)
    road_y = int(H * 0.62)
    img[road_y:, :, :] = (60, 60, 62)
    # lane texture (mildly predictable, periodic)
    for x in range(0, W, 60):
        cv2.line(img, (x, road_y), (x - 40, H), (150, 150, 150), 3)

    obj_mask = np.zeros((H, W), np.float32)

    def paint_texture(x0, y0, x1, y1, base):
        patch = np.clip(base + rng.randint(-40, 40, (y1 - y0, x1 - x0, 3)), 0, 255).astype(np.uint8)
        img[y0:y1, x0:x1] = patch
        obj_mask[y0:y1, x0:x1] = 1.0

    # cars (rectangles) at varied x, some in left (mutually seen) some in right (occluded)
    car_xs = [40, 210, 360, 500]
    for cx in car_xs:
        w, h = rng.randint(70, 110), rng.randint(45, 70)
        y0 = road_y - h + rng.randint(-8, 8)
        base = np.array([rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)])
        paint_texture(cx, max(0, y0), min(W, cx + w), min(H, y0 + h), base)
        cv2.rectangle(img, (cx, max(0, y0)), (min(W, cx + w), min(H, y0 + h)), (20, 20, 20), 2)

    # pedestrians (tall thin ellipses)
    for px in [150, 300, 470, 590]:
        h = rng.randint(48, 70)
        y0 = road_y - h
        base = np.array([rng.randint(40, 210), rng.randint(40, 210), rng.randint(40, 210)])
        cv2.ellipse(img, (px, y0 + h // 2), (10, h // 2), 0, 0, 360,
                    tuple(int(v) for v in base), -1)
        obj_mask[max(0, y0):y0 + h, px - 10:px + 10] = 1.0

    obj_mask = cv2.GaussianBlur(obj_mask, (0, 0), 3)
    obj_mask = np.clip(obj_mask, 0, 1)
    return img, obj_mask


def make_agent_views(img, occlude_frac=0.34):
    """Ego = collaborator scene with a right-side occluded band + mild sensor difference.
    Returns (ego_bgr, collab_bgr, ego_visible_mask bool[H,W], occ_mask uint8[H,W])."""
    H, W = img.shape[:2]
    collab = img.copy()
    ego = img.copy()
    # mild cross-sensor difference so 'mutually observed' is predictable but not identical
    ego = cv2.GaussianBlur(ego, (0, 0), 1.0)
    ego = np.clip(ego.astype(np.int16) + 6, 0, 255).astype(np.uint8)

    x_occ = int(W * (1 - occlude_frac))
    ego[:, x_occ:] = 0  # ego cannot see the right band; collaborator can
    ego_visible = np.ones((H, W), bool)
    ego_visible[:, x_occ:] = False
    occ_mask = np.zeros((H, W), np.uint8)
    occ_mask[:, x_occ:] = 255
    return ego, collab, ego_visible, occ_mask


# --------------------------------------------------------------------------------------
# Per-cell signals
# --------------------------------------------------------------------------------------
def cell_slices(H, W, G):
    ch, cw = H // G, W // G
    for gy in range(G):
        for gx in range(G):
            yield gy, gx, slice(gy * ch, (gy + 1) * ch), slice(gx * cw, (gx + 1) * cw)


def compute_importance_synthetic(obj_mask, G):
    """Task value proxy = object coverage fraction per cell (independent of texture/inpaintability)."""
    H, W = obj_mask.shape
    Imp = np.zeros((G, G), np.float32)
    for gy, gx, ys, xs in cell_slices(H, W, G):
        Imp[gy, gx] = float(obj_mask[ys, xs].mean())
    # light normalization to [0,1]
    if Imp.max() > 0:
        Imp = Imp / Imp.max()
    return Imp


def compute_importance_saliency(collab_bgr, G):
    """Detector-free fallback for real images: Laplacian energy (WARNING: a proxy, not task loss)."""
    H, W = collab_bgr.shape[:2]
    gray = cv2.cvtColor(collab_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    Imp = np.zeros((G, G), np.float32)
    for gy, gx, ys, xs in cell_slices(H, W, G):
        Imp[gy, gx] = float(lap[ys, xs].mean())
    Imp = Imp / (Imp.max() + 1e-9)
    return Imp


def compute_reconstructability(ego_bgr, collab_bgr, ego_visible, occ_mask, G):
    """Graded Rec(u) in [0,1] = how well the EGO can regenerate the collaborator's cell content
    from its own side-information.

      - mutually observed cell: cross-agent predictability = 1 - MAD(ego_cell, collab_cell)
      - ego-occluded cell:      inpaint the occluded band from ego context, then
                                predictability = 1 - MAD(inpaint_cell, collab_cell)
    Also returns the binary overlap proxy Rec_overlap(u) = 1 iff cell is ego-visible."""
    H, W = collab_bgr.shape[:2]
    ego_inpaint = cv2.inpaint(ego_bgr, occ_mask, 5, cv2.INPAINT_TELEA)

    egoY = cv2.cvtColor(ego_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    inpY = cv2.cvtColor(ego_inpaint, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    colY = cv2.cvtColor(collab_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    Rec = np.zeros((G, G), np.float32)
    Rec_overlap = np.zeros((G, G), np.float32)
    for gy, gx, ys, xs in cell_slices(H, W, G):
        visible_frac = ego_visible[ys, xs].mean()
        Rec_overlap[gy, gx] = 1.0 if visible_frac > 0.5 else 0.0
        if visible_frac > 0.5:
            mad = np.abs(egoY[ys, xs] - colY[ys, xs]).mean()
        else:
            mad = np.abs(inpY[ys, xs] - colY[ys, xs]).mean()
        Rec[gy, gx] = float(np.clip(1.0 - mad / 0.25, 0.0, 1.0))  # 0.25 MAD -> Rec 0
    return Rec, Rec_overlap


# --------------------------------------------------------------------------------------
# Analysis / reporting
# --------------------------------------------------------------------------------------
def analyze(Imp, Rec, Rec_overlap, imp_hi=0.10, rec_hi=0.60):
    imp = Imp.flatten()
    rec = Rec.flatten()
    ovl = Rec_overlap.flatten()

    pear = pearsonr(imp, rec)[0]
    spear = spearmanr(imp, rec)[0]

    important = imp > imp_hi
    hi_rec = rec > rec_hi
    n_imp = int(important.sum())

    # 2x2 counts over ALL cells
    q = {
        "HiImp_HiRec": int((important & hi_rec).sum()),
        "HiImp_LoRec": int((important & ~hi_rec).sum()),
        "LoImp_HiRec": int((~important & hi_rec).sum()),
        "LoImp_LoRec": int((~important & ~hi_rec).sum()),
    }
    # fraction of IMPORTANT cells that are also reconstructable (the offload cell)
    offload_frac = (q["HiImp_HiRec"] / n_imp) if n_imp else 0.0

    # G1' : graded Rec vs binary overlap
    ovl_corr = pearsonr(rec, ovl)[0] if ovl.std() > 0 else float("nan")
    rec_hi_bin = (rec > rec_hi).astype(int)
    disagree = float(np.mean(rec_hi_bin != ovl.astype(int)))
    # reconstructable-but-not-overlapping = offload opportunities the overlap mask MISSES
    miss_by_overlap = int(((rec > rec_hi) & (ovl < 0.5)).sum())
    overlap_says_known_but_not = int(((ovl > 0.5) & (rec <= rec_hi)).sum())

    return dict(pear=pear, spear=spear, n_imp=n_imp, quad=q, offload_frac=offload_frac,
                ovl_corr=ovl_corr, disagree=disagree, miss_by_overlap=miss_by_overlap,
                overlap_says_known_but_not=overlap_says_known_but_not,
                imp_hi=imp_hi, rec_hi=rec_hi)


def verdict_lines(a):
    L = []
    g1_corr = abs(a["pear"]) < 0.6
    g1_cell = a["offload_frac"] > 0.10
    g1p = (not np.isnan(a["ovl_corr"])) and (a["ovl_corr"] < 0.8) and (a["disagree"] > 0.05)
    L.append(f"G1  axes distinct        : corr={a['pear']:+.3f} (|corr|<0.6? {'PASS' if g1_corr else 'FAIL'})")
    L.append(f"G1  offload cell present : {100*a['offload_frac']:.1f}% of important cells hi-Rec "
             f"(>10%? {'PASS' if g1_cell else 'FAIL'})")
    L.append(f"G1' Rec != overlap mask  : corr(Rec,overlap)={a['ovl_corr']:+.3f}, disagree={100*a['disagree']:.1f}% "
             f"({'PASS' if g1p else 'FAIL'})")
    L.append(f"      -> {a['miss_by_overlap']} reconstructable cells the OVERLAP mask misses "
             f"(offload opportunities); {a['overlap_says_known_but_not']} overlap-known-but-not-reconstructable")
    overall = g1_corr and g1_cell and g1p
    L.append("")
    L.append(f"OVERALL PHASE-0: {'PASS -> proceed to Phase 1' if overall else 'INSPECT -> see notes below'}")
    return L, overall


def save_scatter(Imp, Rec, Rec_overlap, a, path):
    plt.figure(figsize=(6.4, 5.2))
    imp = Imp.flatten(); rec = Rec.flatten(); ovl = Rec_overlap.flatten()
    for val, c, lab in [(1, "#1f77b4", "ego-visible (overlap=1)"),
                        (0, "#d62728", "ego-occluded (overlap=0)")]:
        m = ovl == val
        plt.scatter(imp[m], rec[m], s=18, c=c, alpha=0.6, edgecolors="none", label=lab)
    plt.axvline(a["imp_hi"], color="gray", ls="--", lw=1)
    plt.axhline(a["rec_hi"], color="gray", ls="--", lw=1)
    plt.text(a["imp_hi"] + 0.02, 0.96, "HiImp/HiRec\n(OFFLOAD to best-effort)", fontsize=8, va="top")
    plt.text(a["imp_hi"] + 0.02, a["rec_hi"] - 0.04, "HiImp/LoRec\n(RELIABLE, mandatory)", fontsize=8, va="top")
    plt.xlabel("Importance  Imp(u)"); plt.ylabel("Reconstructability  Rec(u)")
    plt.title(f"Phase 0 axes   corr={a['pear']:+.2f}   offload-cell={100*a['offload_frac']:.0f}% of important")
    plt.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def save_panel(ego, collab, Imp, Rec, Rec_overlap, path):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax[0, 0].imshow(cv2.cvtColor(collab, cv2.COLOR_BGR2RGB)); ax[0, 0].set_title("collaborator view (full)")
    ax[0, 1].imshow(cv2.cvtColor(ego, cv2.COLOR_BGR2RGB)); ax[0, 1].set_title("ego view (right band occluded)")
    ax[0, 2].axis("off")
    for axx, M, t in [(ax[1, 0], Imp, "Imp (importance)"),
                      (ax[1, 1], Rec, "Rec (graded reconstructability)"),
                      (ax[1, 2], Rec_overlap, "overlap mask (cheap proxy)")]:
        im = axx.imshow(M, cmap="viridis", vmin=0, vmax=1); axx.set_title(t)
        plt.colorbar(im, ax=axx, fraction=0.046)
    for a_ in ax.flat:
        a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Phase 0 axis sanity check.")
    ap.add_argument("--image", default=None, help="optional real image path (else synthetic scene)")
    ap.add_argument("--grid", type=int, default=16, help="grid cells per side (units)")
    ap.add_argument("--occlude", type=float, default=0.34, help="right-band ego occlusion fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "phase0_out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"could not read image: {args.image}")
        img = cv2.resize(img, (640, 384))
        obj_mask = None
        src = f"real image: {args.image}"
    else:
        img, obj_mask = make_synthetic_scene(seed=args.seed)
        src = f"synthetic scene (seed={args.seed})"

    ego, collab, ego_visible, occ_mask = make_agent_views(img, occlude_frac=args.occlude)
    G = args.grid

    if obj_mask is not None:
        Imp = compute_importance_synthetic(obj_mask, G)
        imp_note = "Imp = object coverage (task ground truth)"
    else:
        Imp = compute_importance_saliency(collab, G)
        imp_note = "Imp = Laplacian-saliency PROXY (no detector) -- plug a detector for real results"

    Rec, Rec_overlap = compute_reconstructability(ego, collab, ego_visible, occ_mask, G)
    a = analyze(Imp, Rec, Rec_overlap)

    save_scatter(Imp, Rec, Rec_overlap, a, os.path.join(args.out, "scatter_imp_rec.png"))
    save_panel(ego, collab, Imp, Rec, Rec_overlap, os.path.join(args.out, "panel_scene.png"))

    lines = []
    lines.append("=" * 78)
    lines.append("PHASE 0 — Axis sanity check (Imp vs Rec vs overlap)")
    lines.append("=" * 78)
    lines.append(f"source           : {src}")
    lines.append(f"grid             : {G}x{G} = {G*G} cells   occlusion={args.occlude:.2f}")
    lines.append(f"importance signal: {imp_note}")
    lines.append("")
    lines.append("Correlations (want LOW -> axes independent):")
    lines.append(f"  pearson(Imp,Rec)  = {a['pear']:+.3f}")
    lines.append(f"  spearman(Imp,Rec) = {a['spear']:+.3f}")
    lines.append("")
    lines.append(f"2x2 quadrants (imp_hi>{a['imp_hi']}, rec_hi>{a['rec_hi']}), {a['n_imp']} important cells:")
    q = a["quad"]
    lines.append(f"                 |   Lo Rec        Hi Rec")
    lines.append(f"     Hi Imp      |   {q['HiImp_LoRec']:3d} (reliable)  {q['HiImp_HiRec']:3d} (OFFLOAD)")
    lines.append(f"     Lo Imp      |   {q['LoImp_LoRec']:3d} (best-eff)  {q['LoImp_HiRec']:3d} (drop)")
    lines.append(f"  offload cell = {100*a['offload_frac']:.1f}% of important cells")
    lines.append("")
    vlines, overall = verdict_lines(a)
    lines += vlines
    lines.append("")
    lines.append("Notes:")
    lines.append("  * The OFFLOAD cell (HiImp/HiRec) is the entire source of the dual-link win:")
    lines.append("    content important enough to matter yet reconstructable enough to risk on best-effort.")
    lines.append("  * 'miss_by_overlap' cells are exactly what a HydraCollab-style overlap mask CANNOT")
    lines.append("    identify as offloadable -- that gap is the joint method's differentiator.")
    lines.append("  * Single-scene + synthetic Imp is INDICATIVE only. Re-run on DAIR-V2X/OPV2V paired")
    lines.append("    frames with a real detector (ablation Imp) before trusting the numbers.")
    report = "\n".join(lines)
    with open(os.path.join(args.out, "phase0_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[written] {args.out}/  (scatter_imp_rec.png, panel_scene.png, phase0_report.txt)")


if __name__ == "__main__":
    main()
