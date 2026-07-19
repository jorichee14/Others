#!/usr/bin/env python3
"""
Toy Collaborative-Perception sandbox — learn the basics before touching STAMP/OPV2V.

Four rungs, all self-contained (synthetic scene, frozen "detector", no GPU/dataset/torch).
Each rung teaches one concept and builds toward G0 (the motivating experiment for
task-conditioned communication).

    Rung 1  FUSION       fusing a collaborator's tokens improves the ego task.
    Rung 2  BUDGET       send only K of N tokens -> the bandwidth/accuracy trade-off.
    Rung 3  TWO TASKS    a token's value is a VECTOR over tasks (detection vs segmentation),
                         not a scalar. corr(V_det, V_seg) is low -> tasks want different tokens.
    Rung 4  TOY G0       at a fixed budget, task-BLIND selection starves the second task;
                         task-CONDITIONED allocation recovers both. (The whole thesis, in miniature.)

Run:
    python3 toy_cp_sandbox.py --rung 1
    python3 toy_cp_sandbox.py --rung all       # runs 1..4, writes plots to toy_out/

Concepts map to FINAL_DIRECTION.md:
    Rung 3 = novelty claim #1 (task-conditioned transmit-value)
    Rung 4 = novelty claims #1 + #2 (multi-task budget allocation), the G0 figure.
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUT = os.path.join(os.path.dirname(__file__), "toy_out")


# ======================================================================================
# Scene: one scene, two agents with complementary views. Two kinds of task-relevant stuff:
#   - "objects" (cars/peds)         -> matter for DETECTION
#   - "road/drivable regions"       -> matter for SEGMENTATION
# The point: detection cares about object tokens, segmentation cares about road tokens.
# ======================================================================================
def make_scene(H=384, W=640, seed=0):
    rng = np.random.RandomState(seed)
    img = np.zeros((H, W, 3), np.uint8)
    # sky
    for y in range(H):
        t = y / H
        img[y, :, :] = (np.array([205, 180, 155]) * (1 - t) + np.array([120, 120, 120]) * t)
    road_y = int(H * 0.60)
    img[road_y:, :, :] = (60, 60, 63)

    det_mask = np.zeros((H, W), np.float32)   # where objects are  (detection ground truth)
    seg_mask = np.zeros((H, W), np.float32)   # where drivable road is (segmentation ground truth)
    seg_mask[road_y:, :] = 1.0                # the whole road band is "drivable"

    # objects scattered along the road
    for cx in [40, 180, 330, 470, 560]:
        w, h = rng.randint(60, 100), rng.randint(40, 64)
        y0 = road_y - h + rng.randint(-6, 6)
        base = np.array([rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)])
        patch = np.clip(base + rng.randint(-35, 35, (min(H, y0 + h) - max(0, y0), min(W, cx + w) - cx, 3)), 0, 255)
        img[max(0, y0):min(H, y0 + h), cx:min(W, cx + w)] = patch.astype(np.uint8)
        det_mask[max(0, y0):min(H, y0 + h), cx:min(W, cx + w)] = 1.0
        seg_mask[max(0, y0):min(H, y0 + h), cx:min(W, cx + w)] = 0.0  # object occludes road

    det_mask = np.clip(cv2.GaussianBlur(det_mask, (0, 0), 3), 0, 1)
    return img, det_mask, seg_mask


def split_agents(img, occlude_frac=0.34):
    """Ego occludes a right band; collaborator sees full scene (complementary view)."""
    H, W = img.shape[:2]
    collab = img.copy()
    ego = cv2.GaussianBlur(img, (0, 0), 1.0)
    x = int(W * (1 - occlude_frac))
    ego[:, x:] = 0
    ego_visible = np.ones((H, W), bool)
    ego_visible[:, x:] = False
    return ego, collab, ego_visible


# ======================================================================================
# Tokens + per-task value + a frozen "task scorer"
# ======================================================================================
def grid(H, W, G):
    ch, cw = H // G, W // G
    for gy in range(G):
        for gx in range(G):
            yield gy, gx, slice(gy * ch, (gy + 1) * ch), slice(gx * cw, (gx + 1) * cw)


def per_task_value(det_mask, seg_mask, ego_visible, G):
    """V_t(u) for t in {det, seg}: how much does token u help task t at the EGO.
    A token only helps if it carries info the ego does NOT already have (occluded region)."""
    H, W = det_mask.shape
    Vdet = np.zeros((G, G), np.float32)
    Vseg = np.zeros((G, G), np.float32)
    novel = np.zeros((G, G), np.float32)  # fraction of the token the ego cannot see
    for gy, gx, ys, xs in grid(H, W, G):
        unseen = 1.0 - ego_visible[ys, xs].mean()
        novel[gy, gx] = unseen
        # value = task-relevant content * how novel it is to the ego
        Vdet[gy, gx] = det_mask[ys, xs].mean() * unseen
        Vseg[gy, gx] = seg_mask[ys, xs].mean() * unseen
    return Vdet, Vseg, novel


def task_score(received_value_map, full_value_map):
    """Frozen scorer: fraction of the achievable per-task value that the received tokens deliver.
    (A stand-in for 'AP recovered' / 'mIoU recovered' — bounded in [0,1].)"""
    denom = full_value_map.sum()
    if denom <= 1e-9:
        return 1.0
    return float(received_value_map.sum() / denom)


def select_topk(value_map, K):
    """Return a boolean G x G mask of the K highest-value tokens."""
    flat = value_map.flatten()
    if K >= flat.size:
        return np.ones_like(value_map, bool)
    thr_idx = np.argsort(flat)[::-1][:K]
    mask = np.zeros(flat.size, bool)
    mask[thr_idx] = True
    return mask.reshape(value_map.shape)


# ======================================================================================
# Rungs
# ======================================================================================
def rung1(G=16):
    """FUSION: ego-alone vs ego+collaborator on detection."""
    img, det, seg = make_scene()
    ego, collab, vis = split_agents(img)
    Vdet, _, _ = per_task_value(det, seg, vis, G)
    full = Vdet  # value available from the collaborator
    ego_alone = task_score(np.zeros_like(full), full)          # ego receives nothing
    ego_fused = task_score(full, full)                          # ego receives all tokens
    print("RUNG 1 — FUSION (detection)")
    print(f"  ego alone (no collaboration): score = {ego_alone:.2f}")
    print(f"  ego + collaborator (all tokens): score = {ego_fused:.2f}")
    print(f"  => fusion recovers {100*(ego_fused-ego_alone):.0f} points the ego could not see.\n")
    return ego_alone, ego_fused


def rung2(G=16):
    """BUDGET: sweep K, watch detection score rise with tokens sent."""
    img, det, seg = make_scene()
    ego, collab, vis = split_agents(img)
    Vdet, _, _ = per_task_value(det, seg, vis, G)
    N = G * G
    Ks = [0, 2, 4, 8, 16, 32, 64, N]
    scores = []
    for K in Ks:
        mask = select_topk(Vdet, K)
        scores.append(task_score(Vdet * mask, Vdet))
    print("RUNG 2 — BUDGET (detection): tokens sent -> score")
    for K, s in zip(Ks, scores):
        print(f"  K={K:3d}  score={s:.2f}")
    print("  => the bandwidth/accuracy trade-off: most value is in a few tokens (diminishing returns).\n")
    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(5.4, 4))
    plt.plot(Ks, scores, "o-")
    plt.xlabel("tokens sent (budget K)"); plt.ylabel("detection score recovered")
    plt.title("Rung 2 — bandwidth vs accuracy"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "rung2_budget.png"), dpi=130); plt.close()
    return Ks, scores


def rung3(G=16):
    """TWO TASKS: value is a vector; det and seg want different tokens."""
    img, det, seg = make_scene()
    ego, collab, vis = split_agents(img)
    Vdet, Vseg, _ = per_task_value(det, seg, vis, G)
    d, s = Vdet.flatten(), Vseg.flatten()
    corr = pearsonr(d, s)[0] if d.std() > 0 and s.std() > 0 else float("nan")
    # how different are the top-K sets?
    K = 16
    md, ms = select_topk(Vdet, K), select_topk(Vseg, K)
    overlap = int((md & ms).sum())
    print("RUNG 3 — TWO TASKS (value is a vector, not a scalar)")
    print(f"  corr(V_det, V_seg) = {corr:+.2f}   (low => tasks want different tokens)")
    print(f"  top-{K} token sets: detection & segmentation share only {overlap}/{K} tokens")
    print("  => a single 'importance' score cannot serve both tasks. (novelty claim #1)\n")
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, M, t in [(ax[0], Vdet, "V_det (detection value)"), (ax[1], Vseg, "V_seg (segmentation value)")]:
        im = a.imshow(M, cmap="viridis"); a.set_title(t); plt.colorbar(im, ax=a, fraction=.046)
        a.set_xticks([]); a.set_yticks([])
    ax[2].scatter(d, s, s=16, alpha=.6)
    ax[2].set_xlabel("V_det"); ax[2].set_ylabel("V_seg"); ax[2].set_title(f"corr={corr:+.2f}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "rung3_two_tasks.png"), dpi=130); plt.close()
    return corr, overlap


def rung4(G=16, budget=16):
    """TOY G0: task-blind vs task-conditioned selection at a fixed budget.
    Receiver runs BOTH tasks with equal demand. One shared budget."""
    img, det, seg = make_scene()
    ego, collab, vis = split_agents(img)
    Vdet, Vseg, _ = per_task_value(det, seg, vis, G)

    # Policy A — task-BLIND: select by detection value only (the typical single-task selector)
    mA = select_topk(Vdet, budget)
    A_det = task_score(Vdet * mA, Vdet); A_seg = task_score(Vseg * mA, Vseg)

    # Policy A' — task-agnostic proxy: select by summed generic saliency (det+seg equally, but as
    # a single scalar map committed before knowing demand) -> still not demand-aware per token
    generic = Vdet + Vseg
    mAg = select_topk(generic, budget)
    Ag_det = task_score(Vdet * mAg, Vdet); Ag_seg = task_score(Vseg * mAg, Vseg)

    # Policy A'' — naive weighted-sum of NORMALIZED values (a tempting "joint" that is SCALE-FRAGILE:
    # det value is concentrated, seg value is diffuse, so normalization can tip the whole budget to
    # one task). Kept to show why naive combination is not the answer.
    nd = Vdet / (Vdet.sum() + 1e-9); ns = Vseg / (Vseg.sum() + 1e-9)
    mW = select_topk(0.5 * nd + 0.5 * ns, budget)
    W_det = task_score(Vdet * mW, Vdet); W_seg = task_score(Vseg * mW, Vseg)

    # Policy B — task-CONDITIONED budget allocation: split the shared budget by demand
    # (equal demand -> half to each task), take each task's top tokens within its share, then union.
    # This is the simplest correct multi-task allocation; smarter allocation is the actual research.
    w_det, w_seg = 0.5, 0.5
    k_det = int(round(budget * w_det)); k_seg = budget - k_det
    mB = select_topk(Vdet, k_det) | select_topk(Vseg, k_seg)
    B_det = task_score(Vdet * mB, Vdet); B_seg = task_score(Vseg * mB, Vseg)

    def worst(a, b): return min(a, b)
    print(f"RUNG 4 — TOY G0  (budget = {budget} tokens, receiver runs detection + segmentation)")
    print(f"  {'policy':<36}{'det':>6}{'seg':>7}{'worst-task':>12}")
    print(f"  {'A   task-blind (detection-only)':<36}{A_det:>6.2f}{A_seg:>7.2f}{worst(A_det,A_seg):>12.2f}")
    print(f"  {'A-  task-agnostic (raw sum)':<36}{Ag_det:>6.2f}{Ag_seg:>7.2f}{worst(Ag_det,Ag_seg):>12.2f}")
    print(f"  {'A** naive weighted-sum (scale-fragile)':<36}{W_det:>6.2f}{W_seg:>7.2f}{worst(W_det,W_seg):>12.2f}")
    print(f"  {'B   task-conditioned (budget split)':<36}{B_det:>6.2f}{B_seg:>7.2f}{worst(B_det,B_seg):>12.2f}")
    base = max(worst(A_det, A_seg), worst(Ag_det, Ag_seg), worst(W_det, W_seg))
    gain = worst(B_det, B_seg) - base
    print(f"\n  worst-task: best baseline = {base:.2f}, task-conditioned B = {worst(B_det,B_seg):.2f}  ({gain:+.2f})")
    if worst(B_det, B_seg) > base + 0.03:
        print("  => every task-BLIND / naive policy STARVES a task; task-conditioned allocation")
        print("     protects the worst task. (novelty #1 = tasks diverge; #2 = allocation is the fix). PASS.")
    else:
        print("  => tasks did not diverge enough here; raise object density or occlusion.")
    print("  NOTE: even B (fixed 50/50 split) is naive -- demand-adaptive, multi-task-aware allocation")
    print("        under a per-task guarantee is the actual research contribution.\n")

    os.makedirs(OUT, exist_ok=True)
    labels = ["A\ntask-blind", "A-\nagnostic sum", "A**\nweighted", "B\ntask-cond."]
    dets = [A_det, Ag_det, W_det, B_det]; segs = [A_seg, Ag_seg, W_seg, B_seg]
    x = np.arange(4); wbar = 0.36
    plt.figure(figsize=(6.4, 4.2))
    plt.bar(x - wbar/2, dets, wbar, label="detection")
    plt.bar(x + wbar/2, segs, wbar, label="segmentation")
    plt.xticks(x, labels); plt.ylabel("task score recovered"); plt.ylim(0, 1.05)
    plt.title(f"Rung 4 — toy G0 @ budget {budget}: task-blind starves seg")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rung4_toyG0.png"), dpi=130); plt.close()
    return dict(A=(A_det, A_seg), Ag=(Ag_det, Ag_seg), B=(B_det, B_seg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="all", help="1 | 2 | 3 | 4 | all")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--budget", type=int, default=16)
    args = ap.parse_args()
    print(f"[toy CP sandbox]  grid={args.grid}x{args.grid}  budget={args.budget}\n")
    if args.rung in ("1", "all"): rung1(args.grid)
    if args.rung in ("2", "all"): rung2(args.grid)
    if args.rung in ("3", "all"): rung3(args.grid)
    if args.rung in ("4", "all"): rung4(args.grid, args.budget)
    if args.rung == "all":
        print(f"[plots] {OUT}/  (rung2_budget.png, rung3_two_tasks.png, rung4_toyG0.png)")


if __name__ == "__main__":
    main()
