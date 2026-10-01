#!/usr/bin/env python3
"""
STAGE 03 - locate EVERY anchor board placed during the mapping run, express
them all in one `map` frame, anchor the cloud to the first board, and measure
GLIM drift wherever a board was revisited.

  (1) ONE streaming pass over the bag.  Every configured board DESIGN is
      detected in every sampled frame.  Nothing is buffered but the sightings.

  (2) Sightings -> INSTANCES.  Two of your boards are the same printout, so
      appearance cannot separate them; position does.  Each sighting is lifted
      into GLIM world through the trajectory and the resulting positions are
      single-link clustered with `assoc_radius`.  One cluster = one physical
      board.  This works whether or not the designs carry distinct marker IDs;
      with distinct IDs the clustering merely confirms what the IDs said.

      Requirement: real boards must be further apart than assoc_radius, and
      GLIM drift must be much smaller than it.  Both hold here (>1 m apart,
      mm-level drift) -- and if drift ever approached assoc_radius the
      clustering would visibly merge two boards rather than fail silently.

  (3) Sightings -> SECTIONS.  Within an instance, sightings separated by more
      than `section_gap` seconds are separate visits.  Each section gets its
      own averaged pose, scatter, reprojection stats and time span.  Section
      timestamps are exported.

  (4) ANCHOR.  The cloud is anchored to the FIRST section of the anchor board:
      translation + yaw only, so GLIM's gravity-aligned z is preserved.
      Using section 0 (never a fusion) means the map origin is reproducible
      and does not shift when a later sighting is added.

  (5) FUSION.  Per instance the sections are gated (reprojection, scatter,
      sample count) and inverse-variance fused into the canonical pose that 04
      and 06 consume.  If accepted sections still disagree by more than the
      configured tolerance the board is written anyway, WARNED about, and
      stamped with drift_spanned_mm so downstream knows the uncertainty.
      The anchor board's canonical pose stays pinned to section 0 by
      construction; its fused pose is reported alongside for comparison.

  (6) LOOP CLOSURE, per board rather than once for the run:
        - first section vs last section, against a noise floor built from the
          two sections' own scatter, so a residual smaller than the sampling
          noise is reported as NOT significant instead of as drift;
        - every consecutive section pair;
        - co-visible cross-checks: when two boards appear in one image the
          relative pose is drift-free (no trajectory, no extrinsic, no clock),
          so comparing it against the trajectory-derived relative pose isolates
          GLIM's error at that place and time.
      The result is drift as a function of position along the run instead of a
      single end-to-end number.

  python3 03_anchor.py [pipeline_config.json]
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np
import open3d as o3d

from pipeline_common import (load_pipeline, load_traj, pose_at_interp, R_to_q,
                             qR, ang_deg)
from pipeline_boards import (bank_from_config, avg_T, spread, fmt_spread,
                             pose_delta, cluster_link, sectionize,
                             read_tf_and_info, iter_images, pick_intrinsics, stp)


# --------------------------------------------------------------------------- #
def provenance(cfg_path, s):
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        rev = None
    try:
        h = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:16]
    except Exception:
        h = None
    return {"config": os.path.abspath(cfg_path), "config_sha256_16": h,
            "git_rev": rev, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime()),
            "detect_period": s.get("detect_period"),
            "section_gap": s.get("section_gap"),
            "assoc_radius": s.get("assoc_radius"),
            "time_tol": s.get("time_tol")}


def sig(mm, noise_mm, k=2.0):
    """Is a residual bigger than the sampling noise it sits on?"""
    if noise_mm <= 0:
        return True, float("inf")
    return mm > k * noise_mm, mm / noise_mm


# --------------------------------------------------------------------------- #
def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    S = P.sensor
    s = P.stage("03_anchor")

    detect_period = float(s.get("detect_period", 0.20))
    section_gap = float(s.get("section_gap", 3.0))
    min_sec_n = int(s.get("min_section_samples", 4))
    assoc_r = float(s.get("assoc_radius", 0.40))
    tol = float(s["time_tol"])
    fuse_cfg = s.get("fuse", {})
    g_reproj = float(fuse_cfg.get("max_reproj_px", 1.0))
    g_std = float(fuse_cfg.get("max_std_mm", 20.0))
    g_dis_mm = float(fuse_cfg.get("max_section_disagree_mm", 50.0))
    g_dis_deg = float(fuse_cfg.get("max_section_disagree_deg", 1.0))

    # frame convention for every board pose this stage reports
    board_axes = s.get("board_axes", "opencv")
    board_origin = s.get("board_origin", "corner")

    names = s.get("boards_used")
    if not names:
        names = [s.get("board", "anchor")] + list(s.get("extra_boards", []) or [])
    bank = bank_from_config(P.cfg, names,
                            group_gap_factor=float(s.get("group_gap_factor", 2.5)))
    for b in bank.designs:
        print(b.describe())
    FIX = dict((b.name, b.frame_fix(board_axes, board_origin)) for b in bank.designs)
    if board_axes == "ros":
        print("board frame: ROS convention -- x = outward normal (into the room), "
              "y = left, z = up; map +x points AWAY from the anchor board's wall")
    else:
        print("board frame: OpenCV native -- x along columns, y DOWN, z INTO the "
              "board; map +x points INTO the anchor board's wall, so the scene "
              "sits at negative x. Set 03_anchor.board_axes to \"ros\" to flip this.")
    if board_origin != "corner":
        print("board origin: %s (native OpenCV origin is the TOP-left corner "
              "with +y down, since OpenCV 4.6)" % board_origin)
    HEAD = 0 if board_axes == "ros" else 2   # which board axis defines map +x

    run = bool(P.reference)
    if run:
        # a run (dataset.reference_pass): its trajectory is stage 08's LiDAR-ICP
        # track, registered to the reference pass's anchored map -- already in
        # map, so the boards are measured IN that frame: no re-anchoring, no
        # cloud, and they go to their own file next to the reference's
        traj = P.lidar_track_traj()
        print("run in %s's map: trajectory = stage 08's LiDAR track %s (already in map); "
              "boards measured in that frame, nothing re-anchored" % (P.reference, traj))
        if os.path.basename(s["frame_out"]) == "anchor_frame.json":
            s["frame_out"] = P.outp("boards_{tag}.json")
    else:
        traj = P.outp(P.dataset["traj"])
    tr_t, tr_T = load_traj(traj)
    T_lidar_cam = S.T_lidar_camera
    t0, t1 = float(tr_t[0]), float(tr_t[-1])
    print("trajectory: %d poses, %.1f s\n" % (len(tr_t), t1 - t0))

    # ------------------------- pass 1: TF + intrinsics ---------------------- #
    infos, tree, available = read_tf_and_info(
        P.dataset["bag"], [S.camera_info_topic], want_tf=True,
        tf_max_gap=float(s.get("tf_max_gap", 0.2)))
    K, D, from_bag = pick_intrinsics(infos, S.camera_info_topic, s["rectified"],
                                     S.K, S.dist)
    print("intrinsics: %s | fx=%.2f fy=%.2f cx=%.2f cy=%.2f | rectified=%s"
          % ("bag CameraInfo" if from_bag else "config fallback",
             K[0, 0], K[1, 1], K[0, 2], K[1, 2], s["rectified"]))
    print("TF frames: %s ..." % (tree.frames()[:12],))

    # ------------------------- pass 2: streaming detect --------------------- #
    zed_map, cam_frame = s["zed_map_frame"], s["cam_frame"]
    hits = dict((b.name, []) for b in bank.designs)   # per design, per sighting
    frames = []                                       # co-visible bookkeeping
    n_frames = n_notraj = 0
    notraj_t = []
    t_wall = time.time()

    for st, gray in iter_images(P.dataset["bag"], S.image_topic,
                                detect_period, int(s.get("max_images", 0))):
        n_frames += 1
        dets = bank.detect(gray, K, D, stamp=st)
        if not dets:
            continue
        T_wl, gap = pose_at_interp(tr_t, tr_T, st)
        in_traj = gap <= tol
        if not in_traj:
            n_notraj += len(dets)
            notraj_t.extend([st] * len(dets))
        T_mz_cam = tree.lookup(zed_map, cam_frame, st)
        idx = []
        for d in dets:
            # same physical pose, reported in the configured board frame
            dT = d.T @ FIX[d.design]
            rec = {"t": st, "T_cam": dT, "reproj": d.reproj, "n": d.n,
                   "ratio": d.ratio,
                   "T_world": (T_wl @ T_lidar_cam @ dT) if in_traj else None,
                   "T_mz": (T_mz_cam @ dT) if T_mz_cam is not None else None,
                   "inst": None}
            hits[d.design].append(rec)
            idx.append((d.design, len(hits[d.design]) - 1))
        if len(idx) > 1:
            frames.append(idx)

    print("\nscanned %d sampled frames in %.1f s (period %.2f s)"
          % (n_frames, time.time() - t_wall, detect_period))
    for b in bank.designs:
        print("  %-12s sightings %-5d | %s"
              % (b.name, b.reject["ok"], b.reject_str()))
    if n_notraj:
        nt = np.array(notraj_t)
        before = int((nt < t0).sum()); after = int((nt > t1).sum())
        print("  ! %d sightings dropped: no trajectory pose within %.3f s"
              % (n_notraj, tol))
        print("    %d before the trajectory starts, %d after it ends, %d inside "
              "it (stamps %.1f..%.1f s rel. traj start)"
              % (before, after, n_notraj - before - after, nt.min() - t0,
                 nt.max() - t0))
        if before + after > 0.5 * n_notraj:
            print("    -> most are OUTSIDE the GLIM trajectory span, i.e. the rig "
                  "was looking at a board before GLIM initialised or after it "
                  "stopped. Those frames are unrecoverable here, but they are "
                  "usually the best-dwelt views of the anchor board -- consider "
                  "re-exporting the trajectory to cover the full bag.")
        else:
            print("    -> most are INSIDE the trajectory span, which means a "
                  "clock offset or a trajectory gap, not a coverage problem. "
                  "Raise time_tol or check ZED/Ouster time sync.")

    # ------------------------- sightings -> instances ----------------------- #
    instances = []          # list of dicts
    for b in bank.designs:
        recs = [r for r in hits[b.name] if r["T_world"] is not None]
        if not recs:
            print("\n! design '%s': no trajectory-linked sightings -> no instances"
                  % b.name)
            continue
        pos = np.array([r["T_world"][:3, 3] for r in recs])
        lab = cluster_link(pos, assoc_r)
        nclu = int(lab.max()) + 1
        order = sorted(range(nclu), key=lambda k: min(recs[i]["t"]
                                                      for i in np.where(lab == k)[0]))
        print("\ndesign '%s': %d trajectory-linked sightings -> %d spatial cluster(s) "
              "at %.2f m linkage (config says instances=%d)"
              % (b.name, len(recs), nclu, assoc_r, b.instances))
        if nclu != b.instances:
            print("  ! CLUSTER COUNT MISMATCH. Either a board was never seen, or a "
                  "bad PnP outlier formed a spurious cluster, or two copies are "
                  "closer together than assoc_radius=%.2f m." % assoc_r)
        for slot, k in enumerate(order):
            sel = [recs[i] for i in np.where(lab == k)[0]]
            sel.sort(key=lambda r: r["t"])
            for r in sel:
                r["inst"] = len(instances)
            nm = b.instance_name(slot)
            taken = set(i["name"] for i in instances)
            if nm in taken:
                k = 2
                while "%s_%d" % (nm, k) in taken:
                    k += 1
                print("  ! instance name '%s' is already used -- the registry says "
                      "instances=%d but %d clusters were found, so every cluster "
                      "got the same name and would OVERWRITE the previous one. "
                      "Naming this one '%s_%d'; set instances/instance_names in "
                      "the boards registry." % (nm, b.instances, nclu, nm, k))
                nm = "%s_%d" % (nm, k)
            centre = np.median([r["T_world"][:3, 3] for r in sel], axis=0)
            print("  cluster %d -> '%s': %d sightings, first %.1f s, "
                  "world centre %s" % (k, nm, len(sel), sel[0]["t"] - t0,
                                       centre.round(3).tolist()))
            instances.append({"name": nm, "design": b.name, "board": b,
                              "recs": sel, "slot": slot})

    if not instances:
        raise SystemExit("no board instances localised (check traj / extrinsic / "
                         "boards registry).")

    # ------------------------- sections ------------------------------------ #
    for inst in instances:
        recs = inst["recs"]
        secs = []
        for gi, g in enumerate(sectionize([r["t"] for r in recs], section_gap)):
            sub = [recs[i] for i in g]
            Ts = [r["T_world"] for r in sub]
            w = [r["n"] / max(r["reproj"], 0.05) ** 2 for r in sub]
            Tw, nused = avg_T(Ts, weights=w)
            sp = spread(Ts, Tw)
            secs.append({"index": gi, "T_world": Tw, "n": len(sub), "n_used": nused,
                         "t_start": sub[0]["t"], "t_end": sub[-1]["t"],
                         "mean_reproj": float(np.mean([r["reproj"] for r in sub])),
                         "max_reproj": float(np.max([r["reproj"] for r in sub])),
                         "std_mm": sp[0], "max_mm": sp[1], "max_deg": sp[2],
                         "weak": len(sub) < min_sec_n})
        inst["sections"] = secs

    # ------------------------- anchor selection ----------------------------- #
    want = s.get("anchor_board")
    anchor = None
    if want:
        anchor = next((i for i in instances if i["name"] == want), None)
        if anchor is None:
            anchor = next((i for i in instances
                           if i["design"] == want and i["slot"] == 0), None)
        if anchor is None:
            raise SystemExit("anchor_board '%s' matched no instance (have: %s)"
                             % (want, ", ".join(i["name"] for i in instances)))
    else:
        anchor = min(instances, key=lambda i: i["sections"][0]["t_start"])
    anchor["is_anchor"] = True
    A0 = anchor["sections"][0]
    T_world_boardA = A0["T_world"]
    print("\nanchor board: '%s' section 0 (%d samples, %s), first seen %.1f s"
          % (anchor["name"], A0["n"],
             fmt_spread((A0["std_mm"], A0["max_mm"], A0["max_deg"])),
             A0["t_start"] - t0))

    # ------------------------- map frame ------------------------------------ #
    board_pos_world = T_world_boardA[:3, 3]
    R_world_boardA = T_world_boardA[:3, :3]
    nrm = R_world_boardA[:, HEAD]
    hxy = np.array([nrm[0], nrm[1], 0.0])
    if np.linalg.norm(hxy) < 0.15:
        print("  ! anchor board normal is within %.1f deg of vertical; its heading "
              "is ill-conditioned. Falling back to the board x-axis for map x -- "
              "the map yaw convention CHANGES with this branch."
              % np.degrees(np.arcsin(max(np.linalg.norm(hxy), 1e-9))))
        xb = R_world_boardA[:, 0]
        hxy = np.array([xb[0], xb[1], 0.0])
        if np.linalg.norm(hxy) < 1e-6:
            raise SystemExit("anchor board has no usable horizontal heading")
    new_x = hxy / np.linalg.norm(hxy)
    new_z = np.array([0.0, 0.0, 1.0])
    new_y = np.cross(new_z, new_x)
    R_align = np.column_stack([new_x, new_y, new_z]).T

    T_N_world = np.eye(4)
    T_N_world[:3, :3] = R_align
    T_N_world[:3, 3] = -R_align @ board_pos_world
    if run:
        T_N_world = np.eye(4)            # the trajectory is already in map
    for inst in instances:
        for sec in inst["sections"]:
            sec["T_map"] = T_N_world @ sec["T_world"]

    # ------------------------- anchor the cloud ----------------------------- #
    input_anchored = bool(P.cfg["01_build_map"].get("anchor_camera_start", False))
    cam0 = (tr_T[0] @ T_lidar_cam)[:3, 3]
    if input_anchored and "cam0_shift" in P.cfg["01_build_map"]:
        cam0 = np.array(P.cfg["01_build_map"]["cam0_shift"], float)
    elif input_anchored:
        print("  ! stage-01 anchored the cloud but did not record its shift; "
              "recomputing as tr_T[0] @ T_lidar_camera. If stage 01 anchors on "
              "the first IMAGE stamp rather than trajectory index 0 these differ.")
    board_pos_in = board_pos_world - (cam0 if input_anchored else 0.0)

    pcd_done = False
    if s.get("skip_cloud", False) or run:
        print("\nskip_cloud: board poses only, no cloud read or written")
    else:
        pcd = o3d.io.read_point_cloud(s["input"])
        if len(pcd.points) == 0:
            raise SystemExit("no points in %s" % s["input"])
        pcd.translate(-board_pos_in)
        pcd.rotate(R_align, center=(0, 0, 0))
        o3d.io.write_point_cloud(s["output"], pcd)
        print("\nanchored %d pts -> %s  (upright, origin='%s' section 0)"
              % (len(pcd.points), s["output"], anchor["name"]))
        pcd_done = True

    # ------------------------- fuse + loop closure -------------------------- #
    print("\n=== per-board sections ===")
    for inst in instances:
        secs = inst["sections"]
        print("\n[%s]  (design '%s')" % (inst["name"], inst["design"]))
        for sec in secs:
            gate = []
            if sec["weak"]:
                gate.append("n<%d" % min_sec_n)
            if sec["mean_reproj"] > g_reproj:
                gate.append("reproj>%.2f" % g_reproj)
            if sec["std_mm"] > g_std:
                gate.append("std>%.1fmm" % g_std)
            sec["gated"] = gate
            print("  s%d  t=%.1f..%.1f s (%.1f%%..%.1f%%)  n=%d(%d used)  "
                  "reproj %.3f px  %s  xyz=%s%s"
                  % (sec["index"], sec["t_start"] - t0, sec["t_end"] - t0,
                     100 * (sec["t_start"] - t0) / max(t1 - t0, 1e-9),
                     100 * (sec["t_end"] - t0) / max(t1 - t0, 1e-9),
                     sec["n"], sec["n_used"], sec["mean_reproj"],
                     fmt_spread((sec["std_mm"], sec["max_mm"], sec["max_deg"])),
                     sec["T_map"][:3, 3].round(4).tolist(),
                     ("   REJECTED: " + ",".join(gate)) if gate else ""))

        # Two different gates, on purpose.
        #  good : sections allowed to PULL the fused canonical pose. A section
        #         with two samples has an unrepresentative mean, so it is kept
        #         out of the fusion.
        #  meas : sections allowed to MEASURE drift. Sample count is not a
        #         reason to discard a revisit -- a single sighting as the rig
        #         walks past at the end of the run is very often the only
        #         loop-closure evidence in the whole bag, and it is exactly the
        #         number the stage exists to produce.
        good = [sec for sec in secs if not sec["gated"]]
        meas = [sec for sec in secs
                if sec["mean_reproj"] <= g_reproj and sec["std_mm"] <= g_std]
        inst["meas"] = meas
        if not good:
            print("  ! no section passes the fusion gates -> board NOT written")
            inst["fused"] = None
            if len(meas) > 1:
                a, b_ = meas[0], meas[-1]
                mm, dg = pose_delta(a["T_map"], b_["T_map"])
                print("    (drift is still measurable across s%d->s%d: %.1f mm "
                      "/ %.3f deg)" % (a["index"], b_["index"], mm, dg))
            continue

        Tf, _ = avg_T([sec["T_map"] for sec in good],
                      weights=[sec["n"] / max(sec["std_mm"], 0.5) ** 2
                               for sec in good])
        inst["fused"] = Tf
        inst["good"] = good

        span_mm = span_deg = 0.0
        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                mm, dg = pose_delta(good[i]["T_map"], good[j]["T_map"])
                span_mm = max(span_mm, mm); span_deg = max(span_deg, dg)
        inst["span_mm"] = span_mm
        inst["span_deg"] = span_deg
        if len(good) > 1:
            print("  fused over %d sections -> xyz=%s | section spread %.1f mm / %.3f deg"
                  % (len(good), Tf[:3, 3].round(4).tolist(), span_mm, span_deg))
            if span_mm > g_dis_mm or span_deg > g_dis_deg:
                print("  ! WARNING: sections disagree beyond tolerance "
                      "(%.1f mm > %.1f, %.3f deg > %.3f). Written anyway; "
                      "drift_spanned_mm records it." % (span_mm, g_dis_mm,
                                                        span_deg, g_dis_deg))
        else:
            print("  single accepted section -> xyz=%s" % Tf[:3, 3].round(4).tolist())

        # loop closure for this board, over the measurement-grade sections
        lc = None
        if len(meas) > 1:
            # A thin section reports std ~ 0, which would make any residual look
            # infinitely significant. Charge it the worst scatter this board
            # showed on a well-sampled section instead.
            ref = max([sec["std_mm"] for sec in meas if sec["n"] >= min_sec_n]
                      or [0.0])

            def secnoise(sec):
                return sec["std_mm"] if sec["n"] >= min_sec_n else max(ref, 5.0)

            a, b = meas[0], meas[-1]
            mm, dg = pose_delta(a["T_map"], b["T_map"])
            noise = float(np.hypot(secnoise(a), secnoise(b)))
            ok, ratio = sig(mm, noise)
            thin = [sec["index"] for sec in (a, b) if sec["n"] < min_sec_n]
            lc = {"from_section": a["index"], "to_section": b["index"],
                  "dt_s": round(b["t_start"] - a["t_end"], 3),
                  "mm": round(mm, 2), "deg": round(dg, 4),
                  "noise_mm": round(noise, 2), "ratio": round(ratio, 2),
                  "significant": bool(ok),
                  "n_from": a["n"], "n_to": b["n"],
                  "thin_sections": thin,
                  "fused_from_sections": [sec["index"] for sec in good]}
            print("  loop closure s%d->s%d over %.1f s: %.1f mm / %.3f deg "
                  "(noise %.1f mm, ratio %.1fx) -> %s%s"
                  % (a["index"], b["index"], lc["dt_s"], mm, dg, noise, ratio,
                     "DRIFT" if ok else "not significant",
                     ("   [s%s thin: %d/%d samples -- treat as indicative, "
                      "re-run with a longer dwell to confirm]"
                      % (",".join(str(i) for i in thin), a["n"], b["n"]))
                     if thin else ""))
            cons = []
            for i in range(len(meas) - 1):
                x, y = meas[i], meas[i + 1]
                m2, d2 = pose_delta(x["T_map"], y["T_map"])
                nz = float(np.hypot(secnoise(x), secnoise(y)))
                o2, r2 = sig(m2, nz)
                cons.append({"from_section": x["index"], "to_section": y["index"],
                             "mm": round(m2, 2), "deg": round(d2, 4),
                             "noise_mm": round(nz, 2), "ratio": round(r2, 2),
                             "significant": bool(o2),
                             "n_from": x["n"], "n_to": y["n"]})
            lc["consecutive"] = cons
        inst["lc"] = lc

    # ------------------------- co-visible cross-checks ---------------------- #
    pairs = {}
    for idx in frames:
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                (da, ia), (db, ib) = idx[a], idx[b]
                ra, rb = hits[da][ia], hits[db][ib]
                if ra["inst"] is None or rb["inst"] is None:
                    continue
                if ra["inst"] == rb["inst"]:
                    continue
                key = tuple(sorted((ra["inst"], rb["inst"])))
                ra, rb = (ra, rb) if ra["inst"] == key[0] else (rb, ra)
                pairs.setdefault(key, []).append(
                    np.linalg.inv(ra["T_cam"]) @ rb["T_cam"])

    xcheck = []
    if pairs:
        print("\n=== co-visible cross-checks (drift-free relative poses) ===")
    for (i, j), rels in sorted(pairs.items()):
        if len(rels) < min_sec_n:
            continue
        Trel, _ = avg_T(rels)
        sp = spread(rels, Trel)
        A, B = instances[i], instances[j]
        if A.get("fused") is None or B.get("fused") is None:
            continue
        Ttraj = np.linalg.inv(A["fused"]) @ B["fused"]
        mm, dg = pose_delta(Trel, Ttraj)
        ok, ratio = sig(mm, max(sp[0], 1e-6))
        print("  %s <-> %s : %d co-visible frames (%s) | trajectory disagrees by "
              "%.1f mm / %.3f deg (ratio %.1fx) -> %s"
              % (A["name"], B["name"], len(rels), fmt_spread(sp), mm, dg, ratio,
                 "GLIM error" if ok else "within noise"))
        xcheck.append({"a": A["name"], "b": B["name"], "n_frames": len(rels),
                       "covis_std_mm": round(sp[0], 2),
                       "disagreement_mm": round(mm, 2),
                       "disagreement_deg": round(dg, 4),
                       "ratio": round(ratio, 2), "significant": bool(ok)})

    # ------------------------- map_zed -------------------------------------- #
    mz = [r["T_mz"] for r in anchor["recs"]
          if r["T_mz"] is not None and r["t"] <= A0["t_end"]]
    have_mapzed = len(mz) > 0
    T_mapzed_boardA = None
    if have_mapzed:
        T_mapzed_boardA, _ = avg_T(mz)

    # ------------------------- export --------------------------------------- #
    R_N_boardA = R_align @ R_world_boardA
    T_map_boardA = np.eye(4); T_map_boardA[:3, :3] = R_N_boardA

    boards_out = {}
    for inst in instances:
        if inst.get("fused") is None:
            continue
        is_anchor = inst is anchor
        # the anchor stays pinned to section 0 so `board` is exactly the origin
        Tc = T_map_boardA if is_anchor else inst["fused"]
        b = inst["board"]
        rec = {"frame": b.frame_for(inst["name"], is_anchor),
               "design": inst["design"],
               "is_anchor": is_anchor,
               "xyz": Tc[:3, 3].round(6).tolist(),
               "qxyzw": R_to_q(Tc[:3, :3]).round(6).tolist(),
               "method": "anchor_section0" if is_anchor else "fused_sections",
               "n_sections": len(inst["sections"]),
               "n_sections_used": len(inst["good"]),
               "n_views": int(sum(sec["n"] for sec in inst["good"])),
               "std_mm": round(min(sec["std_mm"] for sec in inst["good"]), 2),
               "max_mm": round(max(sec["max_mm"] for sec in inst["good"]), 2),
               "max_deg": round(max(sec["max_deg"] for sec in inst["good"]), 4),
               "drift_spanned_mm": round(inst["span_mm"], 2),
               "drift_spanned_deg": round(inst["span_deg"], 4),
               "drift_warning": bool(inst["span_mm"] > g_dis_mm
                                     or inst["span_deg"] > g_dis_deg),
               "squares": [b.sx, b.sy], "square_len": b.square_len,
               "dictionary": b.dict_name, "id_offset": b.id_offset,
               "loop_closure": inst.get("lc"),
               "sections": [{"index": sec["index"],
                             "t_start": round(sec["t_start"], 6),
                             "t_end": round(sec["t_end"], 6),
                             "traj_frac_start": round(100 * (sec["t_start"] - t0)
                                                      / max(t1 - t0, 1e-9), 2),
                             "traj_frac_end": round(100 * (sec["t_end"] - t0)
                                                    / max(t1 - t0, 1e-9), 2),
                             "n": sec["n"], "n_used": sec["n_used"],
                             "mean_reproj_px": round(sec["mean_reproj"], 4),
                             "std_mm": round(sec["std_mm"], 2),
                             "max_mm": round(sec["max_mm"], 2),
                             "max_deg": round(sec["max_deg"], 4),
                             "gated_out": sec["gated"],
                             "xyz": sec["T_map"][:3, 3].round(6).tolist(),
                             "qxyzw": R_to_q(sec["T_map"][:3, :3]).round(6).tolist(),
                             "xyz_world": sec["T_world"][:3, 3].round(6).tolist()}
                            for sec in inst["sections"]]}
        if not is_anchor:
            rec["fused_xyz"] = rec["xyz"]
        else:
            rec["fused_xyz"] = inst["fused"][:3, 3].round(6).tolist()
            rec["fused_qxyzw"] = R_to_q(inst["fused"][:3, :3]).round(6).tolist()
        boards_out[inst["name"]] = rec

    frame_out = {"schema": 2,
                 "map_frame": s["map_frame"],
                 # every board pose below is in this convention. A consumer that
                 # detects a board itself (06) MUST apply the same frame_fix or
                 # its poses will disagree with these by a fixed rotation.
                 "board_axes": board_axes,
                 "board_origin": board_origin,
                 "anchor_board": anchor["name"],
                 "R_align": R_align.tolist(),
                 "board_pos_world": board_pos_world.tolist(),
                 "cam0_shift": np.asarray(cam0, float).tolist(),
                 "input_camera_anchored": input_anchored,
                 "cloud_written": pcd_done,
                 "map_to_board_qxyzw": R_to_q(R_N_boardA).tolist(),
                 "T_N_world": T_N_world.tolist(),
                 "boards": boards_out,
                 # design name -> first instance of that design, so a stage that
                 # asks for "anchor" still resolves when instances are named
                 # "anchor_a"/"anchor_b".
                 "board_aliases": dict(
                     (i["design"], i["name"]) for i in reversed(instances)
                     if i["name"] in boards_out),
                 "cross_checks": xcheck,
                 "provenance": provenance(cfg_path, s)}

    alc = anchor.get("lc")
    if alc:
        frame_out["loop_closure_mm"] = alc["mm"]
        frame_out["loop_closure_deg"] = alc["deg"]
        frame_out["loop_closure_significant"] = alc["significant"]
        last = anchor["meas"][-1]
        frame_out["board_pos_world_end"] = last["T_world"][:3, 3].tolist()
        frame_out["board_qxyzw_end"] = R_to_q(last["T_world"][:3, :3]).tolist()
        frame_out["T_map_board_end"] = last["T_world"].tolist()

    if have_mapzed:
        T_N_mapzed = T_map_boardA @ np.linalg.inv(T_mapzed_boardA)
        frame_out["T_map_mapzed"] = T_N_mapzed.tolist()
        frame_out["map_to_mapzed_xyzquat"] = (T_N_mapzed[:3, 3].tolist()
                                              + R_to_q(T_N_mapzed[:3, :3]).tolist())
        print("\nmap->map_zed exported: xyz=%s qxyzw=%s"
              % (T_N_mapzed[:3, 3].round(6).tolist(),
                 R_to_q(T_N_mapzed[:3, :3]).round(6).tolist()))
    else:
        print("\nmap->map_zed NOT exported (no map_zed TF during anchor section 0).")

    with open(s["frame_out"], "w") as f:
        json.dump(frame_out, f, indent=2)
    print("wrote %d board(s) -> %s" % (len(boards_out), s["frame_out"]))

    # ------------------------- drift summary + TF chain --------------------- #
    print("\n=== drift summary (revisited boards only) ===")
    any_lc = False
    for inst in instances:
        lc = inst.get("lc")
        if not lc:
            continue
        any_lc = True
        print("  %-14s s%d->s%d  %7.1f mm  %7.3f deg  noise %5.1f mm  %5.1fx  %s"
              % (inst["name"], lc["from_section"], lc["to_section"], lc["mm"],
                 lc["deg"], lc["noise_mm"], lc["ratio"],
                 "DRIFT" if lc["significant"] else "-"))
    if not any_lc:
        print("  no board was seen in two accepted sections; no drift measurable.")

    MAP, MAPZED = s["map_frame"], s["mapzed_frame"]
    print("\n=== TF chain rooted at %s ===" % MAP)
    for nm, rec in boards_out.items():
        T = np.eye(4); T[:3, :3] = qR(rec["qxyzw"]); T[:3, 3] = rec["xyz"]
        print("\n[%s -> %s]  (%s)" % (MAP, rec["frame"], rec["method"]))
        print(stp(MAP, rec["frame"], T))
    T_boardA_cam0 = np.linalg.inv(T_world_boardA) @ (tr_T[0] @ T_lidar_cam)
    print("\n[%s -> %s]" % (boards_out[anchor["name"]]["frame"], s["cam_frame"]))
    print(stp(boards_out[anchor["name"]]["frame"], s["cam_frame"], T_boardA_cam0))
    if have_mapzed:
        print("\n[%s -> %s]" % (boards_out[anchor["name"]]["frame"], MAPZED))
        print(stp(boards_out[anchor["name"]]["frame"], MAPZED,
                  np.linalg.inv(T_mapzed_boardA)))


if __name__ == "__main__":
    main()
