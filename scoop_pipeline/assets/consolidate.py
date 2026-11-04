#!/usr/bin/env python3
"""The registry's objects made one per physical thing: merged across labels,
small things on bigger ones recorded, and (after views.py) the fragments and
false hits pruned.

    python3 scoop_pipeline/assets/consolidate.py <registry> merge
        [--overlap 0.5] [--size-ratio 3] [--cell 0.05]
    python3 scoop_pipeline/assets/consolidate.py <registry> parts
        [--part-recall 0.3] [--contain 0.6] [--part-ratio 2] [--agree 0.5]
        [--synonyms "desk,table;cabinet,shelf,rack;monitor,computer"]
    python3 scoop_pipeline/assets/consolidate.py <registry> prune
        [--min-views 5] [--min-recall 0.15] [--keep-views 20] [--keep-points 1000]
        [--room YAW XMIN XMAX YMIN YMAX] [--wall-tol 0.12] [--wall-share 0.8]
        [--wall-extent 1.5]

merge  (after objects.py, before views.py). objects.py merges sightings of one
       label only, so a thing SAM 3 calls by several names (table / desk,
       monitor / computer, rack / shelf / cabinet) is several objects. Two
       objects are one when at least --overlap of the smaller one's --cell
       cells are the bigger one's AND they are within --size-ratio of each
       other in size; a much smaller object mostly inside a bigger one is
       kept apart and recorded as "on" it (a monitor on a desk). Merged
       objects pool their points and views; the label is the vote of their
       views (distinct frames x mean score), all names kept in "labels".
parts  (after views.py, before prune). Views decide what is a thing: a real
       thing (a monitor on a desk, a box on a shelf) is found again in most
       frames it is in sight from (views.py's recall); a piece of a bigger
       object that SAM 3 names on its own from some angle (a sofa's end called
       cabinet, a cushion called box) only now and then. An object at most
       1 / --part-ratio the size of another, with --contain of its 5 cm cells
       among the other's (it shares its surfaces; a thing standing on it only
       touches them), and a recall under --part-recall (found in under 30 %
       of the frames it is in sight from: the views do not agree it is a
       thing of its own; ED305's real things: 35 % and up) is a part of it:
       absorbed -- its points and segmented frames pooled
       into the bigger one, its name one more vote there. One that keeps its
       own identity and touches a bigger one (a fifth of its cells or more)
       is recorded "on" it.
       Every object's name is then its views' vote: "agreement" is the share
       of the top name and its --synonyms (desk / table is not a disagreement);
       under --agree the object is marked "uncertain": true.
prune  (after views.py). An object stays when SAM 3 segmented it in at least
       --min-views frames AND in at least --min-recall of the frames where it
       is in sight (views.py's segmented / seen): a real thing in view is
       found again and again, a false hit or a fragment now and then. One
       segmented in --keep-views frames or more stays whatever its recall (a
       real thing SAM 3 finds only from some sides: ED305's washing machine,
       24 of 242 frames) if it holds --keep-points points or more (a piece
       of something bigger that SAM 3 names on its own now and then is small).
       --room (crop_room.py's --yaw and --bounds): an object with --wall-share
       of its points within --wall-tol of a wall and --wall-extent or more
       along it is the room's shell -- a wall, a door, a whiteboard SAM 3
       masked with the wall round it -- and is dropped as "shell" (a cabinet
       against a wall keeps its front 0.4 m or more out and stays).
       The dropped objects go to objects_pruned.json (added to it when
       prune runs again, each with why: "dropped_as").

Each run keeps the registry it started from (objects_before_<step>.json);
ids are renumbered per label (table_00, table_01, ...), points and masks
renamed with them.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from objects import write_ply                                           # noqa: E402


def read_ply(path):
    """(xyz, rgb) of a ply written by objects.write_ply."""
    with open(path, "rb") as fh:
        n = 0
        while True:
            line = fh.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line == "end_header":
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
    return (np.c_[rec["x"], rec["y"], rec["z"]].astype(np.float64),
            np.c_[rec["r"], rec["g"], rec["b"]].astype(np.uint8))


def cells(xyz, size):
    return set(map(tuple, np.floor(xyz / size).astype(np.int64)))


def vote(o):
    return len({v["frame"] for v in o["views"]}) * float(o.get("mean_score", 1.0))


def groups(objs, keys, overlap, size_ratio):
    """Union-find over same-thing pairs; -> (groups as index lists, {small: big} 'on')."""
    n = len(objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    on = {}
    lo = np.array([o["aabb_min"] for o in objs])
    hi = np.array([o["aabb_max"] for o in objs])
    for i in range(n):
        for j in range(i + 1, n):
            if (lo[i] > hi[j] + 0.05).any() or (lo[j] > hi[i] + 0.05).any():
                continue                                       # boxes apart
            a, b = keys[i], keys[j]
            small, big = (i, j) if len(a) <= len(b) else (j, i)
            ks, kb = keys[small], keys[big]
            ov = sum(1 for k in ks if k in kb) / max(len(ks), 1)
            if ov < overlap:
                continue
            if len(kb) <= size_ratio * len(ks):
                parent[find(i)] = find(j)
            elif small not in on or len(keys[on[small]]) > len(kb):
                on[small] = big                                # the tightest container
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values()), on


def renumber(objs):
    """New ids per label, in the order given: {old id: new id}."""
    count, ren = {}, {}
    for o in objs:
        lab = o["label"].replace(" ", "_")
        k = count.get(lab, 0)
        count[lab] = k + 1
        ren[o["id"]] = "%s_%02d" % (lab, k)
    return ren


def rename_files(reg, objs, ren):
    """Points and masks to the new ids (via a temporary name, ids may swap)."""
    moves = []
    for o in objs:
        new = ren[o["id"]]
        for f in [o["points"]] + glob.glob(os.path.join(reg, "masks", o["id"] + "_*.png")):
            src = f if os.path.isabs(f) else os.path.join(reg, f)
            if not os.path.exists(src):
                continue
            base = os.path.basename(src)
            dst = os.path.join(os.path.dirname(src), new + base[len(o["id"]):])
            tmp = src + ".renaming"
            os.rename(src, tmp)
            moves.append((tmp, dst))
        o["points"] = "points/%s.ply" % new
        for key in ("best_for_3d",):
            for b in o.get(key, []):
                for f in ("mask", "segmenter_mask"):
                    if b.get(f):
                        b[f] = b[f].replace(o["id"] + "_", new + "_", 1)
        o["id"] = new
    for tmp, dst in moves:
        os.rename(tmp, dst)


def merge(reg, overlap=0.5, size_ratio=3.0, cell=0.05, log=print):
    path = os.path.join(reg, "objects.json")
    doc = json.load(open(path))
    objs = doc["objects"]
    shutil.copy(path, os.path.join(reg, "objects_before_merge.json"))
    pts = {o["id"]: read_ply(os.path.join(reg, o["points"])) for o in objs}
    keys = [cells(pts[o["id"]][0], cell) for o in objs]
    grp, on = groups(objs, keys, overlap, size_ratio)
    merged, old_of = [], {}
    for g in grp:
        g = sorted(g, key=lambda i: -len(keys[i]))
        head = dict(objs[g[0]])
        labels = {}
        for i in g:
            labels[objs[i]["label"]] = labels.get(objs[i]["label"], 0.0) + vote(objs[i])
        head["label"] = max(labels, key=labels.get)
        head["labels"] = {k: round(v, 2) for k, v in sorted(labels.items(), key=lambda kv: -kv[1])}
        head["merged_from"] = [objs[i]["id"] for i in g]
        if len(g) > 1:
            xyz = np.vstack([pts[objs[i]["id"]][0] for i in g])
            rgb = np.vstack([pts[objs[i]["id"]][1] for i in g])
            k = np.floor(xyz / 0.02).astype(np.int64)
            _, u = np.unique(k, axis=0, return_index=True)
            xyz, rgb = xyz[u], rgb[u]
            write_ply(os.path.join(reg, head["points"]), xyz, rgb)
            for i in g[1:]:
                os.remove(os.path.join(reg, objs[i]["points"]))
                for f in glob.glob(os.path.join(reg, "masks", objs[i]["id"] + "_*.png")):
                    os.rename(f, os.path.join(reg, "masks", head["id"] + "_m" + os.path.basename(f)[len(objs[i]["id"]):]))
            byframe = {}
            for i in g:
                for v in objs[i]["views"]:
                    if v["frame"] not in byframe or v["area"] * v["score"] > byframe[v["frame"]]["area"] * byframe[v["frame"]]["score"]:
                        byframe[v["frame"]] = v
            head["views"] = [byframe[f] for f in sorted(byframe)]
            head["n_views"] = len(byframe)
            head["mean_score"] = round(float(np.mean([v["score"] for v in head["views"]])), 4)
            head["n_points"] = int(len(xyz))
            head["centroid"] = np.round(xyz.mean(0), 4).tolist()
            head["aabb_min"] = np.round(xyz.min(0), 4).tolist()
            head["aabb_max"] = np.round(xyz.max(0), 4).tolist()
        for i in g:
            old_of[objs[i]["id"]] = head["id"]
        merged.append(head)
    merged.sort(key=lambda o: (o["label"], -o["n_points"]))
    ren = renumber(merged)
    for small, big in on.items():
        s = old_of[objs[small]["id"]]
        b = old_of[objs[big]["id"]]
        if s != b:
            for o in merged:
                if o["id"] == s:
                    o["on"] = b
    rename_files(reg, merged, ren)
    for o in merged:
        if "on" in o:
            o["on"] = ren[o["on"]]
    doc["objects"] = merged
    json.dump(doc, open(path, "w"), indent=1)
    nm = sum(1 for o in merged if len(o["merged_from"]) > 1)
    log("merge: %d objects -> %d (%d made of several, %d recorded on another)"
        % (len(objs), len(merged), nm, sum(1 for o in merged if "on" in o)))
    for o in merged:
        if len(o["labels"]) > 1:
            log("  %-22s %s" % (o["id"], ", ".join("%s %.0f" % kv for kv in o["labels"].items())))
    return merged


def dilate(keys):
    """A cell set grown by one cell (26-neighbourhood)."""
    off = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    return {(k[0] + a, k[1] + b, k[2] + c) for k in keys for a, b, c in off}


def recall_of(o):
    seen = {v["frame"] for v in o.get("views_all", [])}
    segm = {v["frame"] for v in o["views"]} & seen
    return len(segm), len(seen), len(segm) / max(len(seen), 1)


def label_votes(o):
    if o.get("labels"):
        return dict(o["labels"])
    return {o["label"]: vote(o)}


SYNONYMS = "desk,table;cabinet,shelf,rack;monitor,computer"


def parts(reg, part_recall=0.3, contain=0.6, part_ratio=2.0, agree=0.5, cell=0.05, touch=0.2,
          synonyms=SYNONYMS, log=print):
    path = os.path.join(reg, "objects.json")
    doc = json.load(open(path))
    objs = doc["objects"]
    if any("views_all" not in o for o in objs):
        raise SystemExit("parts needs views.py's views_all: run views.py first")
    shutil.copy(path, os.path.join(reg, "objects_before_parts.json"))
    syn = [{w.strip() for w in g.split(",") if w.strip()} for g in (synonyms or "").split(";") if g.strip()]
    pts = {o["id"]: read_ply(os.path.join(reg, o["points"])) for o in objs}
    keys = {o["id"]: cells(pts[o["id"]][0], cell) for o in objs}
    grown = {}
    alive = {o["id"]: o for o in objs}
    absorbed = []
    for o in sorted(objs, key=lambda o: len(keys[o["id"]])):          # smallest first
        sid = o["id"]
        _, _, rec = recall_of(o)
        lo, hi = np.array(o["aabb_min"]) - cell, np.array(o["aabb_max"]) + cell
        best, best_c, best_t = None, 0.0, 0.0
        for b in alive.values():
            if b["id"] == sid or len(keys[b["id"]]) < part_ratio * len(keys[sid]):
                continue
            if (lo > np.array(b["aabb_max"])).any() or (hi < np.array(b["aabb_min"])).any():
                continue
            if b["id"] not in grown:
                grown[b["id"]] = dilate(keys[b["id"]])
            t = sum(1 for k in keys[sid] if k in grown[b["id"]]) / max(len(keys[sid]), 1)
            c = sum(1 for k in keys[sid] if k in keys[b["id"]]) / max(len(keys[sid]), 1)
            if (c, t) > (best_c, best_t):
                best, best_c, best_t = b, c, t
        if best is None:
            continue
        # a piece shares the bigger one's surfaces (its cells ARE the other's);
        # a thing on it only touches them (its cells are next to the other's)
        if rec >= part_recall or best_c < contain:
            if best_t >= touch and "on" not in o:          # a thing of its own, on / against it
                o["on"] = best["id"]
            continue
        b = best
        votes = label_votes(b)
        for k, v in label_votes(o).items():
            votes[k] = votes.get(k, 0.0) + v
        b["labels"] = {k: round(v, 2) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}
        byframe = {v["frame"]: v for v in b["views"]}
        for v in o["views"]:
            if v["frame"] not in byframe:
                byframe[v["frame"]] = v
        b["views"] = [byframe[f] for f in sorted(byframe)]
        b.setdefault("parts", []).append({"id": sid, "label": o["label"], "recall": round(rec, 4),
                                          "inside": round(best_c, 3), "n_points": o["n_points"]})
        xyz = np.vstack([pts[b["id"]][0], pts[sid][0]])
        rgb = np.vstack([pts[b["id"]][1], pts[sid][1]])
        u = np.unique(np.floor(xyz / 0.02).astype(np.int64), axis=0, return_index=True)[1]
        pts[b["id"]] = (xyz[u], rgb[u])
        keys[b["id"]] |= keys[sid]
        grown.pop(b["id"], None)
        for q in alive.values():
            if q.get("on") == sid:
                q["on"] = b["id"]
        del alive[sid]
        absorbed.append((o, b["id"], best_c, rec))
    left = list(alive.values())
    os.makedirs(os.path.join(reg, "absorbed"), exist_ok=True)
    for o, _, _, _ in absorbed:
        for f in [os.path.join(reg, o["points"])] + glob.glob(os.path.join(reg, "masks", o["id"] + "_*.png")):
            if os.path.exists(f):
                os.rename(f, os.path.join(reg, "absorbed", os.path.basename(f)))
    for o in left:
        if any(o["id"] == b for _, b, _, _ in absorbed):
            xyz, rgb = pts[o["id"]]
            write_ply(os.path.join(reg, o["points"]), xyz, rgb)
            o["n_points"] = int(len(xyz))
            o["centroid"] = np.round(xyz.mean(0), 4).tolist()
            o["aabb_min"] = np.round(xyz.min(0), 4).tolist()
            o["aabb_max"] = np.round(xyz.max(0), 4).tolist()
            o["n_views"] = len(o["views"])
            o["segmented"], o["seen"], r = recall_of(o)
            o["recall"] = round(r, 4)
        votes = label_votes(o)
        total = sum(votes.values())
        o["label"] = max(votes, key=votes.get)
        same = next((g for g in syn if o["label"] in g), {o["label"]})
        o["agreement"] = round(sum(v for k, v in votes.items() if k in same) / total, 3) if total else 1.0
        o["uncertain"] = o["agreement"] < agree
    left.sort(key=lambda o: (o["label"], -o["n_points"]))
    ren = renumber(left)
    rename_files(reg, left, ren)
    for o in left:
        if o.get("on") in ren:
            o["on"] = ren[o["on"]]
        elif "on" in o:
            o.pop("on")
    doc["objects"] = left
    json.dump(doc, open(path, "w"), indent=1)
    json.dump({"part_recall": part_recall, "contain": contain, "part_ratio": part_ratio,
               "absorbed": [{"id": o["id"], "label": o["label"], "into": ren.get(b, b),
                             "inside": round(c, 3), "recall": round(r, 4), "n_points": o["n_points"]}
                            for o, b, c, r in absorbed]},
              open(os.path.join(reg, "objects_absorbed.json"), "w"), indent=1)
    log("parts (recall < %.2f, >= %.0f %% inside a %.0fx bigger one): %d objects -> %d, %d absorbed"
        % (part_recall, 100 * contain, part_ratio, len(objs), len(left), len(absorbed)))
    into = {}
    for o, b, _, _ in absorbed:
        into.setdefault(ren.get(b, b), []).append(o["label"])
    for b, labs in sorted(into.items(), key=lambda kv: -len(kv[1])):
        log("  %-22s <- %s" % (b, ", ".join(labs)))
    unc = [o for o in left if o["uncertain"]]
    if unc:
        log("names without a majority (agreement < %.2f):" % agree)
        for o in unc:
            log("  %-22s %s" % (o["id"], ", ".join("%s %.0f" % kv for kv in o["labels"].items())))
    return left, absorbed


def on_walls(xyz, room, tol=0.12, share=0.8, extent=1.5):
    """Is a point set the room's shell: most of it within tol of one of the
    room's four walls (room = yaw deg, xmin, xmax, ymin, ymax, as crop_room)
    and extent m or more along it? -> (bool, extent along the wall, m)."""
    yaw, x0, x1, y0, y1 = room
    a = np.radians(-yaw)
    c, s_ = np.cos(a), np.sin(a)
    x = c * xyz[:, 0] - s_ * xyz[:, 1]
    y = s_ * xyz[:, 0] + c * xyz[:, 1]
    best = 0.0
    for d, along in ((np.abs(x - x0), y), (np.abs(x - x1), y), (np.abs(y - y0), x), (np.abs(y - y1), x)):
        near = d <= tol
        if near.mean() >= share:
            lo, hi = np.percentile(along[near], [2, 98])
            best = max(best, float(hi - lo))
    near_any = np.minimum.reduce([np.abs(x - x0), np.abs(x - x1), np.abs(y - y0), np.abs(y - y1)]) <= tol
    if near_any.mean() >= share and best == 0.0:          # along two walls round a corner
        best = float(max(np.ptp(x[near_any]), np.ptp(y[near_any])))
    return best >= extent, best


def prune(reg, min_views=5, min_recall=0.15, keep_views=20, keep_points=1000, room=None,
          wall_tol=0.12, wall_share=0.8, wall_extent=1.5, log=print):
    path = os.path.join(reg, "objects.json")
    doc = json.load(open(path))
    objs = doc["objects"]
    if any("seen" not in o for o in objs):
        raise SystemExit("prune needs views.py's 'seen' / 'segmented': run views.py first")
    shutil.copy(path, os.path.join(reg, "objects_before_prune.json"))
    keep, drop = [], []
    for o in objs:
        segm = len({v["frame"] for v in o["views"]})
        recall = o["segmented"] / max(o["seen"], 1)
        o["recall"] = round(recall, 4)
        ok = ((segm >= min_views and recall >= min_recall)
              or (segm >= keep_views and o["n_points"] >= keep_points))
        o.pop("dropped_as", None)
        if not ok:
            o["dropped_as"] = "rarely found" if segm >= min_views else "too few views"
        if ok and room is not None:
            shell, ext = on_walls(read_ply(os.path.join(reg, o["points"]))[0], room,
                                  wall_tol, wall_share, wall_extent)
            if shell:
                ok = False
                o["dropped_as"] = "shell (%.1f m along a wall)" % ext
        (keep if ok else drop).append(o)
    ids = {o["id"] for o in keep}
    for o in keep:
        if o.get("on") not in ids:
            o.pop("on", None)
    keep.sort(key=lambda o: (o["label"], -o["n_points"]))
    ren = renumber(keep)
    for o in drop:
        for f in [os.path.join(reg, o["points"])] + glob.glob(os.path.join(reg, "masks", o["id"] + "_*.png")):
            if os.path.exists(f):
                os.makedirs(os.path.join(reg, "pruned"), exist_ok=True)
                os.rename(f, os.path.join(reg, "pruned", os.path.basename(f)))
    rename_files(reg, keep, ren)
    for o in keep:
        if "on" in o:
            o["on"] = ren[o["on"]]
    doc["objects"] = keep
    json.dump(doc, open(path, "w"), indent=1)
    pruned_p = os.path.join(reg, "objects_pruned.json")
    earlier = json.load(open(pruned_p)) if os.path.isfile(pruned_p) else {}
    runs = earlier.get("runs", [{k: earlier[k] for k in earlier if k != "objects"}] if earlier else [])
    runs.append({"min_views": min_views, "min_recall": min_recall, "keep_views": keep_views,
                 "keep_points": keep_points, "room": room, "dropped": len(drop)})
    json.dump({"runs": runs, "objects": earlier.get("objects", []) + drop},
              open(pruned_p, "w"), indent=1)
    log("prune (>= %d views and recall >= %.2f, or >= %d views and >= %d points%s): "
        "%d objects -> %d kept, %d dropped"
        % (min_views, min_recall, keep_views, keep_points, "; room shell out" if room else "",
           len(objs), len(keep), len(drop)))
    from collections import Counter
    for why, n in Counter(o["dropped_as"].split(" (")[0] for o in drop).items():
        log("  dropped as %-14s %d" % (why + ":", n))
    for o in drop:
        if o["dropped_as"].startswith("shell"):
            log("    %-22s %s" % (o["id"], o["dropped_as"]))
    return keep, drop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("registry")
    ap.add_argument("step", choices=["merge", "parts", "prune"])
    ap.add_argument("--part-recall", type=float, default=0.3)
    ap.add_argument("--contain", type=float, default=0.6)
    ap.add_argument("--part-ratio", type=float, default=2.0)
    ap.add_argument("--agree", type=float, default=0.5)
    ap.add_argument("--synonyms", default=SYNONYMS)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--size-ratio", type=float, default=3.0)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.15)
    ap.add_argument("--keep-views", type=int, default=20)
    ap.add_argument("--keep-points", type=int, default=1000)
    ap.add_argument("--room", type=float, nargs=5, default=None,
                    metavar=("YAW", "XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--wall-tol", type=float, default=0.12)
    ap.add_argument("--wall-share", type=float, default=0.8)
    ap.add_argument("--wall-extent", type=float, default=1.5)
    a = ap.parse_args(argv)
    reg = os.path.expanduser(a.registry)
    if a.step == "merge":
        objs = merge(reg, a.overlap, a.size_ratio, a.cell)
    elif a.step == "parts":
        objs, _ = parts(reg, a.part_recall, a.contain, a.part_ratio, a.agree, a.cell,
                        synonyms=a.synonyms)
    else:
        objs, _ = prune(reg, a.min_views, a.min_recall, a.keep_views, a.keep_points, a.room,
                        a.wall_tol, a.wall_share, a.wall_extent)
    from collections import Counter
    c = Counter(o["label"] for o in objs)
    print("by label: " + ", ".join("%s %d" % kv for kv in sorted(c.items())))


if __name__ == "__main__":
    sys.exit(main())
