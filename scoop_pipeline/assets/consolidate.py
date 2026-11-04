#!/usr/bin/env python3
"""The registry's objects made one per physical thing: merged across labels,
small things on bigger ones recorded, and (after views.py) the fragments and
false hits pruned.

    python3 scoop_pipeline/assets/consolidate.py <registry> merge
        [--overlap 0.5] [--size-ratio 3] [--cell 0.05]
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
    ap.add_argument("step", choices=["merge", "prune"])
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
    else:
        objs, _ = prune(reg, a.min_views, a.min_recall, a.keep_views, a.keep_points, a.room,
                        a.wall_tol, a.wall_share, a.wall_extent)
    from collections import Counter
    c = Counter(o["label"] for o in objs)
    print("by label: " + ", ".join("%s %d" % kv for kv in sorted(c.items())))


if __name__ == "__main__":
    sys.exit(main())
