# LiDAR map → clean point cloud → Sionna RT mesh

Two-stage pipeline:

1. `01_build_map.py` — merge LiDAR scans placed by GLIM poses into a world
   cloud, **remove dynamic objects**, denoise, colorize.
2. `02_pcd_to_mesh_sionna_v9.py` — turn the cleaned cloud into a flat,
   smooth, accurate mesh split into walls / floor / ceiling / objects for
   per-part ITU materials in Sionna.

```
python3 01_build_map.py pipeline_config.json
python3 02_pcd_to_mesh_sionna_v9.py out/static.pcd mesh_sionna.ply
```

---

## Problem 1 — "GLIM poses, but dynamic points still in the output"

GLIM only gives you *accurate poses*. Good poses make static structure
overlap perfectly across scans, but they do nothing about points that were
returned **from moving objects** — those land wherever the object was at
each scan time and smear into ghost trails / frozen silhouettes.

The old filter used only a **time-span test** per world voxel:

```
static  ⇔  (last_hit − first_hit) ≥ min_span_s  AND  hits ≥ min_hits
```

That catches *fast* movers (a walking person clears a 0.15 m voxel in well
under a second) but is **blind by construction** to anything that dwells
longer than `min_span_s` in one voxel:

- a person standing still for a few seconds,
- someone walking alongside / behind the robot (stays in the same voxels
  relative to nothing — but revisits world voxels slowly),
- a slow mover, a cart, a car that parks and later leaves.

Those pass the span test and survive. That's the residue you're seeing.

### The fix: free-space carving (visibility check)

Every LiDAR return also proves the ray from the sensor to the hit passed
through **empty space**. `01_build_map.py` now marches (subsampled) rays
through the same decision grid and counts, per voxel, in how many distinct
scans it was *seen through*. A voxel that has occupancy hits **and** many
see-through observations held something that was sometimes there and
sometimes not → dynamic, **no matter how long it dwelled**:

```
dynamic  ⇔  free ≥ min_free  AND  free ≥ free_ratio · hits
```

This is the same principle as Removert / ERASOR / OctoMap. Static walls
are never seen through (rays stop at them), so `min_free` / `free_ratio`
absorb noise-level counts. Carving is collected during the merge pass
(one bag read) or replayed on resume from `merged.pcd`.

### Config

```json
"remove_dynamic": {
  "enable": true,
  "voxel": 0.15,
  "min_span_s": 1.0,
  "min_hits": 2,
  "save": true,
  "carve": {
    "enable": true,
    "ray_stride": 4,
    "scan_stride": 2,
    "max_range": 20.0,
    "endpoint_margin": 0.0,
    "min_free": 3,
    "free_ratio": 0.25
  }
}
```

Tuning:

| symptom | change |
|---|---|
| ghosts still survive | lower `free_ratio` (0.1), lower `ray_stride`/`scan_stride` (denser evidence), coarser `voxel` (0.2) |
| thin static stuff disappears (railings, poles, foliage) | raise `min_free` / `free_ratio`, raise `endpoint_margin` |
| too slow | raise `ray_stride` / `scan_stride`, lower `max_range` |

A built-in guard band of 2 voxels before each hit prevents rays from
carving the surface they terminate on; `endpoint_margin` adds to it.

**Two inherent limits:** (a) an object that never moves during the
*entire* recording (a parked car that stays parked) is static as far as
the data can tell — no filter can remove it; (b) carving needs rays to
pass *through* the mover's old position, i.e. some structure behind it
from some viewpoint — a mover silhouetted only against open sky gets no
free-space evidence there. Runs that revisit/sweep the scene maximize
both kinds of evidence.

---

## Layer 3 — object detection & inventory (stage 01, `[6]`)

Optional YOLO stage that turns the single cloud into **layers**:

| layer | file | what it is |
|---|---|---|
| 0 | `map_final_*.pcd` | everything (geometry, unchanged) |
| 1 | `background.pcd` | map **minus** all detected objects — the clean structural shell, the best input for the mesher |
| 2 | `objects.pcd` | object points only, photo colours |
| 2 | `objects_by_instance.pcd` | same points, one distinct colour per instance |
| 2 | `objects/<id>_<label>.pcd` | per-object clouds (`save_per_object: true`) |
| 3 | `objects_inventory.yaml` | the inventory itself |

### How a 2D detection becomes a 3D object

1. Per (strided) image, map points visible from that pose are found by radius
   cull → pinhole projection → **z-buffer**, so an occluded point can never
   take a label meant for the surface in front of it. Same helper colorize
   uses, so both stages agree on which point owns a pixel.
2. Each visible point inherits the class of the mask pixel it lands on
   (segmentation weights strongly preferred; boxes are shrunk by
   `bbox_shrink`, masks eroded by `mask_erode` — the outer ring straddles the
   silhouette). A **depth band** around the detection's median depth stops a
   detection from labelling the wall metres behind it.
3. Votes **accumulate across frames**. A label survives only with multi-view
   agreement: `min_votes` absolute **and** `min_ratio` of the frames in which
   that point was actually visible. This is what kills single-frame false
   positives — one bad detection out of ten views never reaches the map.
4. Surviving points are **DBSCAN-clustered per class** in 3D. Each cluster is
   one inventory entry, measured with a PCA footprint (yaw + length/width +
   height) rather than a full 3D OBB, which degenerates on the flat one-sided
   clusters LiDAR produces.

### Inventory format

```yaml
map:
  source: map_final_20260722.pcd
  frame: map                # or map_anchored, with anchor_shift applied
  floor_z: -1.47
  model: yolo11n-seg.pt
  frames_used: 1204
  background_points: 41203112
  object_points: 2841901
totals:
  chair: 12
  table: 4
objects:
  - id: 1
    label: chair
    class_id: 56
    det_conf: 0.71          # mean YOLO confidence for that class
    view_agreement: 0.62    # median votes / times seen — fusion confidence
    n_points: 5321
    centroid: [3.12, -1.44, -1.02]
    aabb_min: [...]
    aabb_max: [...]
    size: [0.61, 0.58, 0.91]
    footprint: { yaw_deg: 35.5, length: 0.61, width: 0.58, height: 0.91 }
    base_z: -1.44
    height_above_floor: 0.03
    mean_rgb: [0.31, 0.28, 0.26]
```

Coordinates always match `map_final.pcd` — if `anchor_camera_start` is on,
the same shift is applied to the inventory and recorded as `anchor_shift`.

### Notes

- Needs `pip install ultralytics`. Use a **`-seg`** checkpoint; box-only
  weights work but produce much coarser objects.
- Run it **after** dynamic removal — the stages are complementary: carving
  removes what *moved*, detection names what *stayed*. `person` is excluded
  by default since carving already handles them.
- Cost is dominated by YOLO; `img_stride` is the main lever (5 ≈ every 5th
  frame is plenty given multi-view voting).
- **Feed `background.pcd` to the mesher** for a clean structural shell, and
  mesh objects separately (or place them from the inventory as boxes) — that
  gives Sionna per-object materials for free.

### Tuning

| symptom | change |
|---|---|
| objects missing | lower `conf`, lower `min_votes`/`min_ratio`, lower `img_stride`, raise `max_range` |
| walls bleed into objects | raise `mask_erode`/`bbox_shrink`, lower `depth_band`, raise `min_ratio` |
| one object split into several | raise `cluster.eps` |
| touching objects merged | lower `cluster.eps` |
| clutter in the inventory | raise `cluster.min_pts_keep`, or set an explicit `classes` allowlist |

---

## Problem 2 — "the mesh looks awful"

The v8 mesher ran **Poisson over everything first** and then tried to
flatten the result. That inherits every Poisson pathology:

- wavy wall sheets wherever planarization missed an area,
- blobby closures across scan gaps,
- ghost wisps (from the dynamic residue) each wrapped in a surface,
- a decimate → re-project → weld cycle that leaves kinks at plane joints.

`02_pcd_to_mesh_sionna_v9.py` inverts the order — **plane-first**:

1. **Denoise the points before anything sees them**: SOR + *radius*
   outlier removal (kills the low-density ghost wisps SOR misses) +
   iterated MLS projection (collapses the multi-scan "thick wall" skin
   onto the true surface).
2. **Fit planes directly to the points** (two-phase RANSAC: big planes of
   any orientation, then a vertical-only sweep for short wall segments;
   coplanar fragments merged into one plane).
3. **Mesh each plane in 2D**: rasterize inliers on the plane at 0.10 m,
   morphologically close (seals scan gaps ≤ 0.3 m; doorways/windows stay
   open), drop floating islands, dilate one cell to seal wall/wall corners
   and wall/floor joints, grid-triangulate. The structure is **exactly
   flat by construction** — no Poisson involved, nothing to planarize,
   nothing wavy, low uniform triangle count. Extent follows the points, so
   it stays accurate to the map.
4. **Objects only** go through Poisson: fine points > `obj_keep_dist`
   from the structure (anything closer is wall skin → dropped) are
   DBSCAN-clustered (crumbs never reach the mesher), Poisson-meshed at
   depth 10 with a tight crop, Taubin-smoothed, then residual films and
   crumbs are removed.
5. Colour is sampled per-vertex from the fine cloud (cosmetic — Sionna
   materials come from the split files).

### Tuning

| knob | meaning | move it when |
|---|---|---|
| `plane_dist` (0.08) | RANSAC inlier distance | walls still fragment → raise; furniture gets eaten into walls → lower |
| `wall_area` (1.0 m²) | min vertical plane size | short wall stubs stay "objects" → lower; furniture faces become walls → raise |
| `grid_cell` (0.10) | structure mesh pitch | lighter mesh → 0.2; more colour detail → keep |
| `close_cells` (3) | gap sealing | holes in walls → raise; door openings closing up → lower |
| `obj_keep_dist` (0.15) | wall-skin vs object split | wall fuzz becomes objects → raise; switches/frames vanish → lower |
| `obj_min_pts` (300) | crumb threshold | floating debris → raise; small real objects vanish → lower |

### Using it in Sionna

- Units are already meters; PLY is directly importable (or convert to the
  Mitsuba XML scene format).
- Assign ITU materials per split file, e.g. `itu_concrete` for
  `*_walls.ply`, `itu_floorboard` for `*_floor.ply`,
  `itu_ceiling_board` for `*_ceiling.ply`, `itu_metal` or `itu_wood` for
  `*_objects.ply`.
- Watertightness is **not** required by Sionna RT; open doorways and the
  open ceiling/courtyard are fine and physically correct.
- Keep total triangles in the low hundreds of thousands for fast ray
  tracing; `grid_cell` and `obj_depth` are the two levers.

### The workflow that gets you an accurate, smooth, RT-ready mesh

1. Run stage 01 **with carving enabled** and eyeball `static.pcd` — the
   mesh can only ever be as clean as its input cloud. Iterate on the
   `remove_dynamic` block until ghosts are gone.
2. Feed `static.pcd` (or `denoised.pcd` / colored final) to the v9
   mesher.
3. Check the split PLYs in MeshLab/CloudCompare: walls should be perfectly
   flat sheets with real doorways; objects should be recognizable
   furniture, not confetti. Tune with the table above.
4. Import the split files into Sionna with per-part ITU materials.
