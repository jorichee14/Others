# Semantic map synthesis — stages [6] detect and [7] synthesize

Adds YOLO11-seg object detection and asset-conditioned map synthesis to
`01_build_map.py`, as a **side branch** off the colorized cloud:

```
merge → [dynamic] → denoise → colorize → [flatten] → [anchor] → map_final.pcd
                                  └────→ [6] detect → [7] synthesize → map_synth.pcd
                                                                       scene.json
```

With `detect.enable` false the main chain is byte-identical to before, so
turning this on cannot change the map an existing downstream stage reads.

## Files

| file | what it is |
|---|---|
| `01_build_map.py` | your stage script, with `detect()` / `synthesize()` wired in |
| `pipeline_detect.py` | detector wrapper, vote fusion, instance tracking, plane structure |
| `pipeline_assets.py` | asset library, fitting, repainting, surface repair |
| `assets/` | procedural placeholder models + manifest ([contract](assets/README.md)) |
| `test_semantics.py` | synthetic-room self-test — no bag, no weights, no GPU |
| `test_resume.py` | proves the expensive stages are skipped on a resume |
| `pipeline_config.example.json` | every key stage 01 reads, with working defaults |
| `check_config.py` | validates a config before you spend an hour on a bag |
| `pipeline_common.py` | unchanged, included so the folder runs standalone |

## Outputs

**From [6] detect** — the map, and what is in it:

| file | contents |
|---|---|
| `objects_inventory.yaml` | **the map inventory** — structure areas/heights, per-class counts, and every object with its position, size, support and cloud |
| `background.pcd` | **the room without its contents** — structure + unclaimed points |
| `objects.pcd` | every detected object point (the exact complement of the above) |
| `layers/*.pcd` | one per structure kind (`floor`, `wall`, `ceiling`, `support`) and one per detected class (`chair.pcd`, `tv.pcd`, …) |
| `objects/NNN_class.pcd` | one cloud per instance (`save_per_object`) |
| `semantic.pcd` | the map coloured by class, structure muted underneath |
| `instances.json` | machine-readable form of the inventory |
| `labels.npz` | per-point `cls`, `conf`, `frames`, `structure`, `inst`, `n_points` |
| `structure.json` | plane models + kinds, so a resume skips RANSAC |

`background.pcd` / `objects.pcd` **partition the map exactly** — every point is
in one or the other, never both and never neither, which the self-test asserts.
Background is the file to re-mesh or hand to a planner: object clutter is what
makes a map non-reusable between runs.

The inventory is hand-rolled YAML in the same style as `dump_cameras_yaml` in
`pipeline_common.py`, so stage 01 gains no new dependency for one output file:

```yaml
structure:
  floor:   { planes: 1, points: 50175, area_m2: 31.11, height_m: [0.0] }
  wall:    { planes: 4, points: 88140, area_m2: 57.20 }
  room_height_m: 2.600
counts:
  chair: 4
  tv: 1
objects:
  - id: 0
    class: chair
    confidence: 0.910
    centroid: [2.0655, 2.4833, 0.5953]
    extent: [0.51, 0.55, 0.94]
    on_wall: false
    support: floor
    cloud: objects/000_chair.pcd
```

Areas count occupied cells of an in-plane grid, not bounding boxes — a wall
with a doorway or an L-shaped floor is meaningfully smaller than its extent,
and a bbox would overstate a room by a third.

`semantic.pcd` is not an extra. Arrays cannot be reviewed — the only way to
know whether "chair" landed on the chair or on the wall behind it is to open
the map with the labels painted on. Class colours are deterministic (golden-
ratio hue stepping), so two runs are directly comparable and adjacent COCO ids
never collide. The console prints the same thing as a table: every object with
its class, world centroid, extent in metres, point count and view count.

**From [7] synthesize** — optional, off unless enabled:

| file | contents |
|---|---|
| `map_synth.pcd` | the modified cloud |
| `scene.json` | per-instance action, asset, 4×4 pose, scale, fit quality |

`scene.json` carries poses, so stage 03 can instantiate real **meshes** instead
of the point samples baked into `map_synth.pcd`.

## How [6] works

`project_visible()` already returns the map points visible from a camera pose,
one per pixel, z-buffered against occlusion — exactly the operator needed to
lift a 2D mask into 3D. Detect is colorize's loop with a different payload:
instead of writing a colour it casts a class vote.

A single frame's mask is never clean, so four guards apply in order:

1. **mask erosion** — strips the silhouette halo, where a boundary pixel sits
   on the object in one frame and on the wall 3 m behind it in the next.
2. **depth gate** (median + MAD) — drops background genuinely visible *through*
   the mask: between a chair's legs, between a plant's leaves.
3. **multi-frame voting** — bleed lands on a different background point each
   frame because the camera moves, so it collects one vote; the object collects
   one per view.
4. **wall-skirt trim** — see [Limitations](#limitations).

Instance identity comes from the detector rather than being rediscovered
spatially: two chairs side by side are one connected blob to any clustering,
but YOLO separated them in every image, and `InstanceTracker` keeps that.

Detect uses `pose_at_interp`, not stage 01's local `nearest_pose_idx`. At
walking pace a 0.1 s lever arm is tens of millimetres of camera error, which is
exactly what slides a mask off an object onto the surface behind it.

**Walls and floors are not detected.** COCO has no class for them. They come
from RANSAC planes classified by orientation against gravity — floor, ceiling,
wall, support (table/shelf tops). That is what makes "the TV is on a wall" a
computable predicate, and it gives removal a surface to patch afterwards.

## How [7] works

Each instance gets an action from a rules table keyed by class and, optionally,
by what it is attached to:

- **`remove`** — delete the points *and* re-sample the plane behind them. The
  LiDAR never measured the wall behind a TV, so deleting without filling leaves
  a TV-shaped void. The patch is rasterised over exactly the missing footprint
  and coloured from the surviving ring, so it matches the wall rather than the
  object that was there.
- **`replace`** — **keep the measured points** and add an asset fitted to them:
  yaw from horizontal PCA, clamped anisotropic scale, ICP refinement projected
  back onto the gravity axis, base snapped to the classified support plane, and
  every asset point repainted from the measured colours around it. Faces the
  sensor never saw keep the baked colour, tinted toward the instance median.
- **`keep`** — default.

If the fit does not explain the measurement (`min_coverage`), **no asset is
emitted**. A bad fit degrades to "unchanged", never to two overlapping chairs.

## Config

Both blocks go under `01_build_map`. Full annotated defaults are in the
`01_build_map.py` docstring. The rule from the brief:

```json
"rules": {
  "tv":           {"on_wall": "remove", "default": "replace"},
  "clock":        "remove",
  "person":       "remove",
  "potted plant": "replace",
  "chair":        "replace",
  "dining table": "replace",
  "couch":        "replace"
}
```

Path resolution is worth knowing: `P.stage()` resolves `input`/`output`/
`frame_out`/`anchor_frame`/`script_out` against `out_dir`. Everything else is
resolved explicitly, and **inputs (`detect.weights`, `synthesize.assets`)
resolve against the config's directory, not `out_dir`** — the asset library
lives with the config, and `out_dir` is often scratch.

## Running

```bash
pip install ultralytics                       # torch + YOLO
cp pipeline_config.example.json pipeline_config.json
$EDITOR pipeline_config.json                  # set the four dataset paths
python3 check_config.py pipeline_config.json  # validate BEFORE the long run
python3 01_build_map.py pipeline_config.json
```

`check_config.py` exists because stage 01 reads its config **lazily**:
`flatten` isn't touched until after the merge, `detect.weights` not until after
colorize. A missing key therefore fails forty minutes in, having already done
the expensive work — and fails again at the same place on every resume. The
checker walks every key the stage will eventually read and reports all the
problems at once.

### The one trap worth knowing

`image_width` / `image_height` must equal the resolution the **intrinsics were
calibrated at**. Stage 01 resizes images to `(W, H)` but does *not* rescale
`fx/fy/cx/cy`, so a mismatch projects every point to the wrong pixel — colours
smear and detections land on the wrong surfaces, with nothing in the log to say
why. `check_config.py` catches it by comparing `2·cx` against `image_width`.
Either set W/H to the calibration resolution, or rescale the intrinsics.

### Resume — already have a colorized cloud?

Stage 01 resumes from the **furthest** intermediate present, and deleting a
file is how you force it (and everything after it) to recompute:

```
merged.pcd → static.pcd → denoised.pcd → colored.pcd → labels.npz
```

Finding `colored.pcd` skips **merge, dynamic, denoise and colorize in one
step**. That is the case that matters: colorize reads every image in the bag
and projects the whole map per frame, so re-running it merely to reach detect
costs more than detect does. If your existing cloud has another name, point
`colorize.output` at it rather than copying:

```json
"colorize": { "enable": true, "output": "map_final_20260730.pcd", ... }
```

Two rules the resume enforces, both tested in `test_resume.py`:

- a `colored.pcd` is only reused when **`colorize.enable` is true**. Reusing a
  colorized cloud after colorize was switched off would hand back a different
  cloud than the config asks for, so it falls through to merge instead.
- `labels.npz` records the point count it was built for and **refuses to load
  against a different cloud**. If `drop_gray` changed or the merge was rebuilt,
  stale indices would still resolve and every label would land silently on a
  different point. Delete it to re-detect.

## Limitations

Found while building, stated rather than papered over:

- **A depth gate cannot separate a flat object from its own wall.** A TV 6 cm
  proud, seen from 2.6 m and off-axis, has ~40 cm of depth spread across its own
  face — many times the standoff. Multi-view voting does not rescue it either:
  the bleed ring sits in nearly the same world place from every viewpoint that
  can see the object at all, so it accumulates votes just like the object. The
  fix is `Structure.trim_wall_skirt`, which uses the one thing neither test has
  — the wall's plane — and drops points lying *in* a wall that the bulk of the
  instance stands proud of. It self-disables for genuinely flush objects
  (posters, flat panels), whose median residual is near zero. In the self-test
  this takes TV labelling from 93.3% to 100% precision at unchanged recall.
- **ICP must run measurement→asset, not asset→measurement.** A scan is
  one-sided; an asset used as the ICP source drags its never-observed back and
  underside toward whatever is nearest. The same reasoning governs candidate
  scoring. Getting this backwards costs ~25° of yaw error.
- **Assets are placeholders.** Metrically honest and correctly framed, but
  generic. Drop real models into the same manifest slots — no code changes.
- **VRAM is shared with torch.** The model lives outside CuPy's pool and is
  invisible to `memGetInfo()` until allocated, so `detect.gpu_reserve_gib`
  (default 2.0) is held back before deciding whether the cloud goes resident.

## Not changed

`main()`'s anchor step still doesn't write `cam0_shift` into the config, so
stage 03 continues to print its "stage-01 anchored the cloud but did not record
its shift" warning and recompute it. Left alone — it predates this work and
fixing it changes stage 03's inputs.
