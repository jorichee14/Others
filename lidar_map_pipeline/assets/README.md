# Asset library — stage [7] synthesize

Stage [7] does **not** paste these models into the map verbatim. It fits each
asset to the measured cluster and repaints it from what the camera actually
saw, so the asset is a *prior on shape*, not the output geometry:

1. **fit** — yaw from the instance OBB (skipped when `yaw_symmetry` is `0`),
   then anisotropic scale against the asset's own bbox, clamped, then ICP
   refinement against the measured points.
2. **snap** — the base is translated onto the classified support plane
   (floor / table top), which is why `min(z) == 0` is mandatory.
3. **repaint** — every asset point takes the median colour of the measured
   points near it. Where nothing was measured — the back of a couch against a
   wall — the baked placeholder colour survives, tinted toward the instance's
   median.
4. **merge** — the measured points are *kept*; the fitted asset is added
   alongside them.

## Contract

Any asset dropped in here, procedural or hand-made, must honour:

| property | requirement | why |
|---|---|---|
| units | metres | initial scale ≈ 1, so the clamp band is meaningful |
| up axis | `+z` | matches the GLIM world frame; the floor snap is a pure translate |
| front axis | `+y` | gives the yaw search a known starting orientation |
| anchor | origin at footprint centre, `min(z) == 0` | base lands exactly on the support plane |
| colour | vertex colours present | unseen faces need a plausible fallback |
| geometry | no coincident or interior faces | uniform sampling would place points inside the solid, where no surface exists |

`make_assets.py --verify` checks all of these except the last, which is a
review matter.

## Layout

```
assets/
  manifest.json          generated — class → variants, sizes, support, symmetry
  make_assets.py         regenerates every procedural placeholder, deterministically
  meshes/<class>/*.ply   the assets themselves
  models/                YOLO11-seg weights (gitignored — see below)
```

Class keys are **exact COCO names** (`"potted plant"`, `"dining table"`, `"tv"`),
because that is what the detector emits and what the stage `rules` table is
keyed on.

### Manifest fields

- `support` — `floor` sits on the classified floor plane; `surface` prefers a
  table/shelf top under the instance and falls back to the floor; `any` takes
  whichever plane is directly beneath.
- `yaw_symmetry` — rotations of 360/n leaving the shape unchanged. `1` = none,
  so the full yaw search plus the 180° view disambiguation runs; `2` = front
  and back alike; `0` = continuous, skip the search. Both a cost knob and a
  correctness one — searching yaw on a round vase fits noise.
- `size` — measured from the mesh, not declared. `--verify` fails on drift.

## Adding a real model

Drop the file under `meshes/<class>/`, add a variant entry to
`manifest.json`, and re-run `make_assets.py --verify` to check it against the
contract. No pipeline code changes. Multiple variants per class are allowed;
stage [7] picks the one whose bbox aspect ratio is closest to the measured
instance's, so a set of three chairs beats one generic chair.

**Do not regenerate the manifest with `make_assets.py` after hand-adding
variants** — the builder rewrites it from `ASSETS` and would drop them. Add a
builder entry instead, or edit the manifest directly.

## Weights

`models/` is gitignored. Fetch the segmentation weights once:

```bash
pip install ultralytics
yolo settings                      # optional: check the download directory
python3 -c "from ultralytics import YOLO; YOLO('yolo11x-seg.pt')"
mv ~/.config/Ultralytics/yolo11x-seg.pt models/   # or wherever it landed
```

Point `detect.weights` in `pipeline_config.json` at the result. `yolo11m-seg.pt`
is a reasonable speed/quality trade if the run is long; `yolo11n-seg.pt` misses
small furniture.
