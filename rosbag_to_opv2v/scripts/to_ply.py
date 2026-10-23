#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A cloud as a PLY a viewer opens, with the ceiling already off and the
heights coloured, for picking objects by hand.

    python3 scripts/to_ply.py --pcd cloud_in_ref.pcd --out cloud.ply
        [--ceiling 2.0] [--floor-z Z] [--below 0.05] [--no-colour] [--voxel 0.03]

The cut is the one the labeller uses: everything more than --ceiling above the
floor goes, so a scanned ceiling stops hiding the room from above, and
everything below floor + --below goes with it, so the objects stand clear of
the ground. Colours run from blue just off the floor through green and yellow
to red at head height, which is what makes a chair or a pot visible against a
wall. The floor's z is found as in the labeller, or given with --floor-z.

The file opens in CloudCompare, MeshLab or Open3D. A point picked there is a
seed: pass its x, y back as `label_static.py --seed=X,Y`. A cloud cropped
there and saved can also be handed straight to `--pcd`, which reads PLY too --
with `--floor-z` set to the floor printed here, since a cloud cut above the
floor no longer contains the floor the fit would otherwise measure.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ros2opv2v.statics import ground_level, read_cloud_xyz, write_ply     # noqa: E402


def height_colours(z_above: np.ndarray, top: float = 1.6) -> np.ndarray:
    """Blue just off the floor -> green -> yellow -> red at ``top``, white over."""
    import colorsys
    t = np.clip(z_above / max(top, 1e-6), 0, 1)
    lut = np.array([colorsys.hsv_to_rgb(0.66 * (1 - u), 1.0, 1.0) for u in np.linspace(0, 1, 256)])
    rgb = (lut[(t * 255).astype(int)] * 255).astype(np.uint8)
    rgb[z_above > top] = (255, 255, 255)
    return rgb


def convert(pcd: str, out: str, ceiling: float = 2.0, below: float = 0.05,
            floor_z=None, colour: bool = True, voxel: float = 0.0, log=print) -> int:
    cloud = read_cloud_xyz(os.path.expanduser(pcd))
    log(f"{pcd}: {len(cloud)} points")
    ground = float(floor_z) if floor_z is not None else ground_level(cloud)[0]
    log(f"floor z {ground:+.3f} m" + ("" if floor_z is not None else "  (histogram mode; --floor-z overrides)"))
    above = cloud[:, 2] - ground
    keep = above >= below
    if ceiling:
        keep &= above <= ceiling
    cut = cloud[keep]
    log(f"kept {len(cut)} points between {below:.2f} m and "
        f"{('%.2f m' % ceiling) if ceiling else 'the top'} over the floor "
        f"({len(cloud) - len(cut)} dropped: the floor itself, the ceiling and the walls' tops)")
    if voxel > 0 and len(cut):
        key = np.floor(cut / voxel).astype(np.int64)
        _, first = np.unique(key[:, 0] * 1_000_003 ** 2 + key[:, 1] * 1_000_003 + key[:, 2],
                             return_index=True)
        cut = cut[np.sort(first)]
        log(f"thinned to {len(cut)} points at a {voxel} m voxel")
    colours = height_colours(cut[:, 2] - ground) if colour and len(cut) else None
    n = write_ply(os.path.expanduser(out), cut, colours)
    log(f"wrote {out}: {n} points{' with height colours' if colours is not None else ''}")
    log(f"pick in a viewer, then fit with:  --seed=X,Y --floor-z {ground:.3f}"
        f"   (the cut cloud has no floor of its own)")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", required=True, help="the cloud to convert (.pcd or .ply)")
    ap.add_argument("--out", required=True, help="the .ply to write")
    ap.add_argument("--ceiling", type=float, default=2.0,
                    help="drop everything this far above the floor (0 keeps it all)")
    ap.add_argument("--below", type=float, default=0.05,
                    help="drop everything below this over the floor (the floor itself)")
    ap.add_argument("--floor-z", type=float, default=None)
    ap.add_argument("--no-colour", dest="colour", action="store_false")
    ap.add_argument("--voxel", type=float, default=0.0, help="thin to one point per voxel")
    a = ap.parse_args()
    convert(a.pcd, a.out, a.ceiling, a.below, a.floor_z, a.colour, a.voxel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
