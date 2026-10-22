#!/usr/bin/env python3
"""Make LOD-3DGS (github.com/zhaofuq/LOD-3DGS) read the LOD levels that
`lod3dgs.py scene` writes (lod/level_<k>.ply) instead of running its
Windows PotreeConverter: the same change as lod3dgs_lod_levels.patch, put
into scene/dataset_readers.py by this script (whatever the file's line
numbers or line endings), once.

    python3 scoop_pipeline/datasets/lod3dgs_patch.py ~/workspaces/LOD-3DGS
"""
import os
import sys

BLOCK = '''    lod_dir = os.path.join(path, "lod")
    if os.path.isdir(lod_dir):
        # levels made beforehand (scoop_pipeline/datasets/lod3dgs.py scene): level_0.ply the
        # coarsest, each one every point at its resolution; no LAS, no PotreeConverter
        names = [f for f in os.listdir(lod_dir) if f.startswith("level_") and f.endswith(".ply")]
        names.sort(key=lambda f: int(f[len("level_"):-len(".ply")]))
        pcds = [fetchPly(os.path.join(lod_dir, f)) for f in names]
        print("[ Dataloader ] {} LOD levels from {}: {} points".format(
            len(pcds), lod_dir, [len(p.points) for p in pcds]))
        return SceneInfo(point_cloud=pcds,
                         train_cameras=train_cam_infos,
                         test_cameras=test_cam_infos,
                         nerf_normalization=nerf_normalization,
                         ply_path=os.path.join(path, "sparse/0/points3D.ply"),
                         max_level=len(pcds) - 1)

'''


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    p = os.path.join(root, "scene", "dataset_readers.py")
    src = open(p, newline="").read()
    nl = "\r\n" if "\r\n" in src else "\n"
    text = src.replace("\r\n", "\n")
    if 'lod_dir = os.path.join(path, "lod")' in text:
        print(f"{p}: already patched")
        return
    i = text.find("def readoctreeColmapInfo(")
    if i < 0:
        sys.exit(f"{p}: no readoctreeColmapInfo: not LOD-3DGS's scene/dataset_readers.py")
    anchor = "    nerf_normalization = getNerfppNorm(train_cam_infos)\n"
    j = text.find(anchor, i)
    if j < 0:
        sys.exit(f"{p}: the line '{anchor.strip()}' not found in readoctreeColmapInfo")
    j += len(anchor)
    if text[j:j + 1] == "\n":
        j += 1
    out = text[:j] + BLOCK + text[j:]
    open(p, "w", newline="").write(out.replace("\n", nl))
    print(f"{p}: patched (the lod/ levels read)")


if __name__ == "__main__":
    main()
