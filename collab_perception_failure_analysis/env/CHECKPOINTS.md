# Checkpoint manifest (fill in during Step 0.3; one row per checkpoint folder)

Source: OpenCOOD README "Benchmark and model zoo" table (OPV2V LiDAR track),
https://github.com/DerrickXuNu/OpenCOOD#benchmark-and-model-zoo — checkpoints hosted on
UCLA Box (https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V) and Google Drive.

md5 via: `md5sum ~/cpfa/checkpoints/*/net_epoch*.pth`

Downloaded 2026-08-04 into `~/cpfa/checkpoints/` (folder names as shipped):

| Folder | Algorithm (README row) | Source URL | Weights file | md5 |
|--------|------------------------|------------|--------------|-----|
| pointpillar_late_fusion | Late fusion ("Naive Late", PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621128604521 | net_epoch30.pth | eed40b69d9c3c6e3c5ce5787bba0a034 |
| pointpillar_early_fusion | Early fusion ("Cooper", PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621122534978 | latest.pth | 83851f7bf3fe3471cb053e34d5534f92 |
| pointpillar_attentive_fusion | AttFuse ("Attentive Fusion", PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621110356814 | latest.pth | 35c2953c7a168d203c8311c14c75e57b |
| pointpillar_attentive_fusion_compression | AttFuse, compression variant | (same download as above) | latest.pth | 618071b0e2cc50915c99488fce8f5916 |
| pointpillar_fcooper | F-Cooper (PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621123814293 | latest.pth | 39ce8dd1c69d9916ac0d2845ef5c8b0f |
| pointpillar_v2vnet | V2VNet (PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621111444798 | net_epoch83.pth | 1804d01ceab6de22daa9817927c5ca7f |
| point_pillar_coalign | CoAlign (PointPillar) | https://drive.google.com/file/d/1mUEI_Dh4tkG6-LG3QcZ05kK7oOGJzCGK/view?usp=sharing | net_epoch15.pth | b9577e1e8b11b93fc20ad4a1b404bf55 |
| point_pillar_coalign_compression | CoAlign, compression variant | (same download as above) | net_epoch20.pth | 9e0220515a0281eeef570a00f85b8018 |
| pointpillar_CoBEVT_nocompression | CoBEVT (PointPillar, no compression) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/folder/280139625059 | net_epoch19.pth | e90b54f53ce5644bf2765a85827b6663 |
| cobevt_compression | CoBEVT (PointPillar, compression variant) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/folder/280139625059 | net_epoch33.pth | bcd084b3ea557e252f93a853aa29d651 |

Naming caveat: four checkpoints ship weights as `latest.pth`, but OpenCOOD's model loader
scans for `net_epoch*.pth` and picks the highest epoch. Before inference, copy each
`latest.pth` to `net_epoch1.pth` in the same folder (keep `latest.pth` so the md5s above
stay valid). Affected: pointpillar_early_fusion, pointpillar_attentive_fusion,
pointpillar_attentive_fusion_compression, pointpillar_fcooper.

Notes:
- **No-Comm floor:** the model zoo ships no dedicated single-vehicle checkpoint. The late
  fusion model is trained per-vehicle (each CAV detects independently), so the No-Comm
  floor is obtained by evaluating `pointpillar_late` with ego-only input (no collaborator
  messages) — handled by the Phase 1 runner, no separate download.
- Reference AP@0.7 (Default Towns / Culver City) from the README, for the 0.4 smoke test
  and Phase 1 sanity checks: Late 0.781/0.668 · Early 0.800/0.696 · AttFuse 0.815/0.810 ·
  F-Cooper 0.790/0.788 · V2VNet 0.822/0.814 · CoAlign 0.833/0.806 · CoBEVT 0.861/0.836.
