# Checkpoint manifest (fill in during Step 0.3; one row per checkpoint folder)

Source: OpenCOOD README "Benchmark and model zoo" table (OPV2V LiDAR track),
https://github.com/DerrickXuNu/OpenCOOD#benchmark-and-model-zoo — checkpoints hosted on
UCLA Box (https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V) and Google Drive.

md5 via: `md5sum ~/cpfa/checkpoints/*/net_epoch*.pth`

| Folder | Algorithm (README row) | Source URL | Weights file | md5 |
|--------|------------------------|------------|--------------|-----|
| pointpillar_late | Late fusion ("Naive Late", PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621128604521 | _pending_ | _pending_ |
| pointpillar_early | Early fusion ("Cooper", PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621122534978 | _pending_ | _pending_ |
| pointpillar_attentive_fusion | AttFuse ("Attentive Fusion", PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621110356814 | _pending_ | _pending_ |
| pointpillar_fcooper | F-Cooper (PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621123814293 | _pending_ | _pending_ |
| pointpillar_v2vnet | V2VNet (PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621111444798 | _pending_ | _pending_ |
| pointpillar_coalign (optional now) | CoAlign (PointPillar) | https://drive.google.com/file/d/1mUEI_Dh4tkG6-LG3QcZ05kK7oOGJzCGK/view?usp=sharing | _pending_ | _pending_ |
| cobevt (optional now) | CoBEVT (PointPillar) | https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/folder/280139625059 | _pending_ | _pending_ |

Notes:
- **No-Comm floor:** the model zoo ships no dedicated single-vehicle checkpoint. The late
  fusion model is trained per-vehicle (each CAV detects independently), so the No-Comm
  floor is obtained by evaluating `pointpillar_late` with ego-only input (no collaborator
  messages) — handled by the Phase 1 runner, no separate download.
- Reference AP@0.7 (Default Towns / Culver City) from the README, for the 0.4 smoke test
  and Phase 1 sanity checks: Late 0.781/0.668 · Early 0.800/0.696 · AttFuse 0.815/0.810 ·
  F-Cooper 0.790/0.788 · V2VNet 0.822/0.814 · CoAlign 0.833/0.806 · CoBEVT 0.861/0.836.
