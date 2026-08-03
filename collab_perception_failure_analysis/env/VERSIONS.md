# Run-machine environment record (fill in after Step 0.1 gate passes)

| Item | Value |
|------|-------|
| Machine (I1) | `wicomsrobot` (local, Ubuntu + GNOME) |
| GPU / VRAM (I2) | NVIDIA GeForce RTX 3080, 12GB |
| NVIDIA driver / CUDA (from `nvidia-smi`) | 580.173.02 / CUDA 13.0 |
| CUDA toolkit (from `nvcc --version`) | 11.5 (V11.5.119, system) — conda 11.7 fallback if extension build objects |
| Python | 3.8 (conda env `opencood`) |
| torch / torchvision | 1.13.1+cu117 / 0.14.1+cu117 |
| spconv package | spconv-cu117 2.3.6 |
| numpy | 1.23.5 |
| OpenCOOD commit (`git rev-parse HEAD` in the clone) | _pending — run `git -C ~/cpfa/OpenCOOD rev-parse HEAD`_ |
| Dataset root path | `~/cpfa/data/OPV2V` (planned; confirm at Step 0.2) |
| Checkpoint root path | `~/cpfa/checkpoints` (planned; confirm at Step 0.3) |
