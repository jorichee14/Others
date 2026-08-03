# Phase 0 — Testbed setup instructions

Run these on the GPU machine (I1). Every step ends with a **gate** — run it before moving
on, and paste its output back into the session so `IMPLEMENTATION.md` gets updated.
`scripts/verify_phase0.py` automates the gates.

Suggested layout on the run machine (paths are yours to choose; record the real ones in
`env/VERSIONS.md`):

```
~/cpfa/
├── OpenCOOD/            # cloned framework
├── data/OPV2V/          # dataset (test/ at minimum)
└── checkpoints/         # one folder per pretrained model
```

---

## Step 0.1 — Environment

### 0.1.0 Find out what CUDA you have

```bash
nvidia-smi            # top-right "CUDA Version" = max your driver supports
nvcc --version        # CUDA *toolkit* version (needed to compile the NMS op)
```

If `nvcc` is missing, install a CUDA toolkit ≤ the driver's version (conda works:
`conda install -c nvidia cuda-toolkit=11.7` inside the env, after creating it below).

### 0.1.1 Create the environment

```bash
conda create -y -n opencood python=3.8
conda activate opencood
```

### 0.1.2 Install PyTorch + spconv (pick ONE row matching your CUDA)

| Driver CUDA | PyTorch install | spconv install |
|---|---|---|
| 11.3 – 11.6 | `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113` | `pip install spconv-cu113` |
| ≥ 11.7 (incl. 12.x drivers) | `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117` | `pip install spconv-cu117` |

Notes:
- A 12.x **driver** runs cu117 wheels fine (drivers are backward compatible). Stay on
  torch 1.13.x — OpenCOOD is not validated on torch 2.x.
- The spconv suffix must match the torch CUDA build, not the driver.

### 0.1.3 Install OpenCOOD

```bash
git clone https://github.com/DerrickXuNu/OpenCOOD.git
cd OpenCOOD
pip install -r requirements.txt
python setup.py develop
pip install "numpy<1.24"        # AFTER requirements: OpenCOOD uses the removed np.float alias
python opencood/utils/setup.py build_ext --inplace   # CUDA NMS op; needs nvcc
```

Known failure modes:
- `AttributeError: np.float` → numpy pin above didn't stick; re-run it.
- NMS build fails → `nvcc` missing or toolkit newer than the driver supports.
- Runtime CUDA errors inside spconv → wrong spconv suffix for your torch build.
- `open3d` install trouble → skip it for now; it's only needed for visualization, not AP.

### 0.1.4 GATE

```bash
python scripts/verify_phase0.py --stage env
```

Must report: torch imports, `cuda_available=True`, spconv imports, opencood imports,
numpy < 1.24. Then fill in `env/VERSIONS.md` (template provided) and mark 0.1 ✅.

---

## Step 0.2 — Dataset (OPV2V test split)

1. Download the **test** split from the official OPV2V page
   (https://mobility-lab.seas.ucla.edu/opv2v/ — Google Drive, chunked zips; links also in
   the OpenCOOD README under "Data Downloading"). ~30GB unpacked.
   Optional, skippable now: `test_culver_city`.
2. Unzip so the structure is:

```
data/OPV2V/test/<scenario>/<cav_id>/
    ├── <timestamp>.yaml
    ├── <timestamp>.pcd
    └── <timestamp>_camera*.png
```

3. GATE:

```bash
python scripts/verify_phase0.py --stage dataset --dataset-root ~/cpfa/data/OPV2V
```

Checks every scenario/CAV folder for matching yaml/pcd pairs and prints scenario/frame
counts (expect ~16 scenarios, ~2,000 frame-CAV pairs in test). Record the dataset root
in `env/VERSIONS.md`, mark 0.2 ✅.

---

## Step 0.3 — Pretrained checkpoints

1. From the OpenCOOD README model zoo, download (each is a **folder** with
   `config.yaml` + `net_epoch*.pth`) into `~/cpfa/checkpoints/`:
   - `pointpillar_single` (No-Comm floor)
   - `pointpillar_early`
   - `pointpillar_late`
   - `pointpillar_attentive_fusion` (AttFuse)
   - `pointpillar_fcooper`
   - `pointpillar_v2vnet`
   (V2X-ViT / Where2comm / CoAlign come later from their sibling repos — not part of
   Phase 0.)
2. In **each** checkpoint's `config.yaml`, set `validate_dir` to your dataset test path
   (e.g. `~/cpfa/data/OPV2V/test`). This is the standard OpenCOOD workflow.
3. Log source URL + md5 for each in `env/CHECKPOINTS.md`:
   `md5sum ~/cpfa/checkpoints/*/net_epoch*.pth`
4. GATE:

```bash
python scripts/verify_phase0.py --stage checkpoints --checkpoint-root ~/cpfa/checkpoints
```

Loads every `.pth` on CPU and checks `config.yaml` presence + `validate_dir` exists.
Mark 0.3 ✅.

---

## Step 0.4 — Smoke test

```bash
cd ~/cpfa/OpenCOOD
python opencood/tools/inference.py \
    --model_dir ~/cpfa/checkpoints/pointpillar_attentive_fusion \
    --fusion_method intermediate
```

GATE — all three:
1. Completes and prints AP@0.5 / AP@0.7.
2. Numbers near published AttFuse on OPV2V test: **~0.90 / ~0.81**. Large deviation =
   setup problem (wrong split path, wrong checkpoint, spconv mismatch) — stop and report.
3. (Optional) `--show_vis` renders boxes on vehicles, not in the sky.

Paste the AP output back into the session; 0.4 gets marked ✅ and Phase 0 is closed.
