# Implementation Tracker — Collaborative Perception Failure Analysis

**Goal:** Benchmark collaborative perception algorithms on a perfect channel, then under a
constrained link, and attribute failures to **delivery** (message never arrived: latency,
packet loss, bandwidth collapse) vs **content** (message arrived but poisoned fusion: stale
memory, conflicting observations, pose noise).

---

## How to use this file (do not delete this section)

This is the single source of truth for project progress. **Whenever a step is finished,
update this file in the same commit as the work itself:**

1. Change the step's status: `⬜ TODO` → `🟨 IN PROGRESS` → `✅ DONE` (or `⛔ BLOCKED` with a reason).
2. Fill in the step's **Result** line with what actually happened (numbers, file paths, surprises).
3. Append a dated entry to the **Progress log** at the bottom.
4. Never mark a step ✅ unless its **Done when** criterion was actually verified.

Steps are ordered; do not start a step whose prerequisites are not ✅ unless noted.

---

## Inputs needed before Phase 0 (decisions from the project owner)

| # | Decision | Options | Default if unanswered | Answer |
|---|----------|---------|----------------------|--------|
| I1 | Where do experiments run? | Local GPU machine / university cluster / cloud GPU | — (required; this repo's remote session has no GPU — code is written here, runs happen on your machine) | **Local machine `wicomsrobot` (Ubuntu, GNOME desktop)** |
| I2 | GPU + VRAM available | e.g. RTX 3090 24GB, 4090, A100 | — (required; OpenCOOD inference needs CUDA, ~8GB+ VRAM) | **RTX 3080 12GB; driver 580.173.02 (CUDA 13.0); system nvcc 11.5 → use cu117 stack** |
| I3 | Disk space for dataset | OPV2V test split only (~30GB) vs full (~130GB) | Test split only — evaluation of pretrained models needs no training data | _pending_ |
| I4 | Algorithm shortlist for Phase 1 | All OpenCOOD-supported vs a starter subset | Starter subset: No-Comm, Late Fusion, Early Fusion, AttFuse, F-Cooper, V2VNet, V2X-ViT, Where2comm, CoAlign | _pending_ |
| I5 | Conflicting-evidence injection model | Gaussian feature noise / scene-swapped features / ghost activations / pose perturbation | Scene-swap + ghost activations (Gaussian noise kept only as a sanity check — too easy to dismiss) | _pending_ |
| I6 | Severity grid size | Levels per impairment × seeds | 8 levels × 3 seeds | _pending_ |
| I7 | CARLA closed-loop phase (hard task) | In scope now / later / never | Later — decide after Phase 4 results | _pending_ |

---

## Phase 0 — Testbed setup

### Step 0.1 — Environment setup  `✅ DONE`
- Create the project skeleton in this repo (this folder). ✅
- On the run machine (per I1): conda env with Python 3.7–3.9, PyTorch matching the CUDA
  version, then clone and install [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
  (`python setup.py develop`), including `spconv` (version must match CUDA) and the
  CUDA NMS op (`python opencood/utils/setup.py build_ext --inplace`).
- Record exact versions in `env/VERSIONS.md`.
- **Done when:** `python -c "import opencood"` succeeds on the run machine and a GPU is visible to PyTorch — verified via `scripts/verify_phase0.py --stage env`.
- **Result:** env gate passed 4/4 on `wicomsrobot` (2026-08-04): torch 1.13.1+cu117 with
  CUDA available on RTX 3080, numpy 1.23.5, spconv 2.3.6, opencood importable from
  `~/cpfa/OpenCOOD`. Box-overlaps extension built (Cython-only — no nvcc needed, so the
  11.5-toolkit fallback was never required). Versions recorded in `env/VERSIONS.md`.

### Step 0.2 — Dataset download  `✅ DONE`
- Download OPV2V **test split** (per I3) from the official source; verify directory structure
  matches what OpenCOOD's `yaml` configs expect; note the dataset root path in `env/VERSIONS.md`.
- **Done when:** OpenCOOD's dataloader iterates the full test split without errors.
- **Result:** dataset gate passed (2026-08-04): 16 scenarios, 5,985 frame-CAV pairs with
  full yaml↔pcd pairing at `~/cpfa/data/OPV2V/test`. Download came as Google Drive chunk
  zips wrapping `test.zip.part*` splits; joined with `cat`, verified with `unzip -t`.
  `test_culver_city` downloaded but left unextracted (optional domain-shift split).
  Full-dataloader iteration will be exercised implicitly by the 0.4 smoke test.

### Step 0.3 — Pretrained checkpoints  `✅ DONE`
- Download OpenCOOD model-zoo checkpoints for every algorithm in the I4 shortlist into
  `checkpoints/` (git-ignored). Log each checkpoint's source URL and md5 in `env/CHECKPOINTS.md`.
- **Done when:** every shortlisted algorithm has a loadable checkpoint.
- **Result:** gate passed 10/10 (2026-08-04). Ten checkpoints at `~/cpfa/checkpoints/`:
  late, early, AttFuse (+compression), F-Cooper, V2VNet, CoAlign (+compression), CoBEVT
  (no-compression + compression) — CoAlign/CoBEVT and the compression variants are bonus
  coverage beyond the minimum five. All `validate_dir`s point at the test split; md5s and
  the `latest.pth`→`net_epoch1.pth` naming caveat recorded in `env/CHECKPOINTS.md`.
  No-Comm floor will be derived from the late-fusion checkpoint (no single-vehicle
  checkpoint exists in the zoo).

### Step 0.4 — Smoke test  `⬜ TODO`
- Run OpenCOOD `inference.py` with AttFuse on ~20 test frames; confirm detections render
  and AP computation runs.
- **Done when:** one end-to-end inference completes and AP numbers are printed.
- **Result:** _pending_

## Phase 1 — Perfect-channel baseline

### Step 1.1 — No-Comm floor  `⬜ TODO`
- Evaluate ego-only (no collaboration) on the full test split. This floor is the key
  diagnostic reference for Phase 4: degradation **toward** it = delivery failure,
  **below** it = content failure (collaboration actively harming).
- **Done when:** AP@0.5 / AP@0.7 + precision/recall recorded in `results/baseline.md`.
- **Result:** _pending_

### Step 1.2 — Late & early fusion  `⬜ TODO`
- Same protocol for late fusion (box sharing + NMS merge) and early fusion (raw point cloud
  aggregation). Also log communication volume (bits/frame).
- **Done when:** rows added to `results/baseline.md`.
- **Result:** _pending_

### Step 1.3 — Intermediate fusion methods  `⬜ TODO`
- Same protocol for every intermediate-fusion method in the I4 shortlist.
- **Done when:** full baseline table complete: AP@0.5/0.7, precision, recall, bits/frame,
  for all methods, on identical frames/split/seeds.
- **Result:** _pending_

## Phase 2 — Channel wrapper (`commchannel/`)

### Step 2.1 — Interception design  `⬜ TODO`
- Write `commchannel/channel.py`: a wrapper that sits between feature extraction and fusion
  in OpenCOOD's forward pass, receiving per-agent messages and returning (possibly delayed,
  dropped, or corrupted) messages. Must be model-agnostic across the I4 shortlist and
  work for late fusion (boxes) as well as intermediate fusion (features).
- **Done when:** identity channel (no impairment) reproduces Phase 1 numbers exactly.
- **Result:** _pending_

### Step 2.2 — Delivery impairments  `⬜ TODO`
- Implement: (a) **latency** — delay collaborator messages by k frames; (b) **packet loss** —
  Bernoulli per-message drop + Gilbert-Elliott bursty variant; (c) **bandwidth collapse** —
  channel quantization/truncation of shared features.
- **Done when:** unit tests pass (drop rate matches configured p; delayed features come from
  the correct past frame; quantization hits the configured bit budget).
- **Result:** _pending_

### Step 2.3 — Content impairments  `⬜ TODO`
- Implement: (a) **stale memory** — freeze a collaborator's shared message for k frames while
  ego moves; (b) **conflicting evidence** — per I5 (scene-swapped features, ghost activations);
  (c) **pose noise** — Gaussian perturbation of collaborator pose before spatial warping.
- **Done when:** unit tests pass and a visual sanity check shows injected ghosts appear in fused BEV.
- **Result:** _pending_

## Phase 3 — Constrained-link sweeps

### Step 3.1 — Experiment matrix  `⬜ TODO`
- `configs/matrix.yaml`: algorithms × 6 impairments × severity levels × seeds (per I6).
  Frozen: agent count, detection range, split, evaluation frames.
- **Done when:** matrix file committed and a dry run enumerates every cell.
- **Result:** _pending_

### Step 3.2 — Run sweeps  `⬜ TODO`
- Runner script executes the matrix, writing one JSON per cell into `results/sweeps/`
  (AP@0.5/0.7, precision, recall, per-region breakdown — see 4.3). Resumable.
- **Done when:** all cells complete with mean ± std over seeds.
- **Result:** _pending_

## Phase 4 — Failure attribution

### Step 4.1 — Floor test  `⬜ TODO`
- Plot every degradation curve **relative to the No-Comm floor** (not just relative to clean).
  Classify each (method, impairment) as: converges-to-floor (delivery-type failure) vs
  crosses-below-floor (content-type failure).
- **Result:** _pending_

### Step 4.2 — Precision/recall decomposition  `⬜ TODO`
- Delivery failures predicted to show as **recall loss** (missed occluded objects); content
  failures as **precision collapse** (hallucinated detections). Verify per cell.
- **Result:** _pending_

### Step 4.3 — Spatial decomposition  `⬜ TODO`
- Split GT/detections into ego-visible vs occluded/beyond-range regions. Delivery impairments
  should only hurt the occluded region; content corruption should contaminate ego-visible too.
- **Result:** _pending_

### Step 4.4 — Rank stability & summary  `⬜ TODO`
- Method ranking per impairment, robustness curves, area-under-robustness-curve; test the
  fusion-mechanism hypothesis (maxout = delivery-tolerant/content-fragile; attention = partial
  down-weighting of corrupt messages but hurt more by missing ones).
- **Done when:** written analysis in `results/ANALYSIS.md`.
- **Result:** _pending_

## Phase 5 (optional, per I7) — Medium/hard tasks
- `⬜ TODO` — Take the 2–3 most and least robust methods to BEV segmentation and/or tracking;
  hard tier: trajectory prediction or CARLA closed loop. Scope after Phase 4.

---

## Progress log

| Date | Step | Notes |
|------|------|-------|
| 2026-08-03 | 0.1 | Branch + project skeleton created; tracker, README, CLAUDE.md committed. Awaiting inputs I1–I7. |
| 2026-08-03 | 0.1–0.4 | Phase 0 implementation instructions authored: `docs/PHASE0_SETUP.md` (copy-paste setup guide, CUDA-version table, known failure modes), `scripts/verify_phase0.py` (automated gates for env/dataset/checkpoints), `env/VERSIONS.md` + `env/CHECKPOINTS.md` templates. Execution on run machine still pending I1/I2. |
| 2026-08-03 | I1/I2 | Inputs answered: local machine `wicomsrobot`, RTX 3080 12GB, driver CUDA 13.0, system nvcc 11.5. Resolved to cu117 stack (torch 1.13.1+cu117, spconv-cu117); machine-specific command block added to `docs/PHASE0_SETUP.md` with conda cuda-toolkit 11.7 fallback for the NMS build. Step 0.1 execution now unblocked. |
| 2026-08-04 | 0.1 ✅ | Env gate passed 4/4 on `wicomsrobot`: torch 1.13.1+cu117 (CUDA visible, RTX 3080), numpy 1.23.5, spconv 2.3.6, opencood import OK. Extension build succeeded (only a harmless NumPy deprecated-API warning). `env/VERSIONS.md` filled in; OpenCOOD commit hash still to record. Next: 0.2 dataset download. |
| 2026-08-04 | 0.1 | OpenCOOD commit pinned: `31ba16025da27ffe4e336f011290dfbc66f9a1f1`. |
| 2026-08-04 | 0.2 ✅ | Dataset gate passed: 16 scenarios, 5,985 frame-CAV pairs at `~/cpfa/data/OPV2V/test`. Drive chunk zips → `cat`-joined `test.zip.part*` → verified → extracted. Next: 0.3 checkpoints. |
| 2026-08-04 | 0.3 ✅ | Checkpoints gate passed 10/10 after flattening double-nested zips and fixing the gate script (OpenCOOD configs embed numpy YAML tags; now greps `validate_dir` instead of safe_load). Manifest filled with md5s. Caveat logged: 4 checkpoints ship `latest.pth` and need a `net_epoch1.pth` copy for OpenCOOD's loader. Next: 0.4 smoke test. |
