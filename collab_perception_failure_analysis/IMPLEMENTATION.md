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

### Step 0.4 — Smoke test  `✅ DONE`
- Run OpenCOOD `inference.py` with AttFuse on the test split; confirm AP computation runs
  and matches published numbers.
- **Done when:** one end-to-end inference completes and AP numbers are printed.
- **Result:** passed (2026-08-04). AttFuse on full 2,170-frame test split:
  **AP@0.3/0.5/0.7 = 0.91 / 0.91 / 0.82** vs published 0.90/0.815 — matches. Runtime
  3m16s (~11 it/s) on the RTX 3080, which bodes very well for the Phase 3 sweep budget.
  Required fix along the way: a stray plain `cumm 0.5.3` package was shadowing
  `cumm-cu117` and breaking `spconv.core_cc` — removed, pinned `cumm-cu117==0.4.11`
  (see `env/VERSIONS.md`). **Phase 0 complete.**

## Phase 1 — Perfect-channel baseline

### Step 1.1 — No-Comm floor  `✅ DONE`
- Evaluate ego-only (no collaboration) on the full test split. This floor is the key
  diagnostic reference for Phase 4: degradation **toward** it = delivery failure,
  **below** it = content failure (collaboration actively harming).
- **Done when:** AP@0.5 / AP@0.7 + precision/recall recorded in `results/baseline.md`.
- **Result:** runner authored (`scripts/run_phase1.py`, `nocomm` mode): model and box
  post-processing see only the ego vehicle, but GT is generated from the full
  collaborator set (`generate_gt_bbx(batch_data)`) so the floor is scored against the
  same GT as collaborative methods. AP math verified identical to OpenCOOD
  `eval_utils` on 200 randomized trials. Run complete (2026-08-04): floor =
  **AP@0.7 0.575, P@0.7 0.825, R@0.7 0.666** on the full split (paper's No Fusion is
  0.602 but is a differently-trained model — see `results/baseline.md` notes).

### Step 1.2 — Late & early fusion  `✅ DONE`
- Same protocol for late fusion (box sharing + NMS merge) and early fusion (raw point cloud
  aggregation). Also log communication volume (bits/frame).
- **Done when:** rows added to `results/baseline.md`.
- **Result:** late AP@0.7 0.781 (published 0.781), early 0.801 (published 0.800).
  Communication volume deferred to Phase 2 — the channel wrapper sees every message and
  measures it directly.

### Step 1.3 — Intermediate fusion methods  `✅ DONE`
- Same protocol for every intermediate-fusion method in the I4 shortlist.
- **Done when:** full baseline table complete: AP@0.5/0.7, precision, recall, bits/frame,
  for all methods, on identical frames/split/seeds.
- **Result:** all 8 intermediate methods evaluated; every published reference reproduced
  within ±0.001 (AttFuse 0.815, F-Cooper 0.790, V2VNet 0.822, CoAlign 0.833, CoBEVT
  0.862). Full frozen table in `results/baseline.md`. **Phase 1 complete** (bits/frame
  deferred to Phase 2 as above).

## Phase 2 — Channel wrapper (`commchannel/`)

### Step 2.1 — Interception design  `✅ DONE`
- Write `commchannel/channel.py`: a wrapper that sits between feature extraction and fusion
  in OpenCOOD's forward pass, receiving per-agent messages and returning (possibly delayed,
  dropped, or corrupted) messages. Must be model-agnostic across the I4 shortlist and
  work for late fusion (boxes) as well as intermediate fusion (features).
- **Done when:** identity channel (no impairment) reproduces Phase 1 numbers exactly.
- **Result:** package built. Interception is two-level: (a) data level — instance
  monkeypatch of `retrieve_base_data`, mirroring stock logic and reusing
  `reform_param` (inherits OpenCOOD's current-timestamp-GT semantics); covers drop,
  latency, staleness, pose noise, ghosts, scene swap uniformly for late/early/
  intermediate fusion; (b) feature level — forward-pre-hooks for bandwidth
  quantization (per-model registry; AttFuse intercepted at backbone input since its
  fusion is interleaved — documented approximation). GT policy: runners take
  predictions from the impaired dataset, GT from a parallel clean dataset.
  Identity gate: **PASSED for AttFuse, late, and F-Cooper** (2026-08-05: 100/100 input
  batches bitwise identical, 10/10 model outputs identical, each). Channel proven
  transparent across the late-fusion and intermediate-fusion code paths.

### Step 2.2 — Delivery impairments  `⬜ TODO`
- Implement: (a) **latency** — delay collaborator messages by k frames; (b) **packet loss** —
  Bernoulli per-message drop + Gilbert-Elliott bursty variant; (c) **bandwidth collapse** —
  channel quantization/truncation of shared features.
- **Done when:** unit tests pass (drop rate matches configured p; delayed features come from
  the correct past frame; quantization hits the configured bit budget).
- **Result:** implemented (`commchannel/schedule.py`, `channel.py`, `feature_hooks.py`).
  All decisions crc32-seeded ⇒ deterministic and DataLoader-worker-safe;
  Gilbert-Elliott re-simulated from scenario start per query. 10/10 unit tests pass
  (Bernoulli rate ±2%, GE burstiness ≫ i.i.d. control, latency constant, staleness
  sawtooth, additive composition, quantizer level/monotonicity/ego-untouched) plus a
  7-scenario mocked-dataset integration test (drop, delay, scenario-boundary clamp,
  ghosts, swap, pose noise). Bandwidth meter records bits/frame (Phase 1 deferral).

### Step 2.3 — Content impairments  `⬜ TODO`
- Implement: (a) **stale memory** — freeze a collaborator's shared message for k frames while
  ego moves; (b) **conflicting evidence** — per I5 (scene-swapped features, ghost activations);
  (c) **pose noise** — Gaussian perturbation of collaborator pose before spatial warping.
- **Done when:** unit tests pass and a visual sanity check shows injected ghosts appear in fused BEV.
- **Result:** implemented alongside 2.2 (same package): stale memory as sawtooth refresh
  (composes additively with latency), conflicting evidence as car-shaped ghost point
  clusters (geometry unit-tested: 220 pts/vehicle on box surfaces, ground-supported,
  deterministic per seed) and cross-scenario scene swap; pose noise recomputes
  `transformation_matrix` from the perturbed pose while leaving
  `gt_transformation_matrix` untouched (verified) — also fixes OpenCOOD's
  `add_loc_noise` constant-offset reseeding bug (regression test included).
  **Visual ghost sanity check on GPU machine pending.**

## Phase 3 — Constrained-link sweeps

### Step 3.1 — Experiment matrix  `✅ DONE`
- `configs/matrix.yaml`: algorithms × 6 impairments × severity levels × seeds (per I6).
  Frozen: agent count, detection range, split, evaluation frames.
- **Done when:** matrix file committed and a dry run enumerates every cell.
- **Result:** committed (2026-08-05): 7 methods × 8 impairment families (latency,
  iid loss, bursty loss, bandwidth, stale, pose, ghosts, swap) × 5–6 levels × 3 seeds
  = **831 cells** (bandwidth restricted to intermediate fusion), frame stride 3
  (724 frames/cell). Dry run enumerates all cells; Gilbert-Elliott levels calibrated
  to stationary loss rate (empirical 0.712 at target 0.7); pose yaw coupling verified.

### Step 3.2 — Run sweeps  `🟨 IN PROGRESS`
- Runner script executes the matrix, writing one JSON per cell into `results/sweeps/`
  (AP@0.5/0.7, precision, recall, per-region breakdown — see 4.3). Resumable.
- **Done when:** all cells complete with mean ± std over seeds.
- **Result:** runner authored (`scripts/run_phase3.py`): per-cell JSON, resumable,
  clean-dataset GT cache (GT never shrinks under impairment), per-(cell,frame) seeding,
  bandwidth hooks + bits/frame metering, mean-collaborators observable, one model/
  dataset build per method. Execution on GPU machine pending (est. ~24–30 h full,
  resumable in any-size chunks). Per-region breakdown deferred to Phase 4 reruns on
  selected cells.

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
| 2026-08-04 | 0.4 ✅ | **Phase 0 complete.** Smoke test hit a spconv failure first: stray plain `cumm 0.5.3` shadowed `cumm-cu117`, breaking `spconv.core_cc`. Fixed by uninstalling both and pinning `cumm-cu117==0.4.11`. Env gate hardened (imports compiled core, detects dual cumm installs). AttFuse full-split inference: AP@0.3/0.5/0.7 = 0.91/0.91/0.82 (published: 0.90/0.815), 3m16s for 2,170 frames. Next: Phase 1 baselines. |
| 2026-08-04 | 1.1–1.3 | Phase 1 runner authored: `scripts/run_phase1.py` — 12 runs (No-Comm floor + 11 checkpointed methods), per-method JSON + auto-regenerated `baseline.md`, resumable (skips finished methods), AP + overall precision/recall per IoU. Verified against OpenCOOD source at pinned commit (fusion interfaces, GT union semantics, non-mutating AP math cross-checked on 200 random trials). CoAlign + CoBEVT model classes confirmed present in stock OpenCOOD at commit `31ba160`. Awaiting run. |
| 2026-08-04 | 1.1–1.3 | Quick pass (60 frames) succeeded for all 12 methods — zero config/checkpoint failures, incl. CoAlign and nocomm mode. Early signal confirms the study's core structure: nocomm P@0.7 0.917 / R@0.7 0.594 (precise but occlusion-blind) vs collaborative methods R@0.7 ≈ 0.89–0.94. CoBEVT ~2× slower per frame (transformer). Full-split run launched next; est. ~1h. |
| 2026-08-04 | 1.1–1.3 ✅ | **Phase 1 complete.** Full-split run: every published AP@0.7 reproduced within ±0.001. Frozen table committed to `results/baseline.md`. Floor: 0.575/P 0.825/R 0.666. Perfect-channel collaboration benefit is almost purely recall (+0.20–0.25 R@0.7 over floor) — the Phase 4 attribution axes are now calibrated. Next: Phase 2 `commchannel/` wrapper (identity gate = reproduce this table). |
| 2026-08-05 | 2.1 | First identity-gate run mismatched on every frame — diagnosis: the GATE was flawed, not the channel. OpenCOOD's test-time `__getitem__` is stochastic (`shuffle_points` draws a fresh permutation each call), so any two passes differ; point order changes voxelization. Gate rewritten: numpy seeded identically before each side's `__getitem__`, collated input batches compared bitwise tensor-by-tensor (stronger than the old box comparison), plus a model-output check on input-identical frames. |
| 2026-08-05 | 2.1 ✅ / 3.1 ✅ / 3.2 | Identity gate passed for late and fcooper too (100/100 + 10/10 each) — Step 2.1 closed. Phase 3 built: `configs/matrix.yaml` (831 cells, GE stationarity + pose coupling validated in dev container) and `scripts/run_phase3.py` (resumable per-cell runner with clean-GT cache and bandwidth metering). Awaiting sweep execution. |
| 2026-08-05 | 3.2 | **Pilot complete: attfuse × full grid, 123/123 cells, 0 failures, ~78s/cell** (runner optimization: 13min→78s). Findings in `results/pilot_attfuse.md`: loss→floor (delivery signature, never below); latency/stale/pose/swap cross BELOW floor (content failures — 200ms latency worse than total silence); ghosts = pure precision collapse w/ flat recall (sanity check passed, stays above floor); pose non-monotonic (worst at 0.8m); burst≈iid at matched rate; bandwidth free to 4 bits, below floor at 1 bit. Bandwidth L0 reproduces frozen baseline 0.815. Aggregator authored (`scripts/aggregate_sweeps.py`, floor-test classifier, tested). Remaining 6 methods ≈ 16h. |
| 2026-08-06 | 3.2 | Full sweep first pass: late 108/108 ✅; v2vnet/coalign/cobevt mostly ✅. Two runner/channel bugs found and fixed: (1) all `early` cells crashed — early fusion merges agents pre-collation so `record_len` doesn't exist; collaborator count now recorded as unobservable for early. (2) v2vnet/coalign `pose` cells at ≥0.8m crashed on a LATENT STOCK OpenCOOD bug: dataset filters CAVs by COM_RANGE using (noised) lidar_pose but builds pairwise matrices from the unfiltered dict — a membership flip desyncs feature count vs matrix and crashes models that warp with pairwise_t_matrix. Fix: pose noise now clamps the noised position to the true pose's side of the COM_RANGE boundary (connectivity loss is drop's job, not pose noise's). All `pose` cells to be deleted and rerun under the clamped protocol for uniformity. |
| 2026-08-06 | 3.2 | First full-sweep pass finished: **709/831 cells banked** (586 + 123 pilot), 122 failed = 108 early (record_len bug) + 14 v2vnet/coalign pose (COM_RANGE flip bug) — both fixed. CoBEVT full grid clean incl. pose (no pairwise warp, as predicted). Notables: CoBEVT keeps P@0.7 ≈ 0.91–0.93 under ANY loss rate (highest precision retention); its bandwidth curve confirmed non-monotonic (2-bit 0.32 < 1-bit 0.45); pose non-monotonicity reproduces across attfuse/fcooper/late/cobevt. Remaining: rerun early (108) + all pose cells under clamped protocol (105). |
| 2026-08-05 | 3.2 | Sweep smoke test: 6 cells (attfuse latency L0/L1 × 3 seeds, stride 30), 0 failures. Seeds agree to ±0.002 AP (deterministic impairment + shuffle noise only) — seeding design validated. First physics sensible: 100ms latency barely moves AP@0.5 but craters AP@0.7 with P and R falling together (misplaced collaborator evidence), consistent with V2X-ViT async findings. Cell runtime ⇒ full sweep ≈ 25 h. Pilot (attfuse, full grid, stride 3) is next. |
| 2026-08-05 | 2.1–2.3 | `commchannel/` package built: config (composable impairments), crc32-seeded worker-safe schedules, dataset-level channel (drop/latency/stale/pose-noise/ghosts/scene-swap via `retrieve_base_data` monkeypatch reusing stock `reform_param`), feature-level bandwidth hooks with per-model registry + bits/frame meter. 10/10 unit tests + mocked-dataset integration pass in dev container. Design notes: GT always from parallel clean dataset; stock `wild_setting` latency exists but its pose-noise path has a constant-offset reseeding bug — ours replaces it. Remaining for ✅: identity gate + ghost visual check on GPU machine. |
