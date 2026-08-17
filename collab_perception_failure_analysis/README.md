# Collaborative Perception Failure Analysis

Why does collaborative perception fail on a constrained link — because messages fail to
**arrive** (delivery), or because they arrive **wrong** and poison fusion (content)?

Seven pretrained V2V detectors (early, late, AttFuse, F-Cooper, V2VNet, CoAlign, CoBEVT)
are run through a controllable channel across 831 impairment conditions on OPV2V, and
failures are attributed with four independent diagnostics.

- **Findings:** [`results/ANALYSIS.md`](results/ANALYSIS.md) · **Paper:** [`paper/PAPER.md`](paper/PAPER.md)
- **Progress history and gates:** [`IMPLEMENTATION.md`](IMPLEMENTATION.md)

---

## What each piece does

### `commchannel/` — the instrument
Sits between "collaborator produces a message" and "ego fuses it". Proven **bitwise
inert** when disabled, so any degradation is caused by the knob you turned.

| File | Role |
|---|---|
| `config.py` | `ChannelConfig`: one dataclass holding all impairment parameters; they compose freely |
| `schedule.py` | Turns (seed, scenario, frame, agent) into a decision via CRC32 hashing → exactly reproducible, DataLoader-worker-safe. Includes the Gilbert-Elliott burst chain |
| `channel.py` | Attaches to a built OpenCOOD dataset and rewrites messages: drop, delay, stale-refresh, pose noise, ghost injection, scene swap |
| `feature_hooks.py` | Bandwidth impairment: forward-pre-hooks that quantize collaborator BEV features (per-model registry) and meter bits/frame |

Impairments: **delivery** = latency, i.i.d. loss, bursty loss, bandwidth quantization;
**content** = staleness, pose error, ghost vehicles, scene swap.

### `scripts/` — the experiments
Every runner writes one JSON per unit of work and **skips finished units on restart**, so
any run is safe to interrupt.

| Script | What it does | Needs GPU |
|---|---|---|
| `verify_phase0.py` | Setup gates: env (torch/CUDA/spconv), dataset structure, checkpoint loadability | no |
| `test_commchannel.py` | 10 unit tests: loss rates, burst statistics, staleness sawtooth, quantizer, ghost geometry | no |
| `run_phase1.py` | Perfect-channel baseline for all methods + the ego-only **floor** | yes |
| `run_phase2_identity.py` | Inertness gate: identity channel must give bitwise-identical model inputs | yes |
| `run_phase3.py` | The 831-cell sweep over `configs/matrix.yaml` | yes |
| `aggregate_sweeps.py` | Collapses cells into mean±std rows + floor-test classification | no |
| `run_phase43.py` | Spatial decomposition: ego-visible vs occluded recall, ego-visible FP contamination | yes |
| `run_phase5_tracking.py` | Kalman/Hungarian MOT under impairment (burstiness + staleness predictions) | yes |
| `run_blockage_audit.py` | **Phase 6** — model-free test of whether i.i.d. loss is a valid assumption (see below) | no |

### Phase 6 — geometry-conditioned loss (`commchannel/blockage.py`)

Every impairment above drops messages *independently of the scene*. That cannot be
physically right: **the vehicle occluding an agent's lidar is the vehicle obstructing its
radio**, so the messages you lose are disproportionately the ones you needed. If true,
every robustness number in this study — and in the literature — is optimistic.

`blockage.py` answers one geometric question: is a labeled vehicle standing on the
ego↔collaborator chord (inflated by a clearance that stands in for the first Fresnel
radius)? No propagation model, no dB — deliberately, so no modelling choice can
manufacture the correlation. Full rationale and scope caveats:
[`docs/BLOCKAGE.md`](docs/BLOCKAGE.md).

It adds one channel parameter (`blockage_p` = P(drop | chord blocked)), two sweep
families (`loss_blocked`, `loss_iid_matched`), and a `blocked` condition in the spatial
decomposition.

### `configs/matrix.yaml`
The frozen experiment grid: methods × impairment families × severity levels × seeds.
Edit only by **adding**; changing existing levels invalidates comparisons.

---

## Setup

Full instructions with CUDA-version table and known failure modes:
[`docs/PHASE0_SETUP.md`](docs/PHASE0_SETUP.md). Short version:

```bash
conda create -y -n opencood python=3.8 && conda activate opencood
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install spconv-cu117 cumm-cu117==0.4.11     # the plain `cumm` package must NOT be installed
git clone https://github.com/DerrickXuNu/OpenCOOD.git ~/cpfa/OpenCOOD
cd ~/cpfa/OpenCOOD && pip install -r requirements.txt && python setup.py develop
pip install "numpy<1.24"                         # after requirements
python opencood/utils/setup.py build_ext --inplace
```

Then: OPV2V **test** split → `~/cpfa/data/OPV2V/test/`, checkpoints → `~/cpfa/checkpoints/<name>/`
(each folder holding `config.yaml` + `net_epoch*.pth`, with `validate_dir` pointing at the
test split). Sources and MD5s: [`env/CHECKPOINTS.md`](env/CHECKPOINTS.md).

Verify:

```bash
R=~/cpfa/Others/collab_perception_failure_analysis      # this folder
python $R/scripts/verify_phase0.py --stage env
python $R/scripts/verify_phase0.py --stage dataset     --dataset-root ~/cpfa/data/OPV2V
python $R/scripts/verify_phase0.py --stage checkpoints --checkpoint-root ~/cpfa/checkpoints
python $R/scripts/test_commchannel.py                  # expect 10/10
```

---

## Running the study

Run everything from the OpenCOOD repo root (`cd ~/cpfa/OpenCOOD`) so `import opencood`
resolves. `R` is this folder.

**1. Baseline + floor** (~1 h) — must reproduce published AP before anything else counts.

```bash
python $R/scripts/run_phase1.py --checkpoint-root ~/cpfa/checkpoints --out ~/cpfa/results/phase1
# add --max-frames 60 for a 3-minute smoke test first
```

**2. Instrument gate** (~2 min each) — expect `100/100 bitwise identical`.

```bash
for m in attfuse late fcooper; do
  python $R/scripts/run_phase2_identity.py --checkpoint-root ~/cpfa/checkpoints --method $m
done
```

**3. Sweep** (~20 h; ~80 s per cell) — resumable, so run it in any-size chunks.

```bash
python $R/scripts/run_phase3.py --checkpoint-root ~/cpfa/checkpoints --out ~/cpfa/results/sweeps
# subsets: --methods attfuse --impairments latency ghosts --seeds 0 --stride 30 --max-cells 6
```

**4. Aggregate** (seconds) — writes `sweep_summary.{csv,md}` with the floor-test column,
then reshapes it into `sweep_table.md`, the full grid with methods as columns (same
numbers, comparable at a fixed condition).

```bash
python $R/scripts/aggregate_sweeps.py --sweeps ~/cpfa/results/sweeps --out ~/cpfa/results
python $R/scripts/pivot_sweep_table.py --summary ~/cpfa/results/sweep_summary.md \
    --out ~/cpfa/results/sweep_table.md
```

**5. Spatial decomposition** (~4 h) — **one process per method** (see note below).

```bash
for m in attfuse coalign fcooper; do
  python $R/scripts/run_phase43.py --checkpoint-root ~/cpfa/checkpoints \
      --out ~/cpfa/results/spatial --methods $m
done
```

**6. Tracking** (~2 h) — one process per method; GT caches persist to disk after the first run.

```bash
for m in coalign cobevt fcooper; do
  python $R/scripts/run_phase5_tracking.py --checkpoint-root ~/cpfa/checkpoints \
      --out ~/cpfa/results/tracking --methods $m
done
```

**7. Phase 6 — is i.i.d. loss a valid assumption?** Run the audit *first*; it decides
whether the sweep is worth running and supplies the control arm's loss rates.

```bash
# 7a. Model-free audit (~minutes, no GPU). Prints GO / NO-GO.
python $R/scripts/run_blockage_audit.py \
    --config ~/cpfa/checkpoints/pointpillar_attentive_fusion/config.yaml \
    --out ~/cpfa/results/blockage --stride 10

# 7b. On GO: paste the printed `loss_iid_matched.levels` block into configs/matrix.yaml,
#     then run both arms (loss_burst is already banked from step 3 as the third arm).
python $R/scripts/run_phase3.py --checkpoint-root ~/cpfa/checkpoints \
    --out ~/cpfa/results/sweeps --impairments loss_blocked loss_iid_matched

# 7c. The headline prediction: same delivery rate, deficit concentrated in occluded recall.
#     --matched-loss-p comes from the audit's measured realized_drop_rate.
python $R/scripts/run_phase43.py --checkpoint-root ~/cpfa/checkpoints \
    --out ~/cpfa/results/spatial --methods attfuse \
    --conditions identity blocked loss_matched --matched-loss-p <measured>
```

Self-tests for this phase need no dataset or GPU:
`python $R/scripts/run_blockage_audit.py --selftest` (36 checks).

### Operational notes

- **One process per method for steps 5–6.** Shapely-heavy loops fragment the Python heap;
  a third method in the same process ran ~15× slower in testing. The loops above are the fix.
- **Everything is resumable.** Delete a result JSON to force that unit to re-run.
- **Everything is deterministic.** Same config + seed ⇒ identical results, verified by
  reproducing the full spatial tier digit-for-digit in a separate process.
- Two warnings are expected and harmless: `nn.functional.sigmoid is deprecated` and
  shapely's `invalid value encountered in intersection`.

---

## Reproducing a single claim

| Claim | Command |
|---|---|
| Latency is worse than 90% packet loss | `run_phase3.py --methods attfuse --impairments latency loss_iid` |
| Maxout is the content-fragility extreme | `run_phase3.py --methods fcooper coalign --impairments swap` |
| Burstiness matters only with temporal state | `run_phase3.py --impairments loss_iid loss_burst` then `run_phase5_tracking.py --conditions iid70 burst70_long` |
| Corruption contaminates the ego's own view | `run_phase43.py --methods attfuse --conditions identity latency200ms` |
