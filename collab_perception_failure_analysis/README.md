# Collaborative Perception Failure Analysis

Benchmarks collaborative perception algorithms (V2VNet, AttFuse, F-Cooper, Where2comm,
V2X-ViT, CoBEVT, CoAlign, plus early/late fusion baselines) on a **perfect channel**, then
under a **constrained link**, and attributes failures to:

- **Delivery** — the message never arrived usefully: latency, packet loss (Bernoulli and
  bursty Gilbert-Elliott), bandwidth collapse.
- **Content** — the message arrived but poisoned fusion: stale memory, conflicting
  observations (ghost activations / scene-swapped features), pose noise.

Core diagnostics: the **No-Comm floor test** (degrading toward the ego-only floor = delivery
failure; below it = content failure), **precision/recall decomposition** (recall loss vs
precision collapse), and **spatial decomposition** (ego-visible vs occluded regions).

Built on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) with the OPV2V dataset,
evaluating pretrained checkpoints under impairment (no per-condition retraining).

## Repository layout (target)

```
collab_perception_failure_analysis/
├── IMPLEMENTATION.md   # living step tracker — START HERE, keep it updated
├── CLAUDE.md           # working rules for Claude Code sessions in this folder
├── env/                # versions, dataset paths, checkpoint manifests
├── commchannel/        # channel impairment wrapper (Phase 2)
├── configs/            # experiment matrix (Phase 3)
├── scripts/            # runners, evaluation, plotting
└── results/            # baseline tables, sweep JSONs, analysis
```

## Status

See [IMPLEMENTATION.md](IMPLEMENTATION.md) — the authoritative, always-current plan and
progress tracker. **Every finished step must be reflected there in the same commit.**
