# Phase 3 pilot — AttFuse, full impairment grid (2026-08-05)

123 cells (8 impairments × 5–6 levels × 3 seeds), stride 3 (724 frames/cell), 0 failures.
Reference points: clean AttFuse AP@0.7 **0.815**; ego-only floor **0.575** (P 0.825 / R 0.666).
Values below are seed-means of AP@0.7 (seed spread ≤ ±0.01 stochastic, ≤ ±0.001 deterministic).

| level | latency | loss_iid | loss_burst | bandwidth | stale | pose | ghosts | swap |
|-------|---------|----------|------------|-----------|-------|------|--------|------|
| L0 | 0.521 | 0.796 | 0.797 | 0.814 | 0.659 | 0.681 | 0.805 | 0.767 |
| L1 | 0.399 | 0.756 | 0.751 | 0.815 | 0.506 | 0.502 | 0.794 | 0.665 |
| L2 | 0.369 | 0.711 | 0.710 | 0.810 | 0.429 | 0.394 | 0.773 | 0.569 |
| L3 | 0.365 | 0.655 | 0.654 | 0.735 | 0.389 | 0.416 | 0.740 | 0.451 |
| L4 | 0.358 | 0.586 | 0.627 | 0.556 | 0.369 | 0.460 | 0.685 | 0.349 |
| L5 | 0.359 | — | — | — | — | — | — | — |

(Levels per `configs/matrix.yaml`: latency 1–10 frames; loss 10–90%; bandwidth 16→1 bits;
stale refresh 2–32 frames; pose 0.2–3.2 m; ghosts 1–16/message; swap 10–100%.)

## Findings

1. **Floor test separates the families exactly as hypothesized — with one twist.**
   Packet loss (iid and bursty) degrades gracefully TOWARD the floor and never crosses
   it: pure delivery failure (recall erodes 0.90→0.74 toward the floor's 0.666,
   precision holds ≥0.77). Latency, staleness, pose error, and scene swap all cross
   BELOW the floor: content failures where collaboration actively harms.
2. **Latency is a content failure, not a delivery one.** 200 ms of delay (0.399) is
   already worse than total silence (0.575); P and R collapse together — the signature
   of misplaced evidence. Dropping 90% of messages (0.586) beats delivering all of
   them 200 ms late (0.399). Staleness mirrors latency, as expected (same mechanism,
   sawtooth age).
3. **Ghost injection is the pure precision-collapse signature**: P@0.7 falls
   monotonically 0.877→0.738 while R stays flat (0.899→0.884). No real objects lost —
   only hallucinations added (this doubles as the ghost-injection sanity check).
   AttFuse remains ABOVE the floor even at 16 ghosts/message: attention absorbs
   contradictory evidence far better than misaligned evidence.
4. **Pose error is non-monotonic**: worst at 0.8 m (0.394), partial recovery at 3.2 m
   (0.460, precision back to 0.73). Moderately-shifted features form convincing wrong
   boxes; wildly-shifted ones stop overlapping plausible locations.
5. **Burstiness is irrelevant at matched mean loss** (burst ≈ iid at every level;
   slightly better at 90% where bursts leave clean stretches). Consistent with
   detection being frame-independent — no temporal state to disrupt. Validates the
   Gilbert-Elliott implementation and predicts burstiness WILL matter for tracking
   (Phase 5).
6. **Bandwidth is free until it isn't**: 16/8/4 bits indistinguishable from clean
   (0.814/0.815/0.810 — 8× compression for nothing), 2 bits −0.08, 1 bit (0.556)
   dips below the floor — extreme quantization becomes content corruption rather
   than reduced delivery.
7. **Consistency check**: bandwidth L0 (16-bit, near-lossless) reproduces the frozen
   full-split baseline AP@0.7 0.815 on the stride-3 subset.

## Severity ranking for AttFuse (AP@0.7 at max severity)

swap (0.349) < latency (0.359) < stale (0.369) < pose (0.460) < bandwidth (0.556)
< loss_iid (0.586) < loss_burst (0.627) < ghosts (0.685).

Everything that corrupts arriving evidence outranks everything that removes it.
