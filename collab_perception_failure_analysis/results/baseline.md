# Phase 1 — Perfect-channel baseline (OPV2V test split, frozen 2026-08-04)

Produced by `scripts/run_phase1.py` on `wicomsrobot` (full 2,170-frame test split, stock
OpenCOOD at commit `31ba160`, checkpoints per `env/CHECKPOINTS.md`). Precision/recall are
overall values at the deployed operating point (score threshold + NMS), cumulative over
the split. **This table is the frozen reference for all later phases** — Phase 2's
identity-channel gate must reproduce it exactly.

| method | fusion | frames | AP@0.3 | AP@0.5 | AP@0.7 | P@0.7 | R@0.7 | R@0.5 | published AP@0.7 |
|--------|--------|--------|--------|--------|--------|-------|-------|-------|------------------|
| nocomm | nocomm | 2170 | 0.713 | 0.698 | 0.575 | 0.825 | 0.666 | 0.735 | 0.602 |
| late | late | 2170 | 0.866 | 0.859 | 0.781 | 0.847 | 0.871 | 0.914 | 0.781 |
| early | early | 2170 | 0.902 | 0.892 | 0.801 | 0.858 | 0.897 | 0.944 | 0.800 |
| attfuse | intermediate | 2170 | 0.914 | 0.905 | 0.815 | 0.889 | 0.900 | 0.946 | 0.815 |
| attfuse_comp | intermediate | 2170 | 0.902 | 0.899 | 0.811 | 0.897 | 0.886 | 0.931 | — |
| fcooper | intermediate | 2170 | 0.893 | 0.887 | 0.790 | 0.878 | 0.874 | 0.925 | 0.790 |
| v2vnet | intermediate | 2170 | 0.927 | 0.917 | 0.822 | 0.885 | 0.913 | 0.962 | 0.822 |
| coalign | intermediate | 2170 | 0.909 | 0.903 | 0.833 | 0.880 | 0.920 | 0.956 | 0.833 |
| coalign_comp | intermediate | 2170 | 0.895 | 0.885 | 0.806 | 0.859 | 0.909 | 0.950 | — |
| cobevt | intermediate | 2170 | 0.918 | 0.914 | 0.862 | 0.934 | 0.909 | 0.936 | 0.861 |
| cobevt_comp | intermediate | 2170 | 0.892 | 0.890 | 0.836 | 0.942 | 0.884 | 0.910 | — |

## Notes

- **Validation:** every method with a published reference reproduces it within ±0.001.
  The evaluation pipeline is therefore trusted end to end.
- **The floor (nocomm), AP@0.7 = 0.575 vs the OPV2V paper's No Fusion 0.602:** not the
  same measurement. The paper's baseline is a model trained for single-vehicle use; ours
  is the late-fusion checkpoint evaluated ego-only against the full collaborative GT
  (union of all CAVs' annotations). Ours is the right floor for this study — same
  weights, same GT, same pipeline as the collaborative rows — but it should be cited as
  "our ego-only floor," not as the paper's No Fusion number.
- **The attribution signature is already visible:** the floor is precise but blind
  (P@0.7 0.825, R@0.7 0.666) while every collaborative method lifts recall to ~0.87–0.92.
  Collaboration's perfect-channel benefit is almost purely recall — meaning in Phase 3+,
  recall regression toward R≈0.666 signals delivery failure, while precision collapse
  below P≈0.83 signals content corruption.
- **Compression variants cost little when clean:** −0.004 (AttFuse), −0.027 (CoAlign),
  −0.026 (CoBEVT) AP@0.7 vs their uncompressed twins — useful reference points for the
  Phase 2 bandwidth-collapse impairment.
- Communication volume (bits/frame) is not measured here; it falls out naturally of the
  Phase 2 channel wrapper, which sees every message.
