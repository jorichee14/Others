# Working rules for this project

1. **Read `HANDOFF.md` first.** It carries the full context from the parent study
   (`collab_perception_failure_analysis/` on branch
   `claude/collab-perception-failure-analysis-s3bsij`), the machine setup, and the
   pre-registered decision rules.
2. **`IMPLEMENTATION.md` is the source of truth for progress.** Update it in the same
   commit as the work: flip status, fill the Result line with real numbers, append to
   the progress log. Never mark ✅ without verifying the "Done when" criterion.
3. **Pre-register decision rules before running anything expensive.** Phase 6 of the
   parent study produced a useful negative result precisely because its go/no-go
   thresholds were fixed in advance. Do the same here.
4. **Distinguish verified from inferred.** Novelty claims that could not be checked are
   marked ⚠️ UNVERIFIED in `HANDOFF.md` §4. Do not promote them to assertions without a
   literature search.
5. Reuse the parent study's `commchannel` package unchanged for impairments — it is
   proven bitwise inert when disabled, and that property is what makes results
   attributable.
6. Evaluate pretrained checkpoints; do not retrain per condition unless a phase
   explicitly requires it.
7. Operational: one process per method for Shapely-heavy runs; DataLoader workers, never
   in-process loops; everything resumable and seed-deterministic.
8. Datasets and checkpoints are never committed.
