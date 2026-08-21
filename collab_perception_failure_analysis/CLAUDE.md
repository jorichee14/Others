# Working rules for this project

1. **`IMPLEMENTATION.md` is the single source of truth.** Read it before doing anything.
   When any step is finished, update it **in the same commit** as the work: flip the step's
   status, fill in its Result line with actual outcomes (numbers, paths), and append a row
   to the Progress log. Never mark a step ✅ without verifying its "Done when" criterion.
2. Do not start a step whose prerequisites are not ✅, and do not skip ahead of unanswered
   input decisions (I1–I7) that a step depends on.
3. Evaluate **pretrained** checkpoints under impairment — never retrain per condition.
4. Datasets and checkpoints are never committed (git-ignored); record their sources and
   hashes in `env/` instead.
5. All work happens on branch `claude/collab-perception-failure-analysis-s3bsij`.
