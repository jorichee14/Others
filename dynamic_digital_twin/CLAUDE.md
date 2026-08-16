# Working rules for this project

1. **Read `INTRODUCTION.md` first** — thesis, gaps (G1–G3), contributions (C1–C6),
   research questions, positioning, and the reference pack. Parent-study context lives
   in `collab_perception_failure_analysis/` and `temporal_messaging/`.
2. **`IMPLEMENTATION.md` is the source of truth for progress.** Update it in the same
   commit as the work: flip status, fill the Result line with real numbers and paths,
   append to the progress log. Never mark ✅ without verifying the "Done when" criterion.
3. **Pre-register decision rules before anything expensive.** The Phase 1 gate (RQ2)
   must be committed before the Phase 1 run starts, exactly as the parent study did.
4. **Measure, don't model.** The real network is the experimental condition — no
   emulation (netem/tc) unless a step explicitly says so, and then only as a labeled
   control arm.
5. **Clocks before ages.** No age/latency number is reportable until Step 0.1's clock
   error bound is established; every reported age carries that bound.
6. **Record-and-replay discipline.** Capture raw streams (rosbag2) with source
   timestamps on every campaign run so analysis conditions can be recomputed offline
   without re-driving robots. One physical run, many analyses.
7. **Distinguish verified from inferred.** [V]/[S]/[I] tags per the convention in
   `INTRODUCTION.md`; nothing [S] goes into a manuscript before the primary text is read.
8. Large captures (bags, traces) are never committed — record paths, sizes, and md5s
   under `env/` instead.
