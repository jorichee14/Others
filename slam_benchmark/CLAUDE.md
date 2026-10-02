# Working rules for this project

1. **`IMPLEMENTATION.md` is the source of truth for progress.** Update it in the
   same commit as the work: flip the status, fill the Result line with real
   numbers and paths, append to the log. Never mark ✅ without verifying its
   "Done when" line.
2. **Do not start a phase whose input decisions (I1–I7) are unanswered.** A run
   under a guessed extrinsic or a guessed IMU noise model is not a cheap result;
   it is an expensive wrong one that looks like a result in a table.
3. **Refuse rather than default.** A value that cannot be recovered from the data
   is `null` in the config and the loader raises. This project inherits that rule
   from `../rosbag_to_opv2v` for the same reason: the alternative is a run that
   completes, validates, and is geometrically wrong.
4. **Every geometry number comes from `../rosbag_to_opv2v/configs/mirc_coop2.yaml`.**
   Do not re-derive an extrinsic by hand. If that file changes, change
   `configs/coop2.yaml` in the same commit.
5. **Never quote a tier-1 number without its tier-2 and tier-3 companions.** The
   reference is the offline pipeline's own output; ATE against it is agreement,
   not accuracy, and an ATE below the reference's uncertainty is not a win.
6. **Containers do not transform poses.** A method writes its own sensor frame in
   its own world frame. Every re-expression happens in the evaluator, from a
   config that travels with the result.
7. **No method is tuned per run.** Indoor re-parameterisation is declared once in
   the method config, with the published default named beside each changed value,
   so a poor result can be checked against the tuning instead of blamed on it.
8. **A failure is a result.** Lost tracking, a method that needs a sensor this rig
   does not carry, a sequence that cannot evaluate loop closure — each is
   reported with its number, not dropped from the table.
9. Bags, maps, checkpoints and run outputs are never committed.
10. All work happens on branch `claude/collab-perception-failure-analysis-s3bsij`.
