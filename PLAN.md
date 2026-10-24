# Plan: SCooP-iT (with its Gym) and SCooP

SCooP: Sensing and Communication for Cooperative Perception.

## Vision

Multi-agent perception should be developed and proven in the real world:
on real robots in real buildings, with every condition that deployment
brings measured rather than assumed. Simulators and outdoor vehicle datasets
hide what an indoor team faces, and our own recordings show it:
- sensors fall short (the ZED at about 1 Hz) and differ between robots;
- clocks drift (SLAM poses 46 ms early);
- the team is mixed (a LiDAR robot, an RGB-D cart, fixed masts);
- there is no GNSS or motion capture indoors;
- the building occludes people, objects and the radio;
- the network is shared and contended.

## End goal

A real-world indoor platform where researchers can:
1. use synchronized data from a real heterogeneous team, with ground truth
   in one building-scale frame and no motion capture or GNSS;
2. benchmark on what the real robots saw, with labels authored once in the
   map frame and projected into every agent;
3. extend beyond the recordings with a twin built from the measured
   building, whose realism is checked against the real runs;
4. train and test agents in an environment derived from the real site
   (the Gym), on the recordings and in the twin;
5. add their own building with a mapping pass and printed boards.

Two papers share one testbed, one channel model
(`scoop_pipeline/scoop/channel.py`), one set of recordings, and a different
claim each. The Gym is part of the testbed paper (merged 24 Oct); its closed
loop is the extension.

| | SCooP-iT, with SCooP-iT Gym | SCooP | Extension: closed-loop Gym |
|---|---|---|---|
| What | Testbed and environment paper | Dataset paper | Journal extension or short follow-up |
| Folder | `scoop_testbed_iotj/` | `scoop_dataset_paper/` | `scoop_gym/` (+ `isaac_coop_sim/`) |
| Venue | IEEE IoT-J | IEEE T-RO | decided when the closed loop works |
| One claim | A real indoor team captured on one clock and one frame, sensing and communication alike, and offered as a validated environment | Collaborative perception for indoor robot teams studied as findings (Q1-Q4), not only as a leaderboard | Train and score agents against the network the robots actually had |
| State, 24 Oct | Draft compiles (16 pp); results open | Draft written; collection, labels and every experiment open | Link engine, attribution log, `CommSchedule-v0` and tests done; twin and other tasks open |

## Who owns what

- **SCooP-iT:**
  - platform, clock, ground truth, link instrumentation, the replay and its validation (E3a live, E3b twin);
  - SCooP-iT Gym: the interface, replay mode, the twin along the recorded paths, `CommSchedule-v0` and `CoopDetect-v0`.
- **SCooP:**
  - sequences, labels, collection design, collaboration geometry, release, devkit;
  - the offline benchmarks T1 and T2, and Q1-Q4.
  - It cites SCooP-iT for the platform and keeps half a page on it.
- **Extension (closed-loop Gym):**
  - the closed-loop twin and the link indexed by place, with its out-of-run validation;
  - the twin tasks, the agent benchmark (rule, RL, VLM), and attribution of failures down to the link.
  - It cites both papers.
- **Shared, not duplicated:** the channel model, the reference map, and the twin's construction. SCooP-iT describes them; the others cite.

## What blocks what

```
field session (1 h) ---------------------------> E3a, E4 day half, sniffer bias        [SCooP-iT]
labels on coop_2 --+--> CoopDetect-v0 --------> gym v1
                   +--> E3b ΔAP (with twin run) -> SCooP-iT
                   +--> Q1-Q4 training ---------> SCooP
twin run of coop_2 ----> E3b scan agreement --> SCooP-iT; twin open loop -> gym v1
positional link + out-of-run check ------------> closed-loop twin --------> extension
SCooP-iT on arXiv -----------------------------> SCooP submission (cites it)
```

Labels are the shared bottleneck. They unlock E3b, `CoopDetect-v0` and the
whole dataset paper, so start them early (`rosbag_to_opv2v`: `label_static`
for static objects, by hand for moving ones).

## Order of work

### Phase 1: SCooP-iT, compute only (no new data)

From `scoop_testbed_iotj/DIRECTION.md` 5A:
1. confirm which pass each number comes from;
2. the E1 terms and the headroom figure;
3. the E2 confidence interval;
4. the E4 mode distance;
5. the E5 best-fit baseline;
6. the calibration rows;
7. the error budget;
8. `CommSchedule-v0` on coop_2 and survey_1 (the Sec. IX numbers).

Also decide the cross-pass figure: rerun `check.py maps --align` for a cm value, or quote 96.4 % within 10 cm.

### Phase 2: one field session

- Before it, build the payload node and `twin_run.py`.
- Record A1/B1 survey, A2/B2 cooperation, and a probe-off run on the survey_1 route.
- This gives E3a, the E4 day half, the E5 control and the sniffer bias.

### Phase 3: labels on coop_2

- Static boxes in the map frame, plus moving objects by hand.
- Check them projected into every agent (`show_labels.py`).
- This is the first labelled sequence of SCooP, and the input to E3b and `CoopDetect-v0`.

### Phase 4: the twin

1. `make_scenario.py` from coop_2, then `run_isaac.py`.
2. Scan-to-map agreement of synthetic against real scans at the same poses (E3b, first half).
3. With Phase 3: one cooperative detector on real and twin data, under the ideal and logged channels (E3b, second half).

### Phase 5: submit SCooP-iT

- Condense from 16 pages to IoT-J's length (check the Author Portal policy).
- Fig. 1, the headroom, clearance and simgap figures.
- Re-check Table II against the PDFs.
- Remove every `\todo`.
- Post to arXiv at submission.
- Release the pipeline and `scoop_gym` v0 with it.

### SCooP track (runs alongside, after Phase 3)

1. Remaining collection sessions (fill `tables/sessions.tex`).
2. Labels at scale.
3. Release format and devkit.
4. Q1 transfer, Q2 heterogeneity, Q3 gain strata and Q4 link cost (training in the appendix tables).
5. Failure cases.

Submit after SCooP-iT is on arXiv.

### Gym track

- **v0, done:** `CommSchedule-v0` in replay mode.
- **v1, in SCooP-iT (Sec. IX):**
  - `CoopDetect-v0`: needs Phase 3 and recorded frames through the devkit.
  - The twin in open loop: needs Phase 4 (the same run as E3b).
- **v2, the extension:**
  - the positional link (survey radio map plus contention mode), validated out of run;
  - robot control in Isaac instead of pose replay;
  - `ConnectedCoverage-v0` and `RelayPlacement-v0`;
  - agents: rule-based, RL, and VLM;
  - the headline experiment: success under the ideal bus against the measured link, with misses attributed by cause.

## Open decisions

- Deadlines for each submission. Phases 1-5 are in order, but no dates are set.
- The cross-pass figure (Phase 1).
- The venue of the extension (when the closed loop works).
- The project name: keep SCooP or pick a Kimera-style name (see the 24 Oct discussion).
