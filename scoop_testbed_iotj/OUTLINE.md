# SCooP-iT: final outline, tests and scope (25 Oct)

This file supersedes the task lists in `WRITING_GUIDE.md` (Secs. VIII-IX) and
`DIRECTION.md`. Where they disagree, this file wins.

**Title:** SCooP-iT: A Real-World Indoor Testbed and Environment for Embodied
Multi-Agent Perception. SCooP = Sensing and Communication for Cooperative
Perception. IEEE IoT-J.

**Claim:** a real heterogeneous indoor team, captured on one clock and in one
frame, sensing and communication alike. It is offered three ways, each
checked against reality:
- **the recordings**, a replay test bench, for testing;
- **the Gym**, a closed-loop twin built from the recordings, for training;
- **the testbed**, where a Gym-trained policy is deployed, for validation.

**Rule for every test:** it must depend on something only SCooP-iT has, and
it must be able to change an answer. The comparison that matters is the
**fitted** channel (the real link's statistics, without its structure in time
and place) against the **logged** one.

## 1. Outline (target 12 pages, about 8,400 words)

| Sec. | Title | Content | Words |
|---|---|---|---|
| — | Abstract | Real-world gap; what is captured; three uses; E1, E2, E3, E5 results | 250 |
| I | Introduction | Real indoor team vs. vehicles and simulation; IoRT and the surveys; why indoors is hard; no existing platform; SCooP-iT and its three uses; five contributions; seven questions | 900 |
| II | Related Work | IoT testbeds; timeliness and trace replay; channel measurement; robots as IoT nodes; cooperative perception; simulators and environments; the gap (Table II) | 1000 |
| III | Testbed Design | Agents, network, four design decisions, recording | 700 |
| IV | Time and Space Alignment | Clock, sniffer bias, error budget, extrinsics | 800 |
| V | Ground Truth Without Motion Capture | Reference map, fiducial chain, four validation checks, scoring capability | 900 |
| VI | Link Instrumentation | Link state, CSI | 350 |
| VII | Replay: The Recordings as a Test Bench | Model; conditions (link, sensors, pose: ideal / logged / fitted); interface (message models and ROS 2 relay); positioning | 500 |
| VIII | SCooP-iT Gym | World built from real data (Table); interface; conditions and augmentation; OcclusionHandoff; train-test-deploy | 900 |
| IX | Experiments | Spec sheet; E1-E5 (below) | 1600 |
| X | Lessons from the Real World, and Limitations | Five lessons, each now a Gym condition; limits | 400 |
| XI | Conclusion | | 150 |

Change from the current draft: the Gym moves **before** the Experiments
(VIII), so E3 and E5 can test it. In the Gym section, OcclusionHandoff is
the only task, and the task table becomes a one-line "the interface also
supports" sentence.

## 2. Tests

### Trust: can the tools be relied on?

| Test | Question | Pass (fixed before the data) | Data | State |
|---|---|---|---|---|
| **E1 Precise** | Are the testbed's errors small next to the effects it measures? | every term >= 10x below the breaking point of its kind | existing passes | partly; terms missing |
| **E2 Consistent** | Do poses and link logs agree when neither was checked by hand? | clearance vs. detrended RSSI: rho > 0.3 and drop > 5 dB on one link kind | existing passes | passes on mobile_2 (rho 0.60, 5.4 dB); CI to add |
| **E3a Replay vs. live** | Does replaying the logged link reproduce live arrivals? | >= 90 % frame agreement; logged closer than fitted | field session B1 | not run |
| **E3b Gym world vs. real** | Does each part of the twin match real runs that did not build it? | synthetic scans within 3 cm of real at the same poses; fitted dropout and noise match a held-out session; the place-indexed link, fitted on survey_1, predicts a held-out pass's in-frame share within 10 points and better than the fitted channel | twin run of coop_2; existing passes | not run |
| **E3c Gym prediction vs. testbed** | Does the Gym predict what the real robots do? | OcclusionHandoff real success rate within the Gym's 95 % interval for the same scenario | trained policy; field session B2 | not run |
| **E4 Repeatable** | Does the same setting give the same link on another day? | day distance < half the survey-vs-cooperation distance (RTT, goodput) | existing passes (mode half); B1 (day half) | mode half computable |

### Necessity: what the real world changes

| Test | Question | Report | Data | State |
|---|---|---|---|---|
| **E5a Messages** | What does a synthetic channel hide? | in-frame share of 1 KB / 100 KB / 1.5 MB under published synthetic, fitted and logged channels; loaded RTT alongside | existing passes | in hand (86 % / 59-68 % / 100 %); fitted column to add |
| **E5b A decision** | Is a policy tuned on the fitted channel still right on the real one? | `CommSchedule`: the policy best under fitted, scored under logged; regret against the best under logged | existing passes | code ready; not run on real passes |
| **E5c A real system** | Does the real link change an unmodified multi-robot SLAM beyond what its statistics predict? | Swarm-SLAM on coop_2 through the ROS 2 relay, under ideal, fitted and logged (5 runs each): joint ATE, inter-robot loop closures, merge time, map age. Claim only differences larger than run-to-run spread | existing passes; tools exist | not run |
| **E5d A closed-loop policy** | Does training over the measured link change what happens on the real robots? | OcclusionHandoff policy trained over the ideal link vs. over the measured link: success in the Gym and on the testbed | Gym; B2 | not run |

E5 has no pass criterion; each part reports whether fitted and logged
differ, and either outcome is reported.

## 3. Scope

**In this paper**
- The platform and its validation: clock, frame, ground truth, link (E1, E2, E4).
- The replay test bench:
  - the link, plus sensor and pose conditions;
  - message-level and ROS 2 relay interfaces (E3a, E5a-c).
- The Gym:
  - the Isaac twin built from site 1, with every part checked (E3b);
  - the interface, conditions and object augmentation (training only);
  - one closed-loop task, OcclusionHandoff, through train, test and deploy (E3c, E5d).
- One collaborative-SLAM system on one sequence (E5c).

**In the SCooP dataset paper**
- Labels, collaborative detection and segmentation.
- The T1/T2 benchmarks at every site; Q1-Q4.
- "Does augmentation help" as a finding.

**In the extension (after this paper)**
- Further Gym tasks: ActiveCoopDetect, CoopSearch, ConnectedCoverage, RelayPlacement, EdgeOffload / VLA, language agents.
- The agent benchmark (rule-based, RL, VLM); camera-based tasks; other sites in the Gym.

**Not planned**
- More agents, other radios, multi-antenna CSI: listed as next steps only.

## 4. What has to happen, in order

| # | Work | Unlocks |
|---|---|---|
| 1 | Compute on existing passes: E1 terms, E2 CI, E4 mode half, E5a fitted column, E5b on coop_2 and survey_1; decide the cross-pass figure | IV, V, E1, E2, E4, E5a, E5b |
| 2 | Swarm-SLAM through the relay on coop_2, 5 runs per channel (check the camera rates first; use the camera-only setup if LiDAR-to-RGB-D matching fails) | E5c |
| 3 | Field session B1 (payload node, A1/B1 survey, A2/B2 cooperation, probe-off) | E3a, E4 day half, sniffer bias |
| 4 | Isaac world of site 1: scene, Scout Mini differential drive, cart, mast, RTX LiDAR, semantics; fitted conditions; place-indexed link | E3b |
| 5 | OcclusionHandoff in Isaac Lab: scripted baseline, then RL over ideal and over measured link | E5d (Gym half) |
| 6 | Field session B2: deploy both policies | E3c, E5d (testbed half) |
| 7 | Condense, figures, Table II check, no `\todo`; submit and post to arXiv | submission |

Steps 1, 2 and 4 can run in parallel now. Step 3 needs only the payload
node.

## 5. Fallbacks

| If | Then |
|---|---|
| B2 can't happen before the deadline | E3c and E5d report the Gym half only; the deployment is described as the next step |
| The place-indexed link fails E3b | OcclusionHandoff uses the fitted link, and the paper says the twin's link is statistical only |
| Fitted ≈ logged in E5b or E5c | Report it: for that task, the real link's statistics suffice. E5a still stands |
| Swarm-SLAM finds no inter-robot loop closures | Use the camera-only setup; if it still fails, E5c reports map age and per-robot ATE only |
