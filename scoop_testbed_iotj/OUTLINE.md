# SCooP-iT: direction, outline, tests and scope (26 Oct revision)

This revision follows the pre-review. It supersedes `WRITING_GUIDE.md`,
`DIRECTION.md` and the 25 Oct version of this file.

## Direction

**Title:** SCooP-iT: An Indoor Multi-Agent Testbed with a Measured, Replayable
Wireless Link. SCooP = Sensing and Communication for Cooperative Perception.
Venue: IEEE IoT-J (target 10 pages; charges beyond 8).

**One claim:** a real indoor team can be captured precisely enough (sensing
and communication on one clock and in one frame) for its logged link to be
replayed into any collaborative method, and a synthetic channel with the
same statistics does not reproduce it.

**The comparison that carries the paper:** the **fitted** channel (the logged
link's statistics, with no structure in time or place) against the **logged**
one. Published synthetic channels are shown only for reference, because they
deliver everything by construction.

**Three contributions:**
1. The testbed: heterogeneous agents, every link on the sensors' clock, sniffer bias bounded.
2. A building-scale pose reference without motion capture, GNSS or a survey scanner: the map's mesh coloured by 3D Gaussian splatting of the recorded images (to do: PSNR/SSIM on held-out views), checked by a second pass, held-out boards and control distances (about 2 cm RMS).
3. **A replay and a digital twin that let other researchers test and train collaborative methods on the recordings** (Sec. VII):
   - **replay (test):** takes the messages a method sends and predicts, from the recorded link, when each would have arrived or whether it would have been lost; works for message-level models and unmodified ROS 2 systems; checked against messages sent live (E3); a synthetic channel with the same statistics misstates the delivery of large messages (E5);
   - **twin (train):** built from the recordings; generates new scenarios to train on; each part checked against held-out passes before it trains anything (E6). Training in the twin and testing on held-out recordings moved to the SCooP dataset paper (Q5).

The send-size decision and Swarm-SLAM under fitted vs. logged moved to the SCooP dataset paper (Q4), which judges methods; this paper only shows the gap at the message level.

## What was removed, and where it went

| Removed | Why | Where it goes |
|---|---|---|
| Gym deployment on the robots (closed loop), task tables | needs trained motion policies and a second session | next step; the replay and twin are back as Sec. VII and E6 (27 Oct) |
| Embodied / Gym framing in title, abstract, keywords | the paper no longer claims it | follow-up |
| The seven-questions framing and "worth" subsection | uniqueness belongs in Sec. II, reproducibility in the design and release | — |
| Related work on AoI, task-oriented communication, edge, simulators, Lingjing | not needed for the claim | — |
| Stale numbers: 1.06 cm cross-pass, 7.9/2.1 cm, 0.15 cm, `mapping_A`, two mixed RTT regimes | superseded by the 20261015 reference and the 24 Sep passes | replaced |
| "Capacity governed by geometry" | not supported (see the capacity check) | removed |
| "Only a synchronized testbed can measure one-way delay" | the replay uses RTT/2 until E3's payload stamps exist | reworded |

## Outline (compiles to 13 pages with placeholders; about 6,800 words)

| Sec. | Content | Words now | Target |
|---|---|---|---|
| Abstract | problem; capture; checks; replay; logged vs. fitted | 250 | 250 |
| I Introduction | the gap; why indoors is hard; what exists; the one claim; three contributions | 560 | 600 |
| II Related Work | testbeds; trace replay (Mahimahi, Pantheon, NemFi, Syed et al., CausalSim); channel measurement; multi-robot datasets with real links (Lajoie et al.); cooperative perception (CooperScene, EMP); the gap + Table II | 630 | 650 |
| III Testbed Design | agents; network (capacity check flagged); four design decisions; recording | 890 | 650 |
| IV Time and Space Alignment | clock; sniffer bias (lower envelope); error budget at measured speed; extrinsics | 1070 | 750 |
| V Pose Reference | map; fiducial chain; checks (thickness, second pass, control distances, held-out boards); scoring | 1160 | 850 |
| VI Link Instrumentation | link state; CSI (overheard channel; Doppler argument) | 390 | 350 |
| VII-A Replay | model (RTT/2 + s/G, two hops); four channels; message API and ROS 2 relay; assumptions and limits | 470 | 450 |
| VII-B Twin | twin from build passes; measured vs. added parts; the gate; train in twin, test on held-out recordings; interface | 650 | 600 |
| IX Experiments | spec sheet; E1-E5 | 1150 | 1500 when results arrive |
| X Lessons and Limitations | four lessons; limits | 260 | 250 |
| XI Conclusion | results; closed-loop twin as the next step; release | 160 | 150 |

Condense III-V by about 900 words, for example the board-detector
distortion detail and the extrinsics checks (move to the supplementary).

**Figures:**
- Fig. 1 is now one window of coop_2 on one timeline: positions, clearance, RSSI, RTT, goodput, and message arrivals against the frame.
- The others: platforms, reference pipeline, link trace, headroom (E1), clearance (E2), simgap (E3 left, E5a right).
- The replay diagram is dropped; the equation suffices.

## Tests

| Test | Question | Pass (stated before the test) | Data | State |
|---|---|---|---|---|
| E1 Precision | testbed errors vs. breaking points derived for indoor objects (person box: 0.20 m at IoU 0.5, 0.11 m at IoU 0.7) | each term >= 10x below | existing | LiDAR robot, clock, detector pass at IoU 0.5; control distances (2 cm RMS) meet 10x exactly; cart (6.5x), second pass (6.7x), strict clock bound (8.3x) fall short; v95, sniffer bias, radar to fill |
| E2 Consistency | map clearance predicts detrended RSSI | rho > 0.3, drop > 5 dB, block-bootstrap CI excluding 0 | existing | point estimates in hand; CI, exponent sweep, cart-side split to run |
| E3 Replay | replayed vs. live arrivals | >= 90 % frame agreement; logged closer than fitted | field session (payload node at 10 Hz, 1 KB / 100 KB / 1.5 MB, alone and with the other robot, survey and cooperation) | not run |
| E4 Repeatability | same setting, same link on another day | day distance < half the mode distance | existing + the E3 session | mode half computable |
| E5 Gap | in-frame share, logged vs. fitted (published for reference), also against offered load over goodput | none, report either way | existing | logged in hand (86 % / 59-68 %); fitted column to add |
| E6 Twin | twin vs. held-out pass: scans, dropouts, place-indexed link | scans <= 3 cm; dropouts within session spread; link within 10 pt and closer than fitted | twin run on a held-out pass | not run |

## Checks to run before anything else
1. **Capacity:** the logged downlink of 598-676 Mbit/s exceeds the 433 Mbit/s PHY maximum for one stream at VHT-80. Pull NSS and MCS from the radiotap headers. The uplink of 275-282 Mbit/s sits at the practical USB 2.0 ceiling of the RTL8852BU. Every replayed arrival uses G(t), so this goes first.
2. **Reference numbers:** the stage table says 12.3 mm loop and 7.6 / 1.8 cm; the results log says 13.9 mm and 7.8 / 1.9 cm for the same pass. Scan-to-map vs. the reference is 1.3 cm in the paper and 1.5 cm in the log. Pick the final run.
   Control distances: 2 cm RMS is in; add N, the length range, the max and the relative error.
3. **Frame offset:** explain the 0.49 m / 2.95 deg offset between the two passes' board-anchored frames.

## Scope
- **In this paper:** the testbed and its checks (E1, E2, E4); the replay and its validation (E3); logged vs. fitted at the message level (E5); the twin (E6).
- **In the SCooP dataset paper:**
  - the data at every site, labels, detection and segmentation, Q1-Q4;
  - ideal, synthetic, fitted and logged cost across methods and sites, including the T1 C-SLAM benchmark;
  - moved from here: the send-size decision and Swarm-SLAM under fitted vs. logged (Q4); training in the twin, tested on held-out recordings (Q5).
- **In a follow-up paper:**
  - closing the loop: motion policies trained in the twin, deployed on the robots;
  - validated with sim-vs-real correlation across policies (Kadian et al., RA-L 2020).

## Order of work
1. The three checks above.
2. Compute: E1 terms, E2 intervals and sensitivity, E4 mode half, E5 fitted column.
3. The field session: E3, E4 day half, sniffer bias.
4. Condense to about 10 pages, draw the figures, verify Table II (including the new Lajoie et al. row), clear every `\todo`, then submit and post to arXiv.
