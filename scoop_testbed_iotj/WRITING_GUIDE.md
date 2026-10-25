# SCooP-iT: section-by-section writing guide

> **25 Oct:** `OUTLINE.md` holds the final outline, tests and scope. It
> supersedes the task lists for Secs. VIII-IX below; the Gym moves before the
> Experiments.

**Title:** SCooP-iT: A Real-World Indoor Testbed and Environment for Embodied
Multi-Agent Perception. SCooP = Sensing and Communication for Cooperative
Perception. Venue: IEEE IoT-J.

**Claim:** a real heterogeneous indoor team, captured on one clock and in one
frame (sensing and communication alike), offered three ways, each checked
against reality:
- **the recordings** as a test bench (replay);
- **the Gym**, a closed-loop environment built from the recordings;
- **the testbed**, where a Gym-trained policy is deployed.

One file per section in `sections/`; `main.tex` only holds the preamble and
the `\input`s. Compile `main.tex` after each section.

## Budget and status (24 Oct: compiles to 18 pages, about 10,800 words)

Budget for 12 pages with about 14 floats; adjust once you have checked the
IoT-J length policy. IEEE asks for abstracts of 150-250 words; check the
IoT-J author guide.

| Sec. | File | Words now | Target | Status |
|---|---|---|---|---|
| Abstract | `s00_abstract.tex` | 389 | 250 | rewritten 24 Oct; cut to length last |
| I Introduction | `s01_introduction.tex` | 1060 | 900 | rewritten 24 Oct |
| II Related Work | `s02_related.tex` | 1410 | 1000 | drafted; II-F added 24 Oct |
| III Testbed Design | `s03_design.tex` | 850 | 700 | drafted |
| IV Time and Space Alignment | `s04_alignment.tex` | 1080 | 800 | drafted; 9 `\todo` |
| V Ground Truth | `s05_ground_truth.tex` | 1140 | 900 | drafted; stale cross-pass number |
| VI Link Instrumentation | `s06_link.tex` | 390 | 350 | drafted |
| VII Replay | `s07_replay.tex` | 500 | 450 | rewritten 24 Oct |
| VIII Experiments | `s08_experiments.tex` | 2150 | 1500 | drafted; E3 rewritten 24 Oct; results open |
| IX SCooP-iT Gym | `s09_gym.tex` | 1210 | 1000 | rewritten 24 Oct; results open |
| X Lessons and Limitations | `s10_lessons.tex` | 460 | 400 | rewritten 24 Oct |
| XI Conclusion | `s11_conclusion.tex` | 200 | 150 | rewritten 24 Oct |

**Suggested writing order:** first the sections whose content is fixed
(III, IV, V, VI, VII), then II, then IX (design part), then VIII as results
arrive, then I and the abstract last, then X and XI.

---

## Abstract (`s00_abstract.tex`)
Five moves, one or two sentences each:
1. Problem: methods come from vehicles and simulation; a real indoor team differs on every layer.
2. What SCooP-iT is: the team, captured on one clock and one frame, without motion capture or GNSS.
3. What it records: link state, RTT, goodput and CSI per pair; clock below 1 ms; placement 10 mm / 31-54 mm.
4. Three uses: replay test bench, the Gym, deployment on the testbed.
5. Results: headroom 6-101x; map predicts radio (rho 0.60); E3 (todo); the 59-68 % vs. 100 % finding.

**Missing:** the E3 sentence; cut to 250 words.

## I. Introduction (`s01_introduction.tex`)
| Paragraph | Topic sentence |
|---|---|
| 1 | Teams of robots and fixed sensors see beyond any one's occlusions, but their methods come from vehicle data and simulation, and a real indoor team differs on every layer. |
| 2 | IoRT makes robots network nodes; the surveys say sensing, communication and task must be measured together, and name the missing integrated testbed. |
| 3 | Indoors, building such an instrument is hard on every layer: one timeline, one frame without GNSS, every link, and something others can reuse and extend. |
| 4 | No existing platform provides this: IoT testbeds, channel datasets, networked robots, cooperative-perception platforms, embodied simulators. |
| 5 | SCooP-iT and its three uses: recordings as test bench, the Gym, the testbed for deployment. |
| 6 | Five contributions: testbed; ground truth; validated replay; the Gym; what the real world shows. |
| 7 | Seven questions (five trust, two worth) and the roadmap. |

**Missing:** the cross-pass figure in contribution 2; the E3a result in contribution 3.

## II. Related Work (`s02_related.tex`)
| Subsection | One-line message |
|---|---|
| A IoT and wireless testbeds | They measure the radio; no task is scored against ground truth. |
| B Timeliness and networking | Contention, AoI, task-oriented communication, trace replay: no sensing deadline, and replay feeds a socket. |
| C Channel measurement | CSI on sounding links with survey positions; geometry is an input, never an independent check. |
| D Robots as IoT nodes | Real links monitored but not released, or precise positions over an ideal network; indoor GT needs anchors or scanners. |
| E Cooperative perception | Outdoor, GNSS; the link is a scalar delay or a live run; CooperScene is the nearest. |
| F Simulators and environments | Generated worlds (Habitat, CARLA, Isaac Lab), Lingjing's ideal bus, log-based driving sims for one ego. None builds a multi-agent environment from a real team and checks it. |
| G The gap | Table II; SCooP-iT is the integrated testbed the surveys call for, bridged to simulation. |

**Missing:** re-check every Table II cell against the PDFs; add IRIS and CIVAT as rows; verify the new bib entries (Habitat, CARLA, Orbit, Gymnasium, PettingZoo, nuPlan, Waymax). Cut about 400 words: shorten B and C.

## III. Testbed Design (`s03_design.tex`)
| Subsection | Message |
|---|---|
| Opening + Fig. 1 | Two mobile and three fixed agents on one AP; each agent senses, computes, communicates. |
| A Agents | Three classes, each with a role: the LiDAR robot builds the map, the cart is the low-cost agent, the masts give fixed vantage. |
| B Network | One AP on channel 149, VHT-80, 1x1, 802.11ax off (forced by the CSI receiver); capacity 275-282 / 598-676 Mbit/s. |
| C Design decisions | Four: the link is measured, not used; single stream; survey and cooperation modes (contention as a knob); fixed rigs. |
| D Recording | One bag per unit, only what the sensors produce; one pipeline merges onto one timeline. |

**Missing:** Fig. 1 redraw; does `mobile_2` have a radar; the BOM costs and setup effort; confirm which pass the capacity numbers come from.

## IV. Time and Space Alignment (`s04_alignment.tex`)
| Subsection | Message |
|---|---|
| A Clock | One chrony domain; measured offset <= 0.99 ms over coop_2; strict bound 12 ms, carried as the worst case. |
| B Sniffer bias | NTP cannot see path asymmetry; stamped probes measure it. |
| C Error budget | Every timing error becomes displacement v * dt; the budget is dominated by sensor physics, not sync. |
| D Extrinsics | Hand-eye, target-free LiDAR-camera, reflector for radar; each checked on held-out data; detector validated first (8 mm). |

**Missing (9 `\todo`):** repeat the clock over every pass; the sniffer bias value; latency per sensor class; `delta x_max`; the stacked-bar budget; the board split; calibration table rows (LiDAR-camera reprojection, radar reflector, base-LiDAR).

## V. Ground Truth Without Motion Capture (`s05_ground_truth.tex`)
| Subsection | Message |
|---|---|
| Opening + Fig. pipeline | A self-built map carries every pose; boards anchor agents without LiDAR. |
| A Reference map (`refmap_main.tex`) | SLAM -> 46 ms time fix -> plane bundle adjustment -> TSDF; walls 7.6 cm, floors 1.8 cm; loop 12.3 mm. |
| B Fiducial chain | Boards posed from the mapping session; masts placed through a shared board. |
| C Validation | Four checks, each bounding one error: second pass (cross-robot: 96.4 % within 10 cm), control distances, held-out boards (10 mm / 3.1 cm), cross-robot consistency. |
| D Scoring capability | Label once in the map frame and project into every agent. |

**Missing:** the cross-pass figure (the old 1.06 cm compares two superseded maps); the cause of the 0.49 m frame offset; control distances (if rangefinder data exist); p95 and the board split; cross-robot consistency. Update `tables/gt.tex` to the 20261015 reference (7.6 / 1.8 cm).

## VI. Link Instrumentation (`s06_link.tex`)
| Subsection | Message |
|---|---|
| A Link state | 5 Hz passive, 100 Hz RTT, 4 s goodput; outages as labels; RTT idle vs. loaded. |
| B Channel state | Nexmon CSI, 242 subcarriers, 104-141 Hz, valid below 1.2 m/s; aggregation caps the rate. |

**Missing:** one RTT regime stated per pass with its load; extend the link-trace figure with RTT and CSI.

## VII. Replay: The Recordings as a Test Bench (`s07_replay.tex`)
| Paragraph | Message |
|---|---|
| Model | Arrival = t + d(t) + s / G(t), or dropped in an outage; one-way delay from synchronized probes. |
| Beyond the link | Sensors and pose are replayed in the same modes (ideal, logged, fitted). |
| Interface | Any method that emits (sender, receiver, t, size) gets an arrival or a drop. |
| Positioning | Trace replay feeds sockets; SCooP-iT replays per message into perception, on the sensors' clock. |

**Missing:** names of the released modules.

## VIII. Experiments (`s08_experiments.tex`)
| Part | Message | Data |
|---|---|---|
| Opening + spec sheet | Every guarantee, its pass criterion, its bound, its check. | in hand, partly |
| E1 Precise | Testbed errors 6-101x below the breaking points; two short of 10x. | partly: `headroom.py` |
| E2 Consistent | Clearance predicts detrended RSSI: rho 0.60, 5.4 dB on mobile_2. | mobile_2 link in hand; CI to add |
| E3a Replay vs. live | >= 90 % frame agreement. | field session |
| E3b Gym world vs. real | Scans, conditions, link out of run, link-induced dAP. | twin run, models, labels |
| E3c Gym policy on testbed | Real success within the Gym's predicted interval. | trained policy, deployment session |
| E4 Repeatable | Day distance < half the mode distance. | mode half computable; day half from the field session |
| E5 Necessary | Latency looks fine, yet 59-68 % in frame vs. 100 % synthetic. | in hand; best-fit baseline to add |
| Worth | Unique (Table II), reproducible (BOM, pipeline). | bookkeeping |

**Missing:** every E3 result; E1 remaining terms; E2 CI; E4; E5 best-fit; cut about 650 words once results replace the "Remaining" sentences.

## IX. SCooP-iT Gym (`s09_gym.tex`)
| Subsection | Message |
|---|---|
| Opening | Recordings can't say what an agent would see elsewhere; the Gym can, and it is tied to the testbed at both ends. Lingjing's contract, measured world and network. |
| A World from real data + Table gym_world | Building, sensors, motion, localization, link, objects: each built from real data, each checked (E3b). |
| B Interface | Isaac Lab tasks; a shared layer decides arrivals, conditions, reward, attribution; observation and action. |
| C Conditions and augmentation | Ideal / logged / fitted / augmented; objects from free space at real counts; dropouts fitted; train on anything, evaluate on real. |
| D Tasks + Table gym | Five closed-loop tasks in the twin, three replay tasks on the recordings. |
| E Train, test, deploy | Train in the twin, test on the recordings, deploy on the testbed, recalibrate. |
| F Demonstrations | CommSchedule (replay, today); OcclusionHandoff through the full protocol. |
| Scope | Geometric twin; the link indexed by place is a model. |

**Missing:** the CommSchedule numbers on coop_2; the OcclusionHandoff protocol results.

## X. Lessons from the Real World, and Limitations (`s10_lessons.tex`)
One paragraph per lesson, each a condition the Gym now reproduces: CSI shapes the network; the sniffer's second hop; cold-start ICP; the SLAM clock (46 ms); sensors short of their rate (ZED about 1 Hz). Then the limits.

**Missing:** confirm the Ethernet change; the scale line (sites, sessions, hours).

## XI. Conclusion (`s11_conclusion.tex`)
One paragraph: what it gives, the three uses, what the real world showed, next steps.

**Missing:** the E3a result.
