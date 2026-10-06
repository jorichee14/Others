# Paper A (RA-L): the testbed, no dataset release

**Name.** The testbed gets a name tied to the dataset, so the two papers read
as a pair: the testbed records, and SCooP is what it recorded.

| Name | Reads as | Notes |
|---|---|---|
| **SCooP-Bed** (recommended) | "the SCooP testbed" | Says what it is. Paper B can say "recorded on SCooP-Bed". |
| SCooP-Live | "SCooP with the link live" | Stresses measured and replayed links. It could be misread as real-time. |
| SCooP-Lab | "the lab behind SCooP" | Soft; reads as a facility rather than an instrument. |

**Working title.** *SCooP-Bed: An Indoor Multi-Robot Testbed for Collaborative
Perception over Measured Wireless Links*.

**The one claim.** A heterogeneous indoor team with these properties:
- every inter-agent link is measured alongside the sensors;
- all agents run on one clock and sit in one metric frame;
- there is no motion capture or GNSS;
- any collaborative method can be run against the link the agents actually had.

**What the paper is not.** It releases no sequences, labels or benchmark
leaderboard, and it reports no findings about collaborative perception. Those
belong to Paper B. Paper A shows that the instrument works, how precisely it
works, and one example of what it reveals.

**Budget.** RA-L allows 6 pages, plus 2 more with an overlength charge. Plan for
8, with references included.

---

## How to make the testbed the hero

1. **A spec sheet as the central table.** This is the testbed's contract: every
   guarantee it makes, each with a measured bound and the check that produced
   it. Reviewers of a testbed paper read this table first.

   | Guarantee | Bound | Validated by |
   |---|---|---|
   | Clock agreement between agents | about 70 µs (0.2 mm at 3 m/s) | chrony logs, every run |
   | Sniffer clock bias | to be measured | probe-packet stamps |
   | Spatial extrinsics | residual per sensor pair | held-out reflector and board positions |
   | Agent pose in the site frame | about 1 cm surface agreement; board residual | second pass, rangefinder, held-out boards |
   | Link state | 5 Hz passive, 100 Hz RTT, 4 s goodput | explicit outage labels |
   | Channel state (CSI) | 104–141 Hz; valid below 1.2 m/s | λ/2 sampling argument |
   | Replay fidelity | arrival-time error; agreement on which frame each message lands in | experiment E3 |

2. **A tolerance-margin figure (the key figure).** On one log axis, plot the
   testbed's error next to the error at which collaborative perception breaks:
   - pose error: about 1 cm, against the 0.2 m at which late fusion falls below
     ego-only in the OPV2V study;
   - time: 70 µs, against the delay at which every fusion method falls below
     ego-only. Express it as displacement (speed × delay): the 100 ms figure was
     measured at vehicle speed. See E1.

   The point is that the testbed is two to four orders of magnitude more precise
   than the effects it exists to measure. This is the strongest argument a
   testbed paper can make.

3. **Controllable knobs.** A testbed is defined by what you can vary and repeat:
   - load mode: survey, where one agent owns the channel, or cooperation, where
     agents contend;
   - number of agents and agent class;
   - infrastructure placement;
   - trajectory paradigm: concurrent, intersecting or rendezvous;
   - agent speed.

   Each knob should come with a measured effect. For example, load mode moves RTT
   from 3.3 ms to 60–109 ms.

4. **Reproducibility.** Give a bill of materials with cost per agent class, the
   setup effort per new site (mapping-session length, number of boards,
   rangefinder distances), and the software stack. Robotarium and Duckietown both
   lead with these.

5. **Fig. 1 is the testbed, not a result.** Show a building floor plan with:
   - the agents placed in the map frame;
   - the inter-agent links drawn and coloured by measured RTT or RSSI;
   - a strip below with one shared timeline of sensor frames, link samples and
     CSI.

   The figure should say "one clock, one frame, every link" without any text.

---

## Section by section

### Abstract (about 180 words)
Problem, then gap, then the testbed, then the three guarantees with numbers,
then one demonstration sentence. Lead with the measured numbers (clock bound,
cm-level pose, link sample rates), not with adjectives.

### I. Introduction (about 0.9 pages)
**Question it answers:** why does indoor collaborative perception need a new
testbed?
- **Need.** Collaboration is only useful if a neighbour's message arrives in time
  and lands in the right place. Indoors, the link is shared and contended, and
  there is no GNSS.
- **Gap, in three sentences.**
  - Multi-robot testbeds (Robotarium, Duckietown, CPM Lab, ChoiRbot) give
    precise positions but treat the network as ideal.
  - Communication-aware systems (ROS-NetSim, CHORD, ACHORD, MOCHA) model or
    survive the link but do not score perception against ground truth.
  - Cooperative-perception platforms (V2V4Real, V2X-Real, TUMTraf, CoPeD) are
    outdoor or GNSS-referenced, and they idealize or simulate the link.
- **Contributions (three bullets, each with a number).**
  1. A heterogeneous testbed of three agent classes in which every pair's link
     state and channel state is recorded on the sensors' clock.
  2. A procedure that gives building-scale ground truth without motion capture:
     a self-built reference map, plus a fiducial chain for agents without LiDAR,
     validated by measurements that never enter it.
  3. A channel-in-the-loop replay that runs any collaborative method against the
     recorded link, validated against live execution.
- **Highlight sentence.** Under contention, the measured delay moves from
  about 3 ms to 60–109 ms, which crosses one LiDAR frame. Only a testbed that
  records the link can show which messages cross it.

### II. Related Work (about 0.7 pages)
**Question it answers:** what can this testbed do that no existing one can?
- **A. Multi-robot testbeds and platforms:** `wilson2021robotarium`,
  `pickem2017robotarium`, `paull2017duckietown`, `preiss2017crazyswarm`,
  `kloock2021cpmlab`, `hyldmar2019fleet`, `testa2021choirbot`,
  `pichierri2023crazychoir`, `mokhtarian2025survey`.
- **B. Communication in multi-robot systems:** `calvofullana2021rosnetsim`,
  `ginting2021chord`, `saboia2022achord`, `cladera2024mocha`,
  `blumenkamp2022framework`, `tian2023resilient`.
- **C. Cooperative perception platforms and measured links:** `coped`,
  `v2v4real`, `v2xreal`, `zimmer2024tumtrafv2x`, `krammer2022providentia`,
  `jimenez2011integrated`, `qiu2022autocast`, `zhang2021emp`,
  `hawlader2024v2xexp`, `cooperscene`.
- **Capability table** (rows are testbeds, columns are capabilities):
  - indoor;
  - heterogeneous agents, including infrastructure;
  - pose reference without motion capture or GNSS;
  - per-pair link measured;
  - CSI;
  - clock bound reported;
  - link replay into methods;
  - perception scoring.

  Only this testbed fills every column. Word the CSI and replay columns
  carefully, because CooperScene (2026) is the nearest case.

### III. Testbed Design (about 1.2 pages)
**Question it answers:** what is the testbed, and why is it built this way?
- **Agents.**
  - LiDAR robot: Scout Mini with Ouster, ZED 2i, 2× IWR6843 and the CSI sniffer.
  - RGB-D pushcart: D455 with on-board visual SLAM.
  - Infrastructure masts: Arducam, IWR6843 and a Raspberry Pi.
  - Give compute, storage and cost for each.
- **Network:**
  - a single access point on channel 149, VHT-80;
  - no mesh;
  - power save off.
- **Design decisions, each with its reason.** This is what makes it a testbed
  paper rather than a hardware list.
  - *The link is measured, not used to carry sensor data.* Measurement and
    transport stay separate, so any method's traffic can be replayed afterwards
    without the recording having been shaped by one method.
  - *Single stream, 802.11ax off.* The 1×1 CSI receiver requires it. It also
    keeps capacity governed by geometry rather than by scattering richness.
  - *Survey and cooperation modes.* Contention becomes a controlled knob instead
    of an accident.
  - *Fixed sensor rigs.* Calibrate once, not every session.
- **Recording:**
  - one bag per agent;
  - Ouster stored as raw packets for lossless re-decoding;
  - link and clock topics on the same timeline as the sensor topics.
- **Figures:** overview (Fig. 1), platform photos and a data-flow diagram.
- **Tables:** sensors, and the bill of materials.

### IV. Time and Space Alignment (about 1.0 page)
**Question it answers:** how well does the testbed put every measurement from
every agent on one clock and in one rig frame?
- **Clock.**
  - chrony server on the LiDAR robot, 8 s polling, and recording starts only
    after convergence.
  - Offsets and clock events are logged per run, so alignment can be checked at
    every frame.
  - Worst pairwise offset is about 70 µs.
- **Sniffer bias.** NTP cannot observe asymmetric path delay. The bias is
  measured from probe packets that are stamped on transmit and on capture. This
  is a nice testbed-specific trick: the measured link calibrates the clock.
- **Error budget** δx = v·Δt, split into clock, transport latency, exposure and
  LiDAR sweep. Synchronization contributes less than 1 mm, and the budget is
  dominated by sensor physics, which is either compensated or constant.
- **Extrinsics.**
  - hand-eye calibration for base to LiDAR;
  - target-free LiDAR–camera alignment;
  - reflector-board target for radar and camera;
  - each validated on held-out positions (residual table).
- **Highlight:** the error budget as a stacked bar, which feeds the
  tolerance-margin figure.

### V. Ground Truth without Motion Capture (about 1.1 pages)
**Question it answers:** how accurately does the testbed know where every agent
is, at building scale, using only onboard sensors?
- **Reference map.** A dedicated mapping session with GLIM, then the map is
  frozen. Every scan is re-registered against local plane fits. Explain why
  plane fits are used: a 3 cm offset shows only 0.5 cm of nearest-neighbour
  residual.
- **Fiducial chain for agents without LiDAR.**
  - board poses come from the mapping session itself;
  - infrastructure cameras are posed through a shared board view;
  - boards are placed in feature-poor regions;
  - board sightings serve as initialisation and drift bounds.
- **Validation from four independent checks.**
  - a second mapping pass four weeks later: 1.06 cm on walls, scale within
    2×10⁻⁵;
  - laser-rangefinder control distances;
  - held-out boards;
  - cross-robot surface consistency.
- **Scoring capability.** An object labelled once in the map frame projects into
  every agent's frame through these poses. This is how the testbed scores
  collaborative perception without per-frame annotation. Keep it to one
  paragraph: it is a capability, not a label release.
- **Highlight:** centimetre-level and building-scale, from onboard sensing only.
  The independent measurements check the result and never produce it.

### VI. Link Instrumentation (about 0.8 pages)
**Question it answers:** what does the testbed observe of each link, how often,
and within what limits?
- **Link state:**
  - RSSI, MCS, retries and channel-busy ratio at 5 Hz;
  - outages labelled explicitly as `associated=false`;
  - iperf3 goodput in each direction;
  - ICMP RTT at 100 Hz.
- **Channel state:**
  - Nexmon CSI, 242 subcarriers, QoS data frames only;
  - 104–141 Hz;
  - valid up to 1.2 m/s relative speed;
  - explain why the rate cannot simply be raised (aggregation).
- **Contention knob:** RTT is 3.3 ms idle and 60–109 ms under load, caused by
  queueing rather than path loss.
- **Figure:** one link trace with RSSI, goodput, RTT and CSI amplitude against
  time, alongside the agents' positions.

### VII. Channel-in-the-Loop Replay (about 0.6 pages)
**Question it answers:** can any collaborative method be run against the link
the agents actually had?

- **Model.** A message of size s, sent at time t over link (a, b), arrives at
  d_ab(t) + s/G_ab(t), and is dropped when the link is down. Here d_ab is the
  one-way delay.
  - Because the clocks agree to about 70 µs, d_ab can be measured directly from
    stamped probes, where they exist, instead of assumed to be RTT/2. Say this:
    only a synchronized testbed can do it.
  - The ideal, synthetic and logged channels go through one module, so the
    channel is the only variable between runs.
- **Interface.** Any method that emits (sender, receiver, t, size) plugs in. In
  return it gets an arrival time, or a drop, for each message.
- **Implementation.** Extend the existing OpenCOOD wrapper
  (`collab_perception_failure_analysis/commchannel`) with a `Schedule` driven by
  the trace. It already turns latency into staleness and drops collaborators.
  Use `rosbag_to_opv2v` to put SCooP frames into the layout it reads.
- **Literature: where the replay idea comes from and what it improves on.**
  - Trace-driven network emulation: `noble1997trace` (trace modulation, the
    origin), `hemminger2005netem`, `winstein2013sprout` (Cellsim),
    `netravali2015mahimahi` (packet-delivery-opportunity traces),
    `yan2018pantheon` (emulators calibrated against real paths).
  - Robotics and network co-simulation: `calvofullana2021rosnetsim`,
    `sommer2011veins`, `xu2021opencda`.
  - Synthetic channels in collaborative perception (what E5 compares against):
    `where2comm` (bandwidth budget), `v2xvit` (fixed or sampled delay),
    `lei2022syncnet`, `wei2023cobevflow` (asynchrony as a delay parameter).
  - **Positioning.** Network emulators replay a link for network software.
    Collaborative-perception benchmarks simulate a link for perception. This
    testbed replays the measured robot link into perception, on the same clock as
    the sensors.
  - **Expected reviewer question:** "why a fluid model and not Mahimahi-style
    packet traces?" Answer: the iperf windows give goodput, not delivery
    opportunities for each packet. E3 measures how much accuracy the simpler model
    loses.
- **Figure:** replay diagram.

### VIII. Experiments: Validating the Testbed (about 1.5 pages)
Every experiment asks a question about the testbed, not about collaborative
perception. Status below is as of the current pipeline (`scoop_pipeline`,
`mirc_dataset_paper`, `collab_perception_failure_analysis`).

**E1. Is the testbed precise enough for what it is meant to measure?**
- *How.* Put two things on one log axis, in metres.
  - The testbed's error terms: clock offset × speed, sniffer bias × speed,
    extrinsic residuals, pose error from held-out boards and the second pass.
  - The displacement at which fusion breaks.
- *Thresholds.*
  - Pose error: late fusion fell below ego-only at 0.2 m in the OPV2V study.
    That is a geometric threshold and carries over.
  - Latency: in the OPV2V study the damage was displacement, v × delay, at
    20 m/s. Indoors at about 1.5 m/s the same delay moves objects about 13 times
    less.
  - So state the latency threshold as displacement relative to object size, not
    as "100 ms". Use one sentence and one row per error term.
- *Can you do it now?* Mostly.
  - The clock (70 µs) and second-pass (1.06 cm) numbers exist, as do the OPV2V
    thresholds.
  - The held-out board residual, calibration residuals and sniffer bias are still
    TODO. They come from data you have. Plot with placeholders until they land.

**E2. Do the link logs agree with the building layout?**
- *How.* For every link sample (pair, time), compute the Fresnel clearance of
  the straight path through the reference map from both agents' reference poses.
  Then compare it with the measured RSSI and goodput at that time:
  - Spearman correlation per site;
  - RSSI and goodput for clear (above 0.6) against blocked (below 0.6) links;
  - a scatter plot.
- *Why it matters.* The clearance never saw the radio, and the radio never saw
  the map. Agreement validates both the poses and the link logs at once.
- *Can you do it now?* Yes.
  - `characterize_stage2.py` already computes link clearance and a validation
    correlation (`summary.json: link.validation` on `coop2_0828`).
  - The comms tables already carry each machine's map position per row
    (`analysis/comms`).
  - Run it on every processed pass and make one figure.

**E3. Does replay reproduce what the real link does to real messages?**
- *How.* Run it at the message level first. This needs no detector, which also
  keeps Paper A clear of Paper B's findings.
  1. *Run A* (normal pass): link monitors with iperf and ping, as in every
     recording.
  2. *Run B*: same route, same load mode, same day. Replace iperf with a payload
     node on each agent that sends messages over the real link at 10 Hz in three
     sizes: boxes (about 1 KB), compressed features (about 100 KB) and raw clouds
     (about 1–2 MB). Each message is stamped at send and at receive. With the
     clocks synced, the one-way delay is exact.
  3. *Replay*: feed B's send times and sizes through the trace recorded in A, and
     predict each arrival or drop.
- *Report.*
  - Arrival-time error per message size.
  - Agreement in which receiver frame (100 ms at 10 Hz) each message lands in.
    This is the quantity fusion actually sees.
  - Drop agreement.
  - The same comparison with the synthetic channel, which shows how much closer
    the logged trace gets.
- *Optional second level.* A late-fusion run on the received messages against
  the replayed ones. Do it only if an indoor detector exists by then.
- *Can you do it now?* No. It needs one new collection: run B, plus the payload
  node, which is a small ROS 2 publisher and subscriber with stamps. The analysis
  reuses the comms tables. Schedule it together with E4.

**E4. Is the testbed repeatable?**
- *How.* Take the same route and load mode on two days. Compare the link
  statistics:
  - the RTT, goodput and RSSI distributions;
  - RSSI against position along the route;
  - a Wasserstein or Kolmogorov–Smirnov distance between days;
  - the same distance between survey and cooperation modes.
- *Claim.* The day-to-day difference is small next to the mode difference, so a
  load mode is a reproducible setting. The pose half is the second mapping pass,
  four weeks later, 1.06 cm, already measured.
- *Can you do it now?*
  - Yes, if you already have two passes of the same route and mode on different
    days. `wifi_analysis.py` gives the distributions, and the distance is a few
    lines.
  - If not, record the repeat in the same visit as E3. Run A on two days covers
    it.

**E5. Demonstration: what the logged link shows that a synthetic one hides.**
- *How.* Keep it at the message level, from the logs, so it does not pre-empt
  Paper B. For each mode (survey, cooperation) and message size, take the share
  of messages that would arrive within one LiDAR frame (100 ms), computed from
  the logged trace. Next to it, show what the synthetic channels of `where2comm`
  and `v2xvit` assume.
- *Message.* Expect the real link to be bimodal: about 3 ms when idle and
  60–109 ms under contention, with outages tied to position. The synthetic
  channel is one fixed or independently sampled delay. Raw clouds cross the
  one-frame boundary under contention, and boxes do not.
- *One figure.* Delay CDFs per mode and size, with the frame boundary and the
  synthetic assumption drawn in.
- *Can you do it now?* Yes, for the logged side: the ping and iperf tables
  exist, and arrival per size is the replay formula applied to them. The
  synthetic side is the parameter values from those papers. A detection-level
  version is left to Paper B.

### IX. Lessons Learned and Limitations (about 0.4 pages)
Testbed papers are valued for these (see Robotarium and the Kimera-Multi
lessons paper).
- CSI constraints shape the network: single stream, 802.11ax off, aggregation
  caps the CSI rate.
- NTP asymmetry on the sniffer's management link, and how probe packets measure
  it.
- Cold-start ICP falls into wrong basins at building scale, so boards go in
  feature-poor regions.
- Plane-fit refinement versus nearest-neighbour refinement.
- **Limits:**
  - one band and one protocol;
  - one access point per site;
  - one CSI receiver;
  - the self-built map rests on the control distances for absolute scale.

### X. Conclusion (about 0.2 pages)
State the three guarantees with numbers, the replay validation, and the next
steps: more agents, other radios. Point forward to the dataset (Paper B)
without claiming it.

---

## Boundary with Paper B
| Paper A (testbed) | Paper B (dataset) |
|---|---|
| Clock, extrinsics, ground-truth procedure and their validation | One-paragraph summary that cites A |
| Link instrumentation and replay, validated (E2, E3) | Logged link used as a condition in Q4 |
| One method, one site, one demonstration figure (E5) | Multi-method transfer, heterogeneity, occlusion vs range, retained gain, attribution, geometry share |
| No sequences, labels or leaderboard | Release, annotations, splits, benchmarks |

## What exists and what is new
- **Already in the draft or pipeline:** the material for sections III–VII, and
  the numbers for E1 (clock, second pass, RTT) and the OPV2V thresholds.
- **To fill:**
  - sniffer bias;
  - control distances;
  - held-out board residuals;
  - calibration residuals;
  - bill of materials and setup time;
  - E2 (from existing logs);
  - E4 (from existing sessions if the same route was repeated).
- **New collection:** E3 run B (payload node), plus E4's repeat day if you have
  no same-route repeat yet.
