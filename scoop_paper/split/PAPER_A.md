# Paper A (RA-L): the testbed, no dataset release

**Working title.** *An Indoor Multi-Robot Testbed for Collaborative Perception
over Measured Wireless Links*. Give the testbed its own name so that it is not
confused with the SCooP dataset (Paper B).

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
   | Replay fidelity | live vs replayed accuracy gap | experiment E3 |

2. **A tolerance-margin figure (the key figure).** On one log axis, plot the
   testbed's error next to the error at which collaborative perception breaks:
   - pose error: about 1 cm, against the 0.2 m at which late fusion falls below
     ego-only in the OPV2V study;
   - time: 70 µs, against the 100 ms at which every fusion method falls below
     ego-only.

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
- **Highlight sentence.** Under contention, the measured link sits at the 100 ms
  point where fusion stops paying, and only a testbed that records it can show
  that.

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

### VII. Channel-in-the-Loop Replay (about 0.5 pages)
**Question it answers:** can any collaborative method be run against the link
the agents actually had?
- **Model.** Arrival = RTT_ab(t)/2 + s/G_ab(t), and the message is dropped when
  the link is down. Ideal, synthetic and logged channels go through one module,
  so the channel is the only variable.
- **Interface.** Any method that emits messages of size s at time t plugs in.
  Give the API in three lines.
- **Figure:** replay diagram.

### VIII. Experiments: Validating the Testbed (about 1.5 pages)
Every experiment asks a question about the testbed, not about collaborative
perception.

- **E1. Is the testbed precise enough for what it is meant to measure?**
  Tolerance-margin figure: the testbed's clock, pose and extrinsic errors set
  against the latency and pose thresholds at which fusion falls below ego-only.
  These thresholds come from the OPV2V failure study, cited or as an appendix.
  Uses existing numbers only.
- **E2. Do the link logs agree with the physical layout?** Map-derived Fresnel
  clearance against measured RSSI and goodput: Spearman correlation per site,
  and the RSSI drop between clear and blocked links. The geometry never saw the
  signal, so agreement validates both. Uses existing data.
- **E3. Does replay reproduce live execution?**
  - Run late fusion live over the real link on a few sequences, with messages
    actually sent.
  - Then replay the logged trace on the same frames.
  - Report the accuracy gap and the arrival-time error per message.
  - **This is the only new collection the paper needs**, and it is the claim
    that only a testbed can make.
- **E4. Is the testbed repeatable?** Repeat a pass on the same route and in the
  same load mode on another day. Compare:
  - the pose reference (second-pass agreement, already measured);
  - the link-statistics distributions;
  - and show that they differ between load modes more than between days.
- **E5. Demonstration: what the testbed shows that simulation does not.** One
  method on one site under the ideal, synthetic and logged channels. Show that
  the logged link's RTT distribution under contention straddles the 100 ms
  cliff, while the synthetic channel's fixed delay does not. Keep it to one
  figure. The multi-method study, attribution and geometry regression stay in
  Paper B.

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
- **New collection:** E3 live runs only.
