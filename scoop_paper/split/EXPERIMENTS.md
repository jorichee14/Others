# Paper A (SCooP-Bed): plan for the experiments still needed

**Goal of the section.** Show that SCooP-Bed is a trustworthy instrument, and
that it is needed:
- E1 Headroom: precise;
- E2 Map Meets Radio: consistent;
- E3 Twin Run: faithful;
- E4 Same Knob, Same Link: repeatable;
- E5 Sim Gap: necessary.

**Three phases.** Desk work on data you already have, then one recording
session, then assembly. Each thing to do has a tool, an output, and a "done
when" condition. Pass criteria are written here, before anything runs, so that
results are read as tests.

| | Experiment | Data | Tool | Phase |
|---|---|---|---|---|
| E2 | Map Meets Radio | coop2_0828; 24 Sep survey_1, coop_2 | `link_clearance.py` (ready) | 1 |
| E1 | Headroom | existing passes | existing reports + one figure script | 1 |
| E5 | Sim Gap | existing ping/iperf tables | `sim_gap.py` (ready) | 1 |
| E4 | Same Knob, Same Link, mode half | 24 Sep survey_1 vs coop_2 | `link_repeat.py` (to write) | 1 |
| E3 | Twin Run | new session | payload node + `twin_run.py` (to write) | 2 |
| E4 | Same Knob, Same Link, day half | new session vs 24 Sep | `link_repeat.py` | 2 |

## Progress

No schedule: each item gets the date it ended once it ends.

| Item | Status | Result so far | Ended |
|---|---|---|---|
| E1.1 clock (coop_2) | done | measured <= 0.99 ms, bound <= 12 ms | |
| E1.3 pose vs boards, mobile_2 (coop_2) | done | 3.1 cm median; sigma 5.3 cm (284 cm without map fixes) | |
| E1.3 pose vs boards, mobile_1 (coop_2) | open | | |
| E1.6 p95 speeds (coop_2) | open | | |
| E1 Headroom figure | open | | |
| E2.2 robot-to-robot RSSI (coop_2) | done, passes | rho 0.60, 5.4 dB (mobile_2 link); infra_1 0.23, 4.7 dB | |
| E2.3 robot-to-AP (needs AP position) | open | | |
| E2 pooled over passes | open | | |
| E4 mode half (`link_repeat.py`) | open | | |
| E5 Sim Gap (24 Sep) | done | clouds in frame 86 % alone, 59-68 % with two robots; synthetic 100 % | |
| E5 synthetic settings checked in code | done | V2X-ViT constant 100 ms; Where2comm no latency | |
| Payload node + `twin_run.py` | open | | |
| Phase 2 session | open | | |
| E3 Twin Run | open | | |
| E4 day half | open | | |
| Phase 3 assembly | open | | |

---

## Phase 1: desk work on existing data

### 1.1 E2 Map Meets Radio
1. For each site, read the access point's position off the anchored map (the
   MIRC map already contains the AP as structure). Write it to `ap.yaml`.
2. Run:
   `python scoop_pipeline/analysis/comms/link_clearance.py --all data/processed --ap-file ap.yaml`
3. Report, per link and pooled:
   - Spearman ρ between clearance and the residual signal (after the distance
     trend);
   - the residual drop from clear to obstructed windows, in dB.
   - Do this for CSI agent-to-agent links and for agent-to-AP RSSI.
- **Pass:** pooled ρ(residual) > 0.3 and a clear-minus-obstructed drop > 5 dB
  on at least one link kind. The MIRC raw-RSSI result (ρ +0.73 / +0.67, 8 and
  11 dB) suggests it will pass. The residual version is the one to publish.
- **Done when:** `clearance_all.json` and the figure exist for every pass, and
  one paragraph plus the figure is written into Section VIII.

### 1.2 E1 Headroom
1. Gather the testbed's error terms, each with the pass it comes from.
   - **Clock:** `ntp_analysis.py` on every pass. Reconcile the draft text
     ("about 70 µs") with the table (max 42.5 µs).
   - **Sniffer bias:** check whether the probe sender logs its transmit stamp.
     - If it does, take the median of (sniffer receive − mobile_2 send).
     - If it does not, the Phase 2 payload node gives it.
   - **Pose against boards:** 10.0 mm / 0.88° (mobile_1) and 54.0 mm / 1.38°
     (mobile_2) from `slam_benchmark` on coop2_0828. Repeat on 24 Sep with
     `board_table.py`.
   - **Second mapping pass:** confirm the 1.06 cm figure and fill the table cell
     that is still empty.
   - **Extrinsics:** LiDAR–camera reprojection; radar reflector residual with
     `radar_camera_calibration`.
2. Gather the fusion breaking points.
   - Pose: 0.2 m (late fusion, OPV2V study).
   - Latency as displacement: speed × delay, with speed the 95th percentile of
     the reference tracks and size a typical indoor object (chair, 0.5–0.7 m).
3. Write a figure script: one log axis in metres, testbed error bars against the
   breaking points.
- **Pass:** every testbed term is at least 10× below the smallest breaking
  point.
- **Done when:** the figure has no placeholder bars, and each bar cites its pass.

### 1.3 E5 Sim Gap
1. Write `scoop_pipeline/analysis/comms/sim_gap.py`.
   - For each pass and message size (1 KB boxes, 100 KB features, 1.5 MB
     clouds), compute the arrival delay d = RTT(t)/2 + s/G(t). RTT comes from
     the 100 Hz ping; G is the most recent iperf goodput.
   - Count a message as lost during an outage (`associated=false`, or a run of
     lost pings).
   - Outputs: delay CDFs, the share of messages arriving within one LiDAR frame
     (100 ms), and outage durations.
2. Take the synthetic channel parameters exactly as published by Where2comm and
   V2X-ViT, and draw them on the same axes.
3. Split survey mode from cooperation mode.
- **Pass:** this is a demonstration, so there is no pass/fail. The claim to
  check is that cooperation mode pushes raw clouds past the 100 ms boundary
  while boxes stay within it, and that outages exist which the synthetic
  channels do not have. coop2_0828 has three real outages, 12% of frames.
- **Done when:** one CDF figure and one table (share within a frame, per size
  and mode, logged against synthetic) exist.

### 1.4 E4 Same Knob, Same Link: the mode half
1. Write `scoop_pipeline/analysis/comms/link_repeat.py`.
   - Input: two passes.
   - Output: the Wasserstein (or KS) distance between their RTT, goodput and
     RSSI distributions, and the distance between RSSI-against-position on a
     shared grid. Use the `comms_map` grids.
2. Run it on 24 Sep `survey_1` against `coop_2`. This is the mode distance,
   which the day-to-day distance will be compared against in Phase 2.
- **Done when:** the mode distances are tabulated.

### 1.5 Get Phase 2 ready
- Write the **payload node**: a ROS 2 publisher and subscriber on every agent.
  - It sends three message sizes at 10 Hz to every peer.
  - Each message carries (sender, sequence, size, t_send). The receiver logs
    t_recv.
  - Use the transport and QoS a collaborative method would use (DDS, best
    effort). Record that choice.
- Add `twin_run.py`. It replays run B's (t_send, size, link) through run A's
  trace and compares the prediction with B's measured t_recv and drops.
- Dry-run it at your desk: two machines, one AP, a one-minute test.

---

## Phase 2: one recording session

**Where.** The 24 Sep site and route, so that every run doubles as E4's
"another day". Reuse `mapping_A` and check the boards are in place. No new
mapping session is needed.

**Before each run:**
- the clocks have converged;
- 802.11ax is off, single stream, power save off, as on 24 Sep.

| Run | Mode | Traffic | Purpose |
|---|---|---|---|
| A1 | survey | iperf + ping (as usual) | trace to replay; E4 repeat of `survey_1` |
| B1 | survey | payload node + ping | E3 measured arrivals |
| A2 | cooperation | iperf + ping | trace; E4 repeat of `coop_2` |
| B2 | cooperation | payload node + ping | E3 measured arrivals |

- Same route and speed in the A and B runs of a pair. Each run is about 3 min,
  and each pair is back to back, so the building conditions match.
- If time allows, repeat the whole set once. That gives E3 a second sample and
  E4 a within-day baseline.
- About one hour including setup.

**After the session:** process every run with `run_pass.py`, including the comms
tables with positions.

### 2.1 E3 Twin Run
- Replay each B run through its A trace, and through the synthetic channel.
- Report per message size and mode:
  - median and 95th-percentile arrival-time error;
  - agreement on which receiver frame (100 ms) each message lands in;
  - drop agreement.
- **Pass:** frame-assignment agreement ≥ 90%, and the logged replay clearly
  closer than the synthetic channel. If it fails, report the size or mode where
  it fails. Section VII's fluid model is then the limitation, and the
  Mahimahi-style per-packet trace is the stated fix.
- **Done when:** the table and one figure (predicted against measured arrival)
  exist.

### 2.2 E4 Same Knob, Same Link: the day half
- Run `link_repeat.py` on A1 against `survey_1` (24 Sep), and A2 against
  `coop_2`.
- **Pass:** the day-to-day distance is below half the survey-to-cooperation
  distance, for RTT and goodput.
- **Done when:** a table of day distance against mode distance exists.

The sniffer bias for E1 also comes from this session, if it was not available in
1.2: the payload node's t_send on mobile_2 against the sniffer's capture stamp.

---

## Phase 3: assembly
1. Spec-sheet table: every bound filled from E1–E4, each with its pass.
2. Figures:
   - Headroom (E1);
   - clearance against residual (E2);
   - predicted against measured arrival (E3);
   - day against mode distances (E4, a table is fine);
   - delay CDFs (E5).
3. Write Section VIII in the order E1 to E5, each opening with its question and
   pass criterion and closing with the result.
4. Update the abstract and contributions with the numbers.

## Order
```
Phase 1:  E2 ─┐
          E1 ─┼─> Phase 3
          E5 ─┤
          E4 (mode) ─┐
          payload node + twin_run.py ─> Phase 2: session ─> E3, E4 (day), sniffer bias ─> Phase 3
```
Run E2, E1, E5 and the E4 mode half in parallel. The payload node gates the
session, so write it first.
