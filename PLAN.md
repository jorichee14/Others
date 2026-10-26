# Plan: SCooP-iT, SCooP, and the Gym follow-up

> **26 Oct:** after the pre-review, the Gym leaves the SCooP-iT paper and
> becomes the follow-up paper. SCooP-iT is the instrument plus a validated
> replay (`scoop_testbed_iotj/OUTLINE.md`). The work streams below still
> hold; streams D, E and G and milestones M4-M6 now feed the follow-up.

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

## The loop

```
 real recordings ──build──▶ Gym (Isaac twin) ──train──▶ policy
        ▲                     │ checked by E3b              │
        │                     ▼                             ▼
        │              replay test bench ◀────test──── on the recordings
        │                                                   │
        └──── new recordings, refit ◀── testbed ◀──deploy───┘   (E3c)
```

- **Train** in the Gym: a closed-loop twin built from the recordings.
- **Test** on the recordings: the replay test bench, with every condition logged.
- **Validate** on the testbed: deploy and compare with the Gym's prediction. The new recordings refit the Gym.

## Deliverables

| | SCooP-iT (with SCooP-iT Gym) | SCooP | Extension |
|---|---|---|---|
| What | Testbed and environment paper | Dataset paper | Agent benchmark on the Gym |
| Folder | `scoop_testbed_iotj/`, `scoop_gym/`, `isaac_coop_sim/` | `scoop_dataset_paper/` | `scoop_gym/` |
| Venue | IEEE IoT-J | IEEE T-RO | decide after SCooP-iT |
| Claim | A real indoor team captured on one clock and frame, sensing and communication alike, offered as test bench, Gym and testbed, each checked against reality | What changes when collaborative perception meets a real indoor team (Q1-Q4) | Many tasks and agents (rule, RL, VLM) on the Gym, failures attributed down to the link |
| State, 24 Oct | 18 pages, split into section files (`WRITING_GUIDE.md`); results open | Draft; collection, labels, experiments open | Not started |

## Who owns what

- **SCooP-iT:**
  - the platform, clock, ground truth and link;
  - the replay test bench with its conditions (E3a);
  - the Gym: world, interface, conditions, tasks, train-test-deploy;
  - validation of the Gym's world (E3b) and one deployed policy (E3c, `OcclusionHandoff`);
  - the `CommSchedule` demonstration.
- **SCooP:**
  - data at every site, labels, collection design, geometry, release, devkit;
  - the T1 and T2 benchmarks and Q1-Q4;
  - "does augmentation help" as a finding.
- **Extension:** the full set of Gym tasks, the agent benchmark, and large-scale attribution.
- **Shared, described once:** the channel model and conditions (SCooP-iT), the twin construction (SCooP-iT), the labels (SCooP).

## Work streams

| Stream | What | Feeds |
|---|---|---|
| **A. Compute on existing passes** | confirm pass sources; E1 terms and headroom figure; E2 CI; E4 mode distance; E5 best-fit; calibration rows; error budget; `CommSchedule` on coop_2 | SCooP-iT IV, VIII, IX |
| **B. Field sessions** | B1: replay validation session (A1/B1 survey, A2/B2 cooperation, probe-off; needs the payload node and `twin_run.py`). B2: deployment session for E3c | E3a, E4 day half, sniffer bias; E3c |
| **C. Labels** | coop_2 first (static in the map frame, moving by hand), then every sequence | E3b dAP, `CoopDetect`, all of SCooP |
| **D. Gym world (Isaac)** | D1 scene: USD from the map (exists), floor and free space. D2 robots: Scout Mini from URDF as a differential drive, cart and walkers kinematic, masts static. D3 sensors: RTX LiDAR (64/128 beams) and cameras from `isaac_coop_sim`; semantic classes on every prim. D4 objects: placement from free space at label counts, real assets for cameras, material and lighting randomization | E3b, all twin tasks |
| **E. Gym conditions (Python, `scoop_gym`)** | E1 sensor dropout and noise fitted per sensor from bag timestamps and scans. E2 pose error fitted from estimate vs. reference. E3 link indexed by place: survey grids, contention mode, clearance; Sionna fill-in; out-of-run check. E4 attribution causes beyond the link | E3b; every task |
| **F. Replay test bench** | `ReplayEngine.observe()`: recorded frames through the devkit; one detector per agent; `CoopDetect`, `HelperSelect` | E3b dAP; testing Gym-trained policies |
| **G. Gym tasks and training** | Isaac Lab multi-agent tasks calling `scoop_gym` each step. `OcclusionHandoff` first: scripted baseline, then RL; then `ActiveCoopDetect` | IX demonstrations, E3c |
| **H. Writing** | Section by section (`WRITING_GUIDE.md`) | SCooP-iT |
| **S. SCooP dataset track** | remaining sessions, labels at scale, release and devkit, Q1-Q4 training, failure cases | SCooP |

## What blocks what

```
A (compute) ────────────────────────────────▶ E1, E2, E4 mode, E5, IX CommSchedule
B1 (session) ───────────────────────────────▶ E3a, E4 day, sniffer bias
C (labels coop_2) ──┬──▶ F (replay bench) ──▶ E3b dAP, testing policies
                    └──▶ SCooP Q1-Q4
D (Isaac world) ──┬──▶ E3b scans
E (conditions)  ──┴──▶ G (OcclusionHandoff: train) ──▶ test on F ──▶ B2 deploy ──▶ E3c
SCooP-iT on arXiv ──────────────────────────▶ SCooP submission
```

The critical path to a complete SCooP-iT: C, then D and E, then G, then B2. Labels and the Isaac world can start in parallel today.

## Milestones (in order; add dates once deadlines are known)

| M | Done when | Paper sections it completes |
|---|---|---|
| M1 | Stream A finished; cross-pass figure decided | IV, V, VIII (E1, E2, E4 mode, E5), IX CommSchedule |
| M2 | B1 recorded and compared | VIII E3a, E4; abstract and contribution 3 |
| M3 | coop_2 labelled; replay bench reads frames | VIII E3b dAP (with M4) |
| M4 | Isaac world of coop_2 runs; conditions and place-indexed link fitted and checked | VIII E3b; IX world table |
| M5 | `OcclusionHandoff` trained in the twin and tested on the recordings | IX demonstrations |
| M6 | B2: deployed on the testbed, compared with the Gym | VIII E3c; abstract; conclusion |
| M7 | Condensed to length, figures done, Table II verified, no `\todo` | submission; arXiv; release pipeline and `scoop_gym` |

## Decision points

- **After M5, if the deployment can't happen in time:** submit with E3c described and run but its result marked as the next step, and the Gym validated by E3b alone. Weaker, but complete.
- **If E3b's appearance gap is large for cameras:** keep camera tasks out of the paper and lead with LiDAR tasks.
- **If the place-indexed link fails its out-of-run check:** closed-loop tasks use the fitted link (statistics without place), and the paper says so.

## Open decisions

- Deadlines for each submission.
- The cross-pass figure (M1).
- The venue of the extension.
