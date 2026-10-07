# ED305 collaborative perception: plan

Goal: the 16 room cameras and the mobile robots perceive the room together,
so that anything one agent cannot see (out of view or blocked) is still
detected, in one room frame, in time to use.

Each phase ends in something checkable. Later phases depend on earlier ones:
everything is in the room frame (phase 1) on one clock (phase 2).

```
0 task ─► 1 geometry ─► 2 time ─► 3 robots in room ─► 4 data + GT ─► 5 baselines ─► 6 robustness
              └─ coverage tool (this folder)
```

## 0. Fix the task (one page)

- What is perceived: people and robots (add objects only if needed).
- Output: positions on the floor plane (BEV) with track IDs, or 3D boxes, in the room frame.
- Target volume and k: floor to 2 m, k = 2 (two views to triangulate).
- Metrics: MODA / MODP (BEV detection), AP for boxes, IDF1 for tracks,
  and **blind-spot recall**: objects the robot cannot see that the fused system finds.
- Latency budget for the shared result (e.g. under 100 ms end to end).

Done when: agreed with the lab, written down.

## 1. Geometry: digital twin v1

1. **Room frame.** Origin at the red dot, axes as in the README. Mark it on
   the floor (a fiducial at the dot) so every later step can find it.
2. **Room model.** Scan ED305 with a robot's LiDAR (GLIM map), take the room
   size and furniture boxes from the map into `configs/room.yaml`.
3. **Camera intrinsics.** Per camera, from a checkerboard / ChArUco. Check
   the existing March 2026 calibration first; redo only the cameras that fail.
4. **Camera extrinsics in the room frame.** Boards at known positions on the
   floor (positions read from the LiDAR map or the red-dot fiducial),
   `solvePnP` per camera, then one joint refinement over every camera that
   shares a board view. Check: triangulate a point no fit used; error at the
   room centre within a few cm. Re-check cams 4 and 6 and any that moved.
5. **Coverage.** `coverage_map.py` on the real poses and furniture.
6. **Placement.** Step by step in [WORKFLOW.md](WORKFLOW.md): flag cameras
   with assumed ranges, measure only those from video, optimize, re-aim,
   re-calibrate the moved ones, verify coverage.

Done when: `configs/cameras.yaml` and `configs/room.yaml` from measurements,
and a coverage report with no blind spots at k = 2 (or a list of accepted ones).

## 2. Time: one clock

- PTP between the two camera hosts (and the cameras' own timestamps, if the
  a2A1920-51gc supports PTP on GigE: check the Basler spec), chrony/PTP on the robots.
- Measure, don't assume: something every camera sees changing at once
  (an LED, a screen clock) gives the offset per camera.
- Record all cameras (`grab_sync.py` per host) and robots with timestamps on that clock.

Done when: measured offset between any two cameras below ~5 ms
(a person at 1.5 m/s moves under 1 cm).

## 3. Robots in the room frame

- Robot sensor-to-base calibration (existing).
- Robot pose in the room frame over time: LiDAR localization against the phase-1
  map, checked by fiducials on the robots that the cameras see.

Done when: robot pose in the room frame at every timestamp, checked against the
cameras to within a few cm.

## 4. Data and ground truth

- Scenarios built around occlusion: a person hidden from the robot by furniture
  or another person but seen by the cameras; robots crossing; crowded centre;
  the coverage-thin corners.
- Ground truth from all 16 cameras offline (multi-view triangulation, manual fix-up)
  as floor positions / boxes with IDs.
- Export in a format the collaborative-perception models already read
  (OPV2V-style per-agent frames with poses) plus a WILDTRACK-style multi-view set
  for the cameras alone.

Done when: a labelled set with train / test splits and the phase-0 metrics running on it.

## 5. Baselines

1. Cameras only: multi-view detection on the floor plane (MVDet / MVDeTr / EarlyBird family).
2. Robot only: its own sensors.
3. Collaborative, increasing in what is shared:
   late fusion (detections in the room frame) → intermediate fusion (BEV features).

Done when: one table, each method on the phase-0 metrics, blind-spot recall included.

## 6. Robustness (where the research questions are)

- Pose error and time offset injected into the shared data: how fast does fusion degrade?
- Communication: bandwidth, latency, loss between robots and the room server.
- Camera dropout: which cameras matter (leave-one-out from the coverage tool).

## Later (digital twin v2)

Live state in the room model (robot and people poses streaming in), and a simulated
copy of ED305 for synthetic data, once v1 geometry is solid.
