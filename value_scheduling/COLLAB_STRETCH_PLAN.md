# Stretch: coarse-to-fine, what to send at each stage

Principle: the finer the motion, the tighter the deadline and the
smaller the region that matters, so the loop moves onto the robot and
the server gets smaller, rarer messages; the coarser the motion, the
looser the deadline, so the server can take big frames and heavy
models. Bytes scale with the precision the current motion needs, not
with the camera's frame rate.

Stretch hardware assumed: head RealSense D435i (RGB-D, wide), wrist
RealSense D405 (RGB-D, close range), base lidar, telescoping arm,
lift, wrist yaw/pitch/roll, gripper; on-board compute enough for
OpenCV, point-cloud ops and a small detector, not for a large vision-
language model. The server runs the heavy model (segmentation, VLM,
policy). Confirm against their configuration.

## The stages

| Stage | Motion and tolerance | Deadline | What the server needs | Region | Sensor | Form sent | Trigger | Loop lives |
|---|---|---|---|---|---|---|---|---|
| 1 Navigate to the room | base at 0.3–0.5 m/s, 10–20 cm | 0.5–1 s | where the target object is (room, table) | whole scene, low res | head RGB-D + lidar | rung 3: 320×240 JPEG q60 + 8-bit depth, every 1 s; lidar scan on change | on change (robot moved > 0.5 m or scene diff) | server: semantic goal ("the mug is on the left table"); robot: Nav2 on lidar, local |
| 2 Approach the table | base and head to bring target in view, 5 cm | 300–500 ms | the target's 3-D position and a first grasp proposal | the table region, from the server's last answer projected into the image | head RGB-D | rung 1–2: ROI crop at full res, RGB-D, on change | on change inside ROI; one full-res crop when the target first appears | server: detect + propose grasp; robot: base servo to a standoff pose, local |
| 3 Pre-grasp alignment | lift, arm extension, wrist yaw to the standoff, 1–2 cm | 100–300 ms | confirmation that the proposal still holds; refined target pose | a box around the target and gripper, ~20 cm | wrist RGB-D (D405) | rung 2–3: changed pixels inside the box, depth as 8-bit windowed to 10–40 cm | on change, or every 300 ms as a heartbeat | robot: visual servo on the wrist depth to center the target (template or depth centroid); server: one refinement per second at most |
| 4 Grasp | final extension and closing, 2–5 mm | 50 ms or local | nothing in the loop | gripper fingers and target | wrist depth + gripper effort | rung 4: a one-line state at events ("contact", "closed", "slipped"); no images | on event | entirely on the robot: depth-centroid servo, effort-based closing; the server has already said what and where |
| 5 Lift, retreat, verify | lift and retract, 1 cm | 300 ms | did it succeed | the gripper | wrist RGB | one crop after lift, rung 2 | once | server: verify (or the robot checks effort and depth); then back to stage 1 |

## Why this is the optimum shape

- Stages 1–2 need the heavy model and tolerate hundreds of
  milliseconds, so they offload, and the region shrinks from the scene
  to the table as the server's own answer tells the robot where to
  look.
- Stages 3–4 need millimetres at tens of milliseconds, which no Wi-Fi
  loop with a server in it meets reliably; so the fine loop is local
  (wrist depth servo, gripper effort), and the server's role becomes
  one refinement, then silence. The network carries events, not video.
- Stage transitions are the only moments a full-res crop is worth
  sending: when the target first appears (2) and after the lift (5).

Traffic over a pick: a few low-res frames per second for ~20 s of
navigation (~100 KB total), two to five full-res crops at approach
(~300 KB), a trickle of changed pixels during alignment (~50 KB),
events during grasp (~1 KB), one crop to verify. Against streaming
RGB-D at 30 fps for the same minute (~1 GB), the waste number is above
99 %, and the fine stages no longer depend on the link at all.

## Implementation, no training

- **ROI rule per stage** from kinematics and the server's last answer:
  stage 1 whole frame; stage 2 the server's target box projected with
  the head camera's extrinsics; stages 3–4 a fixed box around the
  gripper in the wrist camera; stage 5 the gripper.
- **Change mask** against the server's last received copy (RGB abs-diff
  blurred and thresholded; depth diff > 2 cm; OR), dilated; send the
  mask's boxes and their content; the server patches its copy and runs
  its model on the patched frame unchanged.
- **Rung selection** per frame: the highest rung whose size over the
  measured goodput, plus the server's reported queue delay, plus
  inference, fits the stage's deadline. A lookup, no optimizer.
- **Local fine loop**: depth-centroid or template servo on the D405 in
  stage 3 (OpenCV, ~5 ms); effort-threshold closing in stage 4 (the
  gripper's current sensing). Both are standard Stretch demos and need
  no model.
- **Stage machine** on the robot; stage changes are published to the
  server so it knows what to expect and can drop queued frames from a
  stage that has ended.
- **Stamps** on every message on chrony time; the logger as in package
  A.

## What is measured

Per pick: success; time per stage; bytes per stage; frames the server
processed that arrived after their stage ended (wasted); and for stage
3, the alignment error at the moment of commit against the final grasp
outcome (the AoII-at-decision reading of their task). Baseline: their
current stream at full rate through all stages.

## Where it joins the program

This is the representation ladder and the decision-time score applied
to one robot's own loop rather than a team's. The twin is the server's
patched copy of the workspace; the lifetime is the change mask; the
deadline is the stage's; the "receiver's own future view" is the wrist
camera taking over from the head camera. It is scenario 0 with one
agent, and the two-robot version (a second Stretch or a mast on the
same AP) is the crossing.
