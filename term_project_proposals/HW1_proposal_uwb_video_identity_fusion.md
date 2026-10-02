# HW1 Individual Term Project Proposal

**Emerging Technologies and Applied Innovation, Fall 2026**

## 1. Project Title

Self-Calibrating Multi-Camera UWB-to-Video Fusion: Persistent Player Identity from Any Phone Cameras

## 2. Primary Project Platform

LocusConnect - Spatial Intelligence

## 3. Problem and Target User

Coaches and analysts at university and club teams review video to judge what each player did, but video trackers lose identity at crossings: ByteTrack, our baseline, leaves 3,089 identity switches on the SportsMOT test set. Analysts fix these by hand, hours per match, so training sessions go unanalysed. The LocusConnect UWB stream keeps every player's identity and 3D position through those occlusions, yet commercial tools only align such telemetry to video in time, by a manual sync point. Projecting identity onto players in the frame needs a surveyed, fixed camera, so it never happens at sessions filmed from phones.

## 4. Proposed Project Idea

Input: video from two or more uncalibrated phones and the live Locus RTLS 3D stream (tag ID, x, y, z, timestamp). Processing: (1) each camera self-calibrates to the UWB frame in the first minute of free play, jointly estimating pose, focal length, and clock offset from detected hip keypoints and 3D tag trajectories with RANSAC over unknown tag-to-person correspondences; the shared UWB frame links cameras, so no overlap or mutual calibration is needed; (2) per frame, each tag's 3D position is projected into every view and matched to a detection by gated assignment, so identity stays consistent across cameras; (3) as a stretch, Z-axis jump flags. Output: every view with persistent name labels, plus per-player clips following the least-occluded camera.

## 5. Technology and Data Required

OpenCV for PnP and joint refinement of pose, focal length, and time offset per camera; YOLO-pose and ByteTrack for keypoints and tracks; Python with Hungarian assignment and a Kalman smoother. The work is 3D geometry, estimation, and integration. Data: two-phone video and Locus RTLS 3D logs of three to six tagged people in a gym, with staged crossings, plus manual identity labels. Not yet existing; recorded in two sessions on the course's LocusConnect installation through its MQTT or WebSocket API, confirmed in week one. A ChArUco marker calibration of each camera serves as the reference for the self-calibration metrics.

## 6. Final Prototype / Live Demo

At the end of the semester, I will be able to demonstrate that two phones on tripods at arbitrary spots each calibrate themselves within one minute of four tagged people moving freely, with reprojection error converging on screen. Then each person carries the same name label in both views through repeated crossings; when a player is hidden in one view, the label persists in the other and the per-player clip switches to that camera. Toggling fusion off makes labels swap, toggling it on restores them within a frame. If live UWB access fails, the pipeline runs on a recorded session.

## 7. How Will You Know It Works?

Identity: ID switches per minute, ByteTrack versus fused, per view, on a labelled session; target 90 percent fewer, zero on staged crossings, 100 percent cross-camera agreement; per-frame label accuracy above 95 percent. Calibration, per camera, against the marker reference: rotation error below 1 degree, translation below 20 cm, reprojection below 15 pixels, convergence within 60 seconds, clock offset within 20 ms, estimated per camera so phones need no shared clock. Cross-check: hip keypoints triangulated from two self-calibrated cameras within 25 cm of UWB positions. Jumps (stretch): precision and recall above 0.9 against video labels. Latency below 100 ms per frame.

## 8. Why Should This Project Be Selected?

Identity is the measured bottleneck of sports video tracking, and no published method calibrates arbitrary cameras to UWB from several tagged players with unknown correspondences; because the UWB frame links the cameras, the method combines established parts and fits a semester. Practical value for LocusConnect is direct: coaches already film training on phones, and this puts tag identity into that footage from any number of them. Stable cross-camera identity then unlocks per-player clips and statistics.

## 9. Your Expected Contribution

I work in Python with OpenCV and ROS on multi-sensor robotics data, including camera calibration and synchronization. I will implement the RANSAC calibration solver and per-view association, run both recording sessions, and build the evaluation against the marker reference. I want to develop real-time multi-object tracking and pose-based video analytics.
