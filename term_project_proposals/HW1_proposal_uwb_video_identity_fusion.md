# HW1 Individual Term Project Proposal

**Emerging Technologies and Applied Innovation, Fall 2026**

## 1. Project Title

UWB-Anchored Player Identity for Video Tracking: Keeping Names on Players Through Crossings and Occlusions

## 2. Primary Project Platform

LocusConnect - Spatial Intelligence

## 3. Problem and Target User

Coaches and match analysts at university and amateur clubs review video to see what each player did. Automatic video trackers follow players well until two of them cross or one is hidden behind another; then the tracker often hands the wrong identity to the wrong person. A full match produces dozens of these identity switches, and an analyst repairs them by hand, hours per match, so it is rarely done for training sessions at all. UWB tags already know exactly who each player is, but that identity never reaches the video where coaches actually look.

## 4. Proposed Project Idea

Input: a fixed camera feed of the pitch and the live LocusConnect UWB stream (tag ID, x, y, timestamp). Processing: (1) a one-time calibration that recovers the camera pose in the UWB frame from a single tag walked through the field of view, plus the clock offset between the two systems; (2) an off-the-shelf person detector and tracker on the video; (3) every frame, each tag position is projected into the image and matched to a detection with a gated assignment that tolerates UWB noise and short detection gaps. Output: the video with a persistent name label on every tagged player, a per-player track in pixel and pitch coordinates, and a log of the identity switches avoided.

## 5. Technology and Data Required

OpenCV for camera calibration and PnP pose estimation; YOLO and ByteTrack for detection and tracking; Python with a Kalman smoother and Hungarian assignment for association. These fit because the core work is geometry and integration, not model training. Data: synchronized video and UWB logs of three to six tagged people in a gym or field, including staged crossings, plus manual identity labels for evaluation. This data does not exist yet and will be recorded in two sessions with the course's LocusConnect system and one camera. Access to raw tag positions from LocusConnect will be confirmed in the first weeks.

## 6. Final Prototype / Live Demo

At the end of the semester, I will be able to demonstrate that, with four tagged people repeatedly crossing paths in front of a live camera, the prototype shows each person with a persistent name label that survives crossings and brief occlusions. A toggle switches the UWB fusion off: labels begin swapping after crossings, and switching it back on restores the correct names within a frame. A side panel shows the current assignment and a live count of identity switches in both modes. If live UWB access is unavailable, the same pipeline runs on a recorded synchronized session.

## 7. How Will You Know It Works?

Primary metric: identity switches per minute on a labelled test session, video-only tracker versus fused tracker. Target: at least 90 percent fewer switches, and zero on staged two-person crossings. Secondary metrics: association accuracy, the fraction of frames in which each label sits on the correct person, above 95 percent against manual labels; calibration reprojection error below 15 pixels at 1080p; end-to-end latency below 100 ms per frame so the demo runs live. The baseline is the unmodified ByteTrack output on the same footage.

## 8. Why Should This Project Be Selected?

It solves a concrete problem analysts pay for in hours today, and it brings LocusConnect data into the video where coaches look. The scope is feasible: one camera, a few tags, off-the-shelf detectors. The hard parts, calibration, synchronization, and robust association, are real engineering with measurable results. The demo is immediate and easy to judge. Once identity is stable, per-player clips and statistics follow directly.

## 9. Your Expected Contribution

I have built radar-camera extrinsic calibration with ChArUco targets, SLAM evaluation pipelines, and robust state estimators in Python and ROS. I can lead calibration, synchronization, and association. I want to learn real-time multi-object video tracking and re-identification.
