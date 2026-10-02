# HW1 Individual Term Project Proposal

**Emerging Technologies and Applied Innovation, Fall 2026**

## 1. Project Title

Self-Calibrating UWB-to-Video Fusion: Persistent Player Identity and Jump Clips from Any Camera

## 2. Primary Project Platform

LocusConnect - Spatial Intelligence

## 3. Problem and Target User

Coaches and analysts at university and club teams review video to judge what each player did, but video trackers lose identity at crossings: the best published trackers still leave about 2,600 identity switches on SportsMOT, and SoccerNet association accuracy stays near 60 percent even with perfect detections. The LocusConnect UWB stream keeps every player's identity and 3D position through those occlusions, yet commercial tools only align such telemetry to video in time, by a manual sync point. Projecting identity onto players in the frame needs a surveyed, fixed camera, so it never happens at training sessions filmed from a phone.

## 4. Proposed Project Idea

Input: video from any camera, and the live 3D stream from the Locus RTLS engine (tag ID, x, y, z, timestamp). Processing: (1) markerless self-calibration during the first minute of free play, jointly estimating camera pose, focal length, and the camera-to-UWB clock offset by matching detected hip keypoints to the 3D tag trajectories, with RANSAC over the unknown tag-to-person correspondences; (2) per frame, each tag's 3D position is projected into the image and matched to a detection by gated assignment, so identity survives crossings and occlusions; (3) the Z-axis stream flags jumps above a threshold set from measured vertical noise. Output: video with persistent name labels on every tagged player, plus an auto-cut clip of each jump with height overlaid.

## 5. Technology and Data Required

OpenCV for PnP and joint refinement of pose, focal length, and time offset; YOLO-pose and ByteTrack for keypoints and tracks; Python with Hungarian assignment and a Kalman smoother. The work is 3D geometry, estimation, and integration. Data: synchronized phone video and Locus RTLS 3D logs of three to six tagged people in a gym, with staged crossings and jumps, plus manual identity and jump labels. Not yet existing; recorded in two sessions on the course's LocusConnect installation through its MQTT or WebSocket API, confirmed in week one. A ChArUco marker calibration serves as the reference for the self-calibration metrics.

## 6. Final Prototype / Live Demo

At the end of the semester, I will be able to demonstrate that a phone camera on a tripod at an arbitrary spot calibrates itself within one minute of four tagged people moving freely, with reprojection error shown converging on screen. After that, each person carries a persistent name label through repeated crossings; toggling fusion off makes labels swap, toggling it on restores them within a frame. When someone jumps, a clip appears within seconds showing the jump with its height overlaid. If live UWB access fails, the same pipeline runs on a recorded synchronized session.

## 7. How Will You Know It Works?

Identity: ID switches per minute, ByteTrack alone versus fused, on a labelled session; target 90 percent fewer, and zero on staged crossings; per-frame label accuracy above 95 percent. Calibration: self-calibrated pose versus the marker reference, target rotation error below 1 degree, translation below 20 cm, reprojection below 15 pixels, convergence within 60 seconds, time offset within 20 ms. Jumps: measured UWB vertical noise (published LPS validations report 17 to 21 cm) sets the threshold; precision and recall above 0.9 for jumps above it, against video labels. Latency below 100 ms per frame.

## 8. Why Should This Project Be Selected?

Identity is the measured bottleneck of sports video tracking, and radio identity has been shown to remove it, yet only one published sports system fuses UWB with video, using four fixed cameras. Existing calibration work solves pose and time offset from a single tracked target; none jointly solves pose, focal length, and clock offset from several tagged players with unknown correspondences. That piece combines established methods, so it fits a semester.

## 9. Your Expected Contribution

[What can you contribute now, and what capability do you want to develop? Max. 50 words.]
