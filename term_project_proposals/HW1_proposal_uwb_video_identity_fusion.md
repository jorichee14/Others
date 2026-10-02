# HW1 Individual Term Project Proposal

**Emerging Technologies and Applied Innovation, Fall 2026**

## 1. Project Title

Self-Calibrating UWB-to-Video Fusion: Persistent Player Identity and Z-Axis Jump Clips from Any Camera

## 2. Primary Project Platform

LocusConnect - Spatial Intelligence

## 3. Problem and Target User

Coaches and analysts at university and club teams watch video to judge what each player did, but video trackers lose identity whenever players cross or cluster, exactly the moments that decide games. The LocusConnect UWB stream is immune to that occlusion and already carries identity, 3D position, and heart rate, yet none of it appears in the video coaches actually watch. Linking the two today needs a surveyed, fixed camera and a calibration target, so it is never done at training sessions, where the camera is a phone on a tripod placed wherever there is space.

## 4. Proposed Project Idea

Input: video from any camera, and the live 3D stream from the Locus RTLS engine (tag ID, x, y, z, heart rate, timestamp). Processing: (1) markerless self-calibration: during the first minute of free play, the system jointly estimates camera pose, focal length, and the clock offset between camera and UWB by matching detected player hip keypoints to the 3D tag trajectories, using RANSAC over the unknown tag-to-track correspondences; (2) per frame, each tag's 3D position is projected into the image and matched to a detection by gated assignment, so identity survives crossings and occlusions; (3) the Z-axis stream detects jumps. Output: video with persistent name labels, plus an auto-cut clip of every jump with height and heart rate overlaid.

## 5. Technology and Data Required

OpenCV for PnP and joint refinement of pose and time offset; YOLO-pose and ByteTrack for keypoints and tracks; Python with Hungarian assignment and a Kalman smoother. These fit because the work is 3D geometry, estimation, and integration. Data: synchronized phone video and Locus RTLS 3D logs of three to six tagged people in a gym, with staged crossings and jumps, plus manual identity and jump labels. Not yet existing; recorded in two sessions on the course's LocusConnect installation. Required access: per-tag 3D positions with timestamps from the edge server, confirmed in week one. A marker-based calibration serves as reference.

## 6. Final Prototype / Live Demo

At the end of the semester, I will be able to demonstrate that a phone camera on a tripod at an arbitrary spot calibrates itself within one minute of four tagged people moving freely, with reprojection error shown converging on screen. After that, each person carries a persistent name label through repeated crossings; toggling fusion off makes labels swap, toggling it on restores them within a frame. When someone jumps, a clip appears within seconds showing the jump with its height and heart rate overlaid. If live UWB access fails, the same pipeline runs on a recorded synchronized session.

## 7. How Will You Know It Works?

Identity: ID switches per minute, ByteTrack alone versus fused, on a labelled test session; target 90 percent fewer, and zero on staged crossings; per-frame label accuracy above 95 percent. Calibration: self-calibrated camera pose versus marker-based reference, target rotation error below 1 degree, translation below 20 cm, reprojection below 15 pixels, convergence within 60 seconds of play, time offset within 20 ms. Events: jump-clip precision and recall above 0.9 against manual labels. Latency below 100 ms per frame so the demo runs live.

## 8. Why Should This Project Be Selected?

LocusConnect's own comparison says cameras fail under occlusion and UWB does not; this project is the bridge that puts the unbroken UWB stream into the video coaches watch. Self-calibration removes the surveyed camera that keeps fusion out of training sessions. Scope is one phone, a few tags, off-the-shelf detectors; the engineering is 3D geometry and robust association with hard numbers. The jump clip shows the Z-axis advantage in a way anyone understands.

## 9. Your Expected Contribution

I have built radar-camera extrinsic calibration with ChArUco targets, SLAM evaluation pipelines, and robust state estimators in Python and ROS. I can lead calibration, synchronization, and association. I want to learn real-time multi-object tracking and pose-based video analytics.
