# HW1 Individual Term Project Proposal

**Emerging Technologies and Applied Innovation, Fall 2026**

## 1. Project Title

Self-Calibrating Multi-Camera UWB-to-Video Fusion on the Edge: Persistent Player Identity with Occlusion-Triggered Camera Switching

## 2. Primary Project Platform

LocusConnect - Spatial Intelligence

## 3. Problem and Target User

Coaches and analysts at university and club teams review video to judge what each player did, but video trackers lose identity at crossings: ByteTrack, our baseline, leaves 3,089 identity switches on the SportsMOT test set. When a player is hidden in one camera, analysts switch angles and re-find them by hand, hours per match, so training sessions go unanalysed. The LocusConnect UWB stream keeps every player's identity and 3D position through those occlusions, yet commercial tools only align such telemetry to video in time. Putting identity into phone footage needs a surveyed, fixed camera, so it never happens.

## 4. Proposed Project Idea

Input: two or more uncalibrated phones and the live Locus RTLS 3D stream (tag ID, x, y, z, timestamp). Processing, split across the fog: each phone runs YOLO-pose and sends only hip keypoints, not video; the LocusConnect edge server self-calibrates each camera to the UWB frame in the first minute of play (pose, focal length, clock offset; RANSAC over unknown tag-to-person pairs), projects every tag into every view, associates by gated assignment, and scores every view per player for occlusion from keypoint confidence and box overlap; when a player's view is occluded, the server switches that player's feed to the best other camera within a frame, identity carried by the tag. Output: labelled views and per-player clips across cameras.

## 5. Technology and Data Required

OpenCV for PnP and joint refinement of pose, focal length, and time offset per camera; YOLO-pose on the phones and ByteTrack on the server; Python with Hungarian assignment and a Kalman smoother. Data: two-phone video and Locus RTLS 3D logs of three to six tagged people in a gym, with staged crossings and occlusions, plus manual identity labels. Not yet existing; recorded in two sessions on the course's LocusConnect installation through its MQTT or WebSocket API, confirmed in week one. Reference without a target: PnP on court-line intersections of known size, registered to UWB by a tag on four corners.

## 6. Final Prototype / Live Demo

At the end of the semester, I will be able to demonstrate that two phones on tripods at arbitrary spots each calibrate themselves within one minute of four tagged people playing, with reprojection error converging on screen. Then each person carries the same name label in both views through crossings; when someone steps in front of a player, that player's feed switches to the other camera within a frame, label intact. A meter shows keypoint traffic beside raw video's cost. Toggling fusion off makes labels swap and switching stop. If live UWB access fails, the pipeline runs on a recording.

## 7. How Will You Know It Works?

Identity: ID switches per minute, ByteTrack versus fused, per view on a labelled session; target 90 percent fewer, zero on staged crossings, full cross-camera agreement. Switching: fraction of frames a player is unoccluded in the selected view, above 95 percent versus best single camera; handover within a frame. Calibration per camera, against the court-line reference: rotation below 1 degree, translation below 20 cm, reprojection below 15 pixels, convergence within 60 seconds, clock offset within 20 ms, no shared phone clock. Fog: phone uplink over 95 percent below 720p video; edge latency under 100 ms per frame versus cloud round trip.

## 8. Why Should This Project Be Selected?

Identity is the measured bottleneck of sports video tracking, and no published method calibrates arbitrary cameras to UWB from tagged players with unknown correspondences. LocusConnect already runs its positioning engine on an edge server; this project puts the fusion on that fog node, keeps video on the phones, and measures why. Coaches already film on phones, so the value is direct. Stretch: a learned association affinity trained on the fusion's own labels, ablated against Hungarian.

## 9. Your Expected Contribution

I work in Python with OpenCV and ROS on multi-sensor robotics data, including camera calibration and synchronization. I will implement the RANSAC calibration solver, per-view association, and occlusion switching, run both recording sessions, and build the evaluation against the reference. I want to develop real-time multi-object tracking and edge deployment.
