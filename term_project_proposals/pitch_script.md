# Three-Minute Pitch: Self-Calibrating Multi-Camera UWB-to-Video Fusion on the Edge

Spoken pace is about 130 words per minute, so the script below is about 380 words. Show the system diagram page if a screen is available. Otherwise no slides are needed.

---

## The problem (0:00 to 0:45)

Picture a coach reviewing training video. Players cross, cluster, and block each other, and the video tracker loses who is who. On the SportsMOT benchmark, the standard ByteTrack tracker makes over 3,000 identity mistakes. So analysts fix it by hand, for hours, and most training sessions are never analysed at all.

LocusConnect's UWB tags already know exactly where every player is, and never mix them up, because radio passes through bodies. But today that data only lines up with video in time. Drawing it onto the players in the frame needs a camera surveyed in advance.

## The idea (0:45 to 1:45)

Our idea: point two or more ordinary phones at the court, anywhere. In the first minute of play, the system calibrates each camera by itself, using the players as the calibration target. UWB says where each tag is in metres, the camera sees people moving, and we find the one camera position and lens that make the two agree. Then every player carries a name label that never swaps. And when a player is blocked in one camera, the system switches their feed to the other camera within a frame, because the tag carries the identity across.

This is a fog computing system. Phones run pose detection and send only keypoints, a few kilobytes a second instead of video. The LocusConnect edge server, which already runs the positioning engine, does the calibration, matching, and switching.

## Why it is worth a semester (1:45 to 2:15)

Identity is the measured weak point of sports video tracking, and radio identity fixes it. Calibrating arbitrary cameras from several tagged players at once has not been published. It is real fog engineering with numbers we can measure: bandwidth saved, and edge versus cloud latency. And it runs on hardware we already have: phones and the LocusConnect system.

## What we will demonstrate (2:15 to 2:40)

By the end of the semester we will show it live. Two phones on tripods calibrate themselves in a minute while four people play. Names stay on through crossings. Someone steps in front of a player, and that player's feed jumps to the other camera. Flip fusion off, and the labels swap. We will measure identity switches against ByteTrack, calibration error against the court lines, bandwidth, and latency.

## The team (2:40 to 3:00)

I am looking for teammates for four parts: calibration and geometry, the phone app, the edge server and networking, and evaluation and data collection. If any of that sounds like you, come talk to me after class.

---

## A sentence an endorser can use

"It solves a real problem with hardware the course already has, it is a genuine fog system with measurable bandwidth and latency gains, and the live demo will clearly show whether it works."

## Likely questions and short answers

- **Why not just use a better video tracker?** The best published trackers still make thousands of identity errors on SportsMOT. They guess identity from appearance, and teammates in the same jersey look alike. The tag knows identity for certain.
- **How can a camera calibrate itself without a board?** Every tagged player is a moving point whose position UWB already knows in metres. A minute of play gives thousands of matches between where people are and where the camera sees them. That is the same input a calibration board provides, only spread over the whole court.
- **Why the edge and not the cloud?** Video from several phones is heavy, and the camera switch has to happen within a frame. Sending keypoints to a server in the same building keeps the uplink small and the latency low. We will measure both against a cloud round trip.
- **What if live UWB access fails?** The pipeline runs on recorded sessions, so every measurement still holds. Only the live moment of the demo is at risk.
- **How big should the team be?** Three or four people, one per part. The parts connect only through the edge server, so they can be built in parallel.
