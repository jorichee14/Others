# Stretch 4 communication-aware pick: stress test

## The situation, as facts

- **Robot.** Stretch 4: two hemispherical 3-D lidars, two global-shutter
  fisheye RGB cameras, one central high-resolution RGB camera on the
  gripper workspace, six laser line sensors; no depth camera listed.
  Compute: Intel NUC 15 (main system) and an NVIDIA Jetson Orin NX
  (for researchers' vision/AI). Sources: Hello Robot Stretch 4 hardware
  guide; IEEE Spectrum (2026).
- **Pipeline today.** Raw images over ROS 2 to a server running YOLO;
  2Fast-2Lamaa planned (robot or server); ED305 camera network and a
  USD twin available.
- **Task.** Coarse-to-fine pick: find, drive, arrive, plan grasp,
  align, close, verify.

## Claims and attacks

| # | Claim | Attack | Evidence | Verdict |
|---|---|---|---|---|
| 1 | The link is the bottleneck | Raw images over DDS on Wi-Fi fail because of IP fragmentation, retransmission timing and buffer bursts; JPEG plus DDS QoS tuning fixes most of it | Lee et al., Optimizing ROS 2 Communication for Wireless Robotic Systems (arXiv 2508.11366, 2025) | **Fails as a research problem** for the current pipeline: it is engineering. Survives only with heavy offloaded traffic, shared AP load (camera network, other robots) or weak spots. Must be measured. |
| 2 | Offloading is needed | The Orin NX runs YOLO at frame rate; 2Fast-2Lamaa needs < 16 % of a laptop CPU | 2Fast-2Lamaa (IJRR 2026); Stretch 4 hardware | **Fails for YOLO and 2Fast-2Lamaa.** Survives for heavy models: open-vocabulary detection/segmentation, grasp generation, VLM/VLA. "Full workload stack infeasible on smaller onboard GPUs", larger ones drain batteries hours faster (Pohland et al., MSR-TR-2026-14). |
| 3 | 2Fast-2Lamaa should run on the server | Light enough for the robot; offloading it adds two 3-D lidar streams for nothing | as above | **Fails as a design.** Keep only as a stress condition. Run it on the robot. |
| 4 | Phase-aware compression is the contribution | Configuration adaptation (resolution, quality, region) under bandwidth is established; a state machine on top is engineering | Chameleon, AWStream (SIGCOMM 2018); DDS (SIGCOMM 2020) | **Weak.** Reviewers will call it AWStream with phases. |
| 5 | Choosing where to send from is the contribution | Already proposed: communication map → point of the next cloud request, request–response window | Liu et al. (arXiv 2606.31497, 2026); Gordon et al. (INFOCOM 2026 workshop); RCAMP (IROS 2017) | **Taken**, in simulation. Real-robot evidence is still open. |
| 6 | Slowing down is part of the contribution | Standard in teleoperation and safety practice | teleop practice; AMR safety margins | **Not a contribution.** Keep as a mechanism. |
| 7 | Fine phase is local, link irrelevant | True if the fine loop uses onboard sensing. If the grasp model or fine policy is on the server (as in OK-Robot with AnyGrasp on a workstation), the fine phase is where the link hurts most | OK-Robot (RSS 2024); Pohland et al. (2026): tens of ms hurt manipulation accuracy | **Depends on the stack.** This is where the real problem sits. |
| 8 | Lidar pauses during fine phases | No depth camera: the object's 3-D position comes from lidar points in the YOLO box | Stretch 4 sensor list | **Partly fails.** Robot-pose scans pause; object points stay on, computed on the robot. |
| 9 | Error at the moment of action predicts success better than latency | Plausible; needs ground truth | AoI / AoII literature; camera network gives ground truth | **Holds as a metric**, to be shown. |
| 10 | The gap is open | 2026 work is converging: Liu et al., Pohland et al., Gordon et al., SPO, latency-tolerant VLA | as listed | **Open but closing.** Real-robot, real-link evidence is the differentiator. |

## What survives

The real decision is not how to compress, but **which model answers
each phase: the robot's or the server's.**

- Onboard (Orin NX): small models (YOLO, lidar box points), fast,
  independent of the link, less accurate or closed-vocabulary.
- Server: heavy models (open-vocabulary detection, grasp generation,
  VLM), more accurate, slow, dependent on the link at the robot's
  position.
- In coarse-to-fine, the accuracy needed rises as the deadline
  tightens. Coarse phases tolerate the server's delay; fine phases need
  the server's accuracy but cannot always wait for it.

So per phase there is a crossover: use the server when its answer
arrives in time at the robot's position and the onboard model is not
accurate enough; otherwise use the onboard model, or move to where the
link makes the server answer arrive in time, or slow down.

## The stress-tested idea

**Phase- and position-dependent offloading for coarse-to-fine mobile
manipulation.** For each phase, the robot chooses onboard model or
server model, and for the server the form and send point of the
request, so that the answer meets the phase's accuracy tolerance and
deadline given the link at its position.

| Against | Difference |
|---|---|
| Chinchali et al. (RSS 2019) | offload on local uncertainty, one perception task; no phases, no position, no deadline from motion |
| Liu et al. (2026) | choose where to send from; one request type, fixed form, no onboard alternative, simulation |
| AWStream / Chameleon | form of one stream under bandwidth; no robot motion, no onboard alternative |
| SPO, latency-tolerant VLA, TIC-VLA | hide delay inside the policy (training); here the policy is unchanged and the choice is which model answers |
| Pohland et al. (2026) | measures the onboard / edge / cloud trade-off; does not decide it per phase |

Honest strength: a systems contribution with a clear trade-off and
real-robot evidence; not a new theory. Venue: RA-L / IROS, or IoT-J.

## Kill tests (Tuesday and the week after)

1. **Does the onboard stack meet every phase's tolerance?** Run YOLO +
   lidar box points on the Orin NX; measure object error against the
   grasp tolerance. If it is within tolerance in every phase, there is
   no crossover and no offloading problem for this task; the problem
   moves to multi-robot / shared AP.
2. **Does the server model beat it where it matters?** Run the heavy
   model the group wants (open-vocabulary detector or grasp generator)
   on the server; measure its accuracy gain per phase.
3. **Does the link ever make the server miss?** Server-answer delay
   with JPEG and tuned DDS, at the table and at the weak spot, with the
   camera network loaded. If it never misses a deadline, the position
   part is not needed.

The idea stands only if all three show a gap: onboard not accurate
enough in some phase, server accurate enough, and the link sometimes
too slow for it.
