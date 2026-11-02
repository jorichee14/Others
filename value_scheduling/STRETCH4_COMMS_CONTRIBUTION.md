# Stretch 4: the comms-side contribution

## Split of the surviving idea

The surviving idea: for each phase, the robot decides whether its own
model or the server's answers. That decision has two halves.

| Half | Question | Owner |
|---|---|---|
| Compute | how accurate and how slow is each model, onboard and on the server? | 廖恩莆 (pipeline), Prof. Yeh (compute) |
| **Comms** | **will a server request sent from here come back in time, and how do we make it more likely?** | Jorich |

## The comms contribution

**Predicting, and then raising, the probability that a server request
returns before the phase's deadline, from the robot's position, on real
Wi-Fi under real load.**

Three parts:

1. **Request–response time predictor.** For a request of size S sent
   from position x now, the distribution of the time until the answer
   is back on the robot: uplink + server queue + inference + downlink.
   Inputs: the link map of ED305 in the USD twin (built from survey
   runs), the live link (RTT, goodput, RSSI), the current load on the
   AP (camera network, other traffic), the server's reported queue.
   Output: P[T ≤ D] for that request.
2. **Transport knobs that raise P[T ≤ D].** Form of the request (crop,
   resolution, codec), priority class on the AP (WMM / DSCP), DDS
   settings for large messages.
3. **The decision interface.** The robot's offloading choice reads
   P[T ≤ D] per option (onboard; server with form c, from here
   or from the next stop point) instead of a fixed rule.

## Why this, from the literature

| Work | What it leaves open |
|---|---|
| Liu et al. (arXiv 2606.31497, 2026): request–response window, choose the send point from a communication map | assume the map is known before execution; list moving people, occlusion and network load as open; call for tests over deployed Wi-Fi with online inference |
| Gordon et al. (INFOCOM 2026 workshop): radio map → QoS risk costmap | path planning, not the timing of individual requests |
| RCAMP (IROS 2017): Gaussian-field radio map, connectivity-safe paths | signal strength, not request deadline |
| FogROS2-PLR (2024): choose network interfaces to minimise deadline-miss probability | cloud servers and interfaces, not the robot's position or task phase |
| Lee et al. (arXiv 2508.11366, 2025): DDS tuning for large messages on lossy Wi-Fi | transport only, no deadline or task |
| Pohland et al. (MSR-TR-2026-14): latency hurts manipulation; bandwidth makes naive offload impractical | measurement, no decision |

The gap the comms side fills: a measured, online prediction of
whether a request meets its deadline from where the robot is, on a
real building's Wi-Fi with real load, plus the transport means to raise
it, used by the offloading decision.

## How it is shown

1. **Predictor accuracy.** Predicted vs measured deadline-miss rate over
   ED305 (positions, loads, request sizes). Baselines: RSSI threshold;
   mean RTT; static map only (Liu et al. style).
2. **Knob gains.** Miss rate with and without form change, priority,
   DDS tuning; at the table and at the weak spot; camera
   network on and off.
3. **Task effect.** Picks with the offloading decision driven by each
   predictor: pick success, object error at the moment of action, time
   per pick.

## Kill tests

- If requests never miss in ED305 even with the camera network loaded,
  the predictor has nothing to predict: create the weak spot / load or
  move to the multi-robot setting.
- If the static map predicts as well as the online predictor, the live
  part is not needed: report the map result alone.

## Where it leads

The same predictor, with several robots and the camera network
contending for one AP, is the shared-Wi-Fi setting of the dissertation.
