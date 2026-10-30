# The AP problem and the design that answers it

## The problem

The method puts the selection at the AP inside the 802.11ax uplink
OFDMA trigger loop: stations report demand in buffer status reports,
the AP chooses who sends what in each trigger frame. No commodity AP
exposes that loop.

| Option | Status |
|---|---|
| Asus RT-BE82U (the AP in use; Wi-Fi 7, Broadcom) | OFDMA and MU-MIMO scheduled in firmware; Asuswrt and Asuswrt-Merlin expose airtime fairness, QoS and band steering, no per-station or per-trigger control, no BSR access |
| Any other consumer or enterprise AP (Broadcom, Qualcomm, MediaTek) | same: the trigger scheduler lives in firmware |
| Linux hostapd with a Wi-Fi 6 card (MediaTek MT7915/16 via mt76, Qualcomm via ath11k) | AP mode works; OFDMA and MU scheduling run in the card's firmware; hostapd sets parameters, not the per-trigger decision |
| Intel AX2xx in AP mode | no usable AP-mode OFDMA |
| openwifi (FPGA SDR) | open MAC, but 802.11a/g/n only, no OFDMA |
| Vendor SDKs (Qualcomm, Broadcom under NDA) | out of reach for this project |

So the trigger loop cannot be run on any hardware available. The plan
already names the answer, application-level grants; this file designs
it so that simulation and hardware test the same abstraction.

## The design: grants from an edge scheduler

**Roles.** The AP stays a plain AP (EDCA contention, firmware OFDMA
among whoever transmits). An **edge scheduler** on the server plays the
AP's role at the application layer.

**Loop per frame (period T_f = 100 ms), with trigger interval T_g.**

1. Each agent encodes its base layer and the value report for its
   refinement units (unit id, cell, predicted gain), and sends the
   report to the edge scheduler in a small datagram (≤ 1 MTU) in the
   voice access category (AC_VO), so it never queues behind bulk.
2. The scheduler runs the greedy selection over all reports it holds at
   the next grant instant, computes a grant per agent (which unit ids,
   in which order, how many), and multicasts one grant frame (AC_VO).
3. An agent transmits only the granted units, in order, and only in its
   grant window; base layers go in a protected first window each frame.
4. Units that have not arrived by the deadline are missing at fusion.

**What this reproduces.** Contention-free, value-selected, deadline-
bounded uplink with one decision point and no request round from the
receiver (the value report rides on a message the agent sends anyway,
as the BSR would). The protected base-layer window is the restricted-TWT
analogue.

**What it does not reproduce.** Frequency multiplexing across resource
units (grants serialize agents in time instead of placing them in
parallel RUs), trigger-frame timing at the microsecond scale, and the
BSR's zero-overhead carriage (the report costs one small frame per
agent per interval). Each is stated as a limitation; the second and
third are small at T_g of tens of milliseconds, the first is the
difference between OFDMA and TDMA and is what ns-3 is for.

**Fairness of the comparison.** Every baseline runs on the same AP:

- EDCA baseline: agents send compressed messages whenever ready, no
  grants;
- sender-side selection with round-robin: grants of equal airtime per
  agent, content chosen by the sender;
- receiver-driven two-round: a request frame from the scheduler
  followed by responses, same codec;
- ours: value-selected grants.

Grant and report overhead is charged to our airtime.

## What the Asus AP does underneath

The AP's firmware OFDMA will still schedule among the stations that
transmit in a window. With one granted sender per window, it has nothing
to multiplex and the uplink is a single-user transmission; with several
granted senders (parallel grants, to approximate RUs) the firmware
decides the RU split, which we do not control but can measure. Keep
single-sender windows as the default and parallel grants as an
experiment. Turn off airtime fairness and band steering so the AP does
not reorder what the grants ordered.

## Measurement

Without AP-side logs, measure at the edges and in the air:

- send and receive timestamps per unit at the agents and the server,
  on one clock (chrony, as the SCooP testbed does; offset ≤ 1 ms);
- a monitor-mode sniffer on the channel for airtime per station and
  per access category, retries and collisions (the EDCA baseline needs
  it);
- the AP's own client-rate (MCS) readout per station, polled, as the
  cost input to the greedy step.

## Simulation-to-hardware match

The ns-3 model simulates the same grant abstraction (an application-
layer scheduler over an 802.11ax MAC with the default firmware-like
multi-user scheduler), and separately the ideal trigger-loop version
(the replaced `MultiUserScheduler`). The paper reports three systems:
trigger-loop (simulated, the target), grants (simulated), grants
(hardware). The sim-grant vs hardware-grant agreement validates the
simulator; the trigger-loop vs grant gap is the cost of not owning the
AP, which is the deployment argument for AP vendors.

## Hardware for the check

Two to four agents with Wi-Fi 6 clients (the Jetson Orin NX + USB
dongle rigs exist), one Asus RT-BE82U, one edge server on Ethernet to
the AP, one sniffer. The SCooP testbed's robots can be the agents for
this check; nothing else of SCooP is used.
