# The program: one scenario, four phases

**The scenario.** A robot team on shared Wi-Fi in a changing building,
keeping each robot's picture of the places its next move depends on
correct, in time. Deadlines come from motion, validity from the world,
need from the receiver's own future view, and the channel is shared
and short.

**The quantity everything is scored by.** AoII at the decision: how long
a robot's belief about a place has been wrong at the moment its move
depends on that place. Zero where the robot sees for itself in time,
zero for walls, growing at the place's change rate λ(x) otherwise.

**The one system**, built once and switched on in stages:

```
 holders (masts, robots)        server                         receivers (robots)
 observe → on change ─────▶ store: one copy per cell ─────▶ memory: hold, extrapolate, expire
                            TTL from λ(x), invalidate          approach time from path
                            schedule fetches in windows        speed / path from AoII ahead
                            grants: who sends what when
 ◀──────────── clock slots: each sender's phase in the frame; compute to fit the slot ──────────▶
```

## Phase 0: instrument (weeks 1–2)

One corner, one mast, one robot, one person. Everything else off.
Build the three nodes, the logger, the scorer, λ(x) from the
recordings; bench replay through the logged channel; then one session.

Delivers: the waste fraction (timer traffic whose content had not
changed or whose receiver would see it anyway); bytes and latency per
representation (frame / box / bit); the first AoII-at-decision column.
No method claim. It proves the kitchen works.

## Phase 1: the team's memory under a short link (weeks 3–6)

The crossing: five nodes, overlapping windows, two holders of one
cell, bulk traffic, the link throttled until fetch-on-change misses a
third of its windows. The server stores one copy per cell with TTL
from λ(x), invalidates on observed change, and schedules fetches inside
each (robot, cell) window [deadline − TTL, deadline] by earliest
deadline, λ-weighted under overload, multicast when two robots share a
cell, representation downgraded (frame → box → bit) when the window
would be missed. Senders transmit only on grant.

Delivers: λ-weighted misses at decision time against fetch-on-change,
fetch-on-request and the clairvoyant schedule from the recorded paths;
pose and control deadline hits; the moved-cabinet test (invalidation
and refresh before stopping distance).

Stop rule: if the link cannot be made short on this rig, or plain EDF
is within noise of the full policy, the result is the formulation,
the bound and the waste number; say so.

## Phase 2: the channel takes turns (weeks 3–6, in parallel)

The same five nodes, synthetic then real messages, synchronized
readiness on the shared clock. Measure the damage: in-frame delivery
versus N and message size, Bianchi fit. Then slot each sender's phase
in the frame from the clock, no coordinator; carry a timestamp and
velocity so the receiver extrapolates the known delay; widen slots by
rate; let the slot phase set how deep each sender encodes before it
sends; fuse on the server as messages arrive.

Delivers: the contention-loss surface and how much slotting recovers
at what task cost; encode curves per device class; burst versus
staggered fusion latency.

Stop rule: if device jitter eats the slots or compensation costs more
than it saves, fall back to grants (phase 1 already has them) and
report why.

## Phase 3: the robot drives on it (weeks 6–8)

Switch on the loop: each robot's speed and path respond to the AoII of
the places ahead. Full speed where a valid observation says clear;
slow where the picture cannot be kept correct; route past a mast or
through a good-link place before entering a hidden region; fetch early
where the survey radio map says the link will be good. Three policies:
crawl at every hidden place, drive on AoI, drive on AoII with link-
aware prefetch.

Delivers: mission time at fixed safety for the three; AoII at the
decision against AoI and bytes as predictors of outcome; the same rule
unchanged over a detector, an occupancy map and a one-bit certificate.

Then the joint session: all switches, four runs (today's system;
phase 1; phase 2; both), the system result.

## Phase 4: the theory and the paper (weeks 8–10)

Expected AoII at the decision as a function of λ, the delivery-delay
distribution (phase 2) and the trigger threshold (phase 1); from it
the TTL rule, the speed rule, and the condition under which no
communication is needed. The clairvoyant bound as the regret
reference. One system paper; one each for phases 1 and 2 if their
results stand alone.

## Who

| Phase | Owner | Others |
|---|---|---|
| 0 | person 1 builds; everyone uses | person 3 writes the metric page first |
| 1 | the fetch scheduler: person 3; store, triggers, memory, λ(x), invalidation, logger: person 1 | — |
| 2 | person 2 | — |
| 3 | person 3 | persons 1 and 2 run their switches |
| 4 | person 3 | co-authors |

## Rig, decided once

Two Scout Mini (Jetson Orin NX, Ouster, ZED, radar), three Raspberry
Pi masts with cameras, one Asus RT-BE82U, one server, chrony, one
sniffer, the reference map of site 1 preloaded on every node, one
corner and one T-junction fixed for every session, one JSONL log
format with the belief field, `tc` on the server for throttling. No
digital twin, no Sionna, no labels beyond who was where per trial.
