# Scenario: the crossing

Five agents on one access point; each has something another needs;
the channel cannot carry all of it before the moment it matters, and
the moment is set by how the robots move.

## The place

A T-junction at site 1: a long corridor (the stem) meeting a crossing
corridor (the bar). Three masts: M1 looks down the left arm of the bar,
M2 down the right arm, M3 down the stem toward the junction. Two robots:
R1 comes up the stem toward the junction at 1 m/s and will turn left;
R2 comes along the bar from the right at 1 m/s and will go straight
through. Neither robot can see the other's arm until it is at the
junction. One or two people walk in the arms; a cart is sometimes
parked just past the corner on the left.

```
            M1 ◄── left arm ──┬── right arm ──► M2   ◄── R2
                              │
                              │ stem
                              │
                              M3 (looks up the stem)
                              │
                              ▲
                              R1
```

## Who holds what, who needs what, by when

| Holder | Observation | Needed by | Deadline (approach time) | Spoils in |
|---|---|---|---|---|
| M1 | left arm: person? cart? | R1 (turning left) | when R1's stopping distance reaches the corner: ≈ (d_R1 − 1.5 m)/1 m/s | person: ~2 s; cart: minutes |
| M2 | right arm: person? | R2 (coming from the right, already sees part of it); R1 after it turns? no, R1 goes left | R2's own view covers most of it: low need | ~2 s |
| M3 | stem: R1's position and speed | R2 (R1 will cross R2's path at the junction) | when R2's stopping distance reaches the junction | ~1 s (R1 moves) |
| R1 (LiDAR) | the stem and, once at the junction, the bar | R2 (if R1 arrives first, it sees R2's blind side) | R2's approach time | ~2 s |
| R2 (LiDAR) | the right arm and junction | R1 | R1's approach time | ~2 s |
| R1, R2 | pose, 10 Hz | server, each other | 100 ms | — |
| R1, R2 | keyframes to the live map, 0.5–1 MB/s | server | ~1 s, low value now | — |

Deadlines move with the robots: as R1 accelerates, its approach time
shrinks and M1's window closes sooner; if R2 stops, M3's observation
for R2 has no deadline at all until it moves again. Spoil times move
with what is observed: a parked cart stays valid for minutes, a walking
person for two seconds, so the same cell's window depends on what is
in it.

## The contention

All five transmit to one AP. Without coordination, at every frame M1,
M2, M3 push frames (periodic), both robots push poses and keyframes,
and whoever loses the backoff is late. The link's airtime is shorter
than the sum of what everyone would like to send: a JPEG per mast per
frame plus two keyframe streams exceeds what lands within 100 ms
(measured: 1.5 MB messages land in-frame 59–68 % of the time with two
robots). Something has to give, and the question is what.

## The decision, each interval

For each (holder, observation, receiver) triple with an open window
[deadline − spoil, deadline]:

- **whether** to send it at all: zero if the receiver will see the cell
  itself before its deadline (R2 already sees the right arm; M2's
  observation has no receiver), zero if the window is already closed;
- **what** to send: one bit (clear / occupied), a box with velocity, or
  a frame, with sizes 40 B / 80 B / 100 KB; the smallest that lets the
  receiver decide, and the larger ones only when airtime allows;
- **who** sends it when two holders have the same cell: R1 at the
  junction and M1 both see the left arm; pick the one with the cheaper
  airtime (nearer the AP, higher MCS) or the one whose copy is fresher;
- **when**: inside the window, as late as the link allows so the
  observation is fresh, as early as the deadline requires so it is not
  missed; a person observation for R1 should go at deadline − (a
  margin), not at the moment the person appeared;
- **ahead of what**: poses and the M3→R2 observation (R1 is about to
  cross R2's path) outrank keyframes every time; keyframes get what
  airtime is left.

The deadlines are a function of motion; the windows are a function of
what is observed; the airtime per message is a function of who sends
it; the value of sending is a function of whether the receiver will see
it anyway. All four are known or predictable one interval ahead from
the poses, paths and the static map that every node already has.

## What is scored

Per robot per junction crossing:
- did it hold a valid observation of its blind arm before its stopping
  distance reached the corner (hit), and was that observation still
  true at the moment (not spoiled);
- did it decide correctly (crawl if a person or cart was there, full
  speed if clear); did R1 and R2 avoid each other's path at the
  junction;
- pose deadline misses during the crossing;
- bytes on the air, per sender, from the sniffer; keyframe throughput
  retained.

Per session: hits and spoils against the clairvoyant schedule computed
afterwards from the recorded paths and the logged link.

## Policies compared

1. Periodic broadcast, FIFO (today): masts at 2 Hz frames, robots
   streaming; the AP does what it does.
2. Periodic with priorities: poses in AC_VO, the rest best-effort.
3. On-change broadcast: masts send on change to everyone; no timing.
4. Window scheduling: the decision above, computed at the server from
   poses and paths, with grants to senders (the AP cannot be
   programmed, so the server issues send permissions on a control
   topic).
5. Window scheduling + slotting: the same, with each sender's phase in
   the frame fixed from the shared clock (person 2's piece).
6. Clairvoyant: the offline optimum.

## Variations that make it a result

- **Speed.** R1 at 0.5 / 1.0 / 1.5 m/s: deadlines shrink; which
  policies keep hitting.
- **Load.** Keyframe streams on and off: whether the needed
  observations survive the bulk.
- **Spoil.** A person (fast spoil) versus a parked cart (slow spoil) in
  the left arm: whether the scheduler sends the cart once and the
  person just in time.
- **Redundant holders.** R1 arriving at the junction before R2: whether
  M1's and R1's observations of the left arm are sent once, by the
  cheaper sender.
- **Mast failure.** M1 off: R1 has no source for the left arm; the
  scheduler must report a miss and R1 must crawl; the score counts the
  honest miss, not a false hit.

## Why this scenario and not the single corner

The single corner has one holder, one receiver, one window and no
shortage, so every sensible policy wins. The crossing has five
senders, overlapping windows whose deadlines move with two robots'
motion, two holders of the same cell, two kinds of spoil time, bulk
traffic competing, and a link that cannot carry it all. Each of the
four decisions (whether, what, who, when) has a wrong answer that a
naive policy gives, and the score shows it.

## What it needs

The T-junction with corridors ≥ 3 m in each arm; three masts with
power and Wi-Fi; two robots on scripted paths with a waypoint follower
and a speed command; one person (two for the "both arms" case) and a
cart; chrony on everyone; the sniffer; the static map on every node;
the server with the scheduler and the grant topic. One session of
about three hours gives 20 crossings per policy at one speed; three
sessions cover the speed and load variations.
