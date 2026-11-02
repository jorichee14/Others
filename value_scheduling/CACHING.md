# The world model as a cache: a project for someone who knows caching

The server's twin and the robots' map tiles are copies of one object,
the building, that changes slowly in some places and fast in others,
and is accessed by robots along paths that are known ahead of time.
That is a distributed cache with a known future access sequence, a
measurable per-item volatility, and a contended channel between the
tiers. Everything below is a caching problem first and a robotics
problem second, and each part has a test on the real rig.

## The hierarchy

| Tier | Holds | Capacity | Access pattern |
|---|---|---|---|
| Robot (Jetson / Pi) | map tiles, received observations, its own recent scans | memory-bound (Pi: tight) | its planned path, at ~1 m/s |
| Edge / AP (server) | the twin, every agent's recent observations | large | all robots' paths |
| World | the building | — | changes at rate λ(x) per cell |

Reads: a robot localizes against tiles ahead of it and needs other
agents' observations of regions it cannot see (the shadow). Writes:
keyframes update the twin; the twin invalidates tiles. The channel
between tiers is the contended Wi-Fi link; the uplink is CSMA among
robots, the downlink is owned by the AP.

## Correction (1 Nov): the static map is preloaded, not cached

Site 1's reference map fits on every node (a few hundred MB as TSDF,
a few GB as splat tiles; 16 GB and NVMe on the Orin NX, the SD card on
a Pi). Caching static tiles would solve a problem the rig does not
have. Every node preloads the static map and holds a version number;
the cache is the *dynamic layer only*: changes since mapping and other
agents' observations, which expire. Work package A is deferred to the
multi-site case where capacity is real; B and C are the project.

## Work package A (deferred). Known-future prefetch and eviction on the robot

The robot's planned path gives its future access sequence to tiles. A
cache whose future accesses are known can run Belady's MIN for
eviction and exact prefetch, which no ordinary cache can. The only
uncertainty is deviation from the plan, measurable as a hit-rate loss.

Build: a tile cache on the Jetson and on the Pi with bounded memory;
policies LRU, LFU, path-prefetch + MIN, with prefetch depth as a
parameter. Measure on recorded routes and live: hit rate, bytes
downloaded, localization continuity (map→odom drops) against the
reference trajectory, memory high-water mark, and the hit-rate loss as
a function of plan deviation.

Test on the robot: drive the survey route with each policy; tiles
served from the server over the real link; localization quality from
the real LiDAR against the tiles in cache.

## Work package B. Coherence with measured volatility

A tile or observation is valid until the world changes under it. Each
cell has a change rate λ(x) estimated from the team's own observation
history (how often its occupancy changed between visits); corridors
high, walls zero. TTL(x) = f(λ(x)) is the coherence rule: an item past
its TTL is stale and must be refreshed before use; an item far from
its TTL need not be sent at all. Invalidation is pushed by the twin
when a keyframe shows change; refresh is pulled by the robot before
approach time.

Build: per-cell λ from the recordings (coop_2, survey_1 have repeated
visits); TTL-based validity on both tiers; push invalidation from the
server's change detection; a "staleness budget" per task (localization
tolerates stale walls, obstacle avoidance does not tolerate stale
corridors).

Measure: bytes refreshed versus staleness at use, on recorded data and
live; the fraction of refreshes that were unnecessary (the world had
not changed), which is the waste a coherence rule should remove.

## Work package C. The edge cache of observations (the contention angle)

Today each sender's observation goes to each receiver that needs it,
so a region seen by one robot and needed by two others is sent twice on
the contended uplink. With a cache at the AP: a sender uploads a unit
once; receivers pull it on the downlink, which the AP schedules without
contention; the AP serves a unit to each receiver at that receiver's
approach time, from cache, with no further uplink. Duplicates from two
senders of the same cell are deduplicated by content hash before the
second upload (the AP replies "already have it").

Build: a cache at the server keyed by (cell, time) with volatility TTL;
a sender protocol that announces hashes first and uploads on miss; a
receiver protocol that pulls by region along its path; downlink
scheduling by approach time.

Measure: uplink airtime and collision rate with and without the cache
at N = 5 (the CONTENTION phase 0 rig); deduplication rate on real
multi-robot recordings; delivery-before-approach-time rate on the
downlink.

Test on the robot: two Jetsons as senders, three Pis as receivers
pulling along scripted paths, real AP, sniffer for airtime.

## Work package D (optional). Temporal caching inside one robot

Consecutive LiDAR frames of a static scene are nearly identical. Cache
the static BEV or voxel content against the map and send only the
delta (cells that differ from the cached map prediction). This is
delta encoding with the map as the reference; it is the robot-local
version of B and uses the same λ(x).

Measure: bytes per frame versus full-frame, on real LiDAR; encode time
on Jetson and Pi.

## Prior art to read first

Mobility-aware proactive caching in vehicular networks (RSU content
caching with trajectory prediction) is a large literature and occupies
the generic "prefetch along a predicted path" claim; HD-map tile
prefetching for vehicles exists. What is not in that literature, and is
this project's claim: a *known* plan gives an exact future access
sequence (MIN is implementable, not approximated); validity derived
from *measured* per-cell volatility of the environment rather than a
content popularity model; and perception observations, not media
content, as the cached object, served to receivers at their approach
times instead of re-sent by senders on a contended uplink. Also read
Belady 1966, the LRU/ARC/LIRS line for the eviction baselines, and
ETSI CPM for the standardized observation format.

## Deliverables and order

1. A (two weeks): the tile cache with the four policies, measured on
   recordings, then one live drive. Standalone result: how much a known
   plan buys over LRU on a robot.
2. B (two weeks): λ(x) from the recordings; TTL coherence; the
   unnecessary-refresh fraction. Standalone result: the waste in
   timer-based twin refresh.
3. C (three weeks, needs the contention rig): the edge observation
   cache; uplink airtime with and without it. This is the result that
   joins the contention work.
4. D if time allows.

Each package is one figure and one table; A and B need no new hardware
and no network beyond what exists.
