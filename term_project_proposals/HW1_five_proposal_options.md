# HW1 Term Project: Five Proposal Options

Course: Emerging Technologies and Applied Innovation, Fall 2026 (Prof. Hank Huang, NYCU)

Platform note. In this course "GPS/GEO" is **Generative Engine Optimization**: measuring how
AI models perceive and position an entity (a brand, a school, a product) when asked a fixed
set of questions. It is not GNSS positioning. UWB work lives under **LocusConnect**, energy
work under **KiwiEnergy**.

Each option below gives the problem, the methods, the data, the live demo, the metric, and
the main risk. Word limits in the HW1 template are tight, so each section is written to be
trimmed into the one-page form rather than expanded.

---

## Option 1. NLOS-Aware UWB Positioning: Detecting and Down-Weighting Corrupted Ranges

**Platform:** LocusConnect (UWB)

**Problem.** UWB player tracking is accurate when a tag has a clear line of sight to the
anchors. When another player's body, a goalpost, or a bench blocks the path, the range is
measured too long (non-line-of-sight, NLOS) and the position jumps by a metre or more. The
tracking operator sees jitter and ghost sprints in exactly the crowded moments coaches care
about most. Existing pipelines treat every range as equally trustworthy.

**Methods.**
1. Log raw per-anchor ranges (and channel impulse response, CIR, if the LocusConnect
   hardware exposes it; otherwise received-signal metrics such as first-path power and
   RX level) alongside a reference trajectory.
2. Reference: a tag carried along a taped-out path with known waypoints, or tracked by a
   LiDAR/camera rig, or a total station if one is available. A taped path with timed
   waypoints is enough for a first version.
3. Label each range as LOS or NLOS from the reference (residual above a threshold), then
   train a small classifier (gradient-boosted trees on hand-crafted CIR features, or a 1-D
   CNN on the raw CIR) to predict NLOS per range.
4. Positioning: an EKF or sliding-window factor graph over ranges, where each range is
   weighted by the classifier's LOS probability and wrapped in a robust (Huber) loss.
5. Baseline: plain least-squares trilateration and an unweighted EKF on the same log.

**Data.** Must be collected: a few sessions on the course's UWB setup with deliberately
staged occlusions (people standing between tag and anchor). Public UWB NLOS datasets
(e.g. the eWINE / DW1000 CIR datasets) can pretrain the classifier.

**Live demo.** A tag moves around the room while a person walks between it and the
anchors. Two trails on screen: the naive estimate and the NLOS-aware estimate. The naive
one jumps; the robust one does not. Per-anchor LOS/NLOS flags update live.

**Metric.** Positioning RMSE and 95th-percentile error against the reference path, naive
vs. robust; NLOS classification F1. Target: at least 30% RMSE reduction under occlusion.

**Risk.** If the hardware only exposes final positions and not raw ranges, the project
must fall back to outlier rejection on positions, which is weaker. Confirm raw-range
access in week one.

---

## Option 2. Automatic Tactical Pattern Detection from UWB Player Positions

**Platform:** LocusConnect (UWB)

**Problem.** Coaches of amateur and university teams get a heat map from a positioning
system and little else. The questions they actually ask ("how fast did we press after
losing the ball", "how compact was the back line", "when did we overload the left side")
are answered today by a person scrubbing video. That takes hours per match and does not
happen at all for training sessions.

**Methods.**
1. Define two or three patterns precisely enough to label them. Good candidates because
   they are computable from positions alone: (a) defensive-line compactness breaches
   (line depth over a threshold for more than N seconds), (b) numerical overload in a
   pitch zone (attackers minus defenders in a zone), (c) press trigger (three or more
   players converging on the ball-side zone within a window).
2. Compute per-frame team-shape features: convex hull area, line depth and width,
   centroid, Voronoi/pitch-control share per zone, inter-player distances, velocities.
3. Detection: start rule-based from the feature definitions, then add a lightweight
   classifier (logistic regression or a small temporal CNN over 5-second windows) trained
   on hand-labelled windows. Compare both.
4. Ground truth: label events in a few recorded sessions by watching synchronized video.
5. Output: a timeline of detected events with a replayable 2-D animation of each.

**Data.** UWB position streams from the course's system, recorded during real or staged
small-sided games. If no team sessions are available, public tracking datasets
(Metrica Sports open data, SoccerTrack) provide positions in the same format for
development, with the UWB data used for the demo.

**Live demo.** Replay a recorded session (or run a live 5-v-5 staged drill in a hall):
the animated pitch shows players, and events light up on a timeline as they occur, each
with its supporting numbers (line depth, zone counts).

**Metric.** Event-detection precision, recall, and F1 against manual labels, plus
temporal alignment error in seconds. Target: F1 above 0.8 on at least one pattern.

**Risk.** Ambiguous pattern definitions make labelling inconsistent. Fix definitions and
label a pilot set before building anything.

---

## Option 3. UWB-to-Video Identity Fusion: Calibrating a Camera Against UWB Tags and Keeping Player IDs Through Occlusion

**Platform:** LocusConnect (UWB), combined with video

**Problem.** Video tracking gives rich context but loses player identity every time two
players cross or one is occluded, so a match's "who did what" needs manual fixing. UWB
gives identity for free but no visual context. Nobody in the course setup currently
aligns the two, so the camera and the tags live in different coordinate frames and cannot
help each other.

**Methods.**
1. Extrinsic calibration between the camera and the UWB frame: walk a single tag around
   the field while it is visible in the camera, detect it in the image (a coloured marker
   or a person detector), and solve the camera pose (PnP with RANSAC) from the
   UWB-position-to-pixel correspondences. This is the same idea as ChArUco-based
   radar-camera calibration, with the tag as the moving target.
2. Run an off-the-shelf detector and tracker on the video (YOLO + ByteTrack).
3. Association: project each tag's UWB position into the image every frame and match tags
   to tracklets with the Hungarian algorithm on reprojection distance, with a gating
   threshold and short-term memory so brief detection dropouts do not break assignment.
4. When the video tracker swaps IDs, the UWB assignment overrides it; when UWB is noisy,
   the tracklet smooths the displayed position.
5. Baseline: the video tracker alone.

**Data.** Must be collected: synchronized video and UWB logs of a few people with tags in a
gym or field, including deliberate crossings. Timestamps must be aligned (a clap or a jump
visible in both streams is enough).

**Live demo.** A live camera feed with three or four tagged people crossing paths. Each
person carries a persistent name label sourced from their tag. Toggle the fusion off and
the labels swap after crossings; toggle it on and they hold.

**Metric.** Number of ID switches per minute with and without fusion (target: zero
switches on staged crossings), calibration reprojection error in pixels, and association
accuracy against manual labels.

**Risk.** Time synchronization and camera placement. Plan one calibration session early
and keep the camera fixed afterwards.

---

## Option 4. Next-Hour Load Forecasting with a Recommended Peak-Shaving Action

**Platform:** KiwiEnergy

**Problem.** A building or campus facility manager pays a demand charge set by the
highest 15-minute peak of the month. Peaks are avoidable if the manager knows an hour in
advance that one is coming and shifts a controllable load (chillers, EV charging, a
battery). Today they see consumption after the fact and cannot act. Dashboards show
history; nobody tells them what to do next.

**Methods.**
1. Data ingestion from KiwiEnergy readings at 15-minute or finer resolution, joined with
   weather (temperature, humidity) and calendar features (hour, weekday, holiday).
2. Forecasting: a persistence baseline and a seasonal-naive baseline first; then
   gradient-boosted trees (LightGBM) and a small sequence model (temporal CNN or a compact
   transformer). Report probabilistic intervals using conformal prediction so the action
   layer knows how confident the forecast is.
3. Action layer: given the forecast and a simple model of one controllable load (battery
   capacity and power, or a deferrable load with a deadline), solve a small linear program
   each step that minimizes the predicted peak subject to constraints. Output a concrete
   recommendation ("discharge 20 kW from 14:00 to 15:00").
4. Evaluation in replay: run the whole loop over held-out weeks and measure both forecast
   error and the peak actually achieved by following the recommendations.

**Data.** KiwiEnergy consumption data (access via the course; confirm granularity and
history length). Fallbacks if needed: public building datasets (Building Data Genome 2,
ASHRAE Great Energy Predictor). Weather from an open API.

**Live demo.** Data streams in (live or replayed at high speed). The screen shows the
last day, the next-hour forecast with its interval, a peak warning when one is predicted,
and the recommended action. A toggle applies the action and shows the resulting flattened
curve.

**Metric.** MAPE and RMSE of the next-hour forecast versus the persistence baseline
(target: 25% lower error); interval coverage near the nominal 90%; monthly peak
reduction in replay when recommendations are followed (target: 5 to 10%).

**Risk.** The data may be too coarse or too short for sequence models. Trees and
baselines still work on small data, so the project degrades gracefully.

---

## Option 5. A Repeatable GEO Instrument: Measuring How AI Models Position an Entity, With Test-Retest Reliability

**Platform:** GPS/GEO (Generative Engine Optimization)

**Problem.** Marketing and communications teams (a university admissions office, a
Taiwanese hardware brand) increasingly care what ChatGPT, Gemini, and Claude say when a
prospective student or buyer asks "which university is best for X" or "recommend a
laptop for Y". Today someone types a few questions and screenshots the answers. That is
not a measurement: answers change between runs, models, and phrasings, so nobody can tell
whether a change in the score is real or noise, and nobody can tell if a website edit
moved anything.

**Methods.**
1. Build a fixed query set (30 to 50 prompts) for one entity, with paraphrase variants
   of each prompt to separate phrasing noise from model behaviour.
2. Query several models through their APIs, repeated N times per prompt, with and without
   web retrieval where the API supports it. Store every raw response.
3. Extraction: a structured pass (a second LLM call with a strict schema, checked against
   a hand-labelled subset) that records for each response whether the entity is mentioned,
   its rank among competitors, sentiment, and which attributes and sources are cited.
4. Reliability analysis: test-retest agreement across repeated runs (Krippendorff's alpha
   on categorical fields, rank correlation on rankings), cross-model agreement, and the
   effect of paraphrasing. This gives an error bar on every score instead of a single
   number.
5. Intervention experiment (stretch): change a page the models can retrieve (an
   about-page the user controls) and rerun; test whether the shift exceeds the measured
   noise floor.

**Data.** Generated by the project itself: API access to at least three model providers
(cost is small at this scale) plus a hand-labelled validation set of a few hundred
responses for the extractor.

**Live demo.** Type an entity name and pick a query set. The system runs the panel live
(or shows a cached run for speed) and produces a dashboard: mention rate, mean rank,
sentiment, per-model comparison, and a reliability score with confidence intervals.
Rerun the same entity and show the scores land inside the intervals.

**Metric.** Extractor accuracy against hand labels (target F1 above 0.9); test-retest
reliability (target alpha above 0.8 for mention and rank); a documented noise floor so
any claimed positioning change can be tested for significance.

**Risk.** Model APIs change behaviour over the semester. Storing every raw response and
timestamp turns that into a finding rather than a failure.

---

## Which to pick

Options 1 and 3 fit an estimation and sensor-fusion background most directly and produce
the most convincing live demos (a visible jump that disappears; a label that stops
swapping). Option 2 is the closest to the instructor's own LocusConnect example and
depends most on getting real team sessions. Option 4 is the safest on data if KiwiEnergy
access is confirmed. Option 5 is the least hardware-dependent and the most novel in
framing (a measurement instrument with error bars rather than a dashboard).

If two are submitted as a primary and a fallback, a good pair is Option 1 (primary) and
Option 5 (fallback), since they share no hardware risk.
