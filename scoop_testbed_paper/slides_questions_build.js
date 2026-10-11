const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.author = "Jorich Loquero";
pres.title = "SCooP-iT Testbed Questions";

const SETUP = "444444";
const CYAN = "1A9FD9", CYAN_L = "BFE6F6", GREEN = "2E7D32", ORANGE = "E67E22", GREY = "7F8C8D", TXT = "222222";
const F = "Arial";

function titleBand(slide, text) {
  slide.background = { color: "FFFFFF" };
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 10, h: 0.75, fill: { color: CYAN }, line: { color: CYAN } });
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0.75, w: 10, h: 0.06, fill: { color: CYAN_L }, line: { color: CYAN_L } });
  slide.addText(text, { x: 0.3, y: 0.05, w: 9.4, h: 0.65, fontFace: F, fontSize: 26, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "title" });
}
function footer(slide, n, total) {
  slide.addText(`${n}/${total}`, { x: 8.6, y: 5.25, w: 1.2, h: 0.3, fontFace: F, fontSize: 10, color: GREY, align: "right", isTextBox: true, margin: 0, objectName: "page" });
}
// Bullet hierarchy: L0 green square, L1 orange diamond, L2 arrow; "Setup:" "Answer:" "Lacking:" lines as L3 plain
function body(slide, items, fs) {
  const runs = [];
  items.forEach((it, i) => {
    const lvl = it.l || 0;
    const last = i === items.length - 1;
    const col = it.color || TXT;
    // paragraph-level options must sit on EVERY run of the paragraph
    const para = lvl === 0 ? { bullet: { code: "25A0", indent: 18 }, indentLevel: 0, paraSpaceAfter: 5 }
             : lvl === 1 ? { bullet: { code: "25C6", indent: 18 }, indentLevel: 1, paraSpaceAfter: 3 }
             : lvl === 2 ? { bullet: { code: "25BA", indent: 18 }, indentLevel: 2, paraSpaceAfter: 2 }
             :             { bullet: { code: "2002", indent: 14 }, indentLevel: 3, paraSpaceAfter: 2 };
    const size = lvl === 0 ? fs.l0 : lvl === 1 ? fs.l1 : lvl === 2 ? fs.l2 : fs.l3;
    const base = () => ({ fontFace: F, fontSize: size, ...JSON.parse(JSON.stringify(para)) });
    if (it.label) {
      // pptxgenjs takes paragraph props from the last run and starts a new
      // paragraph on any bulleted run, so a label + text pair cannot share one
      // bulleted paragraph: keep each labelled line as a single run.
      runs.push({ text: it.label + " " + it.t, options: { ...base(), bold: false, color: col, breakLine: !last } });
    } else {
      runs.push({ text: it.t, options: { ...base(), bold: lvl === 0 || !!it.b, color: col, breakLine: !last } });
    }
  });
  slide.addText(runs, { x: 0.45, y: 0.98, w: 9.1, h: 4.2, valign: "top", isTextBox: true, margin: 0, objectName: "body" });
}


// Booktabs-style table: three columns, italic group rows, indented terms,
// "--" where a value is still missing (grey), pass green / short orange.
function bookTable(slide, caption, cols, colW, groups) {
  const none = { type: "none" };
  const rule = (pt) => ({ type: "solid", pt, color: "333333" });
  const cell = (t, o) => ({ text: t, options: { fontFace: F, fontSize: 11, color: TXT, valign: "middle", border: [none, none, none, none], ...(o || {}) } });
  const rows = [];
  rows.push(cols.map((c, i) => cell(c, { bold: true, border: [rule(1.25), none, rule(0.75), none] })));
  groups.forEach(g => {
    rows.push([cell(g.title, { italic: true, bold: true, colspan: cols.length })]);
    g.rows.forEach(r => {
      const [term, qty, val, st] = r;
      const col = st === "pass" ? GREEN : st === "short" ? ORANGE : st === "missing" ? GREY : TXT;
      rows.push([cell("      " + term), cell(qty, { color: "444444" }), cell(val === "" ? "--" : val, { color: col, bold: st === "pass" || st === "short" })]);
    });
  });
  // bottom rule on the last row
  rows[rows.length - 1] = rows[rows.length - 1].map(c => ({ ...c, options: { ...c.options, border: [none, none, rule(1.25), none] } }));
  slide.addText(caption, { x: 0.6, y: 0.95, w: 8.8, h: 0.35, fontFace: F, fontSize: 12, color: "444444", align: "center", isTextBox: true, margin: 0, objectName: "caption" });
  slide.addTable(rows, { x: 0.6, y: 1.35, w: 8.8, colW, rowH: 0.23, margin: 0.02, fontFace: F, objectName: "results" });
}


function metricsSlide(s, title, items) {
  titleBand(s, title);
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    runs.push({ text: it[0] + ": ", options: { fontFace: F, fontSize: 13, bold: true, color: TXT, bullet: { code: "25C6", indent: 16 }, indentLevel: 0, paraSpaceAfter: 5, breakLine: false } });
    runs.push({ text: it[1], options: { fontFace: F, fontSize: 13, color: TXT, bullet: { code: "25C6", indent: 16 }, indentLevel: 0, paraSpaceAfter: 5, breakLine: !last } });
  });
  // pptxgenjs: paragraph props come from the last run; a bulleted non-first run would start a new
  // paragraph, so merge each pair into one run instead (bold label is lost; kept as "Name." prefix).
  const merged = [];
  for (let i = 0; i < runs.length; i += 2) merged.push({ text: runs[i].text + runs[i + 1].text, options: { ...runs[i + 1].options } });
  s.addText(merged, { x: 0.45, y: 0.98, w: 9.1, h: 4.2, valign: "top", isTextBox: true, margin: 0, objectName: "metrics" });
}

const FS = { l0: 17, l1: 15, l2: 14, l3: 13 };
const slides = [];

// ---- 1 overview
slides.push(s => {
  titleBand(s, "SCooP-iT Testbed Questions");
  body(s, [
    { l: 0, t: "A testbed paper is judged on seven questions" },
    { l: 1, t: "Trust: can readers rely on the instrument?", b: true },
    { l: 2, label: "Q1 Precise.", t: "Are its own errors small next to the effects it measures? (E1 Headroom)" },
    { l: 2, label: "Q2 Consistent.", t: "Do the poses and the link logs agree when neither was checked by hand? (E2 Map Meets Radio)" },
    { l: 2, label: "Q3 Faithful.", t: "Does replaying a logged link reproduce what the live link did? (E3 Twin Run)" },
    { l: 2, label: "Q4 Repeatable.", t: "Does the same setting give the same link on another day? (E4 Same Knob, Same Link)" },
    { l: 2, label: "Q5 Necessary.", t: "What does a measured link show that a synthetic one hides? (E5 Sim Gap)" },
    { l: 1, t: "Worth: should anyone care?", b: true },
    { l: 2, label: "Q6 Unique.", t: "What can it do that no existing platform can?" },
    { l: 2, label: "Q7 Reproducible.", t: "Can someone else build and use it?" },
    { l: 1, t: "Pass criteria fixed before the data were read; answers from the 24 Sep session (mapping_A, survey_1, coop_2)" },
  ], FS);
});

// ---- 2 Q1
slides.push(s => {
  titleBand(s, "Q1. Is it precise?");
  body(s, [
    { l: 0, t: "Are the testbed's own errors small next to the effects it measures?" },
    { l: 1, t: "If clock or pose errors were as coarse as the effects under study, every later number would be ambiguous. Thresholds: a tenth of where fusion breaks, 2 cm for pose terms, 10 ms for time terms" },
    { l: 1, t: "Q1.1 How far apart are the agents' clocks?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "chrony offset logs over coop_2, turned into displacement at 1 m/s" },
    { l: 3, label: "Answer:", t: "0.99 ms measured, under the 10 ms threshold; strict bound 12 ms, above it", color: GREEN },
    { l: 1, t: "Q1.2 How well is an agent without LiDAR placed in the LiDAR frame?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "held-out ChArUco boards predict where the board should appear; map built once, frozen" },
    { l: 3, label: "Answer:", t: "RGB-D cart 3.1 cm median, above the 2 cm threshold; LiDAR robot 10 mm (coop2_0828); map repeats to 1.06 cm", color: GREEN },
    { l: 3, label: "Lacking:", t: "LiDAR robot vs boards on coop_2; radar and LiDAR-camera residuals; sniffer bias; measured p95 speed; rangefinder control distances", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q1 results: is it precise?");
  bookTable(s, "Thresholds are a tenth of where fusion breaks: 2 cm for pose terms (0.2 m), 10 ms for time terms (one LiDAR frame)", ["", "Threshold", "Value"], [3.9, 1.5, 3.4], [
    {"title": "Clock", "rows": [
      ["Measured offset (coop_2)", "10 ms", "0.99 ms", "pass"],
      ["Strict chrony bound (coop_2)", "10 ms", "12 ms", "short"],
      ["Sniffer bias (probe stamps)", "10 ms", "", "missing"]]},
    {"title": "Map", "rows": [
      ["Refinement, final correction (mapping_A)", "2 cm", "0.15 cm, 0.017 deg", "pass"],
      ["Repeatability, two maps four weeks apart", "2 cm", "1.06 cm, scale 0.99998", "pass"],
      ["Control distances (rangefinder, per site)", "2 cm", "", "missing"]]},
    {"title": "Agents in the frame", "rows": [
      ["Board detector vs pipeline, ZED / RealSense", "2 cm", "8.1 / 8.3 mm", "pass"],
      ["RGB-D cart vs held-out boards (coop_2)", "2 cm", "3.1 cm median, sigma 5.3 cm", "short"],
      ["LiDAR robot vs held-out boards (coop2_0828)", "2 cm", "10.0 mm, 0.88 deg", "pass"],
      ["LiDAR robot vs held-out boards (coop_2)", "2 cm", "", "missing"],
      ["Extrinsics: radar; LiDAR-camera", "2 cm", "", "missing"],
      ["Agent speed, p95 of reference tracks", "input", "", "missing"]]}]);
});


slides.push(s => metricsSlide(s, "Q1 metrics explained: clock and map", [["Measured clock offset", "how far a client's clock sat from mobile_1's at each chrony poll over the run; the worst value is quoted. At 1 m/s, 1 ms of clock error moves a measurement by 1 mm"], ["Strict chrony bound", "chrony's guaranteed worst case: the offset plus half the network round-trip to the server. Always larger than the measured offset; it is what you quote if you refuse to trust the measurement itself"], ["Sniffer bias", "a constant error NTP cannot see when the path to the server is asymmetric (the CSI Pi syncs over a slow 2.4 GHz link). Measured by stamping the same packet at the sender and at the sniffer"], ["Final pose correction", "how much the last refinement iteration still moved a scan against the map. Small means the registration converged; it shows self-consistency, not accuracy"], ["Repeatability", "wall surfaces of two maps built independently, four weeks apart; 1.06 cm is how far apart the two maps put the same walls. Scale 0.99998 is the size ratio between them"], ["Control distances", "wall-to-wall distances from a laser rangefinder compared with the same distances read in the map. The only check of absolute scale, since both maps could be wrong the same way"], ["Time threshold 10 ms, pose threshold 2 cm", "one tenth of where fusion breaks: 100 ms (one LiDAR frame, every fusion method below ego-only) and 0.2 m (late fusion below ego-only)"]]));

slides.push(s => metricsSlide(s, "Q1 metrics explained: agents in the frame", [["ChArUco board", "a printed checkerboard with markers whose pose a camera can compute. Each board's pose in the map is fixed once, during the mapping session"], ["Board detector check", "re-detecting the mapping session's boards must reproduce the pipeline's own board poses. 8 mm says the instrument used in the next rows is finer than what it checks"], ["Held-out board residual", "from the agent's reference trajectory and its camera extrinsic, predict where a board should appear, then compare with where the camera saw it. Only boards that did not enter the trajectory estimate count, so the check is independent"], ["Median and sigma", "median over all board sightings in a run; sigma is the pose graph's own per-pose uncertainty (marginal covariance). 284 cm without map fixes is the same cart with the map registration switched off"], ["LiDAR robot vs RGB-D cart", "the LiDAR robot's reference uses no boards at all, so its 10 mm is fully independent; the cart's trajectory uses some boards, so it is scored on the held-out ones"], ["Agent speed, p95", "the 95th-percentile speed of the reference tracks. It converts a time error into a position error; 1 m/s is assumed until measured"]]));

// ---- 3 Q2
slides.push(s => {
  titleBand(s, "Q2. Is it consistent?");
  body(s, [
    { l: 0, t: "Do the poses and the link logs agree when neither was checked by hand?" },
    { l: 1, t: "The map and the poses never saw the radio; the radio never saw the map. If map geometry predicts the signal, both are validated at once" },
    { l: 1, t: "Q2.1 Does Fresnel clearance through the map predict RSSI?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "per 0.5 s window, clearance of the link path through the anchored map vs RSSI with the distance trend removed (fixed exponent). Pass: rho > 0.3 and clear-minus-obstructed drop > 5 dB" },
    { l: 3, label: "Answer:", t: "mobile_2 CSI link rho 0.60, 5.4 dB: pass. infra_1 link rho 0.23, 4.7 dB: short. Self-test recovers a planted 12 dB wall as 11.9 dB", color: GREEN },
    { l: 1, t: "Q2.2 Does it hold on the links the benchmark replays?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "agent-to-access-point RSSI and goodput, pooled over survey_1 and coop_2" },
    { l: 3, label: "Lacking:", t: "AP position in the map (ap.yaml); pooled rho with a confidence interval; out-of-run prediction (fit survey_1, score coop_2); CSI power as second signal", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q2 results: is it consistent?");
  bookTable(s, "Map-derived Fresnel clearance against RSSI with the distance trend removed", ["", "Threshold", "Value"], [3.9, 2.3, 2.6], [
    {"title": "Robot-to-robot CSI links (coop_2)", "rows": [
      ["mobile_2 to sniffer: rho, drop", "rho > 0.3, drop > 5 dB", "0.60, 5.4 dB", "pass"],
      ["infra_1 to sniffer: rho, drop", "rho > 0.3, drop > 5 dB", "0.23, 4.7 dB", "short"]]},
    {"title": "Agent-to-access-point links", "rows": [
      ["RSSI and goodput (needs AP position)", "rho > 0.3, drop > 5 dB", "", "missing"],
      ["Pooled over passes and sites", "rho > 0.3, drop > 5 dB", "", "missing"]]},
    {"title": "Method", "rows": [
      ["Self-test, planted 12 dB wall", "recovers 12 dB", "11.9 dB, rho 0.68", "pass"],
      ["Confidence interval on rho", "lower bound > 0.3", "", "missing"],
      ["Out-of-run prediction (fit survey_1, score coop_2)", "rho > 0.3", "", "missing"],
      ["CSI power as second signal", "same", "", "missing"]]}]);
});


slides.push(s => metricsSlide(s, "Q2 metrics explained", [["Fresnel clearance", "along the straight line between two radios, the smallest free distance to the nearest map point, divided by the first Fresnel radius at that point. 1 or more: the zone the signal needs is free; below 0.6: obstructed; near 0: the line crosses a wall"], ["Detrended RSSI", "received signal strength minus the free-space distance trend (fixed exponent 2, offset taken from clear windows). What remains is the part obstruction explains, not distance"], ["Spearman rho", "rank correlation between clearance and detrended RSSI over 0.5 s windows. 0 means no relation, 1 a perfect monotone one; above 0.3 is the pre-set bar"], ["Clear-minus-obstructed drop", "median detrended RSSI in clear windows minus the median in obstructed windows, in dB. Above 5 dB means the wall is visible in the radio"], ["Self-test", "synthetic data with a 12 dB wall planted; the method recovers 11.9 dB, so it measures what it claims"], ["Why it validates two things", "the map and poses produced the clearance; the radio produced the RSSI; neither used the other. Agreement means both are right together"], ["Out-of-run prediction", "fit the detrending offset on one pass, score another: shows the geometry predicts a run it was not tuned on"]]));

// ---- 4 Q3
slides.push(s => {
  titleBand(s, "Q3. Is it faithful?");
  body(s, [
    { l: 0, t: "Does replaying a logged link reproduce what the live link did?" },
    { l: 1, t: "Replay is what the testbed offers every method. If replay does not match reality, the logged channel is just another simulation. This is the headline claim" },
    { l: 1, t: "Q3.1 Do replayed arrivals match live arrivals?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "twin runs on the survey_1 route: run A logs the link (iperf, ping); run B sends stamped 1 KB / 100 KB / 1.5 MB messages at 10 Hz. Replay B through A's trace: arrival = send + d(t) + s/G(t), dropped in outage" },
    { l: 3, label: "Pass:", color: SETUP, t: "at least 90 % agreement on which 100 ms frame each message lands in, and closer than the synthetic channel" },
    { l: 3, label: "Answer:", t: "none yet. The model and its measured inputs (goodput, RTT, outages) exist; no live messages to compare against", color: GREEN },
    { l: 3, label: "Lacking:", t: "payload node; one session (A1/B1 survey, A2/B2 cooperation, back to back); twin_run.py; per-size, per-mode table of arrival error, frame agreement, drop agreement", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q3 results: is it faithful?");
  bookTable(s, "Replay against live arrivals on twin runs (A logs the link, B sends stamped messages)", ["", "Threshold", "Value"], [3.4, 2.4, 3.0], [
    {"title": "Replay model", "rows": [
      ["Arrival rule d(t) + s/G(t), drop in outage", "", "defined", ""],
      ["Inputs: goodput, RTT, outages", "", "measured on every pass", ""]]},
    {"title": "Live comparison (per size and mode)", "rows": [
      ["Frame agreement", ">= 90 %", "", "missing"],
      ["Arrival-time error, median and p95", "below synthetic channel", "", "missing"],
      ["Drop agreement", "", "", "missing"],
      ["Payload node and twin runs", "A1/B1 survey, A2/B2 coop", "", "missing"]]}]);
});


slides.push(s => metricsSlide(s, "Q3 metrics explained", [["Replay rule", "a message of size s sent at time t arrives at t + d(t) + s / G(t): one-way delay plus transmission time at the goodput measured at that moment; dropped if the link was in an outage"], ["d(t) and G(t)", "d is the one-way delay (RTT/2, or a stamped probe where available); G is the goodput of the most recent iperf test on that link"], ["Twin runs", "the same route driven twice the same day: run A logs the link as usual; run B sends stamped messages of 1 KB, 100 KB, 1.5 MB at 10 Hz. B's messages are replayed through A's log and compared with when they really arrived"], ["Frame agreement", "share of messages for which replay and live put the arrival in the same 100 ms LiDAR frame. The pass bar is 90 %"], ["Arrival-time error", "replayed minus live arrival time per message; reported as median and 95th percentile, per size and mode"], ["Drop agreement", "share of messages on which replay and live agree about lost versus delivered"], ["Closer than synthetic", "the same comparison with the published synthetic channel must come out worse than the replay"]]));

// ---- 5 Q4
slides.push(s => {
  titleBand(s, "Q4. Is it repeatable?");
  body(s, [
    { l: 0, t: "Does the same setting give the same link on another day?" },
    { l: 1, t: "A testbed setting is useful only if it repeats; otherwise \"cooperation mode\" is not a controlled condition" },
    { l: 1, t: "Q4.1 Does the frame repeat?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "two maps of the same site, four weeks apart, processed independently" },
    { l: 3, label: "Answer:", t: "1.06 cm wall placement, scale within 2e-5", color: GREEN },
    { l: 1, t: "Q4.2 Does the link repeat?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "Wasserstein and KS distance on RTT, goodput, RSSI (and RSSI on shared map cells) between two days of one mode, against the distance between survey and cooperation mode. Pass: day/mode ratio below 0.5" },
    { l: 3, label: "Answer:", t: "untested. Mode distance (survey_1 vs coop_2) computable from data in hand: link_repeat.py", color: GREEN },
    { l: 3, label: "Lacking:", t: "run link_repeat.py on survey_1 vs coop_2; one repeat day on the same route (the Q3 session provides it); day-vs-mode table", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q4 results: is it repeatable?");
  bookTable(s, "Same setting, another day", ["", "Threshold", "Value"], [3.4, 2.4, 3.0], [
    {"title": "Pose half", "rows": [
      ["Frame across days, two maps four weeks apart", "2 cm", "1.06 cm, scale 0.99998", "pass"]]},
    {"title": "Link half", "rows": [
      ["Mode distance, survey_1 vs coop_2 (W1, KS)", "reference", "", "missing"],
      ["Day distance, same route and mode", "", "", "missing"],
      ["Day / mode ratio, RTT and goodput", "< 0.5", "", "missing"]]}]);
});


slides.push(s => metricsSlide(s, "Q4 metrics explained", [["W1, Wasserstein-1 distance", "the average shift between two distributions, in the unit measured (ms for RTT, Mbit/s for goodput, dB for RSSI). It answers by how much the link changed"], ["KS statistic", "the largest gap between two cumulative distributions, from 0 to 1. It answers whether the shapes differ at all, regardless of how much"], ["Mode distance", "survey_1 against coop_2 on the same day: the distance one knob (one sender versus all senders) produces. It is the reference the day distance is compared with"], ["Day distance", "the same route and the same mode on two different days"], ["Day / mode ratio", "pass when below 0.5: a day changes the link by less than half of what the mode changes it, so the mode is a controlled condition"], ["RSSI on grid", "per-cell median RSSI in the map frame, compared only on cells both passes occupied, so a different route is not counted as a different link"], ["Pose half", "the two maps four weeks apart are the same test for the frame instead of the link"]]));

// ---- 6 Q5
slides.push(s => {
  titleBand(s, "Q5. Is it necessary?");
  body(s, [
    { l: 0, t: "What does a measured link show that a synthetic one hides?" },
    { l: 1, t: "Every collaborative-perception benchmark models the channel synthetically: constant or i.i.d. delay, quantized to whole frames, no outages. If that described the real link, nobody would need to measure it" },
    { l: 1, t: "Q5.1 How much benefit is left when everyone talks at once?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "1.5 MB messages at 10 Hz through the logged link, delay = RTT/2 + size/goodput, lost in outages; one sender (survey_1) vs two robots (coop_2); synthetic channels as configured in their code" },
    { l: 3, label: "Answer:", t: "within one LiDAR frame: 86 % alone, 59 to 68 % crowded, 100 % under V2X-ViT (constant 100 ms) and Where2comm (no latency); RTT 5 to 10 ms idle, 25 to 33 ms loaded; 3 real outages, 12 % of frames (coop2_0828)", color: GREEN },
    { l: 1, t: "Q5.2 Would a tuned synthetic channel do?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "constant or Binomial delay fitted to the measured delays; same in-frame share" },
    { l: 3, label: "Lacking:", t: "best-fit synthetic baseline; more passes and a second site; probe-off control run (iperf itself loads the link)", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q5 results: is it necessary?");
  bookTable(s, "1.5 MB messages at 10 Hz through the logged link against the synthetic channels of prior benchmarks", ["", "Reference", "Value"], [4.0, 1.9, 2.9], [
    {"title": "Share arriving within one LiDAR frame (100 ms)", "rows": [
      ["One robot (survey_1)", "100 % synthetic", "86 %", "pass"],
      ["Two robots contending (coop_2)", "100 % synthetic", "59 to 68 %", "short"],
      ["Synthetic: V2X-ViT 100 ms; Where2comm none", "", "100 %", ""]]},
    {"title": "Delay and outages", "rows": [
      ["RTT idle to loaded (coop_2)", "100 ms frame", "5 to 10 ms to 25 to 33 ms", ""],
      ["Real outages (coop2_0828)", "none in synthetic", "3, covering 12 % of frames", ""]]},
    {"title": "Strengthening", "rows": [
      ["Best-fit synthetic baseline, fitted delay", "in-frame share", "", "missing"],
      ["More passes, second site", "", "", "missing"],
      ["Probe-off control run (ping only)", "", "", "missing"]]}]);
});


slides.push(s => metricsSlide(s, "Q5 metrics explained", [["In-frame share", "percentage of 1.5 MB messages, one every 100 ms, whose delay RTT/2 + size/goodput is at most 100 ms and that were not sent during an outage. 100 ms is one LiDAR frame"], ["Synthetic channel share", "what the benchmarks' own settings imply: V2X-ViT adds a constant 100 ms then rounds down to whole frames, so every message counts as on time; Where2comm has no latency model at all. Both give 100 %"], ["One robot vs two robots", "survey_1 has one sender owning the channel; coop_2 has both robots contending at the same access point. The drop from 86 % to 59 to 68 % is contention, which i.i.d. per-agent delay models cannot express"], ["RTT idle vs loaded", "ping round-trip time with no traffic and with iperf running. Delay depends on offered load, which a constant delay cannot represent"], ["Outage", "a run of unanswered pings lasting at least one frame; messages sent inside it are lost. Synthetic channels have none"], ["Best-fit synthetic baseline", "a constant or Binomial delay tuned to the measured delays; if it still mispredicts the in-frame share, the gap is not a matter of parameter choice"], ["Probe-off control", "a run with ping only, no iperf, to separate the queueing the measurement adds from the queueing the team would face"]]));

// ---- 7 Q6 + Q7
slides.push(s => {
  titleBand(s, "Q6. Is it unique?  Q7. Is it reproducible?");
  body(s, [
    { l: 0, t: "Q6 What can it do that no existing platform can?" },
    { l: 3, label: "Setup:", color: SETUP, t: "capability table: indoor; heterogeneous incl. infrastructure; pose reference without mocap or GNSS; every pair's link measured; CSI; clock bound reported; replay into methods; perception scoring" },
    { l: 3, label: "Answer:", t: "SCooP-iT fills all eight columns. Nearest: CooperScene (outdoor C-V2X, no CSI, replay unknown), CoPeD (heterogeneous, estimator pose, no link)", color: GREEN },
    { l: 3, label: "Lacking:", t: "verify every cell against the cited paper", color: ORANGE },
    { l: 0, t: "Q7 Can someone else build and use it?" },
    { l: 3, label: "Setup:", color: SETUP, t: "bill of materials, setup effort per site, released pipeline, spec sheet with every bound" },
    { l: 3, label: "Answer:", t: "pipeline released with the analysis tools for E1, E2, E4, E5; spec sheet drafted; component list per agent class", color: GREEN },
    { l: 3, label: "Lacking:", t: "cost per agent class; mapping time, boards, control distances, software versions per site; payload node and twin_run.py", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q6 and Q7 results: unique, reproducible?");
  bookTable(s, "Q6: fills every capability column no other platform fills together. Q7: bill of materials, setup effort, pipeline, spec sheet", ["", "Nearest other / how", "SCooP-iT"], [3.4, 3.4, 2.0], [{"title": "Q6 Capability", "rows": [["Indoor, no mocap or GNSS", "CoPeD (estimator pose only)", "yes", "pass"], ["Heterogeneous incl. infrastructure", "CoPeD, TUMTraf-V2X", "yes", "pass"], ["Every pair's link on the sensors' clock", "CooperScene (C-V2X, per scene)", "yes", "pass"], ["Channel state (CSI)", "none", "yes", "pass"], ["Clock bound; replay into methods", "CooperScene (no bound); simulators", "yes", "pass"], ["Cells verified against cited papers", "CooperScene replay, probe rate unknown", "", "missing"]]}, {"title": "Q7 Reproducibility", "rows": [["Pipeline and E1/E2/E4/E5 tools", "released", "yes", "pass"], ["Spec sheet, component list", "drafted", "yes", "pass"], ["Cost per agent class", "bookkeeping", "", "missing"], ["Per-site effort, software versions", "bookkeeping", "", "missing"]]}]);
});

// ---- 8 status table
slides.push(s => {
  titleBand(s, "Where each question stands (24 Sep)");
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: CYAN }, fontFace: F, fontSize: 13, align: "left", valign: "middle" } });
  const c = (t, col) => ({ text: t, options: { fontFace: F, fontSize: 11, color: col || TXT, valign: "middle" } });
  const rows = [
    [hdr("Question"), hdr("Answer now"), hdr("Verdict"), hdr("Still lacking")],
    [c("Q1 Precise"), c("clock 0.99 ms; map 0.15 / 1.06 cm; detector 8 mm; cart 3.1 cm; LiDAR robot 10 mm"), c("pass, 2 terms short", GREEN), c("mobile_1 on coop_2; radar; LiDAR-cam; sniffer bias; p95 speed; control distances")],
    [c("Q2 Consistent"), c("rho 0.60, 5.4 dB on mobile_2 link"), c("pass on one link", GREEN), c("AP links; pooling; CI; out-of-run; CSI power")],
    [c("Q3 Faithful"), c("model and inputs only"), c("no data", ORANGE), c("payload node; twin-run session; comparison table")],
    [c("Q4 Repeatable"), c("frame repeats: 1.06 cm, 2e-5"), c("half", ORANGE), c("mode distance (tool ready); repeat day")],
    [c("Q5 Necessary"), c("86 % / 59-68 % / 100 % synthetic; RTT 5-10 to 25-33 ms"), c("shown", GREEN), c("tuned synthetic baseline; more passes; probe-off run")],
    [c("Q6 Unique"), c("all 8 capability columns"), c("pass", GREEN), c("verify cells")],
    [c("Q7 Reproducible"), c("pipeline, tools, spec sheet, component list"), c("partial", ORANGE), c("costs; per-site effort; payload node")],
  ];
  s.addTable(rows, { x: 0.4, y: 1.0, w: 9.2, colW: [1.35, 3.6, 1.35, 2.9], border: { type: "solid", pt: 0.5, color: "CCCCCC" }, rowH: 0.36, fontFace: F, objectName: "status", margin: 0.04 });
  s.addText("Green: answered on the pre-set criterion. Orange: needs data. One field session (twin runs on the survey_1 route) closes Q3, the Q4 day half, and the sniffer bias.",
    { x: 0.4, y: 4.95, w: 8.1, h: 0.3, fontFace: F, fontSize: 10, color: GREY, italic: true, isTextBox: true, margin: 0, objectName: "note" });
});

const total = slides.length;
slides.forEach((fn, i) => { const s = pres.addSlide(); fn(s); footer(s, i + 1, total); });
pres.writeFile({ fileName: "SCooP-iT_testbed_questions.pptx" }).then(f => console.log("wrote", f));
