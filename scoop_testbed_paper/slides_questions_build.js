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
    { l: 1, t: "If clock or pose errors were as coarse as the effects under study, every later number would be ambiguous. Pass: each term 10x below the breaking point of its kind (0.2 m pose; one LiDAR frame for time)" },
    { l: 1, t: "Q1.1 How far apart are the agents' clocks?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "chrony offset logs over coop_2, turned into displacement at 1 m/s" },
    { l: 3, label: "Answer:", t: "0.99 ms measured = 1 mm, 101x below one frame; strict bound 12 ms = 1.2 cm, 8x (short)", color: GREEN },
    { l: 1, t: "Q1.2 How well is an agent without LiDAR placed in the LiDAR frame?", b: true },
    { l: 3, label: "Setup:", color: SETUP, t: "held-out ChArUco boards predict where the board should appear; map built once, frozen" },
    { l: 3, label: "Answer:", t: "RGB-D cart 3.1 cm median (6.5x below 0.2 m, short of 10x); LiDAR robot 10 mm (coop2_0828); map repeats to 1.06 cm", color: GREEN },
    { l: 3, label: "Lacking:", t: "LiDAR robot vs boards on coop_2; radar and LiDAR-camera residuals; sniffer bias; measured p95 speed; rangefinder control distances", color: ORANGE },
  ], FS);
});




slides.push(s => {
  titleBand(s, "Q1 results: is it precise?");
  bookTable(s, "Each term against the breaking point of its kind: 0.2 m for pose, one LiDAR frame (0.1 m at 1 m/s) for time; pass at 10x", ["", "Quantity", "Value"], [2.7, 3.0, 3.1], [{"title": "Clock", "rows": [["Measured offset", "coop_2, chrony logs", "0.99 ms = 1 mm (101x)", "pass"], ["Strict chrony bound", "coop_2, offset + root delay/2", "12 ms = 1.2 cm (8.3x)", "short"], ["Sniffer bias", "probe stamps, both ends", "", "missing"]]}, {"title": "Map", "rows": [["Refinement", "final pose correction", "0.15 cm, 0.017 deg (133x)", "pass"], ["Repeatability", "two maps, four weeks apart", "1.06 cm, scale 0.99998 (19x)", "pass"], ["Control distances", "rangefinder, per site", "", "missing"]]}, {"title": "Agents in the frame", "rows": [["Board detector", "vs pipeline, ZED / RealSense", "8.1 / 8.3 mm (24x)", "pass"], ["RGB-D cart", "held-out boards, coop_2", "3.1 cm (6.5x), sigma 5.3 cm", "short"], ["LiDAR robot", "held-out boards, coop2_0828", "10.0 mm, 0.88 deg (20x)", "pass"], ["LiDAR robot", "held-out boards, coop_2", "", "missing"], ["Extrinsics", "radar; LiDAR-camera", "", "missing"], ["Agent speed", "p95 of reference tracks", "", "missing"]]}]);
});

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
  bookTable(s, "Map-derived Fresnel clearance against RSSI with the distance trend removed; pass: rho > 0.3 and clear-minus-obstructed drop > 5 dB", ["", "Quantity", "Value"], [2.7, 3.0, 3.1], [{"title": "Robot-to-robot CSI links (coop_2)", "rows": [["mobile_2 to sniffer", "Spearman rho, drop", "0.60, 5.4 dB", "pass"], ["infra_1 to sniffer", "Spearman rho, drop", "0.23, 4.7 dB", "short"]]}, {"title": "Agent-to-access-point links", "rows": [["RSSI and goodput", "needs AP position (ap.yaml)", "", "missing"], ["Pooled over passes and sites", "link_clearance.py --all", "", "missing"]]}, {"title": "Method", "rows": [["Self-test", "planted 12 dB wall", "recovered 11.9 dB, rho 0.68", "pass"], ["Confidence interval", "bootstrap over windows", "", "missing"], ["Out-of-run prediction", "fit survey_1, score coop_2", "", "missing"], ["CSI power", "second signal, same test", "", "missing"]]}]);
});

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
  bookTable(s, "Replay against live arrivals; pass: at least 90 % agreement on the receiver frame, closer than the synthetic channel", ["", "Quantity", "Value"], [2.7, 3.0, 3.1], [{"title": "Replay model", "rows": [["Arrival rule", "d(t) + s/G(t), drop in outage", "defined", ""], ["Inputs", "goodput, RTT, outages", "measured on every pass", ""]]}, {"title": "Live comparison", "rows": [["Payload node", "1 KB / 100 KB / 1.5 MB at 10 Hz, stamped", "", "missing"], ["Twin runs", "A1/B1 survey, A2/B2 cooperation", "", "missing"], ["Arrival-time error", "median, p95 per size and mode", "", "missing"], ["Frame agreement", "share landing in the same 100 ms frame", "", "missing"], ["Drop agreement", "per size and mode", "", "missing"]]}]);
});

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
  bookTable(s, "Pass: day-to-day distance below half the survey-to-cooperation distance, for RTT and goodput", ["", "Quantity", "Value"], [2.7, 3.0, 3.1], [{"title": "Pose half", "rows": [["Frame across days", "two maps, four weeks apart", "1.06 cm, scale 0.99998", "pass"]]}, {"title": "Link half", "rows": [["Mode distance", "survey_1 vs coop_2, W1 and KS", "", "missing"], ["Day distance", "same route and mode, second day", "", "missing"], ["Day / mode ratio", "RTT and goodput, pass < 0.5", "", "missing"]]}]);
});

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
  bookTable(s, "1.5 MB messages at 10 Hz through the logged link against the synthetic channels of prior benchmarks; demonstration", ["", "Quantity", "Value"], [2.7, 3.0, 3.1], [{"title": "Share arriving within one LiDAR frame", "rows": [["One robot", "survey_1", "86 %", "pass"], ["Two robots contending", "coop_2", "59 to 68 %", "short"], ["Synthetic channels", "V2X-ViT 100 ms; Where2comm none", "100 %", ""]]}, {"title": "Delay and outages", "rows": [["RTT idle to loaded", "coop_2", "5 to 10 ms to 25 to 33 ms", ""], ["Real outages", "coop2_0828", "3, covering 12 % of frames", ""]]}, {"title": "Strengthening", "rows": [["Best-fit synthetic baseline", "constant or Binomial delay fitted", "", "missing"], ["More passes, second site", "sim_gap.py", "", "missing"], ["Probe-off control run", "ping only, no iperf", "", "missing"]]}]);
});

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
    [c("Q1 Precise"), c("clock 0.99 ms (101x); map 0.15 / 1.06 cm; detector 8 mm; cart 3.1 cm; LiDAR robot 10 mm"), c("pass, 2 terms short", GREEN), c("mobile_1 on coop_2; radar; LiDAR-cam; sniffer bias; p95 speed; control distances")],
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
