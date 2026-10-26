// One slide: the digital twin built from real data, as editable PowerPoint
// shapes in the deck's style (blue title band, page number bottom right).
// The diagram takes the left half; the right half is left free.
//   node slides_media/digital_twin_slide.js slides_media/digital_twin_slide.pptx
const pptxgen = require("pptxgenjs");

const out = process.argv[2] || "digital_twin_slide.pptx";
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.author = "Jorich Loquero";
pres.title = "Digital Twin from Real Data";

const F = "Arial";
const INK = "1B2430", SOFT = "5A6472", ACCENT = "C2571A", CHECK = "2E7D4F";
const BAND = "009AD8", BAND2 = "5EC4EB", BAND3 = "D9F1FB";
const K = {               // fill, shadow, edge
  real:  ["E9EAEC", "DADCE0", "B9BEC6"],
  sim:   ["FBEFD9", "F1E1C2", "E2C58E"],
  label: ["DFF0E3", "CDE5D3", "9CCBA8"],
  data:  ["DDE8F6", "CCDAEE", "9DB7DD"],
};

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// title band
s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 10, h: 0.68, fill: { color: BAND }, line: { color: BAND }, objectName: "band" });
s.addShape(pres.ShapeType.rect, { x: 0, y: 0.68, w: 10, h: 0.06, fill: { color: BAND2 }, line: { color: BAND2 }, objectName: "band strip" });
s.addShape(pres.ShapeType.rect, { x: 0, y: 0.74, w: 10, h: 0.06, fill: { color: BAND3 }, line: { color: BAND3 }, objectName: "band strip 2" });
s.addText("Digital Twin from Real Data", { x: 0.3, y: 0.04, w: 9.4, h: 0.6, fontFace: F, fontSize: 28, bold: true,
  color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "title" });
s.addText("2/40", { x: 8.9, y: 5.28, w: 0.9, h: 0.28, fontFace: F, fontSize: 12, color: INK, align: "right",
  isTextBox: true, margin: 0, objectName: "page" });

function panel(x, y, w, h, kind, name) {
  const [fill, shadow, edge] = K[kind];
  s.addShape(pres.ShapeType.roundRect, { x: x + 0.03, y: y + 0.03, w, h, rectRadius: 0.07,
    fill: { color: shadow }, line: { color: shadow }, objectName: name + " shadow" });
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.07,
    fill: { color: fill }, line: { color: edge, width: 1 }, objectName: name });
}
function card(x, y, w, h, kind, name) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.04, fill: { color: "FFFFFF" },
    line: { color: K[kind][2], width: 0.9 }, objectName: name });
}
function text(runs, opts, name) {
  s.addText(runs, Object.assign({ fontFace: F, valign: "middle", isTextBox: true, margin: 0, objectName: name }, opts));
}
function dot(x, y, filled, name) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.09, h: 0.09, fill: { color: filled ? ACCENT : "FFFFFF" },
    line: { color: ACCENT, width: 1.1 }, objectName: name });
}
function twoLines(x, y, w, h, t, sub, name) {
  text([
    { text: t, options: { bold: true, fontSize: 9.5, color: INK, breakLine: true } },
    { text: sub, options: { fontSize: 8, color: SOFT } },
  ], { x, y, w, h }, name);
}
function down(x, y0, y1, name) {
  s.addShape(pres.ShapeType.line, { x, y: y0, w: 0, h: y1 - y0,
    line: { color: INK, width: 1.25, endArrowType: "triangle" }, objectName: name });
}

const X = 0.6, W = 4.35, PAD = 0.12, GAP = 0.12;
const cw2 = (W - 2 * PAD - 0.08) / 2;          // two cards across

// 1 real data
const y1 = 0.95, h1 = 0.98;
panel(X, y1, W, h1, "real", "real data");
text([
  { text: "1  Real data", options: { bold: true, fontSize: 12, color: INK } },
  { text: "   one recorded pass", options: { italic: true, fontSize: 9, color: SOFT } },
], { x: X + PAD, y: y1 + 0.03, w: W - 2 * PAD, h: 0.26 }, "real data title");
const items = [["Scanned building", "reference map, cm-level"], ["Robots and sensors", "real LiDARs, cameras, mounts"],
  ["Real trajectories", "where the robots drove"], ["Measured Wi-Fi link", "RTT, goodput, outages"]];
items.forEach(([t, sub], i) => {
  const ix = X + PAD + (i % 2) * (cw2 + 0.08), iy = y1 + 0.3 + Math.floor(i / 2) * 0.33;
  card(ix, iy, cw2, 0.3, "real", "real item " + (i + 1));
  dot(ix + 0.08, iy + 0.105, true, "real dot " + (i + 1));
  twoLines(ix + 0.24, iy + 0.02, cw2 - 0.27, 0.26, t, sub, "real item text " + (i + 1));
});

// 2 digital twin
const y2 = y1 + h1 + GAP, h2 = 2.1;
panel(X, y2, W, h2, "sim", "digital twin");
text("2  Digital Twin · Isaac Sim", { x: X + PAD, y: y2 + 0.03, w: W - 1.4, h: 0.26, fontSize: 12, bold: true,
  color: INK }, "digital twin title");
const knobs = [["Comms", "latency · loss · bandwidth", true], ["Sensor dropouts", "LiDAR or camera off", false],
  ["Pose source", "ground truth · noisy · SLAM", false], ["People and objects", "walkers · furniture", true],
  ["Robot paths", "new routes · speed", true], ["Sensor models", "LiDAR beams · resolution", true]];
knobs.forEach(([t, sub, built], i) => {
  const kx = X + PAD + (i % 2) * (cw2 + 0.08), ky = y2 + 0.32 + Math.floor(i / 2) * 0.36;
  card(kx, ky, cw2, 0.32, "sim", "knob " + t);
  dot(kx + 0.08, ky + 0.115, built, "knob dot " + t);
  twoLines(kx + 0.24, ky + 0.025, cw2 - 0.27, 0.27, t, sub, "knob text " + t);
});
// a deck of scenarios
const sy = y2 + 0.32 + 3 * 0.36 + 0.04, sw = W - 2 * PAD - 0.1, sh = 0.5;
card(X + PAD + 0.08, sy + 0.08, sw, sh, "sim", "scenario card 3");
card(X + PAD + 0.04, sy + 0.04, sw, sh, "sim", "scenario card 2");
card(X + PAD, sy, sw, sh, "sim", "scenario card 1");
text([
  { text: "Scenario 1", options: { bold: true, fontSize: 9.5, color: INK } },
  { text: "  4 people · 30% loss · camera off · SLAM poses", options: { fontSize: 8, color: SOFT, breakLine: true } },
  { text: "✓ ", options: { bold: true, fontSize: 8.5, color: CHECK } },
  { text: "sensor data   ", options: { fontSize: 8.5, color: INK } },
  { text: "✓ ", options: { bold: true, fontSize: 8.5, color: CHECK } },
  { text: "GT boxes   ", options: { fontSize: 8.5, color: INK } },
  { text: "✓ ", options: { bold: true, fontSize: 8.5, color: CHECK } },
  { text: "link per frame", options: { fontSize: 8.5, color: INK } },
], { x: X + PAD + 0.1, y: sy + 0.04, w: sw - 0.65, h: sh - 0.08 }, "scenario text");
text("× N", { x: X + PAD + sw - 0.55, y: sy, w: 0.48, h: sh, fontSize: 14, bold: true, color: ACCENT, align: "right" },
  "n scenarios");
// legend for the dots, top right of the panel
dot(X + W - 1.18, y2 + 0.115, true, "legend built dot");
text("built", { x: X + W - 1.06, y: y2 + 0.06, w: 0.35, h: 0.2, fontSize: 8, color: SOFT }, "legend built");
dot(X + W - 0.7, y2 + 0.115, false, "legend planned dot");
text("planned", { x: X + W - 0.58, y: y2 + 0.06, w: 0.5, h: 0.2, fontSize: 8, color: SOFT }, "legend planned");

// 3 train, 4 test on real
const y3 = y2 + h2 + GAP, h3 = 0.42;
panel(X, y3, W, h3, "data", "train");
text([
  { text: "3  Train", options: { bold: true, fontSize: 12, color: INK } },
  { text: "   on digital-twin scenarios", options: { fontSize: 9.5, color: INK } },
], { x: X + PAD, y: y3, w: W - 0.9, h: h3 }, "train text");
text("next step", { x: X + W - 0.85, y: y3, w: 0.73, h: h3, fontSize: 8.5, italic: true, color: SOFT, align: "right" },
  "train status");

const y4 = y3 + h3 + GAP, h4 = 0.42;
panel(X, y4, W, h4, "label", "test on real");
text([
  { text: "4  Test on real", options: { bold: true, fontSize: 12, color: INK } },
  { text: "   held-out real recording", options: { fontSize: 9.5, color: INK } },
], { x: X + PAD, y: y4, w: W - 2 * PAD, h: h4 }, "test text");

// arrows down the middle
const mid = X + W / 2;
down(mid, y1 + h1, y2, "arrow real to twin");
down(mid, y2 + h2, y3, "arrow twin to train");
down(mid, y3 + h3, y4, "arrow train to test");

// held-out real recording, down the left edge
const lx = 0.42, top = y1 + 0.45, bot = y4 + h4 / 2;
const dash = { color: SOFT, width: 1.1, dashType: "dash" };
s.addShape(pres.ShapeType.line, { x: lx, y: top, w: X - lx, h: 0, line: dash, objectName: "held-out top" });
s.addShape(pres.ShapeType.line, { x: lx, y: top, w: 0, h: bot - top, line: dash, objectName: "held-out down" });
s.addShape(pres.ShapeType.line, { x: lx, y: bot, w: X - lx, h: 0, line: Object.assign({ endArrowType: "triangle" }, dash),
  objectName: "held-out in" });
text("held-out real recording", { x: 0.15, y: 2.1, w: 0.22, h: 2.0, fontSize: 8.5, italic: true, color: SOFT,
  align: "center", vert: "vert270" }, "held-out label");

s.addNotes("Start from one real recorded pass: the scanned building, our real robots and sensor mounts, the trajectories they drove, and the Wi-Fi link we measured. The digital twin in Isaac Sim keeps all of that real and lets us change one thing at a time: the comms (latency, loss, bursts, bandwidth), sensor dropouts, where the poses come from (ground truth, noisy, or a SLAM algorithm's output), people and furniture, robot paths, and sensor models. Each setting is a scenario that comes with sensor data from every robot, ground-truth boxes for free, and the link per frame, so we can make as many training scenarios as we want. The models are then tested on a held-out real recording with its own measured link, so sim-trained and real-trained models can be compared on the same scene. Built today: comms, people and objects, robot paths, sensor models. Planned: sensor dropouts, SLAM as the pose source, and the training step itself.");

pres.writeFile({ fileName: out }).then(f => console.log(f));
