// One slide: the digital twin built from real data, as editable PowerPoint
// shapes in the deck's style (blue title band, page number bottom right).
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
  s.addShape(pres.ShapeType.roundRect, { x: x + 0.04, y: y + 0.04, w, h, rectRadius: 0.08,
    fill: { color: shadow }, line: { color: shadow }, objectName: name + " shadow" });
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: edge, width: 1 }, objectName: name });
}
function card(x, y, w, h, kind, name, dashed) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05, fill: { color: "FFFFFF" },
    line: { color: K[kind][2], width: 0.9, dashType: dashed ? "dash" : "solid" }, objectName: name });
}
function heading(x, y, text, sub, name) {
  s.addText([
    { text: text, options: { bold: true, fontSize: 13, color: INK } },
    { text: "    " + sub, options: { italic: true, fontSize: 10, color: SOFT } },
  ], { x, y, w: 7, h: 0.28, fontFace: F, valign: "middle", isTextBox: true, margin: 0, objectName: name });
}
function dot(x, y, filled, name) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.11, h: 0.11, fill: { color: filled ? ACCENT : "FFFFFF" },
    line: { color: ACCENT, width: 1.25 }, objectName: name });
}
function twoLines(x, y, w, h, t, sub, name, size) {
  s.addText([
    { text: t, options: { bold: true, fontSize: size || 10.5, color: INK, breakLine: true } },
    { text: sub, options: { fontSize: 9, color: SOFT } },
  ], { x, y, w, h, fontFace: F, valign: "middle", isTextBox: true, margin: 0, objectName: name });
}
function down(x, y0, y1, name) {
  s.addShape(pres.ShapeType.line, { x, y: y0, w: 0, h: y1 - y0,
    line: { color: INK, width: 1.5, endArrowType: "triangle" }, objectName: name });
}

const X = 0.6, W = 8.95;

// 1 real data
const y1 = 0.95, h1 = 0.8;
panel(X, y1, W, h1, "real", "real data");
heading(X + 0.15, y1 + 0.05, "1  Real data", "one recorded pass", "real data title");
const items = [["Scanned building", "reference map, cm-level"], ["Robots and sensors", "real LiDARs, cameras, mounts"],
  ["Real trajectories", "where the robots drove"], ["Measured Wi-Fi link", "RTT, goodput, outages"]];
const iw = (W - 0.3 - 0.3) / 4;
items.forEach(([t, sub], i) => {
  const ix = X + 0.15 + i * (iw + 0.1), iy = y1 + 0.34;
  card(ix, iy, iw, 0.38, "real", "real item " + (i + 1));
  dot(ix + 0.1, iy + 0.135, true, "real dot " + (i + 1));
  twoLines(ix + 0.28, iy + 0.02, iw - 0.32, 0.34, t, sub, "real item text " + (i + 1));
});

// 2 digital twin
const y2 = y1 + h1 + 0.15, h2 = 1.95;
panel(X, y2, W, h2, "sim", "digital twin");
heading(X + 0.15, y2 + 0.05, "2  Digital Twin · Isaac Sim", "change what you want, keep the rest real", "digital twin title");
const knobs = [["Comms", "latency · loss · bandwidth", true], ["Sensor dropouts", "LiDAR or camera off", false],
  ["Pose source", "ground truth · noisy · SLAM", false], ["People and objects", "walkers · furniture", true],
  ["Robot paths", "new routes · speed", true], ["Sensor models", "LiDAR beams · resolution", true]];
const kw = 1.72, kh = 0.6, kg = 0.08;
knobs.forEach(([t, sub, built], i) => {
  const kx = X + 0.15 + (i % 3) * (kw + kg), ky = y2 + 0.38 + Math.floor(i / 3) * (kh + kg);
  card(kx, ky, kw, kh, "sim", "knob " + t);
  dot(kx + 0.1, ky + 0.12, built, "knob dot " + t);
  s.addText(t, { x: kx + 0.27, y: ky + 0.05, w: kw - 0.3, h: 0.24, fontFace: F, fontSize: 10.5, bold: true, color: INK,
    valign: "middle", isTextBox: true, margin: 0, objectName: "knob title " + t });
  s.addText(sub, { x: kx + 0.1, y: ky + 0.3, w: kw - 0.15, h: 0.26, fontFace: F, fontSize: 9, color: SOFT,
    valign: "middle", isTextBox: true, margin: 0, objectName: "knob text " + t });
});
// legend for the dots
dot(X + 0.17, y2 + h2 - 0.25, true, "legend built dot");
s.addText("built", { x: X + 0.33, y: y2 + h2 - 0.31, w: 0.5, h: 0.22, fontFace: F, fontSize: 9, color: SOFT,
  valign: "middle", isTextBox: true, margin: 0, objectName: "legend built" });
dot(X + 0.8, y2 + h2 - 0.25, false, "legend planned dot");
s.addText("planned", { x: X + 0.96, y: y2 + h2 - 0.31, w: 0.8, h: 0.22, fontFace: F, fontSize: 9, color: SOFT,
  valign: "middle", isTextBox: true, margin: 0, objectName: "legend planned" });

// the deck of scenarios
const cx = X + 0.15 + 3 * (kw + kg) + 0.1, cy = y2 + 0.38, cw = X + W - 0.3 - cx, ch = 1.18;
card(cx + 0.12, cy + 0.12, cw - 0.12, ch, "sim", "scenario card 3");
card(cx + 0.06, cy + 0.06, cw - 0.12, ch, "sim", "scenario card 2");
card(cx, cy, cw - 0.12, ch, "sim", "scenario card 1");
s.addText([
  { text: "Scenario 1", options: { bold: true, fontSize: 12, color: INK, breakLine: true } },
  { text: "4 people · 30% loss · camera off · SLAM poses", options: { fontSize: 8.5, color: SOFT } },
], { x: cx + 0.15, y: cy + 0.06, w: cw - 0.4, h: 0.42, fontFace: F, valign: "top", isTextBox: true, margin: 0,
  objectName: "scenario title" });
const checks = [["LiDAR + camera data", "every robot"], ["Ground-truth boxes", "every frame, free"], ["Link per frame", "measured or tuned"]];
checks.forEach(([t, sub], j) => {
  s.addText([
    { text: "✓  ", options: { bold: true, color: CHECK } },
    { text: t, options: { bold: true, color: INK } },
    { text: "   " + sub, options: { color: SOFT, fontSize: 8.5 } },
  ], { x: cx + 0.15, y: cy + 0.52 + j * 0.2, w: cw - 0.35, h: 0.2, fontFace: F, fontSize: 9.5, valign: "middle",
    isTextBox: true, margin: 0, objectName: "scenario check " + (j + 1) });
});
s.addText("× N scenarios", { x: X + W - 2.0, y: y2 + h2 - 0.33, w: 1.85, h: 0.26, fontFace: F, fontSize: 13, bold: true,
  color: ACCENT, align: "right", valign: "middle", isTextBox: true, margin: 0, objectName: "n scenarios" });

// 3 train, 4 test on real
const y3 = y2 + h2 + 0.15, h3 = 0.5;
panel(X, y3, W, h3, "data", "train");
s.addText([
  { text: "3  Train", options: { bold: true, fontSize: 13, color: INK } },
  { text: "      collaborative perception models on digital-twin scenarios built from real data", options: { fontSize: 10.5, color: INK } },
], { x: X + 0.15, y: y3, w: W - 1.3, h: h3, fontFace: F, valign: "middle", isTextBox: true, margin: 0, objectName: "train text" });
s.addText("next step", { x: X + W - 1.15, y: y3, w: 1.0, h: h3, fontFace: F, fontSize: 9.5, italic: true, color: SOFT,
  align: "right", valign: "middle", isTextBox: true, margin: 0, objectName: "train status" });

const y4 = y3 + h3 + 0.15, h4 = 0.5;
panel(X, y4, W, h4, "label", "test on real");
s.addText([
  { text: "4  Test on real", options: { bold: true, fontSize: 13, color: INK } },
  { text: "      a held-out real recording with its measured link", options: { fontSize: 10.5, color: INK } },
], { x: X + 0.15, y: y4, w: W - 3.0, h: h4, fontFace: F, valign: "middle", isTextBox: true, margin: 0, objectName: "test text" });
s.addText("sim-trained vs real-trained", { x: X + W - 2.65, y: y4, w: 2.5, h: h4, fontFace: F, fontSize: 9.5, italic: true,
  color: SOFT, align: "right", valign: "middle", isTextBox: true, margin: 0, objectName: "test note" });

// arrows down the middle
const mid = X + W / 2;
down(mid, y1 + h1, y2, "arrow real to twin");
down(mid, y2 + h2, y3, "arrow twin to train");
down(mid, y3 + h3, y4, "arrow train to test");

// held-out real recording, down the left edge
const lx = 0.42;
s.addShape(pres.ShapeType.line, { x: lx, y: y1 + 0.4, w: X - lx, h: 0, line: { color: SOFT, width: 1.25, dashType: "dash" }, objectName: "held-out top" });
s.addShape(pres.ShapeType.line, { x: lx, y: y1 + 0.4, w: 0, h: y4 + h4 / 2 - (y1 + 0.4), line: { color: SOFT, width: 1.25, dashType: "dash" }, objectName: "held-out down" });
s.addShape(pres.ShapeType.line, { x: lx, y: y4 + h4 / 2, w: X - lx, h: 0,
  line: { color: SOFT, width: 1.25, dashType: "dash", endArrowType: "triangle" }, objectName: "held-out in" });
s.addText("held-out real recording", { x: 0.12, y: 2.0, w: 0.24, h: 2.2, fontFace: F, fontSize: 9, italic: true, color: SOFT,
  align: "center", valign: "middle", vert: "vert270", isTextBox: true, margin: 0, objectName: "held-out label" });

s.addNotes("Start from one real recorded pass: the scanned building, our real robots and sensor mounts, the trajectories they drove, and the Wi-Fi link we measured. The digital twin in Isaac Sim keeps all of that real and lets us change one thing at a time: the comms (latency, loss, bursts, bandwidth), sensor dropouts, where the poses come from (ground truth, noisy, or a SLAM algorithm's output), people and furniture, robot paths, and sensor models. Each setting is a scenario that comes with sensor data from every robot, ground-truth boxes for free, and the link per frame, so we can make as many training scenarios as we want. The models are then tested on a held-out real recording with its own measured link. Built today: comms, people and objects, robot paths, sensor models. Planned: sensor dropouts, SLAM as the pose source, and the training step itself.");

pres.writeFile({ fileName: out }).then(f => console.log(f));
