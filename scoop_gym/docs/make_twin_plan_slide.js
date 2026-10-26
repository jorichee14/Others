// One 16:9 slide: the SCooP-iT Gym plan drawn as native shapes in the left
// Run: NODE_PATH=<dir with pptxgenjs> APPLY_THEME=<pptx skill>/scripts/apply_theme.js node make_twin_plan_slide.js
// half (x <= 6.67"), the right half left empty for the presenter's text.
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("process.env.APPLY_THEME");

const THEME = {
  name: "SCooP-iT",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: { dk1: "222222", lt1: "FFFFFF", dk2: "3B3B3B", lt2: "F2F2F2",
            accent1: "C45A1C", accent2: "E0B46A", accent3: "A582C4",
            accent4: "7D9CCC", accent5: "7CB37C", accent6: "62A9A9",
            hlink: "C45A1C", folHlink: "A582C4" },
};

const INK = "222222", MUTED = "666666", ORANGE = "C45A1C";
const STAGE = {
  data:   ["ECECEC", "9A9A9A"], twin: ["FBF0DC", "E0B46A"], gate: ["EFE6F6", "A582C4"],
  train:  ["E3ECF8", "7D9CCC"], test: ["E4F3E4", "7CB37C"], deploy: ["DFF1F1", "62A9A9"],
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.333" x 7.5"
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "SCooP-iT Gym: plan from real data to a validated twin";
const S = pres.shapes;
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

let n = 0;
const name = (k) => `${k}_${++n}`;

function text(x, y, w, h, runs, opts = {}) {
  slide.addText(runs, Object.assign({ x, y, w, h, margin: 0, isTextBox: true, fontFace: "Calibri",
    fontSize: 10, color: INK, valign: "middle", objectName: name("text") }, opts));
}

function marker(x, y, st, d = 0.13) {
  const line = { color: ORANGE, width: 1.25 };
  if (st === "validated") {
    slide.addShape(S.OVAL, { x, y, w: d, h: d, fill: { color: ORANGE }, line: { color: ORANGE, width: 1.25 }, objectName: name("validated") });
  } else {
    slide.addShape(S.OVAL, { x, y, w: d, h: d, fill: { color: "FFFFFF" }, line, objectName: name(st) });
    if (st === "implemented")
      slide.addShape(S.PIE, { x, y, w: d, h: d, fill: { color: ORANGE }, line: { color: ORANGE, width: 0.5 }, angleRange: [90, 270], objectName: name("half") });
  }
}

function tag(x, y, kind) {
  const solid = kind === "M";
  slide.addText(kind, { x, y, w: 0.2, h: 0.17, shape: S.ROUNDED_RECTANGLE, rectRadius: 0.03, margin: 0,
    fill: { color: solid ? "3B3B3B" : "FFFFFF" }, line: { color: "3B3B3B", width: 0.75 },
    fontFace: "Calibri", fontSize: 8, bold: true, color: solid ? "FFFFFF" : "3B3B3B",
    align: "center", valign: "middle", isTextBox: true, objectName: name("tag") });
}

// a one-line card: status marker, bold title, muted note, optional M/A tag
function chip(x, y, w, st, title, note, kind) {
  const h = 0.27;
  slide.addShape(S.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: "FFFFFF" },
    line: { color: "BBBBBB", width: 0.75 }, objectName: name("card") });
  marker(x + 0.08, y + (h - 0.13) / 2, st);
  const runs = [{ text: title, options: { bold: true } }];
  if (note) runs.push({ text: "  " + note, options: { color: MUTED } });
  text(x + 0.28, y, w - 0.3 - (kind ? 0.24 : 0), h, runs);
  if (kind) tag(x + w - 0.26, y + 0.05, kind);
}

function stage(y, h, kind, num, title, note) {
  const [fill, stroke] = STAGE[kind];
  slide.addShape(S.ROUNDED_RECTANGLE, { x: X0, y, w: W0, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: stroke, width: 1.25 }, objectName: name("stage") });
  text(X0 + 0.15, y + 0.05, 3.4, 0.28, [{ text: num + "  ", options: { bold: true } }, { text: title, options: { bold: true } }],
    { fontSize: 13 });
  if (note) text(X0 + 2.5, y + 0.05, W0 - 2.65, 0.28, note, { color: MUTED, italic: true, align: "right" });
}

// one-row stage: title (and note below it) on the left, chips on the right
function rowStage(y, kind, num, title, note, chips) {
  const [fill, stroke] = STAGE[kind];
  const h = 0.55;
  slide.addShape(S.ROUNDED_RECTANGLE, { x: X0, y, w: W0, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: stroke, width: 1.25 }, objectName: name("stage") });
  text(X0 + 0.15, y + 0.06, 1.75, note ? 0.24 : 0.43, [{ text: num + "  " + title, options: { bold: true } }], { fontSize: 13 });
  if (note) text(X0 + 0.15, y + 0.3, 1.8, 0.2, note, { color: MUTED, italic: true });
  const cx = X0 + 1.95, cw = W0 - 1.95 - 0.12, gap = 0.08;
  const each = (cw - gap * (chips.length - 1)) / chips.length;
  chips.forEach((c, k) => chip(cx + k * (each + gap), y + (h - 0.27) / 2, each, c[0], c[1], c[2]));
  return y + h;
}

function down(y1, y2) {
  slide.addShape(S.LINE, { x: XC, y: y1, w: 0, h: y2 - y1, line: { color: "333333", width: 1.25, endArrowType: "triangle" }, objectName: name("arrow") });
}

// ------------------------------------------------------------------ geometry
const X0 = 0.6, W0 = 5.5, XC = X0 + W0 / 2;       // stage boxes span 0.6"-6.1"
const GAP = 0.15;

// title and legend
text(X0, 0.25, W0, 0.32, "Plan: from real data to a validated twin", { fontSize: 16, bold: true });
let lx = X0;
for (const [st, lab, w] of [["validated", "validated", 0.75], ["implemented", "implemented, not yet run", 1.65], ["planned", "planned", 0.7]]) {
  marker(lx, 0.66, st);
  text(lx + 0.18, 0.62, w, 0.2, lab, { color: MUTED });
  lx += w + 0.22;
}
tag(lx, 0.64, "M"); text(lx + 0.25, 0.62, 0.65, 0.2, "measured", { color: MUTED });
tag(lx + 0.95, 0.64, "A"); text(lx + 1.2, 0.62, 0.5, 0.2, "added", { color: MUTED });

// 1 real data
let y = 0.95;
const half = (W0 - 0.3 - 0.12) / 2, xL = X0 + 0.15, xR = X0 + 0.15 + half + 0.12;
stage(y, 1.13, "data", "1", "Real data", "never mixed");
text(xL, y + 0.34, half, 0.18, "Build passes (site 1)", { bold: true, color: MUTED });
text(xR, y + 0.34, half, 0.18, "Held-out passes (another day)", { bold: true, color: MUTED });
chip(xL, y + 0.54, half, "validated", "Reference map", "checked");
chip(xL, y + 0.84, half, "implemented", "Robots, sensors, link logs");
chip(xR, y + 0.54, half, "planned", "Survey + coop passes", "checks only");
chip(xR, y + 0.84, half, "planned", "Labels on held-out frames");
const y1 = y + 1.13;

// 2 digital twin
y = y1 + GAP;
stage(y, 1.97, "twin", "2", "Digital twin · Isaac Sim", "built from build passes only");
const rows = [
  [["implemented", "Building", "USD from map", "M"], ["implemented", "Sensor models", "real mounts", "M"]],
  [["implemented", "Paths: recorded", "replayed", "M"], ["planned", "Paths: new routes", "robot control", "A"]],
  [["implemented", "Link: logged, fitted", null, "M"], ["planned", "Link: by place", "survey cells", "M"]],
  [["planned", "Dropouts, pose error", "fitted", "M"], ["implemented", "People, objects", "boxes now", "A"]],
];
rows.forEach((r, k) => {
  chip(xL, y + 0.38 + k * 0.32, half, ...r[0]);
  chip(xR, y + 0.38 + k * 0.32, half, ...r[1]);
});
const sy = y + 0.38 + 4 * 0.32 + 0.02;
slide.addShape(S.ROUNDED_RECTANGLE, { x: xL, y: sy, w: W0 - 0.3, h: 0.27, rectRadius: 0.05, fill: { color: "FFFFFF" },
  line: { color: "D8C39A", width: 0.75 }, objectName: name("scenario") });
text(xL + 0.1, sy, W0 - 0.55, 0.27, [{ text: "Scenario × N  ", options: { bold: true, color: ORANGE } },
  { text: "a setting per part: link, dropouts, pose, people", options: { color: MUTED } }]);
const y2 = y + 1.97;

// 3-6
const y3 = y2 + GAP;
const e3 = rowStage(y3, "gate", "3", "Check the twin", "train only on what passes",
  [["planned", "Scans", "≤ 3 cm"], ["planned", "Dropouts", "fit"], ["planned", "Link", "±10 pt"]]);
const y4 = e3 + GAP;
const e4 = rowStage(y4, "train", "4", "Train in the twin", null,
  [["planned", "Perception + comms"], ["planned", "Motion policies"]]);
const y5 = e4 + GAP;
const e5 = rowStage(y5, "test", "5", "Test on held-out", "real data only",
  [["planned", "Perception + comms", "on real frames"]]);
const y6 = e5 + GAP;
const e6 = rowStage(y6, "deploy", "6", "Deploy on testbed", null,
  [["planned", "Motion policies", "vs. twin prediction"]]);

for (const [a, b] of [[y1, y1 + GAP], [y2, y3], [e3, y4], [e4, y5], [e5, y6]]) down(a + 0.01, b - 0.01);

// held-out passes -> 3 and 5 (dashed, right margin)
const xr = X0 + W0 + 0.18;
const dash = { color: "777777", width: 1.25, dashType: "dash" };
slide.addShape(S.LINE, { x: X0 + W0, y: 0.95 + 0.68, w: 0.18, h: 0, line: dash, objectName: name("held") });
slide.addShape(S.LINE, { x: xr, y: 0.95 + 0.68, w: 0, h: y5 + 0.275 - (0.95 + 0.68), line: dash, objectName: name("held") });
for (const yy of [y3 + 0.275, y5 + 0.275])
  slide.addShape(S.LINE, { x: X0 + W0 + 0.01, y: yy, w: xr - X0 - W0 - 0.01, h: 0, flipH: true,
    line: Object.assign({}, dash, { endArrowType: "triangle" }), objectName: name("held") });
text(xr + 0.03, 3.0, 0.22, 1.6, "held-out passes", { rotate: 90, color: "777777", align: "center", fontSize: 10, w: 1.6, h: 0.22, x: xr + 0.12 - 0.8, y: 3.7 });

// deploy -> new recordings refit the twin (left margin)
const xl = X0 - 0.2;
const fb = { color: ORANGE, width: 1.25 };
slide.addShape(S.LINE, { x: xl, y: y6 + 0.275, w: X0 - xl, h: 0, line: fb, objectName: name("fb") });
slide.addShape(S.LINE, { x: xl, y: 0.95 + 0.4, w: 0, h: y6 + 0.275 - (0.95 + 0.4), line: fb, objectName: name("fb") });
slide.addShape(S.LINE, { x: xl, y: 0.95 + 0.4, w: X0 - xl - 0.01, h: 0, line: Object.assign({}, fb, { endArrowType: "triangle" }), objectName: name("fb") });
text(xl - 0.95, 3.9, 1.9, 0.22, "new recordings refit the twin", { rotate: 270, color: ORANGE, align: "center", fontSize: 10, x: xl - 0.2 - 0.95, y: 4.0, w: 1.9, h: 0.22 });

slide.addNotes("Left half: the plan. Right half is free for the talking points. Status: filled = validated, half = implemented but not yet run in Isaac, hollow = planned. M = measured from the recordings, A = added.");

(async () => {
  const out = "/home/user/Others/scoop_gym/docs/twin_plan_halfslide.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out, "bottom of stage 6 at", e6.toFixed(2), "in");
})();
