// Fig. 3 (reference pipeline) as native, editable PowerPoint shapes.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.title = "Reference pipeline";

const INK = "14213D", MUTED = "5A6475", LANE = "EEF1F5", LANE_E = "A7B0C0";
const OUTF = "FBE6D0", OUTE = "D9822B", OUTT = "9A5A14";
const FONT = "Calibri", FS = 12;

// figure units (100 x 40, y up) -> inches
const S = 0.1233, X0 = 0.5, TOP = 1.75, HU = 40;
const X = u => X0 + u * S, Y = u => TOP + (HU - u) * S, L = u => u * S;

const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };
slide.addText("Reference pipeline", { x: 0.5, y: 0.45, w: 12.3, h: 0.7, fontFace: "Cambria",
  fontSize: 36, bold: true, color: INK, margin: 0, isTextBox: true, objectName: "Title" });

function lane(y0, y1, title, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X(0.6), y: Y(y1), w: L(98.8), h: L(y1 - y0),
    fill: { color: LANE }, line: { color: LANE_E, width: 1 }, rectRadius: 0.1, objectName: name });
  slide.addText(title, { x: X(1.8), y: Y(y1) + 0.08, w: 6, h: 0.32, fontFace: FONT, fontSize: 14,
    bold: true, color: INK, margin: 0, isTextBox: true, objectName: name + " title" });
}
function box(x0, x1, yc, h, text, name, bold) {
  slide.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, x: X(x0), y: Y(yc + h / 2), w: L(x1 - x0),
    h: L(h), fill: { color: "FFFFFF" }, line: { color: INK, width: 1.25 }, rectRadius: 0.06,
    fontFace: FONT, fontSize: FS, bold: !!bold, color: INK, align: "center", valign: "middle",
    margin: 2, objectName: name });
  return { x0, x1, yc, h };
}
function out(x0, x1, yc, h, text, name, bold) {
  slide.addText(text, { shape: pres.shapes.PARALLELOGRAM, x: X(x0), y: Y(yc + h / 2), w: L(x1 - x0),
    h: L(h), fill: { color: OUTF }, line: { color: OUTE, width: 1.5 },
    fontFace: FONT, fontSize: FS, bold: !!bold, color: INK, align: "center", valign: "middle",
    margin: 2, objectName: name });
  return { x0, x1, yc, h };
}
// a straight line between two figure points, arrow at the second
function line(p, q, o) {
  o = o || {};
  const x = Math.min(p[0], q[0]), w = Math.abs(q[0] - p[0]);
  const ytop = Math.max(p[1], q[1]), h = Math.abs(q[1] - p[1]);
  const goesLeft = q[0] < p[0], goesUp = q[1] > p[1];
  slide.addShape(pres.shapes.LINE, { x: X(x), y: Y(ytop), w: L(w), h: L(h),
    flipH: goesLeft, flipV: goesUp,
    line: { color: o.color || INK, width: o.width || 1.25, dashType: o.dash ? "dash" : "solid",
            endArrowType: o.noArrow ? "none" : "triangle" }, objectName: o.name || "Arrow" });
}
const right = (a, b, n) => line([a.x1, a.yc], [b.x0, a.yc], { name: n });

// ---- mapping session
lane(19.2, 39.4, "Mapping session (once per site)", "Mapping lane");
let yc = 30.6;
const m1 = box(2.0, 11.5, yc, 6.0, "Mapping\nrecording", "Mapping recording");
const m2 = box(13.5, 30.5, yc, 6.0, "LiDAR-inertial SLAM\n+ pose-time correction", "SLAM");
const m3 = box(32.5, 44.5, yc, 6.0, "Plane bundle\nadjustment", "Bundle adjustment");
const m4 = box(46.5, 56.5, yc, 6.0, "TSDF fusion", "TSDF fusion");
const m5 = out(58.5, 74.5, yc, 7.0, "Frozen reference\nmap + trajectory", "Frozen reference", true);
const b1 = box(80.0, 98.4, 34.9, 4.8, "Board poses", "Board poses");
const b2 = box(80.0, 98.4, 24.9, 4.8, "Infrastructure\ncamera poses", "Camera poses");
right(m1, m2, "Arrow 1"); right(m2, m3, "Arrow 2"); right(m3, m4, "Arrow 3"); right(m4, m5, "Arrow 4");
line([74.0, yc + 1.0], [b1.x0, b1.yc], { name: "Arrow to boards" });
line([74.0, yc - 1.0], [b2.x0, b2.yc], { name: "Arrow to cameras" });
line([84.0, b1.yc - b1.h / 2], [84.0, b2.yc + b2.h / 2], { name: "Boards to cameras" });
slide.addText("through the boards", { x: X(85.0), y: Y(31.2), w: L(13), h: L(2.6), fontFace: FONT,
  fontSize: 10, color: MUTED, margin: 0, valign: "middle", isTextBox: true, objectName: "Through label" });
slide.addText("checks: wall thickness · loop closure · pose time · control distances", {
  shape: pres.shapes.ROUNDED_RECTANGLE, x: X(13.5), y: Y(23.6), w: L(43.0), h: L(3.0),
  fill: { color: LANE }, line: { color: MUTED, width: 0.75, dashType: "dash" }, rectRadius: 0.05,
  fontFace: FONT, fontSize: 11, italic: true, color: MUTED, align: "center", valign: "middle",
  margin: 0, objectName: "Checks" });

// ---- every sequence
lane(0.6, 17.6, "Every sequence", "Sequence lane");
yc = 7.4;
const s1 = box(2.0, 11.5, yc, 6.0, "Sequence\nrecording", "Sequence recording");
const s2 = box(13.5, 30.5, yc, 6.0, "LiDAR-inertial SLAM\n+ pose-time check", "Sequence SLAM");
const s3 = box(32.5, 56.5, yc, 8.4, "Per-agent registration\nLiDAR: scan-to-map ICP\nRGB-D: odometry + ICP + boards", "Registration");
const s4 = box(58.5, 72.0, yc, 8.4, "Independent\nvalidation\n(held-out boards)", "Validation");
const s5 = out(74.0, 87.5, yc, 7.0, "Reference\nposes (site)", "Reference poses", true);
const s6 = out(89.0, 98.4, yc, 7.0, "Labels and\ngeometry", "Labels");
right(s1, s2, "Arrow 5"); right(s2, s3, "Arrow 6"); right(s3, s4, "Arrow 7");
right(s4, s5, "Arrow 8"); right(s5, s6, "Arrow 9");

// ---- the frozen reference feeds every sequence
const xf = 66.5, ybus = 14.2, d = { color: OUTE, dash: true, width: 1.5 };
line([xf, yc + 0 + m5.yc - m5.h / 2 - yc], [xf, ybus], Object.assign({ noArrow: true, name: "Bus down" }, d));
line([xf, ybus], [44.5, ybus], Object.assign({ noArrow: true, name: "Bus across" }, d));
line([44.5, ybus], [44.5, s3.yc + s3.h / 2], Object.assign({ name: "To registration" }, d));
line([65.25, ybus], [65.25, s4.yc + s4.h / 2], Object.assign({ name: "To validation" }, d));
slide.addText("frozen map, boards", { x: X(46.5), y: Y(16.9), w: L(17), h: L(2.4), fontFace: FONT,
  fontSize: 11, color: OUTT, align: "center", valign: "bottom", margin: 0, isTextBox: true,
  objectName: "Bus label" });

slide.addNotes("Top: the mapping session, once per site. GLIM with its pose-time correction, plane bundle adjustment and TSDF fusion give the frozen reference map and trajectory; the boards are posed in it and the infrastructure cameras placed through them. Wall thickness, loop closure, pose time and control distances check it. Bottom: every sequence is registered into the frozen map per agent, validated on held-out boards, and exported in the site frame.");
pres.writeFile({ fileName: "fig_refpipeline.pptx" }).then(f => console.log("wrote " + f));
