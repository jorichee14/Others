// Fig. 3 (reference pipeline) as native, editable PowerPoint shapes, with
// the processing of each robot's recording on a slide before it.
//   node figs/src/refpipeline_pptx.js   -> fig_refpipeline.pptx (cwd)
// Slide 1: recording processing (scoop_pipeline/processing/process_recording.py,
// merge_session.py; configs/recording.yaml). Slide 2: the reference pipeline
// (the paper's figs/fig_refpipeline.pdf, from refpipeline.py).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.title = "Reference pipeline";

const INK = "14213D", MUTED = "5A6475", LANE = "EEF1F5", LANE_E = "A7B0C0";
const OUTF = "FBE6D0", OUTE = "D9822B", OUTT = "9A5A14";
const FONT = "Calibri";

// figure units (100 wide, y up) -> inches
const S = 0.1233, X0 = 0.5;

function canvas(title, top, hu) {
  const slide = pres.addSlide();
  slide.background = { color: "FFFFFF" };
  slide.addText(title, { x: 0.5, y: 0.4, w: 12.3, h: 0.7, fontFace: "Cambria", fontSize: 32,
    bold: true, color: INK, margin: 0, isTextBox: true, objectName: "Title" });
  const X = u => X0 + u * S, Y = u => top + (hu - u) * S, L = u => u * S;
  const d = { slide, X, Y, L };

  d.lane = (x0, x1, y0, y1, title, name) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X(x0), y: Y(y1), w: L(x1 - x0), h: L(y1 - y0),
      fill: { color: LANE }, line: { color: LANE_E, width: 1 }, rectRadius: 0.1, objectName: name });
    slide.addText(title, { x: X(x0 + 1.2), y: Y(y1) + 0.07, w: L(x1 - x0 - 2.4), h: 0.3, fontFace: FONT,
      fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true, objectName: name + " title" });
  };
  d.box = (x0, x1, yc, h, text, name, o) => {
    o = o || {};
    slide.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, x: X(x0), y: Y(yc + h / 2), w: L(x1 - x0),
      h: L(h), fill: { color: "FFFFFF" }, line: { color: INK, width: 1.25 }, rectRadius: 0.06,
      fontFace: FONT, fontSize: o.fs || 12, bold: !!o.bold, color: INK, align: "center",
      valign: "middle", margin: 2, objectName: name });
    return { x0, x1, yc, h };
  };
  d.out = (x0, x1, yc, h, text, name, o) => {
    o = o || {};
    slide.addText(text, { shape: pres.shapes.PARALLELOGRAM, x: X(x0), y: Y(yc + h / 2), w: L(x1 - x0),
      h: L(h), fill: { color: OUTF }, line: { color: OUTE, width: 1.5 },
      fontFace: FONT, fontSize: o.fs || 12, bold: !!o.bold, color: INK, align: "center",
      valign: "middle", margin: 2, objectName: name });
    return { x0, x1, yc, h };
  };
  // a straight line between two figure points, arrow at the second
  d.line = (p, q, o) => {
    o = o || {};
    const x = Math.min(p[0], q[0]), w = Math.abs(q[0] - p[0]);
    const ytop = Math.max(p[1], q[1]), h = Math.abs(q[1] - p[1]);
    slide.addShape(pres.shapes.LINE, { x: X(x), y: Y(ytop), w: L(w), h: L(h),
      flipH: q[0] < p[0], flipV: q[1] > p[1],
      line: { color: o.color || INK, width: o.width || 1.25, dashType: o.dash ? "dash" : "solid",
              endArrowType: o.noArrow ? "none" : "triangle" }, objectName: o.name || "Arrow" });
  };
  d.right = (a, b, n) => d.line([a.x1, a.yc], [b.x0, a.yc], { name: n });
  d.note = (x0, x1, yc, h, text, name, o) => {
    o = o || {};
    slide.addText(text, { x: X(x0), y: Y(yc + h / 2), w: L(x1 - x0), h: L(h), fontFace: FONT,
      fontSize: o.fs || 10, italic: !!o.italic, color: o.color || MUTED, align: o.align || "left",
      valign: "middle", margin: 0, isTextBox: true, objectName: name });
  };
  return d;
}

// ============================================================ slide 1: recordings
{
  const d = canvas("Recording processing, per robot and pass", 1.45, 46);
  const C = [[1.8, 14.2], [16.2, 30.2], [32.2, 48.2], [50.2, 63.2], [65.2, 77.2]];  // columns
  const MERGE = [81.2, 98.6];

  // ---- mobile_1: Ouster packets + ZED 2i SVO2
  d.lane(0.6, 79.2, 24.0, 45.4, "mobile_1  ·  Ouster LiDAR + ZED 2i", "mobile_1 lane");
  let y = 36.6;
  const a1 = d.box(...C[0], y, 7.2, "Raw bag\nlidar + IMU packets,\nmetadata, ZED IMU", "m1 raw bag", { fs: 11 });
  const a2 = d.box(...C[1], y, 7.2, "Decode\npackets → points + IMU\n(sensor time, _mobile1)", "m1 decode", { fs: 11 });
  const a3 = d.box(...C[2], y, 7.2, "Retime\nsensor → PC clock (clock fit);\nZED IMU re-synced to Ouster", "m1 retime", { fs: 11 });
  const a4 = d.box(...C[3], y, 7.2, "GLIM\nLiDAR-inertial\nSLAM", "m1 GLIM", { fs: 11 });
  const a5 = d.out(...C[4], y, 7.2, "LiDAR\ntrajectory", "m1 trajectory", { fs: 11, bold: true });
  d.right(a1, a2); d.right(a2, a3); d.right(a3, a4); d.right(a4, a5);
  y = 28.0;
  const z1 = d.box(...C[0], y, 6.0, "ZED SVO2\n(stereo video)", "m1 svo", { fs: 11 });
  const z2 = d.box(C[1][0], C[2][1], y, 6.0, "ZED replay: images, depth, odometry, pose\non the capture clock", "m1 zed replay", { fs: 11 });
  const z3 = d.out(...C[3], y, 6.0, "ZED bags", "m1 zed bags", { fs: 11 });
  d.right(z1, z2); d.right(z2, z3);

  // ---- mobile_2: Ouster packets + RealSense + Isaac VSLAM, one raw bag
  d.lane(0.6, 79.2, 0.6, 22.0, "mobile_2  ·  Ouster LiDAR + RealSense + Isaac VSLAM", "mobile_2 lane");
  y = 13.4;
  const b1 = d.box(...C[0], y, 7.2, "Raw bag\nlidar + IMU packets,\nmetadata", "m2 raw bag", { fs: 11 });
  const b2 = d.box(...C[1], y, 7.2, "Decode\npackets → points + IMU\n(sensor time, _mobile2)", "m2 decode", { fs: 11 });
  const b3 = d.box(...C[2], y, 7.2, "Retime\nsensor → PC clock\n(clock fit)", "m2 retime", { fs: 11 });
  const b4 = d.box(...C[3], y, 7.2, "GLIM, own config\n+ pose-time\ncorrection +46 ms", "m2 GLIM", { fs: 11 });
  const b5 = d.out(...C[4], y, 7.2, "LiDAR\ntrajectory", "m2 trajectory", { fs: 11, bold: true });
  d.right(b1, b2); d.right(b2, b3); d.right(b3, b4); d.right(b4, b5);
  y = 4.8;
  const r1 = d.box(...C[0], y, 6.0, "RealSense, IMU,\nIsaac VSLAM", "m2 realsense", { fs: 11 });
  const r2 = d.box(C[1][0], C[2][1], y, 6.0, "copied as recorded: already on the PC clock;\nframes renamed per robot", "m2 copy", { fs: 11 });
  d.right(r1, r2);

  // ---- each pass: the merge
  d.lane(80.4, 99.4, 0.6, 45.4, "Each pass", "pass lane");
  const m = d.box(...MERGE, 25.0, 13.0, "Merge\nevery robot's\nretimed + ZED bags,\nchecked against\nrecord.yaml;\ncalibrations\nto /tf_static", "merge", { fs: 11 });
  const mb = d.out(MERGE[0], MERGE[1], 11.0, 6.4, "Merged bag", "merged bag", { fs: 11, bold: true });
  d.line([(MERGE[0] + MERGE[1]) / 2, m.yc - m.h / 2], [(MERGE[0] + MERGE[1]) / 2, mb.yc + mb.h / 2], { name: "to merged bag" });
  d.line([76.6, 36.6], [MERGE[0], 28.0], { name: "mobile_1 bags" });
  d.line([76.6, 13.4], [MERGE[0], 22.0], { name: "mobile_2 bags" });
  d.note(MERGE[0], MERGE[1], 3.2, 5.0, "one robot alone (20261015):\nits own retimed bag, no merge", "single robot note", { fs: 9.5, italic: true, align: "center" });
  d.note(C[2][0], C[4][1], 22.95, 1.6, "trajectories → the mapping session or the sequence (next slide)", "to next", { fs: 9.5, italic: true, align: "right" });

  d.slide.addNotes("Each robot's recording is processed on its own, then the pass is merged. mobile_1: its raw bag holds the Ouster's lidar and IMU packets; they are decoded into points and IMU in sensor time, retimed onto the PC clock with a clock fit, and the ZED IMU is re-synced to the Ouster IMU (its driver stamps it late, 2.12 s on 20260924). GLIM gives the LiDAR trajectory. The ZED's SVO2 is replayed into image, depth and odometry bags on the capture clock. mobile_2: the same Ouster chain; GLIM runs with its own config and its poses are moved 46 ms on export. Its RealSense and Isaac VSLAM topics are already on the PC clock and are copied. The merge checks every robot's bags against record.yaml and adds the calibrations; a pass with one robot uses its own retimed bag.");
}

// ============================================================ slide 2: reference pipeline
{
  const d = canvas("Reference pipeline", 1.75, 40);
  // ---- mapping session
  d.lane(0.6, 99.4, 19.2, 39.4, "Mapping session (once per site)", "Mapping lane");
  let yc = 30.6;
  const m1 = d.box(2.0, 11.5, yc, 6.0, "Processed\nrecording", "Mapping recording");
  const m2 = d.box(13.5, 30.5, yc, 6.0, "LiDAR-inertial SLAM\n+ pose-time correction", "SLAM");
  const m3 = d.box(32.5, 44.5, yc, 6.0, "Plane bundle\nadjustment", "Bundle adjustment");
  const m4 = d.box(46.5, 56.5, yc, 6.0, "TSDF fusion", "TSDF fusion");
  const m5 = d.out(58.5, 74.5, yc, 7.0, "Frozen reference\nmap + trajectory", "Frozen reference", { bold: true });
  const b1 = d.box(80.0, 98.4, 34.9, 4.8, "Board poses", "Board poses");
  const b2 = d.box(80.0, 98.4, 24.9, 4.8, "Infrastructure\ncamera poses", "Camera poses");
  d.right(m1, m2); d.right(m2, m3); d.right(m3, m4); d.right(m4, m5);
  d.line([74.0, yc + 1.0], [b1.x0, b1.yc], { name: "Arrow to boards" });
  d.line([74.0, yc - 1.0], [b2.x0, b2.yc], { name: "Arrow to cameras" });
  d.line([84.0, b1.yc - b1.h / 2], [84.0, b2.yc + b2.h / 2], { name: "Boards to cameras" });
  d.note(85.0, 98.0, 29.9, 2.6, "through the boards", "Through label");
  d.slide.addText("checks: wall thickness · loop closure · pose time · control distances", {
    shape: pres.shapes.ROUNDED_RECTANGLE, x: d.X(13.5), y: d.Y(23.6), w: d.L(43.0), h: d.L(3.0),
    fill: { color: LANE }, line: { color: MUTED, width: 0.75, dashType: "dash" }, rectRadius: 0.05,
    fontFace: FONT, fontSize: 11, italic: true, color: MUTED, align: "center", valign: "middle",
    margin: 0, objectName: "Checks" });

  // ---- every sequence
  d.lane(0.6, 99.4, 0.6, 17.6, "Every sequence", "Sequence lane");
  yc = 7.4;
  const s1 = d.box(2.0, 11.5, yc, 6.0, "Processed\nrecording", "Sequence recording");
  const s2 = d.box(13.5, 30.5, yc, 6.0, "LiDAR-inertial SLAM\n+ pose-time check", "Sequence SLAM");
  const s3 = d.box(32.5, 56.5, yc, 8.4, "Per-agent registration\nLiDAR: scan-to-map ICP\nRGB-D: odometry + ICP + boards", "Registration");
  const s4 = d.box(58.5, 72.0, yc, 8.4, "Independent\nvalidation\n(held-out boards)", "Validation");
  const s5 = d.out(74.0, 87.5, yc, 7.0, "Reference\nposes (site)", "Reference poses", { bold: true });
  const s6 = d.out(89.0, 98.4, yc, 7.0, "Labels and\ngeometry", "Labels");
  d.right(s1, s2); d.right(s2, s3); d.right(s3, s4); d.right(s4, s5); d.right(s5, s6);

  // ---- the frozen reference feeds every sequence
  const xf = 66.5, ybus = 14.2, dsh = { color: OUTE, dash: true, width: 1.5 };
  d.line([xf, m5.yc - m5.h / 2], [xf, ybus], Object.assign({ noArrow: true, name: "Bus down" }, dsh));
  d.line([xf, ybus], [44.5, ybus], Object.assign({ noArrow: true, name: "Bus across" }, dsh));
  d.line([44.5, ybus], [44.5, s3.yc + s3.h / 2], Object.assign({ name: "To registration" }, dsh));
  d.line([65.25, ybus], [65.25, s4.yc + s4.h / 2], Object.assign({ name: "To validation" }, dsh));
  d.note(46.5, 63.5, 15.7, 2.4, "frozen map, boards", "Bus label", { fs: 11, color: OUTT, align: "center" });

  d.slide.addNotes("Top: the mapping session, once per site. GLIM with its pose-time correction, plane bundle adjustment and TSDF fusion give the frozen reference map and trajectory; the boards are posed in it and the infrastructure cameras placed through them. Wall thickness, loop closure, pose time and control distances check it. Bottom: every sequence is registered into the frozen map per agent, validated on held-out boards, and exported in the site frame.");
}

// ============================================================ slides 3-6: how each mapping step works
// drawings from refmethod.py (figs/fig_refmethod_{a,b,c,d}.png); 20261015 numbers from the tables
const STEPS = [
  ["a", "Pose-time correction", [
    "GLIM stamps each pose with the start of its LiDAR sweep; the pose is the robot about half a sweep (50 ms) later.",
    "A scan placed with the pose stamped at its own time lands off the walls whenever the robot turns.",
    "Fix: every pose moved +46 ms on export (recording.yaml, pose_time_offset)."],
   "20261015: 4 ms left; checked on every pass (heading against turning rate)",
   "GLIM writes each pose with the time its sweep started, but the pose describes the robot about half a sweep later. Looking up the pose at a scan's own time therefore gives where the robot was 50 ms later; while turning, that scan lands beside the walls. We move every pose 46 ms on export, and check the remaining offset on every pass."],
  ["b", "Plane bundle adjustment", [
    "Keyframes: 1 s of scans each. Planes: flat 0.5 m cubes seen by at least 3 keyframes.",
    "Each plane is solved in closed form, so only the poses are unknown; the cost is every plane's thickness squared.",
    "All keyframe poses move together; odometry keeps what no plane sees, gravity holds roll and pitch; 3 rounds."],
   "20261015: whole-run walls 12.3 cm (GLIM as output) → 9.0 cm (reference)",
   "Like bundle adjustment with cameras, but the shared features are planes instead of points. Each keyframe sees part of a plane; if its pose is off, its points sit above or below the others. The plane is fitted in closed form, and all poses are moved together until every plane is as thin as the scans allow."],
  ["c", "TSDF fusion", [
    "Each point is a ray: every 2 cm cell within ±8 cm of the hit gets its signed distance (+ in front, − behind).",
    "Averaged over all rays; the surface is where the average crosses zero (cells seen by at least 3 rays).",
    "One surface instead of every cell any scan hit: the union's band grows with every pass."],
   "20261015: map walls 7.8 cm, against 12.8 cm for the union",
   "The union of points keeps every cell any scan ever hit, so a rarely hit cell counts as much as the dense centre and walls grow thicker with every pass. The TSDF instead averages, along each ray, how far each cell is in front of or behind the hit, and puts the surface where that average is zero."],
  ["d", "Wall thickness", [
    "Each 0.25 m patch: fit a plane, take the points' offsets along its normal.",
    "Thickness = p95 − p5 of the offsets (3.29σ for Gaussian noise).",
    "Points from one scan, 1 s or the whole run, as placed: separates sensor, motion and pose error."],
   "20261015 reference: sensor alone 3.0 · 1 scan 4.7 · 1 s 6.7 · whole run 9.0 cm",
   "This is how every thickness in the tables is measured. The same patches are measured with the points of one scan, of one second, or of the whole run as placed by the trajectory; each scan against its own plane gives the sensor alone."],
];
for (const [key, title, bullets, result, notes] of STEPS) {
  const slide = pres.addSlide();
  slide.background = { color: "FFFFFF" };
  slide.addText(title, { x: 0.5, y: 0.4, w: 12.3, h: 0.7, fontFace: "Cambria", fontSize: 32, bold: true,
    color: INK, margin: 0, isTextBox: true, objectName: "Title" });
  const h = 5.7, w = h * 1440 / 1180;
  slide.addImage({ path: require("path").join(__dirname, "..", `fig_refmethod_${key}.png`), x: 0.35, y: 1.35, w, h,
    altText: title + ": schematic drawing", objectName: "Drawing" });
  const x = 0.35 + w + 0.35, tw = 13.333 - x - 0.5;
  slide.addText(bullets.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < bullets.length - 1 } })), {
    x, y: 1.7, w: tw, h: 3.6, fontFace: FONT, fontSize: 17, color: INK, valign: "top", paraSpaceAfter: 12,
    margin: 0, isTextBox: true, objectName: "Bullets" });
  slide.addText(result, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y: 5.6, w: tw, h: 0.9, fill: { color: OUTF },
    line: { color: OUTE, width: 1 }, rectRadius: 0.08, fontFace: FONT, fontSize: 15, bold: true, color: INK,
    align: "left", valign: "middle", margin: 8, objectName: "Result" });
  slide.addNotes(notes);
}

pres.writeFile({ fileName: "fig_refpipeline.pptx" }).then(f => console.log("wrote " + f));
