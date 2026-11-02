// Pringles task pipeline + architecture deck (pptxgenjs, structured: theme, layouts, sections).
const pptxgen = require('pptxgenjs');
const { applyTheme } = require('/root/.claude/skills/synced/2ea7b1e4-184a-4243-87ea-e20b1a578b5f_6491705d-d951-4c59-9af7-0e3d5a286022/pptx/scripts/apply_theme.js');
const path = require('path');
const sizeOf = (f) => { const b = require('fs').readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };

const THEME = {
  name: 'Pick Comms', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: '1D1D1F', lt1: 'FFFFFF', dk2: '0B3D3A', lt2: 'EEF6F4',
    accent1: '0B7F74', accent2: '1A5FD0', accent3: 'A3171D', accent4: '5B3FB0',
    accent5: '1D7A3A', accent6: 'E9C46A', hlink: '1A5FD0', folHlink: '5B3FB0' },
};
const OUT = path.join(__dirname, 'Pringles_Pipeline_Architecture.pptx');
const IMG = (f) => path.join(__dirname, 'deck_img', f);

(async () => {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = 'Pringles from the cabinet: pipeline and architecture';
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: 'TITLE_DARK', background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.3, w: 11.7, h: 1.4,
        fontSize: 44, bold: true, color: C.background1, valign: 'bottom', align: 'left' }, text: '' } },
      { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 3.85, w: 11.7, h: 1.4,
        fontSize: 20, color: C.background2, valign: 'top' }, text: '' } },
    ],
  });
  pres.defineSlideMaster({
    title: 'CONTENT', background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: 'title', type: 'title', x: 0.5, y: 0.3, w: 12.33, h: 0.8,
        fontSize: 32, bold: true, color: C.text2, valign: 'middle', align: 'left' }, text: '' } },
      { text: { text: 'Pringles pick · pipeline and comms', options: { x: 0.5, y: 7.05, w: 6, h: 0.3,
        fontSize: 10, color: '7A7A7A' } } },
    ],
    slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: '7A7A7A' },
  });

  const fitImage = (slide, file, box, name) => {
    const { w, h } = sizeOf(IMG(file));
    let iw = box.w, ih = (box.w * h) / w;
    if (ih > box.h) { ih = box.h; iw = (box.h * w) / h; }
    slide.addImage({ path: IMG(file), x: box.x + (box.w - iw) / 2, y: box.y, w: iw, h: ih,
      altText: name, objectName: name });
    return { x: box.x + (box.w - iw) / 2, y: box.y, w: iw, h: ih };
  };
  const card = (slide, x, y, w, h, head, body, tint, headColor, name) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
      fill: { color: tint }, line: { color: tint }, objectName: name + ' card' });
    slide.addText([
      { text: head, options: { bold: true, fontSize: 16, color: headColor, breakLine: true } },
      { text: body, options: { fontSize: 14, color: C.text1 } },
    ], { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2, valign: 'top', isTextBox: true,
      paraSpaceAfter: 4, objectName: name + ' text' });
  };

  // 1 Title
  pres.addSection({ title: 'Overview' });
  let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: 'Overview' });
  s.addText('Pringles from the Cabinet', { placeholder: 'title' });
  s.addText('Task pipeline and system architecture\nStretch robot · wired room cameras · GPU server · what crosses the robot’s Wi-Fi',
    { placeholder: 'body' });
  s.addNotes('Two parts: the task pipeline step by step, then the system architecture with the comms layer.');

  // 2 Pipeline overview
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
  s.addText('The task, step by step', { placeholder: 'title' });
  const r = fitImage(s, 'pipe_full.png', { x: 0.5, y: 1.2, w: 12.33, h: 4.6 }, 'Task pipeline');
  const stats = [
    ['3 steps', 'nothing crosses the Wi-Fi', 'open · grasp + lift · hand over', C.text2],
    ['3 steps', 'a few bytes', 'acknowledge · global navigation · return', C.accent1],
    ['2 steps', 'one photo (~50 KB)', 'local navigation · detect', C.accent3],
  ];
  stats.forEach(([big, mid, small, col], i) => {
    const x = 0.5 + i * 4.18;
    s.addText([
      { text: big, options: { fontSize: 30, bold: true, color: col, breakLine: true } },
      { text: mid, options: { fontSize: 16, bold: true, color: C.text1, breakLine: true } },
      { text: small, options: { fontSize: 12, color: '666666' } },
    ], { x, y: r.y + r.h + 0.15, w: 3.95, h: 1.1, isTextBox: true, valign: 'top', margin: 0,
      objectName: 'Stat ' + (i + 1) });
  });
  s.addNotes('Read the Wi-Fi badges across the steps: most steps need nothing or a few bytes; only two steps send one photo each.');

  // 3-4 Pipeline halves
  pres.addSection({ title: 'Pipeline' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pipeline' });
  s.addText('Steps 1–4: acknowledge, navigate, open the cabinet', { placeholder: 'title' });
  fitImage(s, 'pipe_left.png', { x: 0.5, y: 1.2, w: 12.33, h: 5.7 }, 'Pipeline steps 1 to 4');
  s.addNotes('Acknowledge and global navigation exchange only targets and poses. Nav2, AMCL and obstacle avoidance run on the robot. Local navigation sends one photo and gets the handle position back. Opening the cabinet is entirely local.');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pipeline' });
  s.addText('Steps 5–8: pick up, return, hand over', { placeholder: 'title' });
  fitImage(s, 'pipe_right.png', { x: 0.5, y: 1.2, w: 12.33, h: 5.7 }, 'Pipeline steps 5 to 8');
  s.addNotes('Detect sends one photo and gets the best mask back; depth stays on the robot. Grasp and hand over are local. Return uses the person position from the room cameras, with its time, so the robot knows how old it is.');

  // 5 Architecture
  pres.addSection({ title: 'Architecture' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  s.addText('System architecture', { placeholder: 'title' });
  const a = fitImage(s, 'arch_full.png', { x: 0.5, y: 1.2, w: 8.9, h: 5.7 }, 'System architecture');
  const ax = a.x + a.w + 0.35, aw = 12.83 - ax;
  card(s, ax, 1.2, aw, 1.75, 'Robot', 'Sensors, localization, task executive, object fusion, motion, and the robot side of the comms bridge.', 'E4EEFB', C.accent2, 'Robot');
  card(s, ax, 3.1, aw, 1.75, 'Server', 'Gaussian SLAM map, room cameras, object retrieval, asset layer, and the server side of the comms bridge.', 'EFEAFB', C.accent4, 'Server');
  card(s, ax, 5.0, aw, 1.75, 'Link', 'Four tiers: request (pick), background (keyframe packs), offline (bags at the dock), once (calibration, map).', 'DDF3F0', C.accent1, 'Link');
  s.addNotes('Left: everything on the robot. Right: everything on the server. The red dashed line is the Wi-Fi.');

  // 6 Robot side
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  s.addText('Robot side: time-critical work stays local', { placeholder: 'title' });
  const rb = fitImage(s, 'arch_robot.png', { x: 0.5, y: 1.2, w: 4.2, h: 5.7 }, 'Robot side of the architecture');
  const rx = rb.x + rb.w + 0.45, rw = 12.83 - rx;
  card(s, rx, 1.2, rw, 1.65, 'Runs on the robot', 'Localization (AMCL / LiDAR-inertial odometry), Nav2 with obstacle avoidance, mask × depth → 3-D points, IK, arm and gripper.', 'E4EEFB', C.accent2, 'Local');
  card(s, rx, 3.0, rw, 1.95, 'Comms bridge (robot)', 'offload_client: one photo per look, with ID and capture time · keyframe_packer: background, paused during the pick · recorder: full-rate data to disk, uploaded at the dock · link_monitor: RTT, RSSI, server queue · policy: wait, retry or fall back.', 'DDF3F0', C.accent1, 'Bridge');
  card(s, rx, 5.1, rw, 1.65, 'When an answer is late', 'Retry once with a new photo, keep driving to the current goal, slow down near the person, or use the onboard detector.', 'FBE9E7', C.accent3, 'Late');
  s.addNotes('The robot never needs the server to stay safe or to move; it needs the server only for the big models.');

  // 7 Server side
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  s.addText('Server side: map, retrieval, assets', { placeholder: 'title' });
  const sb = fitImage(s, 'arch_server.png', { x: 0.5, y: 1.2, w: 6.2, h: 5.7 }, 'Server side of the architecture');
  const sx = sb.x + sb.w + 0.45, sw = 12.83 - sx;
  card(s, sx, 1.2, sw, 1.4, 'Offline', 'Gaussian SLAM map (~16 h per run), link map seeded from its Sionna export.', 'E4F4E9', C.accent5, 'Offline');
  card(s, sx, 2.75, sw, 1.65, 'Online', 'Keyframe search → SAM 3 → object cloud → object cache → target with its age. Room cameras are RGB only: depth from the map.', 'FFF7E0', '8A6A00', 'Online');
  card(s, sx, 4.55, sw, 1.0, 'Background', 'SAM 3D asset per object (≥32 GB GPU).', 'EFEAFB', C.accent4, 'Assets');
  card(s, sx, 5.7, sw, 1.05, 'Comms bridge (server)', 'GPU scheduler serves pick requests before training and SAM 3D.', 'DDF3F0', C.accent1, 'ServerBridge');
  s.addNotes('The room cameras are wired to the server, so none of their images use the robot’s Wi-Fi.');

  // 8 What crosses the Wi-Fi
  pres.addSection({ title: 'Link' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Link' });
  s.addText('What crosses the robot’s Wi-Fi', { placeholder: 'title' });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const rows = [
    [hdr('Tier'), hdr('What'), hdr('When'), hdr('Priority')],
    ['Request', 'targets with their age · one JPEG per look · masks and keypoints back · robot pose', 'during the pick', 'high'],
    ['Background', 'keyframe packs: lidar scan + JPEG + pose, ~2 Hz, only when the view changes', 'outside the pick; paused while detecting and grasping', 'low'],
    ['Offline', 'full-rate frames, bags, IMU and trajectory for map training', 'at the dock or overnight', 'lowest'],
    ['Once', 'calibration; 2-D grid per map version', 'set-up', '—'],
  ];
  s.addTable(rows, { x: 0.5, y: 1.3, w: 12.33, colW: [1.6, 6.0, 3.33, 1.4], fontSize: 15,
    color: C.text1, border: { type: 'solid', pt: 0.75, color: 'C8D6D3' },
    fill: { color: C.background1 }, valign: 'middle', rowH: [0.5, 0.75, 0.75, 0.65, 0.55],
    objectName: 'Wi-Fi tiers' });
  s.addText('Never sent during a task: raw image streams, depth images, raw lidar scans. Every request carries an ID and capture time; every answer echoes them.',
    { x: 0.5, y: 5.0, w: 12.33, h: 0.8, fontSize: 16, color: C.accent1, bold: true, isTextBox: true,
      margin: 0, objectName: 'Rule' });
  s.addNotes('Pick requests always go first; background and offline traffic give way while the robot is detecting and grasping.');

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(OUT);
})();
