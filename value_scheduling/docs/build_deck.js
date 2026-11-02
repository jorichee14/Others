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
  // Fonts are set explicitly so the text keeps its size when slides are pasted into another deck.
  const FONT = 'Calibri';
  const addT = (sl, t, o) => sl.addText(t, { fontFace: FONT, ...o });

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
    addT(slide, [
      { text: head, options: { bold: true, fontSize: 16, color: headColor, breakLine: true } },
      { text: body, options: { fontSize: 14, color: C.text1 } },
    ], { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2, valign: 'top', isTextBox: true,
      paraSpaceAfter: 4, objectName: name + ' text' });
  };

  // 1 Title
  pres.addSection({ title: 'Overview' });
  let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: 'Overview' });
  addT(s, 'Pringles from the Cabinet', { placeholder: 'title' });
  addT(s, 'Task pipeline and system architecture\nStretch robot · wired room cameras · GPU server · what crosses the robot’s Wi-Fi',
    { placeholder: 'body' });
  s.addNotes('Two parts: the task pipeline step by step, then the system architecture with the comms layer.');

  // ---------- pipeline as native shapes ----------
  const PIC = (f) => path.join(__dirname, 'pick_img', f);
  const GRP = {
    task: { bg: 'EFEAFB', fg: '5B3FB0' }, nav: { bg: 'E8F0FB', fg: '1A5FD0' },
    man: { bg: 'FBE9E7', fg: 'A3171D' }, hand: { bg: 'E6F4EA', fg: '1D7A3A' },
  };
  const RUNC = { robot: ['Robot', 'DFF3E6', '1F6F43'], server: ['Server', 'ECE6FB', '5B3FB0'] };
  const LOADC = { none: ['Wi-Fi: none', 'E9ECEF', '555555'], tiny: ['Wi-Fi: tiny (bytes)', 'CFEEE9', '0B5F57'],
    photo: ['Wi-Fi: one photo (~50 KB)', '0B7F74', 'FFFFFF'] };
  const STEPS = [
    { g: 'task', group: 'Acknowledge', n: 1, title: 'Acknowledge', img: null, ph: 'room-camera view', run: ['server'], load: 'tiny',
      does: 'Turns “give me the Pringles from the cabinet” into two targets.',
      inp: 'request text · room-camera images', out: 'cabinet view pose and person position, each with its time',
      cam: 'find the cabinet and the person', up: 'request text', down: 'the two targets with their times', late: 'ask again' },
    { g: 'nav', group: 'Navigate to the cabinet', n: 2, title: 'Global navigation', img: 'global_nav.png', run: ['robot'], load: 'tiny',
      does: 'Nav2 drives to the cabinet view pose; AMCL and the robot’s lidar avoid obstacles.',
      inp: 'goal · 2D lidar scan', out: 'robot near the cabinet',
      cam: 'check the cabinet is still where the map says', up: 'robot pose (status only)', down: 'new goal, only if the target changed', late: 'keep driving to the current goal' },
    { g: 'nav', group: 'Navigate to the cabinet', n: 3, title: 'Local navigation', img: 'local_nav.png', run: ['robot', 'server'], load: 'photo',
      does: 'One photo → handle point (server); door plane from depth (robot); rotate – translate – rotate.',
      inp: 'one RGB-D image, robot stopped', out: 'robot in position to open the door',
      cam: '—', up: 'one JPEG photo', down: 'the handle’s pixel position', late: 'new photo, ask again' },
    { g: 'man', group: 'Open the cabinet', n: 4, title: 'Grasp · open · retract', img: ['grasp_handle.png', 'open.png', 'retract.png'], run: ['robot'], load: 'none',
      does: 'Grasp the handle, swing the door open along its arc, retract the arm.',
      inp: 'handle position · door plane', out: 'cabinet open, arm clear of the door', cam: '—' },
    { g: 'man', group: 'Pick up the Pringles', n: 5, title: 'Detect', img: 'retract.png', run: ['robot', 'server'], load: 'photo',
      does: 'One photo → Pringles mask (YOLOE); mask × robot depth → 3-D centre.',
      inp: 'one RGB-D image · “pringles”', out: 'Pringles centre in the robot frame',
      cam: 'hint where the Pringles are, if a camera sees inside', up: 'one JPEG photo', down: 'the best mask', late: 'retry once, then YOLOE on the robot' },
    { g: 'man', group: 'Pick up the Pringles', n: 6, title: 'Grasp + lift', img: 'pick.png', run: ['robot'], load: 'none',
      does: 'IK: pre-grasp → approach → close the gripper → lift 10 cm.',
      inp: 'Pringles centre · joint states', out: 'Pringles in the gripper', cam: '—' },
    { g: 'nav', group: 'Return to the person', n: 7, title: 'Return', img: null, ph: 'room-camera view of the person', run: ['robot'], load: 'tiny',
      does: 'Nav2 drives to the person; the goal moves when the person moves.',
      inp: 'person position · 2D lidar scan', out: 'robot in front of the person',
      cam: 'track the person', up: 'robot pose (status only)', down: 'person position when it changes, with its time', late: 'slow down, ask for an update' },
    { g: 'hand', group: 'Hand over', n: 8, title: 'Hand over', img: 'handover.png', run: ['robot'], load: 'none',
      does: 'Hold the Pringles at hand height; open the gripper when the person pulls.',
      inp: 'gripper effort', out: 'the person holds the Pringles', cam: '—' },
  ];
  const groupsOf = (steps) => {
    const out = [];
    steps.forEach((st) => {
      const last = out[out.length - 1];
      if (last && last.name === st.group) last.steps.push(st); else out.push({ name: st.group, g: st.g, steps: [st] });
    });
    return out;
  };
  const chip = (slide, x, y, label, bg, fg, w) => {
    addT(slide, label, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x, y, w, h: 0.22,
      fill: { color: bg }, line: { color: bg }, fontSize: 10, bold: true, color: fg, align: 'center',
      valign: 'middle', margin: 0, objectName: 'chip ' + label });
  };
  const taskLine = (slide, y) => {
    const groups = groupsOf(STEPS);
    const runs = [{ text: 'Robot: ', options: { bold: true, color: '1D1D1F' } }];
    groups.forEach((gr, i) => {
      runs.push({ text: gr.name, options: { bold: true, color: GRP[gr.g].fg } });
      if (i < groups.length - 1) runs.push({ text: '  →  ', options: { bold: true, color: '555555' } });
    });
    addT(slide, runs, { x: 0.5, y, w: 12.33, h: 0.35, fontSize: 15, isTextBox: true, margin: 0, objectName: 'Task line' });
  };

  // detailed card
  const CARD_H = 5.1;
  const drawCard = (slide, st, x, y, w) => {
    const ix = x + 0.1, iw = w - 0.2;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: CARD_H, rectRadius: 0.08,
      fill: { color: 'FFFFFF' }, line: { color: 'D6D6D6', width: 1 }, objectName: `Step ${st.n} card` });
    // picture
    const ph = 1.1;
    if (Array.isArray(st.img)) {
      const pw = (iw - 0.1) / 3;
      st.img.forEach((f, k) => slide.addImage({ path: PIC(f), x: ix + k * (pw + 0.05), y: y + 0.1, w: pw, h: ph,
        sizing: { type: 'cover', w: pw, h: ph }, altText: `${st.title} photo ${k + 1}`, objectName: `Step ${st.n} photo ${k + 1}` }));
    } else if (st.img) {
      slide.addImage({ path: PIC(st.img), x: ix, y: y + 0.1, w: iw, h: ph, sizing: { type: 'cover', w: iw, h: ph },
        altText: st.title, objectName: `Step ${st.n} photo` });
    } else {
      addT(slide, [{ text: st.ph, options: { breakLine: true } }, { text: '(image to add)', options: { fontSize: 9, color: '999999' } }],
        { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x: ix, y: y + 0.1, w: iw, h: ph,
          fill: { color: 'FAFAFA' }, line: { color: 'AAAAAA', dashType: 'dash', width: 1 }, fontSize: 11, color: '777777',
          align: 'center', valign: 'middle', objectName: `Step ${st.n} placeholder` });
    }
    // number + title
    let cy = y + 0.1 + ph + 0.08;
    addT(slide, String(st.n), { shape: pres.shapes.OVAL, x: ix, y: cy + 0.02, w: 0.28, h: 0.28,
      fill: { color: '333333' }, line: { color: '333333' }, fontSize: 11, bold: true, color: 'FFFFFF',
      align: 'center', valign: 'middle', margin: 0, objectName: `Step ${st.n} number` });
    addT(slide, st.title, { x: ix + 0.36, y: cy, w: iw - 0.36, h: 0.32, fontSize: 15, bold: true, color: '1D1D1F',
      valign: 'middle', isTextBox: true, margin: 0, objectName: `Step ${st.n} title` });
    // runs on
    cy += 0.38;
    addT(slide, 'Runs on', { x: ix, y: cy, w: 0.6, h: 0.22, fontSize: 10, color: '666666', valign: 'middle',
      isTextBox: true, margin: 0, objectName: `Step ${st.n} runs label` });
    st.run.forEach((r, k) => chip(slide, ix + 0.62 + k * 0.68, cy, RUNC[r][0], RUNC[r][1], RUNC[r][2], 0.62));
    // does
    cy += 0.3;
    addT(slide, st.does, { x: ix, y: cy, w: iw, h: 0.62, fontSize: 11.5, color: '333333', valign: 'top',
      isTextBox: true, margin: 0, objectName: `Step ${st.n} description` });
    // input / output
    cy += 0.62;
    addT(slide, [
      { text: 'Input: ', options: { bold: true, color: '1A5FD0' } }, { text: st.inp, options: { color: '444444', breakLine: true } },
      { text: 'Output: ', options: { bold: true, color: '1A5FD0' } }, { text: st.out, options: { color: '444444' } },
    ], { x: ix, y: cy, w: iw, h: 0.62, fontSize: 10.5, valign: 'top', isTextBox: true, margin: 0,
      paraSpaceAfter: 2, objectName: `Step ${st.n} input output` });
    // room cameras
    cy += 0.62;
    addT(slide, [{ text: 'Room cameras: ', options: { bold: true } }, { text: st.cam }],
      { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x: ix, y: cy, w: iw, h: 0.42,
        fill: { color: 'F3EFFC' }, line: { color: 'F3EFFC' }, fontSize: 10.5, color: '3B2A7A', valign: 'middle',
        margin: [2, 5, 2, 5], objectName: `Step ${st.n} room cameras` });
    // Wi-Fi box
    cy += 0.48;
    const wh = y + CARD_H - 0.1 - cy;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ix, y: cy, w: iw, h: wh, rectRadius: 0.05,
      fill: { color: 'FFFFFF' }, line: { color: 'B9E3DC', width: 1.25 }, objectName: `Step ${st.n} wifi box` });
    const [lab, bg, fg] = LOADC[st.load];
    chip(slide, ix + 0.07, cy + 0.07, lab, bg, fg, st.load === 'photo' ? 1.75 : st.load === 'tiny' ? 1.3 : 0.85);
    const lines = st.up || st.down ? [
      ...(st.up ? [{ text: 'Robot → server: ', options: { bold: true, color: '0B7F74' } }, { text: st.up, options: { breakLine: true } }] : []),
      ...(st.down ? [{ text: 'Server → robot: ', options: { bold: true, color: '0B7F74' } }, { text: st.down, options: { breakLine: true } }] : []),
      { text: 'If late: ', options: { bold: true, color: 'A3171D' } }, { text: st.late },
    ] : [{ text: 'nothing crosses the Wi-Fi' }];
    addT(slide, lines, { x: ix + 0.07, y: cy + 0.33, w: iw - 0.14, h: wh - 0.38, fontSize: 10, color: '174F49',
      valign: 'top', isTextBox: true, margin: 0, paraSpaceAfter: 2, objectName: `Step ${st.n} wifi text` });
  };

  const drawRow = (slide, steps, y0) => {
    const groups = groupsOf(steps);
    const PAD = 0.12, GAP = 0.15, ARR = 0.3, ncards = steps.length;
    const intra = groups.reduce((a, gr) => a + (gr.steps.length - 1) * GAP, 0);
    const cw = (12.33 - intra - groups.length * 2 * PAD - (groups.length - 1) * ARR) / ncards;
    let x = 0.5;
    groups.forEach((gr, gi) => {
      const pw = gr.steps.length * cw + (gr.steps.length - 1) * GAP + 2 * PAD;
      slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: pw, h: CARD_H + 0.55, rectRadius: 0.12,
        fill: { color: GRP[gr.g].bg }, line: { color: GRP[gr.g].bg }, objectName: gr.name + ' panel' });
      addT(slide, gr.name, { x: x + PAD, y: y0 + 0.06, w: pw - 2 * PAD, h: 0.36, fontSize: 15, bold: true,
        color: GRP[gr.g].fg, valign: 'middle', isTextBox: true, margin: 0, objectName: gr.name + ' heading' });
      gr.steps.forEach((st, k) => drawCard(slide, st, x + PAD + k * (cw + GAP), y0 + 0.45, cw));
      x += pw;
      if (gi < groups.length - 1) {
        slide.addShape(pres.shapes.RIGHT_ARROW, { x: x + 0.05, y: y0 + 0.45 + CARD_H / 2 - 0.1, w: ARR - 0.1, h: 0.2,
          fill: { color: '666666' }, line: { color: '666666' }, objectName: 'arrow ' + gi });
        x += ARR;
      }
    });
  };

  // 2 Overview as shapes
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
  addT(s, 'The task, step by step', { placeholder: 'title' });
  taskLine(s, 1.15);
  {
    const groups = groupsOf(STEPS);
    const PAD = 0.1, GAP = 0.1, ARR = 0.22;
    const intra = groups.reduce((a, gr) => a + (gr.steps.length - 1) * GAP, 0);
    const cw = (12.33 - intra - groups.length * 2 * PAD - (groups.length - 1) * ARR) / STEPS.length;
    let x = 0.5; const y0 = 1.7, ch = 2.5;
    groups.forEach((gr, gi) => {
      const pw = gr.steps.length * cw + (gr.steps.length - 1) * GAP + 2 * PAD;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: pw, h: ch + 0.5, rectRadius: 0.1,
        fill: { color: GRP[gr.g].bg }, line: { color: GRP[gr.g].bg }, objectName: gr.name + ' panel' });
      addT(s, gr.name === 'Return to the person' ? 'Return' : gr.name, { x: x + PAD, y: y0 + 0.05, w: pw - 2 * PAD, h: 0.36, fontSize: 12, bold: true,
        color: GRP[gr.g].fg, valign: 'middle', isTextBox: true, margin: 0, objectName: gr.name + ' heading' });
      gr.steps.forEach((st, k) => {
        const cx = x + PAD + k * (cw + GAP), cy = y0 + 0.42;
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: cw, h: ch, rectRadius: 0.06,
          fill: { color: 'FFFFFF' }, line: { color: 'D6D6D6', width: 1 }, objectName: `Overview ${st.n} card` });
        addT(s, String(st.n), { shape: pres.shapes.OVAL, x: cx + 0.08, y: cy + 0.1, w: 0.3, h: 0.3,
          fill: { color: '333333' }, line: { color: '333333' }, fontSize: 11, bold: true, color: 'FFFFFF',
          align: 'center', valign: 'middle', margin: 0, objectName: `Overview ${st.n} number` });
        addT(s, st.title, { x: cx + 0.08, y: cy + 0.45, w: cw - 0.16, h: 0.55, fontSize: 13, bold: true,
          color: '1D1D1F', valign: 'top', isTextBox: true, margin: 0, objectName: `Overview ${st.n} title` });
        st.run.forEach((r, kk) => chip(s, cx + 0.08, cy + 1.05 + kk * 0.27, RUNC[r][0], RUNC[r][1], RUNC[r][2], 0.7));
        const [lab, bg, fg] = LOADC[st.load];
        addT(s, lab.replace('Wi-Fi: ', ''), { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06,
          x: cx + 0.08, y: cy + ch - 0.62, w: cw - 0.16, h: 0.5, fill: { color: bg }, line: { color: bg },
          fontSize: 11, bold: true, color: fg, align: 'center', valign: 'middle', margin: 2,
          objectName: `Overview ${st.n} wifi` });
      });
      x += pw;
      if (gi < groups.length - 1) {
        s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 0.03, y: y0 + 0.42 + ch / 2 - 0.08, w: ARR - 0.06, h: 0.16,
          fill: { color: '666666' }, line: { color: '666666' }, objectName: 'overview arrow ' + gi });
        x += ARR;
      }
    });
    addT(s, 'Wi-Fi load per step', { x: 0.5, y: 4.85, w: 4, h: 0.3, fontSize: 12, color: '666666', italic: true,
      isTextBox: true, margin: 0, objectName: 'Overview caption' });
  }
  const stats = [
    ['3 steps', 'nothing crosses the Wi-Fi', 'open · grasp + lift · hand over', C.text2],
    ['3 steps', 'a few bytes', 'acknowledge · global navigation · return', C.accent1],
    ['2 steps', 'one photo (~50 KB)', 'local navigation · detect', C.accent3],
  ];
  stats.forEach(([big, mid, small, col], i) => {
    addT(s, [
      { text: big, options: { fontSize: 30, bold: true, color: col, breakLine: true } },
      { text: mid, options: { fontSize: 16, bold: true, color: C.text1, breakLine: true } },
      { text: small, options: { fontSize: 12, color: '666666' } },
    ], { x: 0.5 + i * 4.18, y: 5.3, w: 3.95, h: 1.3, isTextBox: true, valign: 'top', margin: 0, objectName: 'Stat ' + (i + 1) });
  });
  s.addNotes('Overview: each step, where it runs, and how much it uses the Wi-Fi. Details on the next two slides.');

  // 3-4 detailed halves as shapes
  pres.addSection({ title: 'Pipeline' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pipeline' });
  addT(s, 'Steps 1–4: acknowledge, navigate, open the cabinet', { placeholder: 'title' });
  drawRow(s, STEPS.slice(0, 4), 1.2);
  s.addNotes('Acknowledge and global navigation exchange only targets and poses; Nav2, AMCL and obstacle avoidance run on the robot. Local navigation sends one photo and gets the handle position back. Opening the cabinet is local.');

  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pipeline' });
  addT(s, 'Steps 5–8: pick up, return, hand over', { placeholder: 'title' });
  drawRow(s, STEPS.slice(4), 1.2);
  s.addNotes('Detect sends one photo and gets the best mask back; depth stays on the robot. Grasp and hand over are local. Return uses the person position from the room cameras, with its time.');

  // ---------- our own system architecture (multi-robot mapping, twin, simulation, radio) ----------
  pres.addSection({ title: 'Architecture' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'System architecture', { placeholder: 'title' });
  {
    const k = 0.00648, X = (p) => 0.32 + p * k, Y = (p) => 1.05 + p * k, L = (p) => p * k;
    const PAN = {
      hw: ['F0F1F3', 'C3C7CF'], slam: ['E6EEF9', 'A9C3EC'], scene: ['FBEFD9', 'EFCB96'],
      twin: ['E6F4EA', '9FD3AE'], sim: ['FCF3E3', 'EFCB96'], radio: ['EEEAF8', 'C3B2EC'],
    };
    const panel = (x1, y1, x2, y2, kind, title, right) => {
      const [bg, bd] = PAN[kind];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X(x1), y: Y(y1), w: L(x2 - x1), h: L(y2 - y1), rectRadius: 0.05,
        fill: { color: bg }, line: { color: bd, width: 1 }, objectName: title + ' panel' });
      addT(s, title, { x: X(x1) + 0.08, y: Y(y1) + 0.03, w: L(x2 - x1) * 0.62, h: 0.24, fontSize: 11, bold: true,
        color: '1D1D1F', valign: 'middle', isTextBox: true, margin: 0, objectName: title + ' title' });
      if (right) addT(s, right, { x: X(x1) + L(x2 - x1) * 0.45, y: Y(y1) + 0.03, w: L(x2 - x1) * 0.55 - 0.08, h: 0.24,
        fontSize: 8, color: '555555', align: 'right', valign: 'middle', isTextBox: true, margin: 0, objectName: title + ' note' });
    };
    const card = (x1, y1, x2, y2, title, body, dashed) => {
      s.addShape(pres.shapes.RECTANGLE, { x: X(x1), y: Y(y1), w: L(x2 - x1), h: L(y2 - y1), fill: { color: 'FFFFFF' },
        line: { color: dashed ? 'BDBDBD' : 'B8BCC4', width: 1, dashType: dashed ? 'dash' : 'solid' }, objectName: title + ' card' });
      const runs = [{ text: title, options: { bold: true, fontSize: 9.5, color: '1D1D1F', breakLine: !!body } }];
      if (body) runs.push({ text: body, options: { fontSize: 7.5, color: '444444' } });
      addT(s, runs, { x: X(x1) + 0.03, y: Y(y1) + 0.02, w: L(x2 - x1) - 0.06, h: L(y2 - y1) - 0.04, align: 'center',
        valign: 'middle', isTextBox: true, margin: 0, objectName: title + ' text' });
    };
    const seg = (x1, y1, x2, y2, arrow, color, dash) => {
      const o = { x: X(Math.min(x1, x2)), y: Y(Math.min(y1, y2)), w: L(Math.abs(x2 - x1)), h: L(Math.abs(y2 - y1)),
        line: { color: color || '222222', width: 1.25, dashType: dash || 'solid' }, objectName: 'arrow' };
      if (arrow) { if (x2 > x1 || y2 > y1) o.line.endArrowType = 'triangle'; else o.line.beginArrowType = 'triangle'; }
      s.addShape(pres.shapes.LINE, o);
    };
    const path = (pts, color, dash) => pts.slice(1).forEach((p, i) => seg(pts[i][0], pts[i][1], p[0], p[1], i === pts.length - 2, color, dash));
    const lab = (x, y, t, w) => addT(s, t, { x: X(x), y: Y(y), w: L(w || 150), h: 0.16, fontSize: 7.5, bold: true, color: '333333',
      isTextBox: true, margin: 0, objectName: 'label ' + t });

    // headers + network line
    addT(s, 'ROBOT (each of 2) · Jetson Orin NX 16 GB · ROS 2 Humble', { x: X(12), y: Y(2), w: L(680), h: 0.2, fontSize: 9.5, bold: true, color: '333333', isTextBox: true, margin: 0, objectName: 'robot header' });
    addT(s, 'SERVER · workstation with RTX GPU · ROS 2, Isaac Sim, Sionna RT', { x: X(750), y: Y(2), w: L(900), h: 0.2, fontSize: 9.5, bold: true, color: '333333', isTextBox: true, margin: 0, objectName: 'server header' });
    s.addShape(pres.shapes.LINE, { x: X(725), y: Y(0), w: 0, h: L(880), line: { color: 'D6262C', width: 1.75, dashType: 'dash' }, objectName: 'network line' });

    // ---- robot ----
    panel(12, 42, 700, 183, 'hw', 'Sensors + robot');
    [['Ouster LiDAR · 10 Hz', 37], ['IMU · 100 Hz', 256], ['Drive base', 475]].forEach(([t, x]) => card(x, 87, x + 200, 120, t));
    [['RGB-D camera', 37], ['Wheel enc. (backup)', 256], ['Radio antenna', 475]].forEach(([t, x]) => card(x, 133, x + 200, 167, t));
    panel(12, 212, 700, 360, 'slam', 'Start + odometry');
    card(37, 257, 350, 345, 'Odometry source', 'LiDAR odometry ATE 1.0–4.4 cm\nfallback RKO-LIO · cuVSLAM 23–47 cm');
    card(381, 257, 688, 345, 'Board start', 'ChArUco → initial guess in map\ndetection online: to build', true);
    panel(12, 383, 700, 551, 'slam', 'Map localization', 'offline equivalent: 1.3 cm');
    card(37, 428, 350, 527, 'Fused pose', '2Fast-2Lamaa / FAST-LIO2 odometry\n10 Hz, map frame, bounded error', true);
    card(381, 428, 688, 527, 'map→odom', 'in map: scan-to-map on map tiles, 1–2 Hz\noutside: server SLAM correction\nserver lost: peer Swarm-SLAM', true);
    panel(12, 573, 700, 815, 'slam', 'Link to server', 'rates estimated');
    s.addShape(pres.shapes.RECTANGLE, { x: X(37), y: Y(616), w: L(651), h: L(188), fill: { color: 'FFFFFF' },
      line: { color: 'BDBDBD', width: 1, dashType: 'dash' }, objectName: 'link table' });
    const links = [
      ['↑ Keyframe packs', 'scan + JPEG + pose · ~2 Hz · 0.5–1 MB/s → SLAM'],
      ['↑ Map-frame pose', 'fused pose · 10 Hz · ~1 KB/s → robot prims'],
      ['↓ Map tiles', 'twin map + new SLAM areas + scene edits · 1–5 MB, rare'],
      ['↓ SLAM corrections', 'keyframe ID + pose · ~10 s · <100 B · outside map'],
      ['↓ Control commands', 'from Isaac Sim / operator → drive base'],
      ['↕ Peer (robot j)', 'Swarm-SLAM descriptors, scans · server lost only'],
    ];
    links.forEach(([a, b], i) => {
      const yy = 626 + i * 29;
      addT(s, a, { x: X(48), y: Y(yy), w: L(185), h: 0.17, fontSize: 8.5, bold: true, color: '1D1D1F', isTextBox: true, margin: 0, objectName: 'link ' + a });
      addT(s, b, { x: X(240), y: Y(yy), w: L(440), h: 0.17, fontSize: 8, color: '444444', isTextBox: true, margin: 0, objectName: 'link value ' + a });
    });
    card(12, 838, 350, 892, 'Robot j', 'same stack · peer Swarm-SLAM');
    card(375, 838, 700, 892, 'Clock sync', 'PTP / chrony: robots, server, sniffer', true);
    // robot arrows
    seg(313, 183, 313, 254, true); seg(450, 183, 450, 254, true);
    seg(313, 345, 313, 425, true); lab(323, 352, 'odometry');
    seg(450, 345, 450, 425, true); lab(460, 352, 'initial guess');
    seg(381, 478, 352, 478, true);
    seg(275, 527, 275, 613, true);
    seg(181, 806, 181, 836, true); seg(181, 836, 181, 806, true);

    // ---- server ----
    panel(750, 42, 1146, 203, 'slam', 'Multi-Agent SLAM', 'centralised');
    card(770, 90, 945, 181, 'Map store', 'keyframe packs in;\ntiles + corrections out', true);
    card(957, 90, 1132, 181, 'Swarm-SLAM', 'inter-robot + map\n→ SLAM corrections');
    seg(945, 135, 956, 135, true);
    panel(1165, 42, 1562, 203, 'slam', 'Live mapping', 'GPU');
    card(1185, 90, 1360, 181, 'Splat mapper', 'Gaussian-LIC\nROS 2 port', true);
    card(1372, 90, 1547, 181, 'Live map', 'Gaussian splats\n+ LiDAR points', true);
    seg(1132, 135, 1184, 135, true); seg(1360, 135, 1371, 135, true);
    panel(1587, 42, 1887, 408, 'twin', 'Offline twin', 'versioned');
    card(1607, 87, 1867, 145, 'Twin update', 're-run on stored keyframe packs', true);
    card(1607, 165, 1867, 225, 'GLIM + BA', 'GLIM map, board check: 4 mm');
    card(1607, 243, 1867, 302, 'Anchored map', 'reference ~1.3 cm · version n');
    card(1607, 322, 1867, 382, 'LOD-3DGS + mesh', 'twin splats, radio mesh');
    seg(1737, 145, 1737, 164, true); seg(1737, 225, 1737, 242, true); seg(1737, 302, 1737, 321, true);
    panel(750, 228, 1562, 408, 'scene', 'Scene update');
    card(775, 278, 1000, 380, 'Decomposition', 'segments, materials\nmaintenance: prune, reuse', true);
    card(1025, 278, 1250, 380, 'Object retrieval', 'pose, point cloud, class\nfrom the asset library', true);
    card(1275, 278, 1500, 380, 'Change detection', 'live scans vs current twin\nfree-space carving', true);
    seg(1275, 330, 1251, 330, true); seg(1025, 330, 1001, 330, true);
    seg(1411, 181, 1411, 276, true); lab(1421, 205, 'live map');
    path([[1500, 352], [1575, 352], [1575, 115], [1605, 115]], '222222', 'dash');
    panel(750, 437, 1887, 603, 'sim', 'Isaac Sim', 'USD stage');
    card(775, 487, 1030, 580, 'Robot + antenna prims', 'one per robot, map-frame pose\nevery frame', true);
    card(1053, 487, 1308, 580, 'Semantic layer', 'class + instance ID per prim\nfrom scene update', true);
    card(1331, 487, 1586, 580, 'Gaussian layer', 'twin splats + live map\nRTX visuals', true);
    card(1609, 487, 1865, 580, 'Mesh layer', 'physics, radio materials', true);
    seg(893, 380, 893, 434, true); lab(903, 395, 'USD edits');
    seg(1075, 380, 1075, 484, true); lab(1085, 395, 'classes, IDs');
    seg(1350, 484, 1350, 382, true); lab(1238, 407, 'expected view');
    seg(1527, 181, 1527, 484, true);
    seg(1737, 382, 1737, 484, true); lab(1637, 446, 'splats, mesh', 95);
    panel(750, 628, 1887, 782, 'radio', 'Sionna RT');
    card(775, 676, 1025, 758, 'Scene export', 'USD to Mitsuba scene');
    card(1050, 676, 1300, 758, 'Path solver', 'all Tx/Rx links');
    card(1325, 676, 1575, 758, 'Channel output', 'CIR per link, radio map');
    card(1600, 676, 1865, 758, 'Validation', 'measured CSI / RSSI (sniffer)\nvs simulated channel', true);
    seg(1025, 717, 1049, 717, true); seg(1300, 717, 1324, 717, true); seg(1575, 717, 1599, 717, true);
    seg(950, 603, 950, 673, true); lab(960, 630, 'meshes, materials, all Tx/Rx poses', 260);

    // ---- across the network ----
    path([[688, 636], [705, 636], [705, 110], [768, 110]]);         // keyframe packs -> map store
    path([[770, 157], [715, 157], [715, 478], [691, 478]]);         // tiles + corrections -> map->odom
    path([[688, 662], [735, 662], [735, 533], [773, 533]]);         // map-frame pose -> robot prims

    // legend
    const LGD = [['hw', 'robot hardware'], ['slam', 'SLAM, mapping'], ['scene', 'scene update'], ['twin', 'offline, per site'], ['sim', 'simulation'], ['radio', 'radio']];
    LGD.forEach(([kk, t], i) => {
      const [bg, bd] = PAN[kk], lx = 750 + i * 190;
      s.addShape(pres.shapes.RECTANGLE, { x: X(lx), y: Y(812), w: L(28), h: L(16), fill: { color: bg }, line: { color: bd, width: 1 }, objectName: 'legend ' + t });
      addT(s, t, { x: X(lx + 36), y: Y(808), w: L(150), h: 0.16, fontSize: 8.5, color: '333333', isTextBox: true, margin: 0, objectName: 'legend text ' + t });
    });
    s.addShape(pres.shapes.LINE, { x: X(750), y: Y(858), w: L(26), h: 0, line: { color: 'D6262C', width: 1.75, dashType: 'dash' }, objectName: 'legend network' });
    addT(s, 'network: robot ↔ server and robot ↔ robot', { x: X(786), y: Y(849), w: L(340), h: 0.16, fontSize: 8.5, color: '333333', isTextBox: true, margin: 0, objectName: 'legend network text' });
    s.addShape(pres.shapes.RECTANGLE, { x: X(1188), y: Y(851), w: L(28), h: L(16), fill: { color: 'FFFFFF' }, line: { color: 'AAAAAA', width: 1, dashType: 'dash' }, objectName: 'legend not built' });
    addT(s, 'not built yet', { x: X(1224), y: Y(849), w: L(120), h: 0.16, fontSize: 8.5, color: '333333', isTextBox: true, margin: 0, objectName: 'legend not built text' });
    addT(s, 'one map→odom owner at a time; server takes over on reconnect', { x: X(1360), y: Y(849), w: L(520), h: 0.16, fontSize: 8.5, italic: true, color: '666666', isTextBox: true, margin: 0, objectName: 'legend note' });
  }
  s.addNotes('Our own two-robot system: robots run odometry and map localization; the server runs multi-agent SLAM, live mapping, scene update, the offline twin, Isaac Sim and Sionna RT. Arrows across the red line are what crosses the network.');

  // ---------- architecture with data flow (native shapes) ----------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'Pick task: architecture and data flow', { placeholder: 'title' });
  const ST = {
    existing: { fill: 'F1F3F5', line: 'B8BEC6', dash: 'solid', w: 1, fg: '333333' },
    nav: { fill: 'E4EEFB', line: 'A9C3EC', dash: 'solid', w: 1, fg: '1A5FD0' },
    task: { fill: 'EFEAFB', line: 'C3B2EC', dash: 'solid', w: 1, fg: '5B3FB0' },
    per: { fill: 'FFF7E0', line: 'E9C46A', dash: 'solid', w: 1, fg: '8A6A00' },
    offline: { fill: 'E4F4E9', line: '9FD3AE', dash: 'solid', w: 1, fg: '1D7A3A' },
    neu: { fill: 'DDF3F0', line: '0B7F74', dash: 'solid', w: 2, fg: '0B7F74' },
    edit: { fill: 'FFFFFF', line: '0B7F74', dash: 'dash', w: 1.75, fg: '0B7F74' },
  };
  const BADGE = { neu: ['NEW', '0B7F74', 'FFFFFF'], edit: ['EDITED', 'FFFFFF', '0B7F74'] };
  const node = (x, y, w, h, kind, title, body, name, small) => {
    const st = ST[kind];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: st.fill },
      line: { color: st.line, width: st.w, dashType: st.dash }, objectName: name + ' box' });
    let tx = x + 0.08;
    if (BADGE[kind]) {
      const [lab, bg, fg] = BADGE[kind];
      const bw = lab === 'NEW' ? 0.42 : 0.62;
      addT(s, lab, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.04, x: x + w - bw - 0.08, y: y + 0.07, w: bw, h: 0.2,
        fill: { color: bg }, line: { color: '0B7F74', width: 0.75 }, fontSize: 8.5, bold: true, color: fg,
        align: 'center', valign: 'middle', margin: 0, objectName: name + ' badge' });
    }
    addT(s, title, { x: tx, y: y + 0.04, w: w - 0.8, h: 0.27, fontSize: small ? 11 : 12, bold: true, color: st.fg,
      valign: 'middle', isTextBox: true, margin: 0, objectName: name + ' title' });
    if (body) addT(s, body, { x: tx, y: y + 0.3, w: w - 0.16, h: h - 0.33, fontSize: 9.5, color: '333333',
      valign: 'top', isTextBox: true, margin: 0, objectName: name + ' body' });
  };
  // arrow: from (x1,y1) to (x2,y2), straight; label optional
  const flow = (x1, y1, x2, y2, color, label, lx, ly, lw, align, dash) => {
    const x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
    const toEnd = (x2 >= x1 && y2 >= y1);
    const opts = { x, y, w, h, line: { color, width: 1.75, dashType: dash || 'solid' }, objectName: 'flow ' + (label || '') };
    if (toEnd) opts.line.endArrowType = 'triangle'; else opts.line.beginArrowType = 'triangle';
    if ((x2 < x1) !== (y2 < y1) && w > 0 && h > 0) opts.flipV = true; // diagonal up-right / down-left
    s.addShape(pres.shapes.LINE, opts);
    if (label) addT(s, label, { x: lx, y: ly, w: lw, h: 0.24, fontSize: 9, color, bold: true, align: align || 'center',
      valign: 'middle', isTextBox: true, margin: 0, objectName: 'label ' + label });
  };

  // column headers and the Wi-Fi band
  const RX = 0.5, RW = 4.5, SX = 8.33, SW = 4.5, BX = 5.25, BW = 2.83;
  addT(s, 'ROBOT · Stretch · ROS 2', { x: RX, y: 1.08, w: RW, h: 0.28, fontSize: 12, bold: true, color: '333333', isTextBox: true, margin: 0, objectName: 'Robot header' });
  addT(s, 'SERVER (GPU PC) + ROOM', { x: SX, y: 1.08, w: SW, h: 0.28, fontSize: 12, bold: true, color: '333333', isTextBox: true, margin: 0, objectName: 'Server header' });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: BX, y: 1.08, w: BW, h: 5.0, rectRadius: 0.08, fill: { color: 'FFF7F7' },
    line: { color: 'D6262C', width: 1.25, dashType: 'dash' }, objectName: 'WiFi band' });
  addT(s, 'Wi-Fi', { x: BX, y: 1.12, w: BW, h: 0.26, fontSize: 12, bold: true, color: 'D6262C', align: 'center', isTextBox: true, margin: 0, objectName: 'WiFi header' });

  // ---- robot ----
  node(RX, 1.45, RW, 0.62, 'task', 'Task executive', 'acknowledge → navigate → open → pick → return → hand over', 'Task executive');
  node(RX, 2.4, 1.75, 1.05, 'nav', 'Nav2 + AMCL', 'drives to the goal · avoids obstacles with its own costmap (2D lidar)', 'Nav2');
  node(2.85, 2.4, 2.15, 1.05, 'neu', 'target_client', 'keeps each target + whether it is fresh · goal to Nav2 · heartbeat → link alive, time limit · near an unverified target: slow, look', 'target_client');
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: RX, y: 3.78, w: RW, h: 1.32, rectRadius: 0.06, fill: { color: 'FFF7E0' },
    line: { color: 'E9C46A', width: 1 }, objectName: 'grasp_server box' });
  addT(s, 'grasp_server  (open · pick)', { x: RX + 0.08, y: 3.82, w: 2.1, h: 0.27, fontSize: 12, bold: true, color: '8A6A00', valign: 'middle', isTextBox: true, margin: 0, objectName: 'grasp_server title' });
  addT(s, 'snapshot (D435 / D405) · mask × depth → 3-D point · IK · arm, gripper · door opening', { x: RX + 0.08, y: 4.1, w: 2.2, h: 0.95, fontSize: 9.5, color: '333333', valign: 'top', isTextBox: true, margin: 0, objectName: 'grasp_server body' });
  node(2.85, 3.95, 2.07, 1.05, 'edit', 'detect client', 'resize · ID + capture time · time limit · one retry · drop old answers', 'detect client', true);
  node(RX, 5.4, RW, 0.55, 'existing', 'CSV log', 'one line per request: times, sizes, step, RSSI, RTT, pose', 'Robot log', true);

  // ---- server ----
  node(SX, 1.45, 2.0, 0.62, 'offline', 'Map (offline)', 'Gaussian SLAM · 2D grid', 'Map');
  node(10.83, 1.45, 2.0, 0.62, 'existing', 'Room cameras', 'wired · RGB · depth from the map', 'Room cameras', true);
  node(SX, 2.4, 2.0, 1.05, 'neu', 'target_server', 'watches only the camera that sees the target · all cameras when lost · sends moved / lost · heartbeat', 'target_server');
  node(10.83, 2.4, 2.0, 1.05, 'per', 'Object retrieval', 'keyframe search → SAM 3 → object cloud → object cache (pose, ID, time seen)', 'Object retrieval');
  node(SX, 3.95, SW, 1.05, 'edit', 'detector (mm_perception)', 'decode JPEG → YOLOE / SAM 3 → best mask · echo ID + server times · drop requests already too old · report queue', 'detector');
  node(SX, 5.4, 2.0, 0.55, 'existing', 'CSV log', 'one line per request', 'Server log', true);
  node(10.83, 5.4, 2.0, 0.55, 'offline', 'Link-map script', 'request time p50 / p95 per cell', 'Link map script', true);

  // ---- flows on the robot ----
  const K = '555555', T = '0B7F74';
  flow(3.92, 2.07, 3.92, 2.4, K, 'step', 4.0, 2.11, 0.6, 'left');
  flow(2.85, 2.92, 2.25, 2.92, T, '', 0, 0, 0);                 // target_client -> Nav2
  addT(s, 'goal ·\nspeed', { x: 2.25, y: 2.47, w: 0.6, h: 0.42, fontSize: 9, bold: true, color: T, align: 'center', valign: 'bottom', isTextBox: true, margin: 0, objectName: 'label goal' });
  // task exec -> grasp_server, routed along the left margin
  s.addShape(pres.shapes.LINE, { x: 0.32, y: 1.9, w: 0.18, h: 0, line: { color: K, width: 1.75 }, objectName: 'route a' });
  s.addShape(pres.shapes.LINE, { x: 0.32, y: 1.9, w: 0, h: 2.55, line: { color: K, width: 1.75 }, objectName: 'route b' });
  s.addShape(pres.shapes.LINE, { x: 0.32, y: 4.45, w: 0.18, h: 0, line: { color: K, width: 1.75, endArrowType: 'triangle' }, objectName: 'route c' });
  flow(3.92, 3.45, 3.92, 3.95, T, 'time limit · pause', 3.98, 3.58, 1.0, 'left');
  flow(2.85, 4.48, 2.6, 4.48, K, '', 0, 0, 0);                  // detect client -> grasp (mask)
  flow(2.2, 5.1, 2.2, 5.4, K, '', 0, 0, 0, null, 'dash');       // grasp -> log
  // ---- flows on the server ----
  flow(11.83, 2.07, 11.83, 2.4, K, 'images', 11.9, 2.11, 0.8, 'left');
  flow(10.33, 2.07, 10.95, 2.4, K, '', 0, 0, 0);                // map -> retrieval (diagonal)
  flow(10.83, 2.92, 10.33, 2.92, K, '', 0, 0, 0);               // retrieval -> target_server
  addT(s, 'poses +\ntime', { x: 10.33, y: 2.47, w: 0.5, h: 0.42, fontSize: 9, bold: true, color: K, align: 'center', valign: 'bottom', isTextBox: true, margin: 0, objectName: 'label poses' });
  flow(9.4, 5.0, 9.4, 5.4, K, '', 0, 0, 0, null, 'dash');       // detector -> log
  flow(10.33, 5.67, 10.83, 5.67, K, '', 0, 0, 0, null, 'dash'); // log -> link map script

  // ---- flows across the Wi-Fi ----
  flow(SX, 2.68, 5.0, 2.68, T, 'target moved / lost · heartbeat', BX + 0.05, 2.42, BW - 0.1);
  flow(5.0, 3.18, SX, 3.18, T, 'robot pose', BX + 0.05, 2.92, BW - 0.1);
  flow(4.92, 4.23, SX, 4.23, T, 'one JPEG + ID + capture time', BX + 0.05, 3.97, BW - 0.1);
  flow(SX, 4.73, 4.92, 4.73, T, 'best mask + ID + server times', BX + 0.05, 4.47, BW - 0.1);
  flow(5.0, 5.67, SX, 5.67, K, 'logs, after the run', BX + 0.05, 5.41, BW - 0.1, 'center', 'dash');

  // legend
  const LG = [['neu', 'new node'], ['edit', 'edit to their code'], ['existing', 'existing'], ['offline', 'offline']];
  LG.forEach(([k, lab], i) => {
    const st = ST[k], lx = 0.5 + i * 2.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: 6.3, w: 0.36, h: 0.22, rectRadius: 0.04, fill: { color: st.fill },
      line: { color: st.line, width: st.w, dashType: st.dash }, objectName: 'legend ' + lab });
    addT(s, lab, { x: lx + 0.45, y: 6.28, w: 1.8, h: 0.26, fontSize: 11, color: '444444', valign: 'middle', isTextBox: true, margin: 0, objectName: 'legend text ' + lab });
  });
  s.addShape(pres.shapes.LINE, { x: 9.9, y: 6.41, w: 0.4, h: 0, line: { color: T, width: 1.75, endArrowType: 'triangle' }, objectName: 'legend comms arrow' });
  addT(s, 'comms flow', { x: 10.38, y: 6.28, w: 1.2, h: 0.26, fontSize: 11, color: '444444', valign: 'middle', isTextBox: true, margin: 0, objectName: 'legend comms text' });
  s.addShape(pres.shapes.LINE, { x: 11.55, y: 6.41, w: 0.4, h: 0, line: { color: K, width: 1.75, endArrowType: 'triangle' }, objectName: 'legend local arrow' });
  addT(s, 'local flow', { x: 12.03, y: 6.28, w: 0.85, h: 0.26, fontSize: 11, color: '444444', valign: 'middle', isTextBox: true, margin: 0, objectName: 'legend local text' });
  s.addNotes('Teal arrows cross the Wi-Fi: targets down (only when they change), robot pose up, one photo per look up, the best mask back. Grey arrows stay on one machine. New: target_client and target_server. Edited: their detect client and detector. Offline: map and link-map script.');

  // ---------- the comms pieces ----------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'The comms pieces', { placeholder: 'title' });
  const wcell = (t, bg, fg) => ({ text: t, options: { bold: true, color: fg, fill: { color: bg } } });
  const hd = (t) => ({ text: t, options: { bold: true, color: 'FFFFFF', fill: { color: '0B3D3A' } } });
  const R_ = (t) => wcell(t, 'DFF3E6', '1F6F43'), S_ = (t) => wcell(t, 'ECE6FB', '5B3FB0'), O_ = (t) => wcell(t, 'E4F4E9', '1D7A3A');
  const nm = (t, kind) => ({ text: t, options: { bold: true, color: kind === 'later' ? '4F8F87' : '1D1D1F' } });
  const crow = [
    [hd('Piece'), hd('Type'), hd('Where'), hd('What it does')],
    [nm('detect client'), 'edit to locator.py', R_('Robot'), 'One photo per look (resized, with ID, capture time and pose); time limit; one retry; drops answers for old photos; logs a CSV line'],
    [nm('detector'), 'edit to detector_node.py', S_('Server'), 'Echoes the ID; adds server times; returns the best mask only; drops requests already too old; reports its queue; logs a CSV line'],
    [nm('target_server'), 'new node', S_('Server'), 'Finds each target once on all room cameras, then watches only the camera that sees it; if lost, searches all cameras; sends a target when it moves or is lost; heartbeat ~1 Hz'],
    [nm('target_client'), 'new node', R_('Robot'), 'Keeps each target and whether it is fresh; goal to Nav2; heartbeat → link alive and time limit; near an unverified target: slows down and looks with its own camera'],
    [nm('link-map script'), 'offline script', O_('Offline'), 'From the survey drive and the CSV logs: request time p50 / p95 and failures per cell; only if the survey shows slow spots'],
    [nm('keyframe_packer / ingest', 'later'), 'later, if needed', R_('Both'), 'Only if map updates run during tasks: sent when the view changes, paused during detect and grasp'],
  ];
  s.addTable(crow, { x: 0.5, y: 1.25, w: 12.33, colW: [2.2, 1.95, 1.0, 7.18], fontSize: 12, fontFace: FONT, color: '1D1D1F',
    border: { type: 'solid', pt: 0.75, color: 'C8D6D3' }, fill: { color: 'FFFFFF' }, valign: 'middle',
    rowH: [0.4, 0.62, 0.62, 0.62, 0.72, 0.62, 0.62], objectName: 'Comms pieces' });
  addT(s, 'Two edits to their code, two new nodes, one offline script.', { x: 0.5, y: 5.75, w: 12.33, h: 0.4, fontSize: 16,
    bold: true, color: '0B7F74', isTextBox: true, margin: 0, objectName: 'Summary line' });
  s.addNotes('This is the whole comms side for the full task. Everything else (Nav2, perception, motion) stays as it is.');

  // 8 What crosses the Wi-Fi
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'What crosses the robot’s Wi-Fi', { placeholder: 'title' });
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
    objectName: 'Wi-Fi tiers', fontFace: FONT });
  addT(s, 'Never sent during a task: raw image streams, depth images, raw lidar scans. Every request carries an ID and capture time; every answer echoes them.',
    { x: 0.5, y: 5.0, w: 12.33, h: 0.8, fontSize: 16, color: C.accent1, bold: true, isTextBox: true,
      margin: 0, objectName: 'Rule' });
  s.addNotes('Pick requests always go first; background and offline traffic give way while the robot is detecting and grasping.');

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(OUT);
})();
