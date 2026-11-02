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

  // 5 Architecture
  pres.addSection({ title: 'Architecture' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'System architecture', { placeholder: 'title' });
  const a = fitImage(s, 'arch_full.png', { x: 0.5, y: 1.2, w: 8.9, h: 5.7 }, 'System architecture');
  const ax = a.x + a.w + 0.35, aw = 12.83 - ax;
  card(s, ax, 1.2, aw, 1.75, 'Robot', 'Sensors, localization, task executive, object fusion, motion, and the robot side of the comms bridge.', 'E4EEFB', C.accent2, 'Robot');
  card(s, ax, 3.1, aw, 1.75, 'Server', 'Gaussian SLAM map, room cameras, object retrieval, asset layer, and the server side of the comms bridge.', 'EFEAFB', C.accent4, 'Server');
  card(s, ax, 5.0, aw, 1.75, 'Link', 'Four tiers: request (pick), background (keyframe packs), offline (bags at the dock), once (calibration, map).', 'DDF3F0', C.accent1, 'Link');
  s.addNotes('Left: everything on the robot. Right: everything on the server. The red dashed line is the Wi-Fi.');

  // 6 Robot side
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'Robot side: time-critical work stays local', { placeholder: 'title' });
  const rb = fitImage(s, 'arch_robot.png', { x: 0.5, y: 1.2, w: 4.2, h: 5.7 }, 'Robot side of the architecture');
  const rx = rb.x + rb.w + 0.45, rw = 12.83 - rx;
  card(s, rx, 1.2, rw, 1.65, 'Runs on the robot', 'Localization (AMCL / LiDAR-inertial odometry), Nav2 with obstacle avoidance, mask × depth → 3-D points, IK, arm and gripper.', 'E4EEFB', C.accent2, 'Local');
  card(s, rx, 3.0, rw, 1.95, 'Comms bridge (robot)', 'offload_client: one photo per look, with ID and capture time · keyframe_packer: background, paused during the pick · recorder: full-rate data to disk, uploaded at the dock · link_monitor: RTT, RSSI, server queue · policy: wait, retry or fall back.', 'DDF3F0', C.accent1, 'Bridge');
  card(s, rx, 5.1, rw, 1.65, 'When an answer is late', 'Retry once with a new photo, keep driving to the current goal, slow down near the person, or use the onboard detector.', 'FBE9E7', C.accent3, 'Late');
  s.addNotes('The robot never needs the server to stay safe or to move; it needs the server only for the big models.');

  // 7 Server side
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'Server side: map, retrieval, assets', { placeholder: 'title' });
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
