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

  // ---------- architecture as native shapes ----------
  pres.addSection({ title: 'Architecture' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'System architecture', { placeholder: 'title' });
  const KIND = {
    hw: ['EEF0F3', 'C3C7CF', '333333'], loc: ['E4EEFB', 'A9C3EC', '1A5FD0'], task: ['EFEAFB', 'C3B2EC', '5B3FB0'],
    per: ['FFF7E0', 'E9C46A', '8A6A00'], man: ['FDF3E4', 'EFCB96', '9A5B00'], twin: ['E4F4E9', '9FD3AE', '1D7A3A'],
    com: ['DDF3F0', '0B7F74', '0B7F74'],
  };
  const block = (sl, x, y, w, h, kind, title, right, body, name) => {
    const [bg, bd, fg] = KIND[kind];
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: bg },
      line: { color: bd, width: kind === 'com' ? 1.75 : 1, dashType: kind === 'com' ? 'dash' : 'solid' },
      objectName: name + ' block' });
    addT(sl, title, { x: x + 0.1, y: y + 0.04, w: w * 0.55, h: 0.28, fontSize: 13, bold: true, color: fg,
      valign: 'middle', isTextBox: true, margin: 0, objectName: name + ' title' });
    if (right) addT(sl, right, { x: x + w * 0.45, y: y + 0.04, w: w * 0.55 - 0.1, h: 0.28, fontSize: 10,
      color: '666666', align: 'right', valign: 'middle', isTextBox: true, margin: 0, objectName: name + ' note' });
    addT(sl, body, { x: x + 0.1, y: y + 0.32, w: w - 0.2, h: h - 0.36, fontSize: 11, color: '333333',
      valign: 'top', isTextBox: true, margin: 0, paraSpaceAfter: 1, objectName: name + ' body' });
  };
  // column headers
  const RX = 0.5, RW = 5.05, MX = 5.75, MW = 2.05, SX = 8.0, SW = 4.83, Y0 = 1.5;
  addT(s, 'ROBOT · Stretch · ROS 2', { x: RX, y: 1.12, w: RW, h: 0.3, fontSize: 13, bold: true, color: '333333', isTextBox: true, margin: 0, objectName: 'Robot header' });
  addT(s, 'Wi-Fi', { x: MX, y: 1.12, w: MW, h: 0.3, fontSize: 13, bold: true, color: 'D6262C', align: 'center', isTextBox: true, margin: 0, objectName: 'WiFi header' });
  addT(s, 'SERVER (GPU PC) + ROOM', { x: SX, y: 1.12, w: SW, h: 0.3, fontSize: 13, bold: true, color: '333333', isTextBox: true, margin: 0, objectName: 'Server header' });

  // robot column
  let y = Y0;
  const rob = [
    ['hw', 'Sensors + actuators', '', 'head RGB-D (D435) · gripper RGB-D (D405) · 2D lidar · IMU · base, lift, arm, wrist, gripper', 0.72],
    ['loc', 'Localization + navigation', 'on the robot', 'AMCL on the 2D grid · Nav2 with its own costmap (obstacle avoidance)', 0.62],
    ['task', 'Task executive', 'stage machine', 'acknowledge → navigate → open → pick → return → hand over', 0.55],
    ['per', 'Perception', 'on the robot', 'mask × depth → 3-D points → centre · handle point + door plane · onboard YOLOE as fallback', 0.72],
    ['man', 'Motion', 'on the robot', 'IK (pink) · arm trajectories · small base moves · gripper by effort', 0.55],
  ];
  rob.forEach(([k, t, r, b, h], i) => { block(s, RX, y, RW, h, k, t, r, b, 'Robot ' + i); y += h + 0.1; });
  const rbY = y, rbH = 6.85 - y;
  block(s, RX, rbY, RW, rbH, 'com', 'Comms bridge · robot', 'ROS 2 nodes', '', 'Robot bridge');
  // comms nodes: [name, later?]
  const drawNodes = (nodes, bx, by, bw, cols, prefix) => {
    const gw = (bw - 0.2 - 0.08 * (cols - 1)) / cols;
    nodes.forEach(([n, later], i) => {
      const cx = bx + 0.1 + (i % cols) * (gw + 0.08), cy = by + 0.36 + Math.floor(i / cols) * 0.42;
      addT(s, later ? [{ text: n, options: { breakLine: false } }, { text: '  later', options: { fontSize: 9, bold: false, italic: true } }] : n,
        { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x: cx, y: cy, w: gw, h: 0.34,
          fill: { color: later ? 'F4FAF9' : 'FFFFFF' }, line: { color: later ? '8CC9C1' : '0B7F74', width: 1, dashType: later ? 'dash' : 'solid' },
          fontSize: 11, bold: true, color: later ? '4F8F87' : '0B7F74', align: 'center', valign: 'middle', margin: 0,
          objectName: prefix + ' node ' + n });
    });
  };
  drawNodes([['offload_client'], ['link_monitor'], ['comms_policy'], ['target_tracker'], ['keyframe_packer', 1], ['recorder', 1]],
    RX, rbY, RW, 3, 'robot');

  // server column
  y = Y0;
  const srv = [
    ['twin', 'Map (offline)', '~16 h per run', 'Gaussian SLAM map · reference map + 2D grid · link map (from the map’s Sionna export + survey)', 0.78],
    ['hw', 'Room cameras', 'wired · RGB only', 'calibrated in the map frame · depth rendered from the map · object and person detections', 0.72],
    ['per', 'Object retrieval', 'online', 'keyframe search → SAM 3 → object cloud → object cache → target + view pose, with its age', 0.78],
    ['twin', 'Asset layer', 'background · ≥32 GB GPU', 'SAM 3D asset per object · surfaces labelled measured / generated', 0.62],
  ];
  srv.forEach(([k, t, r, b, h], i) => { block(s, SX, y, SW, h, k, t, r, b, 'Server ' + i); y += h + 0.1; });
  const sbY = y, sbH = 6.85 - y;
  block(s, SX, sbY, SW, sbH, 'com', 'Comms bridge · server', 'ROS 2 nodes', '', 'Server bridge');
  drawNodes([['offload_server'], ['target_server'], ['link_mapper'], ['logger'], ['keyframe_ingest', 1], ['gpu_scheduler', 1]],
    SX, sbY, SW, 3, 'server');

  // Wi-Fi column
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: Y0, w: MW, h: 6.85 - Y0, rectRadius: 0.08,
    fill: { color: 'FFF5F5' }, line: { color: 'D6262C', width: 1.25, dashType: 'dash' }, objectName: 'WiFi column' });
  const tiers = [
    ['Request · high', '0B7F74', 'FFFFFF', '↓ target + its age\n↑ one JPEG per look\n↓ mask / keypoint\n↑ robot pose\n↕ heartbeat', 1.55],
    ['Background · low', 'CFEEE9', '0B5F57', '↑ keyframe packs, ~2 Hz, when the view changes; paused while detecting and grasping', 1.15],
    ['Offline · lowest', 'E9ECEF', '444444', '↑ bags and full-rate frames, at the dock', 0.8],
    ['Once', 'F3F3F3', '444444', '↑ calibration\n↓ 2D grid per map version', 0.72],
  ];
  y = Y0 + 0.1;
  tiers.forEach(([t, bg, fg, body, h], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.08, y, w: MW - 0.16, h, rectRadius: 0.05,
      fill: { color: bg }, line: { color: bg }, objectName: 'tier ' + i });
    addT(s, [{ text: t, options: { bold: true, fontSize: 11, breakLine: true } }, { text: body, options: { fontSize: 10 } }],
      { x: MX + 0.14, y: y + 0.04, w: MW - 0.28, h: h - 0.08, color: fg, valign: 'top', isTextBox: true, margin: 0,
        objectName: 'tier text ' + i });
    y += h + 0.1;
  });
  addT(s, [{ text: 'Never sent', options: { bold: true, breakLine: true } }, { text: 'raw video, depth images, lidar scans' }],
    { x: MX + 0.08, y, w: MW - 0.16, h: 6.85 - 0.08 - y, fontSize: 10, color: 'A3171D', valign: 'top',
      isTextBox: true, margin: 2, objectName: 'never sent' });
  // links between bridges and the Wi-Fi column
  s.addShape(pres.shapes.LINE, { x: RX + RW, y: rbY + 0.6, w: MX + 0.08 - (RX + RW), h: 0,
    line: { color: '0B7F74', width: 2.5, beginArrowType: 'triangle', endArrowType: 'triangle' }, objectName: 'robot link' });
  s.addShape(pres.shapes.LINE, { x: MX + MW - 0.08, y: sbY + 0.6, w: SX - (MX + MW - 0.08), h: 0,
    line: { color: '0B7F74', width: 2.5, beginArrowType: 'triangle', endArrowType: 'triangle' }, objectName: 'server link' });
  addT(s, 'messages: Target (pose, time seen, source, validity) · LinkState (RTT, RSSI, queue, predicted request time) · Detect (+ ID, times)',
    { x: RX + 0.1, y: rbY + rbH - 0.5, w: RW - 0.2, h: 0.42, fontSize: 11, color: '174F49', italic: true,
      valign: 'bottom', isTextBox: true, margin: 0, objectName: 'Robot bridge note' });
  addT(s, 'targets sent only on change, with their age · link map = request time p50/p95 + failures per cell, not just RSSI',
    { x: SX + 0.1, y: sbY + sbH - 0.5, w: SW - 0.2, h: 0.42, fontSize: 11, color: '174F49', italic: true,
      valign: 'bottom', isTextBox: true, margin: 0, objectName: 'Server bridge note' });
  s.addNotes('Left: everything on the robot. Middle: the Wi-Fi, with four traffic tiers. Right: the server and the wired room cameras. The teal dashed blocks are the comms side.');

  // ---------- the comms side ----------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Architecture' });
  addT(s, 'The comms side', { placeholder: 'title' });
  const wcell = (t, bg, fg) => ({ text: t, options: { bold: true, color: fg, fill: { color: bg } } });
  const hd = (t) => ({ text: t, options: { bold: true, color: 'FFFFFF', fill: { color: '0B3D3A' } } });
  const R_ = (t) => wcell(t, 'DFF3E6', '1F6F43'), S_ = (t) => wcell(t, 'ECE6FB', '5B3FB0'), B_ = (t) => wcell(t, 'EEF0F3', '333333');
  const nm = (t, later) => ({ text: later ? t + '  (later)' : t, options: { bold: true, color: later ? '4F8F87' : '1D1D1F' } });
  const crow = [
    [hd('Node'), hd('Where'), hd('What it does')],
    [nm('offload_client'), R_('Robot'), 'One photo per look (resized or cropped JPEG, with ID, capture time and pose); waits up to the time limit; retries; drops old answers'],
    [nm('link_monitor'), R_('Robot'), 'Heartbeat (RTT, server queue) + RSSI + link map at the current and next position → predicted request time (p50 / p95)'],
    [nm('comms_policy'), R_('Robot'), 'Per step: time limit, retry or onboard fallback, speed limit when targets are old, pause / resume background traffic'],
    [nm('target_tracker'), R_('Robot'), 'Latest target per type (object, person) with its age and source; checks validity; asks for a refresh when old; goal to Nav2'],
    [nm('keyframe_packer', 1), R_('Robot'), 'Map updates when the view changes; paused while detecting and grasping'],
    [nm('recorder', 1), R_('Robot'), 'Full-rate data and bags to disk; uploaded at the dock'],
    [nm('offload_server'), S_('Server'), 'Wraps detect: runs YOLOE / SAM 3, answers with the same ID and server times, drops requests already too old, reports its queue'],
    [nm('target_server'), S_('Server'), 'Room-camera detections + object cache → targets, sent only on change, with time seen, source, validity, verified / last seen'],
    [nm('link_mapper'), S_('Server'), 'Link map per cell: request time p50 / p95, failure rate, RTT, RSSI; from a survey drive, then from every logged request'],
    [nm('logger'), S_('Server'), 'Every request: time at each hop, sizes, step, RSSI, RTT, pose → where the time goes'],
    [nm('keyframe_ingest', 1), S_('Server'), 'Receives map updates, fills gaps after a pause'],
    [nm('gpu_scheduler', 1), S_('Server'), 'Runs pick requests before map training and SAM 3D'],
  ];
  s.addTable(crow, { x: 0.5, y: 1.2, w: 12.33, colW: [2.35, 1.0, 8.98], fontSize: 12, fontFace: FONT, color: '1D1D1F',
    border: { type: 'solid', pt: 0.75, color: 'C8D6D3' }, fill: { color: 'FFFFFF' }, valign: 'middle',
    rowH: [0.38, ...Array(12).fill(0.42)], objectName: 'Comms nodes' });
  s.addNotes('The comms side is two small sets of ROS 2 nodes, one on each machine, plus the link map. They wrap the one network call in the existing grasp code and manage the background traffic.');

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
