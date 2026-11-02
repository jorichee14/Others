// Markdown report -> Word in the user's formal style: Liberation Serif 10.5 pt, A4, numbered
// section headings, booktabs tables with "Table N." captions above, links kept as hyperlinks.
//   node md_to_docx.js <report.md> <out.docx> [--course "line above the title"] [--author "Student ID: 314540011"]
const fs = require("fs");
const d = require("docx");
const args = process.argv.slice(2);
const opt = (k, dflt) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : dflt; };
const [src, out] = args;
const course = opt("--course", "Research survey");
const author = opt("--author", "Student ID: 314540011");
const FONT = "Liberation Serif", MONO = "Liberation Mono";
const PAGE_W = 11906, MARGIN = 964, TEXT_W = PAGE_W - 2 * MARGIN;          // A4, 17 mm margins

// inline: **bold**, *italic*, `code`, [text](url)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(`([^`]+)`)|(\[([^\]]+)\]\(([^)]+)\))/g;
  let last = 0, m;
  const plain = (t, o) => t && out.push(new d.TextRun({ text: t, font: FONT, ...base, ...o }));
  while ((m = re.exec(text))) {
    plain(text.slice(last, m.index), {});
    if (m[1]) out.push(...runs(m[2], { ...base, bold: true }));
    else if (m[3]) out.push(...runs(m[4], { ...base, italics: true }));
    else if (m[5]) out.push(new d.TextRun({ text: m[6], font: MONO, size: 18 }));
    else if (m[7]) out.push(new d.ExternalHyperlink({ link: m[9],
      children: [new d.TextRun({ text: m[8], font: FONT, style: "Hyperlink", ...base })] }));
    last = re.lastIndex;
  }
  plain(text.slice(last), {});
  return out;
}

const body = (text) => new d.Paragraph({ children: runs(text), alignment: d.AlignmentType.JUSTIFIED,
  spacing: { after: 80, line: 305 } });                                      // 1.27 line spacing

const NONE = { style: d.BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(rows, caption, n) {
  const ncol = rows[0].length;
  // column widths in proportion to the longest cell text, bounded, summing to the text width
  const len = Array.from({ length: ncol }, (_, c) =>
    Math.min(60, Math.max(8, ...rows.map(r => (r[c] || "").replace(/\*|\[|\]\([^)]*\)/g, "").length))));
  const tot = len.reduce((a, b) => a + b, 0);
  const widths = len.map(l => Math.floor(TEXT_W * l / tot));
  widths[ncol - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const numeric = (s) => /^[\s~±<>≈+\-−]*[\d.,]+\s*(%|dB|cm|m|x|×)?\s*$/.test(s);
  const trs = rows.map((r, i) => new d.TableRow({
    tableHeader: i === 0,
    children: r.map((cell, c) => new d.TableCell({
      width: { size: widths[c], type: d.WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      borders: {
        top: i === 0 ? { style: d.BorderStyle.SINGLE, size: 10, color: "000000" } : NONE,
        bottom: i === 0 ? { style: d.BorderStyle.SINGLE, size: 5, color: "000000" }
              : i === rows.length - 1 ? { style: d.BorderStyle.SINGLE, size: 10, color: "000000" } : NONE,
        left: NONE, right: NONE,
      },
      children: [new d.Paragraph({
        alignment: i === 0 ? (c === 0 ? d.AlignmentType.LEFT : d.AlignmentType.CENTER)
                 : c > 0 && numeric(cell) ? d.AlignmentType.RIGHT : d.AlignmentType.LEFT,
        spacing: { after: 0, line: 250 },
        children: runs(cell, { size: 18, bold: i === 0 ? true : undefined }),
      })],
    })),
  }));
  return [
    new d.Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, alignment: d.AlignmentType.JUSTIFIED,
      children: [new d.TextRun({ text: `Table ${n}. `, bold: true, font: FONT, size: 19 }),
                 new d.TextRun({ text: caption, font: FONT, size: 19 })] }),
    new d.Table({ width: { size: TEXT_W, type: d.WidthType.DXA }, columnWidths: widths, rows: trs,
      borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE } }),
    new d.Paragraph({ spacing: { after: 60 }, children: [] }),
  ];
}

const lines = fs.readFileSync(src, "utf8").split(/\r?\n/);
const kids = [];
let sec = 0, sub = 0, tab = 0, heading = "", para = [];
const flush = () => { if (para.length) { kids.push(body(para.join(" "))); para = []; } };
const strip = (t) => t.replace(/\*\*|\*/g, "");
let title = "";
for (let i = 0; i < lines.length; i++) {
  const L = lines[i];
  if (/^# /.test(L)) { flush(); title = strip(L.slice(2).trim()); continue; }
  if (/^## /.test(L)) {
    flush(); sec++; sub = 0; heading = strip(L.slice(3).trim());
    kids.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 200, after: 70 },
      children: [new d.TextRun({ text: `${sec}. ${heading}`, bold: true, font: FONT, size: 23, color: "000000" })] }));
    continue;
  }
  if (/^### /.test(L)) {
    flush(); sub++; heading = strip(L.slice(4).trim());
    kids.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 140, after: 50 },
      children: [new d.TextRun({ text: `${sec}.${sub} ${heading}`, bold: true, italics: true, font: FONT, size: 21, color: "000000" })] }));
    continue;
  }
  if (/^\|/.test(L)) {
    flush();
    const rows = [];
    while (i < lines.length && /^\|/.test(lines[i])) {
      const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map(s => s.trim());
      if (!cells.every(c => /^:?-{2,}:?$/.test(c))) rows.push(cells);
      i++;
    }
    i--;
    kids.push(...table(rows, heading, ++tab));
    continue;
  }
  const li = L.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
  if (li) {
    flush();
    kids.push(new d.Paragraph({ numbering: { reference: /^\s*\d+\./.test(L) ? "num" : "bul", level: 0 },
      alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 40, line: 290 }, children: runs(li[1]) }));
    continue;
  }
  if (!L.trim()) { flush(); continue; }
  para.push(L.trim());
}
flush();

const head = [
  new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 30 },
    children: [new d.TextRun({ text: course, font: FONT, size: 20 })] }),
  new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 50 },
    children: [new d.TextRun({ text: title, font: FONT, size: 30, bold: true })] }),
  new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 100 },
    border: { bottom: { style: d.BorderStyle.SINGLE, size: 6, color: "000000", space: 6 } },
    children: [new d.TextRun({ text: author, font: FONT, size: 21 })] }),
];

const doc = new d.Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    characterStyles: [{ id: "Hyperlink", name: "Hyperlink", run: { color: "000000", underline: { type: d.UnderlineType.SINGLE } } }],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
    { reference: "num", levels: [{ level: 0, format: d.LevelFormat.DECIMAL, text: "%1.", alignment: d.AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 850, bottom: 794, left: MARGIN, right: MARGIN } } },
    footers: { default: new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.CENTER,
      children: [new d.TextRun({ children: [d.PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: [...head, ...kids],
  }],
});
d.Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log(`wrote ${out}: ${sec} sections, ${tab} tables`); });
