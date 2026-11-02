const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  for (const [n, h] of [['fig_architecture', 1230], ['fig_pipeline', 1140]]) {
    const p = await b.newPage({ viewport: { width: 2000, height: h } });
    await p.goto('file://' + __dirname + '/' + n + '.html');
    await p.waitForTimeout(300);
    await p.screenshot({ path: __dirname + '/' + n + '.png' });
    await p.close();
  }
  await b.close();
})();
