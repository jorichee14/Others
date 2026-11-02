const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 2000, height: 1410 } });
  await p.goto('file://' + __dirname + '/stretch_comms_architecture.html');
  await p.waitForTimeout(300);
  await p.screenshot({ path: __dirname + '/stretch_comms_architecture.png' });
  await b.close();
})();
