const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 3200, height: 1100 } });
  await p.goto('file://' + __dirname + '/fig_task_pipeline.html');
  await p.waitForTimeout(400);
  await p.locator('body').screenshot({ path: __dirname + '/fig_task_pipeline.png' });
  await b.close();
})();
