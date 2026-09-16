'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright-core');
const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const OUT = path.join(__dirname, '..', 'shots');
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto('http://127.0.0.1:8899/', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);

  /* 悬浮球展开菜单 */
  await page.click('#fabBall');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'wb_fab_open.png') });

  /* 滤镜工作台：切聚光 + 强度 80 + 应用，再切雾效叠加 */
  await page.click('#fabMenu button[data-fab="filter"]');
  await page.waitForTimeout(1000);
  await page.keyboard.press('6');
  await page.waitForTimeout(300);
  await page.evaluate(() => {
    const r = document.querySelector('#wbFilterRange');
    r.value = '80';
    r.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(600);
  await page.click('#wbFilterApply');
  await page.waitForTimeout(300);
  await page.keyboard.press('7');
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'wb_filter_fx.png') });

  /* 抠图工作台：圈选后选区可见 */
  await page.keyboard.press('Escape');
  await page.click('#fabBall');
  await page.waitForTimeout(300);
  await page.click('#fabMenu button[data-fab="cutout"]');
  await page.waitForTimeout(1000);
  const box = await page.locator('#wbCutoutCanvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height * 0.42, r2 = Math.min(box.width, box.height) * 0.3;
  await page.mouse.move(cx - r2, cy - r2);
  await page.mouse.down();
  for (let a = 0; a <= 14; a++) {
    const th = (a / 14) * Math.PI * 2;
    await page.mouse.move(cx + Math.cos(th) * r2 * 0.8, cy + Math.sin(th) * r2);
  }
  await page.mouse.up();
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'wb_cutout_lasso.png') });

  await browser.close();
  console.log('shots done');
})().catch(e => { console.error(e); process.exit(2); });
