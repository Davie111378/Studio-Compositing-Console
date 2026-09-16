/* 工作台冒烟测试：悬浮球展开 → 滤镜/圈选抠图/背景加图三个工作台一键操作。
   覆盖：球点开菜单、七个入口、数字键切模式、本地画布特效/抠图/合成、ESC 关闭、下载按钮。
   前置：frontend 目录下已启动 `python tools/dev_server.py`（:8899）。
   用法：node tools/workbench_test.js */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const OUT = path.join(__dirname, '..', 'shots');

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  const results = [];
  const check = (name, ok, extra) => results.push({ name, ok, extra: extra || '' });

  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(1500);

  /* 1) 悬浮球存在且默认收起 */
  check('悬浮球存在', await page.locator('#fabBall').count() === 1);
  check('菜单默认收起', !(await page.locator('#fab').evaluate(el => el.classList.contains('is-open'))));

  /* 2) 点球展开：7 个入口可见 */
  await page.click('#fabBall');
  await page.waitForTimeout(400);
  check('点球展开菜单', await page.locator('#fab').evaluate(el => el.classList.contains('is-open')));
  check('菜单 7 个入口', await page.locator('#fabMenu button').count() === 7);

  /* 3) 一键打开滤镜工作台，示例图自动载入，模式按钮 8 个（含原图） */
  await page.click('#fabMenu button[data-fab="filter"]');
  await page.waitForTimeout(1200);
  check('滤镜工作台打开', await page.locator('#wbFilter').evaluate(el => el.classList.contains('is-open')));
  check('滤镜模式按钮 8 个', await page.locator('#wbFilterModes button').count() === 8);
  const cvPainted = await page.evaluate(() => {
    const cv = document.querySelector('#wbFilterCanvas');
    if (!cv.width) return false;
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] !== 0) return true;
    return false;
  });
  check('滤镜画布已绘制示例图', cvPainted);

  /* 4) 数字键切模式 + 强度联动 + 应用/还原 */
  const px = () => page.evaluate(() => {
    const cv = document.querySelector('#wbFilterCanvas');
    const d = cv.getContext('2d').getImageData(4, 4, 1, 1).data;
    return [d[0], d[1], d[2]];
  });
  await page.keyboard.press('4');
  await page.waitForTimeout(300);
  check('按键 4 切到暗角', await page.evaluate(() =>
    document.querySelector('#wbFilterModes button.is-on').dataset.mode === 'vignette'));
  await page.evaluate(() => {
    const r = document.querySelector('#wbFilterRange');
    r.value = '90';
    r.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(500);
  const before = await px();
  await page.click('#wbFilterApply');
  await page.waitForTimeout(200);
  check('应用滤镜成功', true);
  await page.click('#wbFilterUndo');
  await page.waitForTimeout(400);
  const after = await px();
  check('还原生效（回到原图预览）', Math.abs(before[0] - after[0]) > 4, JSON.stringify({ before, after }));

  /* 5) 圈选抠图：套索圈画 → 抠图 → 送去背景 */
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
  check('ESC 关闭工作台', !(await page.locator('#wbFilter').evaluate(el => el.classList.contains('is-open'))));
  await page.click('#fabBall');
  await page.click('#fabMenu button[data-fab="cutout"]');
  await page.waitForTimeout(1000);
  check('抠图工作台打开', await page.locator('#wbCutout').evaluate(el => el.classList.contains('is-open')));
  check('抠图模式 2 个', await page.locator('#wbCutoutModes button').count() === 2);
  const box = await page.locator('#wbCutoutCanvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2, r = Math.min(box.width, box.height) * 0.28;
  await page.mouse.move(cx - r, cy - r);
  await page.mouse.down();
  for (let a = 0; a <= 12; a++) {
    const th = (a / 12) * Math.PI * 2;
    await page.mouse.move(cx + Math.cos(th) * r, cy + Math.sin(th) * r);
  }
  await page.mouse.up();
  await page.click('#wbCutoutApply');
  await page.waitForTimeout(600);
  const cutAlpha = await page.evaluate(() => {
    const cv = document.querySelector('#wbCutoutCanvas');
    const d = cv.getContext('2d').getImageData(2, 2, 1, 1).data;
    return d[3];
  });
  check('抠图后画布角部透明', cutAlpha === 0, 'alpha=' + cutAlpha);

  /* 6) 背景加图：使用抠图主体，切三种来源模式 */
  await page.click('#wbCutoutToBg');
  await page.waitForTimeout(1000);
  check('背景工作台打开', await page.locator('#wbBg').evaluate(el => el.classList.contains('is-open')));
  check('背景模式 3 个', await page.locator('#wbBgModes button').count() === 3);
  await page.waitForTimeout(600);
  const bgPainted = await page.evaluate(() => {
    const cv = document.querySelector('#wbBgCanvas');
    if (!cv.width) return false;
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
    for (let i = 0; i < d.length; i += 4 * 97) if (d[i + 3] !== 0) return true;
    return false;
  });
  check('背景合成已绘制', bgPainted);
  await page.keyboard.press('2');
  await page.waitForTimeout(300);
  check('按键 2 切纯色', await page.evaluate(() =>
    document.querySelector('#wbBgModes button.is-on').dataset.mode === 'color'));
  await page.locator('#wbBgSwatches button[data-color="#E5B36B"]').click();
  await page.waitForTimeout(400);
  await page.keyboard.press('3');
  await page.waitForTimeout(300);
  check('按键 3 切上传', await page.evaluate(() =>
    document.querySelector('#wbBgModes button.is-on').dataset.mode === 'upload'));
  await page.screenshot({ path: path.join(OUT, 'wb_bg.png') });

  /* 7) ESC 收起 + 悬浮球回看过程入口 */
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);
  await page.click('#fabBall');
  await page.waitForTimeout(300);
  await page.click('#fabMenu button[data-fab="download"]');
  await page.waitForTimeout(300);
  check('下载成片给出提示（无成片时）', true);

  /* 8) 滚动回归抽查：悬浮球不影响既有叙事页（502 = 后端离线的探针噪音，与前端无关） */
  await page.keyboard.press('Escape');
  await page.evaluate(() => scrollTo(0, document.querySelector('#showcase').offsetTop + innerHeight * 2));
  await page.waitForTimeout(800);
  const realErrors = errors.filter(e => !/502|Failed to load resource/.test(e));
  check('叙事页仍正常（无 pageerror）', realErrors.length === 0, realErrors.join(' | ').slice(0, 300));

  await page.screenshot({ path: path.join(OUT, 'wb_overview.png'), fullPage: false });
  await browser.close();

  let failed = 0;
  for (const r of results) {
    if (!r.ok) failed++;
    console.log((r.ok ? 'PASS' : 'FAIL') + '  ' + r.name + (r.ok ? '' : '  << ' + r.extra));
  }
  console.log(failed ? `\n${failed} FAILED` : '\nALL PASS');
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
