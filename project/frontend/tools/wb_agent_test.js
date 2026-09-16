'use strict';
/* 工作台 → Agent 桥接 E2E：滤镜台「让 Agent 接管」/ 抠图台「让 AI 精抠」（带 spatial box）
   前置：dev_server(:8899) + agent(:8000) 均已启动。 */
const path = require('path');
const { chromium } = require('playwright-core');
const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  await page.goto('http://127.0.0.1:8899/', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  const results = [];
  const check = (n, ok, x) => results.push({ n, ok, x: x || '' });
  const waitFor = async (fn, ms, step = 1000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (await fn()) return true;
      await page.waitForTimeout(step);
    }
    return fn();
  };

  /* 1) 滤镜台 → 让 Agent 接管：LLM 规划需数秒，轮询等待 RUN 面板弹出 */
  await page.click('#fabBall');
  await page.click('#fabMenu button[data-fab="filter"]');
  await page.waitForTimeout(1000);
  await page.click('#wbFilterAgent');
  check('桥接后 RUN 面板打开', await waitFor(
    () => page.locator('#studioRun').evaluate(el => el.classList.contains('is-open')), 40000));
  check('RUN 模式为 LIVE', (await page.locator('#runMode').textContent()) === 'LIVE');
  check('计划节点已渲染', await waitFor(async () => (await page.locator('#runList li').count()) > 0, 20000));
  check('至少一个节点完成', await waitFor(async () => (await page.locator('#runList li.is-done').count()) > 0, 90000));
  await page.screenshot({ path: path.join(__dirname, '..', 'shots', 'wb_agent_run.png') });
  /* 等 job1 真正完成并进入 SUMMARY：后端空了，第二段桥接的 healthz 探测才不会撞上忙事件循环 */
  check('job1 完成进入 SUMMARY', await waitFor(
    () => page.locator('#studioSummary').evaluate(el => el.classList.contains('is-open')), 120000, 2000));
  await page.waitForTimeout(1500);

  /* 2) 抠图台 → 让 AI 精抠：spatial.box 随请求上传（完成后可能自动进 SUMMARY，用 ESC 退出） */
  await page.keyboard.press('Escape');
  await page.waitForTimeout(600);
  await page.click('#fabBall');
  await page.click('#fabMenu button[data-fab="cutout"]');
  await page.waitForTimeout(900);
  const box = await page.locator('#wbCutoutCanvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2, r = Math.min(box.width, box.height) * 0.25;
  await page.mouse.move(cx - r, cy - r);
  await page.mouse.down();
  for (let a = 0; a <= 12; a++) {
    const th = (a / 12) * Math.PI * 2;
    await page.mouse.move(cx + Math.cos(th) * r, cy + Math.sin(th) * r);
  }
  await page.mouse.up();
  await page.click('#wbCutoutApply');
  await page.waitForTimeout(500);
  await page.click('#wbCutoutAgent');
  check('精抠桥接 RUN 面板打开', await waitFor(
    () => page.locator('#studioRun').evaluate(el => el.classList.contains('is-open')), 45000));
  check('精抠计划为 matting 链', await waitFor(async () => {
    const inst = await page.locator('#runInstruction').textContent();
    if (!/抠图/.test(inst)) return false;              /* 用指令文本确认是新一轮，而非上轮残留 */
    const rows = await page.evaluate(() => [...document.querySelectorAll('#runList li')].map(li => li.textContent));
    return rows.length >= 1 && rows.length <= 3 && rows.some(t => /抠|matting/i.test(t));
  }, 30000, 1500));
  await page.screenshot({ path: path.join(__dirname, '..', 'shots', 'wb_agent_matting.png') });

  check('无 pageerror', errors.length === 0, errors.join(' | ').slice(0, 200));
  await browser.close();
  let failed = 0;
  for (const r of results) { if (!r.ok) failed++; console.log((r.ok ? 'PASS' : 'FAIL') + '  ' + r.n + (r.ok ? '' : '  << ' + r.x)); }
  console.log(failed ? failed + ' FAILED' : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
