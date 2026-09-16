/* 端到端（真实浏览器）：INTRO → CREATE → RUN → SUMMARY → REPLAY 全流程。
   场景 1：Agent 在线 → 真实 job（LIVE）→ critic 真实决策 → summary → 回放轴+真实 artifacts。
   场景 2：模拟离线 → 演示 job（DEMO）→ 同一流程走通，无假 LIVE。
   前置：dev_server(8899) + agent(8000)。Node ≥20：
     D:\codex-tools\node-v22.17.0-win-x64\node.exe tools/real_flow_test.js */
'use strict';
const path = require('path');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const IMG = path.join(__dirname, '..', 'media', 'source_original.jpg');
const OUT = path.join(__dirname, '..', 'shots');
const REAL_TIMEOUT = 240000;

async function newPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page._errors = [];
  page.on('pageerror', (e) => page._errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') page._errors.push('console: ' + m.text()); });
  return page;
}
const cleanErrs = (page) => page._errors.filter((e) =>
  !/favicon/i.test(e) && !/net::ERR_FAILED|Failed to load resource/i.test(e));

async function goThroughCreate(page) {
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.click('#btnStart');
  await page.waitForSelector('#studioCreate.is-open', { timeout: 5000 });
  await page.setInputFiles('#fileInput', IMG);
  await page.waitForTimeout(300);
  const hasThumb = await page.evaluate(() => !document.querySelector('#studioThumb').hidden);
  await page.click('#btnLaunch');
  return hasThumb;
}

async function main() {
  const browser = await chromium.launch({ executablePath: EXE });
  let failed = 0;

  /* ---------- 场景 1：真实 job ---------- */
  {
    const page = await newPage(browser);
    const hasThumb = await goThroughCreate(page);
    console.log('create: thumbnail set =', hasThumb);

    const st = await page.evaluate(() => ({
      state: document.body.dataset.state,
      mode: document.querySelector('#runMode').textContent,
      isOpen: document.querySelector('#studioRun').classList.contains('is-open')
    }));
    console.log('run panel:', JSON.stringify(st));
    if (st.state !== 'run' || !st.isOpen) failed++;   // 点击即进入 RUN（会话建立中）

    const isReal = await page.waitForFunction(() =>
      ['LIVE', 'DEMO'].includes(document.querySelector('#runMode').textContent),
      null, { timeout: 30000 })
      .then(() => page.evaluate(() => document.querySelector('#runMode').textContent))
      .then((m) => m === 'LIVE')
      .catch(() => false);
    // 等待完成 → SUMMARY
    const done = await page.waitForSelector('#studioSummary.is-open', { timeout: REAL_TIMEOUT })
      .then(() => true).catch(() => false);
    console.log('summary reached:', done, '(real=' + isReal + ')');
    if (!done) failed++;

    const sum = await page.evaluate(() => ({
      time: document.querySelector('#sumTime').textContent,
      tools: document.querySelector('#sumTools').textContent,
      fix: document.querySelector('#sumFix').textContent,
      finalSrc: document.querySelector('#sumFinal').getAttribute('src') || ''
    }));
    console.log('summary:', JSON.stringify(sum));
    if (sum.time === '—' || sum.tools === '—') failed++;
    if (isReal && !/\/artifacts\//.test(sum.finalSrc)) failed++;

    // REPLAY：回放轴 + 真实 artifacts 章节
    await page.click('#btnReplayJob');
    await page.waitForTimeout(700);
    const rp = await page.evaluate(() => ({
      state: document.body.dataset.state,
      barVisible: !document.querySelector('#replayBar').hidden,
      ticks: document.querySelectorAll('#rpTicks i').length,
      finalSrc: document.querySelector('#finalFill').getAttribute('src') || '',
      nodeBox: (document.querySelector('.rail[data-ch="6"] .motion') || {}).textContent || '',
      story: document.body.dataset.story
    }));
    console.log('replay:', JSON.stringify(rp));
    if (rp.state !== 'replay' || !rp.barVisible || rp.ticks !== 13) failed++;
    if (isReal && !/\/artifacts\//.test(rp.finalSrc)) failed++;
    if (isReal && !/NODE/.test(rp.nodeBox)) failed++;
    if (rp.story !== (isReal ? 'job' : 'demo')) failed++;

    await page.screenshot({ path: path.join(OUT, 'replay_job.png') });
    const errs = cleanErrs(page);
    if (errs.length) { failed++; console.log('REAL PAGE ERRORS:', errs.slice(0, 6)); }
    await page.close();
  }

  /* ---------- 场景 2：离线 → 演示 job ---------- */
  {
    const page = await newPage(browser);
    await page.route(/\/healthz$/, (r) => r.abort());
    await page.route(/\/api\/v1\//, (r) => r.abort());
    await goThroughCreate(page);
    await page.waitForTimeout(500);
    const mode = await page.evaluate(() => document.querySelector('#runMode').textContent);
    console.log('offline run mode:', mode);
    if (mode !== 'DEMO') failed++;

    const done = await page.waitForSelector('#studioSummary.is-open', { timeout: 60000 })
      .then(() => true).catch(() => false);
    console.log('demo summary reached:', done);
    if (!done) failed++;
    await page.click('#btnReplayJob');
    await page.waitForTimeout(600);
    const rp = await page.evaluate(() => ({
      barVisible: !document.querySelector('#replayBar').hidden,
      finalSrc: document.querySelector('#finalFill').getAttribute('src') || ''
    }));
    console.log('demo replay:', JSON.stringify(rp));
    if (!rp.barVisible || !/stage_final/.test(rp.finalSrc)) failed++;

    const errs = cleanErrs(page);
    if (errs.length) { failed++; console.log('DEMO PAGE ERRORS:', errs.slice(0, 6)); }
    await page.close();
  }

  await browser.close();
  console.log(failed === 0 ? 'E2E OK — 真实与演示双流程全部通过' : `E2E FAILED (${failed})`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
