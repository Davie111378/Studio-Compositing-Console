/* 工作室五态截图：INTRO / CREATE / RUN（含 critic 重跑）/ SUMMARY / REPLAY。
   离线拦截走 DEMO job，节奏可控；真实链路断言见 real_flow_test.js。 */
'use strict';
const path = require('path');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const OUT = path.join(__dirname, '..', 'shots');

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.route(/\/healthz$/, (r) => r.abort());
  await page.route(/\/api\/v1\//, (r) => r.abort());

  // INTRO（含 CTA）
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, 's1_intro.png') });

  // CREATE
  await page.click('#btnStart');
  await page.waitForSelector('#studioCreate.is-open');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 's2_create.png') });

  // RUN：demo job 全自动
  await page.click('#btnLaunch');
  await page.waitForTimeout(2600);   // T01/T02 执行中
  await page.screenshot({ path: path.join(OUT, 's3_run_nodes.png') });
  await page.waitForTimeout(13200);  // 第一轮 critic 失败 → 重跑面板（≈16s 处）
  await page.screenshot({ path: path.join(OUT, 's4_run_critic.png') });

  // SUMMARY
  await page.waitForSelector('#studioSummary.is-open', { timeout: 60000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 's5_summary.png') });

  // REPLAY（demo 故事）+ 回放轴
  await page.click('#btnReplayJob');
  await page.waitForTimeout(900);
  await page.screenshot({ path: path.join(OUT, 's6_replay_top.png') });
  // 拖动回放轴到 CRITIC 章
  const track = await page.evaluate(() => {
    const r = document.querySelector('#rpTrack').getBoundingClientRect();
    return { x: r.left, w: r.width, y: r.top };
  });
  await page.mouse.click(track.x + track.w * 0.68, track.y);
  await page.waitForTimeout(900);
  await page.screenshot({ path: path.join(OUT, 's7_replay_critic.png') });

  await browser.close();
  console.log(errors.length ? 'PAGE ERRORS: ' + errors.join(' | ') : 'studio shots done, no page errors');
})();
