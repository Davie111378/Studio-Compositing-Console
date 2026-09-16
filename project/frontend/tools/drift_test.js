/* 响应漂移容错测试：mock 后端故意返回「接入算法模型 API 当天」可能出现的异常形态，
   断言前端全程零 pageerror 且优雅降级：
   A) 轮询持续断连 → 8s 后显示「连接不稳定」，恢复后继续跑完；
   B) 模型输出缺 light_dir 字段 → 光照徽章保持占位，不出现 NaN；
   C) run 中途 status=failed → SUMMARY 如实显示「执行中断」，不伪装成功。
   前置：dev_server(8899)。用法：
     D:\codex-tools\node-v22.17.0-win-x64\node.exe tools/drift_test.js */
'use strict';
const path = require('path');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const IMG = path.join(__dirname, '..', 'media', 'source_original.jpg');

const json = (route, obj) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(obj) });

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  let stall = false, failedOnce = false, polls = 0, goodPolls = 0;

  await page.route('**/healthz', (r) => json(r, { status: 'ok' }));
  await page.route('**/api/v1/sessions', (r) => json(r, { session_id: 'mock-s1' }));
  await page.route('**/uploads', (r) => json(r, { asset_id: 'a1', url: '/media/source_original.jpg', width: 1024, height: 1536 }));
  await page.route('**/instructions', (r) => json(r, { run_id: 'mock-r1', session_id: 'mock-s1', kind: 'create', planner: 'mock', rollback: null, plan: { nodes: [], edges: [] } }));
  await page.route('**/api/v1/runs/mock-r1', (r) => {
    polls++;
    if (stall) return r.abort();
    if (failedOnce) {
      return json(r, { status: 'failed', error: { code: 'E_TOOL_FAILED', message: 'mock: 模型推理失败' },
        dag: { nodes: [], edges: [] }, critic_history: [], created_at: Date.now(), finished_at: Date.now() });
    }
    goodPolls++;
    const prog = goodPolls >= 2;              // 第 2 次有效轮询起 lighting 完成（但 outputs 缺 light_dir）
    const st = (role, s, v) => ({ id: 'n_' + role, role, label: role, status: s, version: v || 1, outputs: {}, artifact_urls: [] });
    const nodes = [
      st('matting', 'done'),
      st('background_generate', 'done'),
      st('lighting_estimate', prog ? 'done' : 'running'),  // outputs 始终为 {} → 缺 light_dir
      st('relight', 'pending'), st('shadow_generate', 'pending'),
      st('harmonize', 'pending'), st('export', 'pending')
    ];
    nodes[0].artifact_urls = ['/media/subject_clean.png'];
    nodes[1].artifact_urls = ['/media/scene_lakeside.jpg'];
    json(r, { status: 'running', dag: { nodes, edges: [] },
      critic_history: [], created_at: Date.now() - 3000, finished_at: null });
  });

  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.click('#btnStart');
  await page.waitForSelector('#studioCreate.is-open');
  await page.setInputFiles('#fileInput', IMG);
  await page.click('#btnLaunch');
  await page.waitForSelector('#studioRun.is-open');

  // —— A) 断连 10s：不崩溃，8s 后提示重连 ——
  stall = true;
  await page.waitForTimeout(10500);
  const capStall = await page.textContent('#agentCaption');
  stall = false;

  // —— B) 缺字段：完成态 lighting_estimate.outputs={} ——
  await page.waitForTimeout(4000);
  const deg = await page.textContent('#rfDeg');
  const temp = await page.textContent('#rfTemp');

  // —— C) failed 收尾 ——
  failedOnce = true;
  await page.waitForSelector('#studioSummary.is-open', { timeout: 15000 });
  const sumFix = (await page.textContent('#sumFix')).trim();
  const sumQuote = (await page.textContent('#sumQuote')).trim();

  const clean = errors.filter((e) => !/favicon|net::ERR_FAILED|Failed to load resource/i.test(e));
  console.log('A) 断连提示:', JSON.stringify(capStall), '(期望含「重试」)');
  console.log('B) 光照徽章:', JSON.stringify({ deg, temp }), '(期望占位 —，无 NaN)');
  console.log('C) failed 收尾:', JSON.stringify({ sumFix, sumQuote }), '(期望 执行中断)');
  console.log('轮询次数:', polls, '页面错误:', clean.length ? clean : '无');
  await browser.close();

  const okA = /重试|中断/.test(capStall);
  const okB = deg !== 'NaN°' && temp !== 'NaNK' && !deg.includes('NaN') && !temp.includes('NaN');
  const okC = sumFix === '执行中断';
  if (okA && okB && okC && clean.length === 0) {
    console.log('DRIFT OK — 三类异常形态全部优雅降级，零 pageerror');
  } else {
    console.log('DRIFT FAILED', { okA, okB, okC });
    process.exit(1);
  }
})();
