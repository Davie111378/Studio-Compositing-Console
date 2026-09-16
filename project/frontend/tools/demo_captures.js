// -*- coding: utf-8 -*-
// demo_captures.js —— 规范 §3.4 四个 Demo 剧本的截图归档（真实链路，mock 引擎）
//
//   Demo01 主链路+语音入口：CREATE（麦克风可见）-> RUN -> SUMMARY
//   Demo02 空间指代：API 级验证（instructions 带 spatial），截图 PLAN 卡
//   Demo03 自评修正：RUN 中 critic 面板（未达标 -> reroll）
//   Demo04 条件回滚：版本树 -> 回滚指令 -> 重跑 -> SUMMARY
//
// 前置：agent :8000 + dev_server :8899 在跑。
// 用法：D:\codex-tools\node-v22.17.0-win-x64\node.exe tools/demo_captures.js
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const IMG = path.join(__dirname, '..', 'media', 'source_original.jpg');
const OUT = path.join(__dirname, '..', '..', 'demo', 'screenshots');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));

  // ---- Demo01 主链路 ----
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.click('#btnStart');
  await page.waitForSelector('#studioCreate.is-open');
  await page.setInputFiles('#fileInput', IMG);
  await page.fill('#instructionInput', '把这个人放进傍晚的咖啡馆，光从左边照过来，阴影自然一点');
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'demo01_create_voice_entry.png') });

  await page.click('#btnLaunch');
  await page.waitForSelector('#studioRun.is-open');
  await page.waitForTimeout(6000);   // RUN 中段：PLAN 推进
  await page.screenshot({ path: path.join(OUT, 'demo01_run_plan.png') });

  // critic 面板出现（自检/重跑）时抓一张
  await page.waitForSelector('#criticLive.is-on', { timeout: 40000 }).catch(() => {});
  const criticOn = await page.$('#criticLive.is-on');
  if (criticOn) await page.screenshot({ path: path.join(OUT, 'demo03_critic_selfcheck.png') });

  await page.waitForSelector('#studioSummary.is-open', { timeout: 90000 });
  await page.screenshot({ path: path.join(OUT, 'demo01_summary.png') });

  // ---- Demo02 空间指代（API 级：spatial -> matting trimap）----
  const demo02 = await page.evaluate(async () => {
    const s = await (await fetch('/api/v1/sessions', { method: 'POST' })).json();
    const img = await fetch('/media/source_original.jpg').then(r => r.blob());
    const fd = new FormData();
    fd.append('file', new File([img], 'demo02.jpg', { type: 'image/jpeg' }));
    await fetch(`/api/v1/sessions/${s.session_id}/uploads`, { method: 'POST', body: fd });
    const r = await (await fetch(`/api/v1/sessions/${s.session_id}/instructions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: '点击这里，把这个主体抠出来', quality: 'draft',
                             spatial: { click: { x: 240, y: 320 } } }),
    })).json();
    return r;
  });
  fs.writeFileSync(path.join(OUT, 'demo02_spatial_run.json'),
                   JSON.stringify(demo02, null, 2));
  console.log('Demo02 spatial run_id:', demo02.run_id || '(见 json)');

  // ---- Demo04 条件回滚（继续编辑 + 版本树）----
  await page.fill('#refineInput', '背景换回上一版，但是保留现在的光线');
  await page.click('#btnRefine');
  await page.waitForSelector('#studioRun.is-open', { timeout: 20000 });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: path.join(OUT, 'demo04_rollback_rerun.png') });
  await page.waitForSelector('#studioSummary.is-open', { timeout: 90000 });
  await page.click('#versionsBox summary');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'demo04_versions_tree.png') });

  if (errors.length) throw new Error('页面错误: ' + errors[0]);
  console.log('DEMO CAPTURES OK ->', OUT);
  await browser.close();
})().catch(e => { console.error('DEMO CAPTURES FAIL —', e.message); process.exit(1); });
