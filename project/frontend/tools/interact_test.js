/* 交互链路验证：试用示例 → UPLOAD；开始生成 → 回到 EXTRACT。 */
'use strict';
const path = require('path');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const OUT = path.join(__dirname, '..', 'shots');

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errs = [];
  page.on('pageerror', (e) => errs.push(e.message));

  // 1) CREATE 章「先试用示例」→ 打开工作室 CREATE 面板
  await page.goto(BASE + '?ch=11&cp=0.6', { waitUntil: 'networkidle' });
  await page.waitForTimeout(600);
  await page.click('#btnDemo');
  await page.waitForTimeout(800);
  const st1 = await page.evaluate(() => ({
    state: document.body.dataset.state,
    open: document.querySelector('#studioCreate').classList.contains('is-open'),
    input: document.querySelector('#instructionInput').value
  }));
  console.log('after demo click:', JSON.stringify(st1), '(期望 state=create 且指令已填)');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);

  // 2) YOUR TURN → 点击「开始生成」（demo 故事：回到 02 循环）
  await page.goto(BASE + '?ch=13&cp=0.85', { waitUntil: 'networkidle' });
  await page.waitForTimeout(600);
  await page.click('#btnReplay');
  await page.waitForTimeout(2500);
  const st2 = await page.evaluate(() => {
    const u = (window.scrollY - document.querySelector('#showcase').offsetTop)
      / (document.querySelector('#showcase').offsetHeight - innerHeight) * 2060;
    return { u };
  });
  console.log('after replay click: u =', st2.u.toFixed(1), '(期望 ≈110，即 02 章起点)');

  // 3) 索引点击 → 05 章
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.waitForTimeout(400);
  await page.evaluate(() => document.querySelectorAll('.chapter-index a')[5].click());
  await page.waitForTimeout(2500);
  const st3 = await page.evaluate(() => {
    const u = (window.scrollY - document.querySelector('#showcase').offsetTop)
      / (document.querySelector('#showcase').offsetHeight - innerHeight) * 2060;
    return { u };
  });
  console.log('after index 05 click: u =', st3.u.toFixed(1), '(期望 ≈535)');

  await browser.close();
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'interactions OK, no page errors');
})();
