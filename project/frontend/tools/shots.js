/* 批量截图：按章节/进度走查页面渲染。
   用法：node tools/shots.js */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright-core');

const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const BASE = 'http://127.0.0.1:8899/';
const OUT = path.join(__dirname, '..', 'shots');

const CHECKS = [
  { name: 'hero', q: '' },
  { name: 'hero_clear', q: 'SCROLL:0.45' },
  { name: 'ch01_ask', q: '?ch=1&cp=0.14' },
  { name: 'ch01_mid', q: '?ch=1&cp=0.62' },
  { name: 'ch02_bg', q: '?ch=2&cp=0.25' },
  { name: 'ch02_edge', q: '?ch=2&cp=0.75' },
  { name: 'ch02_end', q: '?ch=2&cp=0.9' },
  { name: 'ch03_mid', q: '?ch=3&cp=0.5' },
  { name: 'ch03_end', q: '?ch=3&cp=0.97' },
  { name: 'ch04', q: '?ch=4&cp=0.8' },
  { name: 'ch05_mid', q: '?ch=5&cp=0.5' },
  { name: 'ch05_end', q: '?ch=5&cp=0.92' },
  { name: 'ch06', q: '?ch=6&cp=0.7' },
  { name: 'ch07', q: '?ch=7&cp=0.65' },
  { name: 'ch08', q: '?ch=8&cp=0.7' },
  { name: 'ch09_rows', q: '?ch=9&cp=0.3' },
  { name: 'ch09_reroll', q: '?ch=9&cp=0.6' },
  { name: 'ch09_pass', q: '?ch=9&cp=0.98' },
  { name: 'ch10', q: '?ch=10&cp=0.8' },
  { name: 'ch11', q: '?ch=11&cp=0.5' },
  { name: 'ch12', q: '?ch=12&cp=0.85' },
  { name: 'ch13', q: '?ch=13&cp=0.8' },
  { name: 'footer', q: null }   /* 滚到页底 */
];

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  for (const c of CHECKS) {
    if (c.q === null) {
      await page.goto(BASE, { waitUntil: 'networkidle' });
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    } else if (c.q.startsWith('SCROLL:')) {
      await page.goto(BASE, { waitUntil: 'networkidle' });
      await page.evaluate((frac) => window.scrollTo(0, innerHeight * frac), parseFloat(c.q.split(':')[1]));
    } else {
      await page.goto(BASE + c.q, { waitUntil: 'networkidle' });
    }
    await page.waitForTimeout(900);
    const file = path.join(OUT, c.name + '.png');
    await page.screenshot({ path: file });
    console.log('shot', c.name);
  }

  /* RELIGHT 拖光交互：把光源从默认位拖到右上 */
  await page.goto(BASE + '?ch=5&cp=0.6', { waitUntil: 'networkidle' });
  await page.waitForTimeout(900);
  const box = await page.evaluate(() => {
    const r = document.querySelector('#stage').getBoundingClientRect();
    return { x: r.left + r.width * 0.175, y: r.top + r.height * 0.24, w: r.width, h: r.height };
  });
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.w * 0.25, box.y - box.h * 0.12, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'ch05_drag.png') });
  console.log('shot ch05_drag');
  const glowOn = await page.evaluate(() => {
    const g = document.querySelector('#relightGlow');
    return g && g.classList.contains('is-on');
  });
  console.log('glow after drag:', glowOn);

  await browser.close();
  if (errors.length) {
    console.log('PAGE ERRORS:\n' + errors.slice(0, 10).join('\n'));
  } else {
    console.log('no page errors');
  }
})();
