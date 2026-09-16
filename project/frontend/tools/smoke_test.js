/* 冒烟测试（浏览器版）：在真实 chromium 里加载首页，
   以 5vh 步长遍历整条滚动旅程 + 深链抽查每章关键进度，捕获 pageerror / console.error。
   覆盖 13 章全部渲染路径（含几何 morph、图层溶解、视频 scrub、深浅色切换）。

   前置：frontend 目录下已启动 `python tools/dev_server.py`（:8899）。
   用法（需 Node ≥20 跑 playwright-core）：
     D:\codex-tools\node-v22.17.0-win-x64\node.exe tools/smoke_test.js */
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
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  /* 1) 常规加载 + 5vh 步进扫完整条滚动旅程（每步推 4 个 rAF，让阻尼引擎收敛） */
  await page.goto(BASE, { waitUntil: 'networkidle' });
  const totalVh = await page.evaluate(() => {
    const sc = document.querySelector('#showcase');
    return (sc.offsetHeight - innerHeight) / innerHeight * 100 - 100;
  });
  for (let u = 0; u <= totalVh; u += 5) {
    await page.evaluate((uvh) => {
      const sc = document.querySelector('#showcase');
      const len = sc.offsetHeight - innerHeight;
      window.scrollTo(0, sc.offsetTop + (uvh / 2060) * len);
      return new Promise((res) => {
        let n = 0;
        const tick = () => ((n++ < 4) ? requestAnimationFrame(tick) : res());
        tick();
      });
    }, u).catch((e) => errors.push('sweep@' + u + 'vh: ' + e.message));
  }

  /* 2) 深链抽查：每章 起点/中段/末段 三个状态直达（含 09 CRITIC 的 PASS 末态） */
  for (let ch = 1; ch <= 13; ch++) {
    for (const cp of [0.02, 0.5, 0.98]) {
      await page.goto(BASE + `?ch=${ch}&cp=${cp}`, { waitUntil: 'load', timeout: 15000 })
        .catch((e) => errors.push(`goto ch=${ch} cp=${cp}: ${e.message}`));
      await page.waitForTimeout(350);
    }
  }

  /* 3) 深链 ?p= 与页脚 */
  await page.goto(BASE + '?p=0.45', { waitUntil: 'load' }).catch((e) => errors.push('goto p=0.45: ' + e.message));
  await page.waitForTimeout(400);
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'smoke_footer.png') });

  await browser.close();

  const real = errors.filter((e) => !/favicon/i.test(e));
  if (real.length === 0) {
    console.log('SMOKE OK — 全章节扫掠 + 深链抽查无运行时异常');
    process.exit(0);
  }
  console.error('SMOKE FAILED — ' + real.length + ' 处异常：\n' + real.slice(0, 12).join('\n'));
  process.exit(1);
})();
