// -*- coding: utf-8 -*-
// multiround_test.js —— C4 多轮交互验收：规范 §3.5 五轮剧本（全 UI 驱动）
//
//   R1 把人物放到咖啡馆（初始创作）
//   R2 改成傍晚
//   R3 光线太冷了
//   R4 阴影轻一点
//   R5 背景换回上一版，但是保留现在的光线   <- 条件回滚语义（C6/Demo04）
//
// 断言：5 轮全部到达 SUMMARY（done）；版本树出现 background_generate 多版本；
//       全程无 pageerror。
//
// 前置：agent :8000（mock 引擎即可）+ dev_server :8899 在跑。
// 用法：D:\codex-tools\node-v22.17.0-win-x64\node.exe tools/multiround_test.js
const { chromium } = require('playwright-core');

const SITE = process.env.IMC_SITE || 'http://127.0.0.1:8899';
const EXE = process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1217\\chrome-win64\\chrome.exe';
const IMG = process.env.IMC_IMG ||
  require('path').join(__dirname, '..', 'media', 'source_original.jpg');
const ROUND_MS = 90000;

const ROUNDS = [
  '把人物放到咖啡馆',
  '改成傍晚',
  '光线太冷了',
  '阴影轻一点',
  '背景换回上一版，但是保留现在的光线',
];

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));

  await page.goto(SITE, { waitUntil: 'domcontentloaded' });
  await page.click('#btnStart');
  await page.waitForSelector('#studioCreate.is-open', { timeout: 5000 });

  for (let i = 0; i < ROUNDS.length; i++) {
    const text = ROUNDS[i];
    const t0 = Date.now();
    if (i === 0) {
      // 首轮：上传 + 指令
      await page.setInputFiles('#fileInput', IMG);
      await page.waitForTimeout(200);
      await page.fill('#instructionInput', text);
      await page.click('#btnLaunch');
    } else {
      // 后续轮：SUMMARY 里继续编辑（同 session 多轮）
      await page.fill('#refineInput', text);
      await page.click('#btnRefine');
    }
    // 等 RUN 开始并回到 SUMMARY（本轮 done）
    await page.waitForSelector('#studioRun.is-open', { timeout: 20000 });
    await page.waitForSelector('#studioSummary.is-open', { timeout: ROUND_MS });
    const round = await page.evaluate(() => ({
      fix: (document.querySelector('#sumFix') || {}).textContent || '',
      time: (document.querySelector('#sumTime') || {}).textContent || '',
    }));
    console.log(`R${i + 1} done ${((Date.now() - t0) / 1000).toFixed(1)}s · ${text} · 修正:${round.fix} 耗时:${round.time}`);
  }

  // 版本树应已出现（多轮 -> 节点多版本）
  await page.click('#versionsBox summary');
  await page.waitForTimeout(400);
  const verInfo = await page.evaluate(() => {
    const roles = Array.from(document.querySelectorAll('#versionsList .ver-role'))
      .map(e => e.textContent.trim());
    return { roles, btns: document.querySelectorAll('#versionsList .ver-btn:not(.is-current)').length };
  });
  if (!verInfo.roles.length) throw new Error('版本树为空');
  if (!verInfo.roles.some(r => /background_generate/.test(r))) {
    throw new Error('版本树缺少 background_generate: ' + verInfo.roles.join(','));
  }
  console.log('版本树:', verInfo.roles.join(', '), '· 可回滚按钮', verInfo.btns, '个');

  if (errors.length) throw new Error('页面运行时错误: ' + errors[0]);
  console.log('MULTIROUND OK — 5 轮连续编辑全部通过，状态无错乱');
  await browser.close();
})().catch(e => { console.error('MULTIROUND FAIL —', e.message); process.exit(1); });
