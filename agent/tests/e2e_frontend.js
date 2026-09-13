// e2e_frontend.js — /agent 对话前端端到端测试 (用户测试约定: Agent 功能测试一律走对话前端)
// 场景: 上传两张图 A(全景背景) + B(绿幕人像) → 指令"将A图片作为B图片的背景" → 出合成片
// 运行: NODE_PATH=<workspace>/node_modules node e2e_frontend.js
const { chromium } = require("playwright-core");

const BASE = "http://127.0.0.1:8765";
const IMG_A = "D:/AIcode/生产实习/data/ai_generated/studio_bg/bg_03_panorama.png";   // A=背景
const IMG_B = "D:/AIcode/生产实习/data/ai_generated/green_fg/fg_03_glasses.png";     // B=前景
const SHOT = "D:/AIcode/生产实习/outputs/e2e_two_image_agent.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
  page.setDefaultTimeout(60000);
  try {
    await page.goto(BASE + "/agent", { waitUntil: "load" });
    // 1) 一次选两张 (multiple) — 前端循环上传, 等 2 个 chip 渲染
    await page.setInputFiles("#fileInput", [IMG_A, IMG_B]);
    await page.waitForFunction(
      () => document.querySelectorAll("#pending .chip").length === 2,
      { timeout: 60000 });
    const chips = await page.$$eval("#pending .chip",
      els => els.map(e => e.innerText.replace(/\s+/g, " ").trim()));
    console.log("[chips]", JSON.stringify(chips, null, 0));
    // 2) 指令 + 发送
    await page.fill("#instr", "将A图片作为B图片的背景");
    await page.click("#sendBtn");
    // 3) 等成片 (LLM 规划 + T01~T06 全链, 上限 8 分钟)
    await page.waitForSelector(".msg.agent .final", { timeout: 480000 });
    await page.waitForTimeout(1500);   // 等汇报文本渲染
    const rows = await page.$$eval(".msg.agent .nodes .row",
      els => els.map(e => e.innerText.replace(/\s+/g, " ").trim()));
    console.log("[dag]");
    rows.forEach(r => console.log("  ", r));
    const meta = await page.$$eval(".msg.agent .meta",
      els => els.map(e => e.innerText.replace(/\s+/g, " ").trim()));
    meta.forEach(m => console.log("[meta]", m));
    const brief = await page.$$eval(".msg.agent .bubble > div:last-child",
      els => (els[els.length - 1] || {}).innerText || "");
    console.log("[report]", brief.trim().slice(0, 300));
    const src = await page.$eval(".msg.agent .final", el => el.getAttribute("src"));
    console.log("[final]", src);
    await page.screenshot({ path: SHOT, fullPage: true });
    console.log("[shot]", SHOT);
    console.log("E2E_PASS");
  } catch (e) {
    console.error("E2E_FAIL:", (e && e.message) || e);
    try { await page.screenshot({ path: SHOT, fullPage: true }); } catch (_) {}
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
