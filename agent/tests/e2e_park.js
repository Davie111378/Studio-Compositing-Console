// e2e_park.js — 真实场景: 图1=北海公园背景(用户上传) + 图2=B站截图人像
// 指令用用户原话, 验证 agent 双图角色解析 + 全链路
const { chromium } = require("playwright-core");

const BASE = "http://127.0.0.1:8765";
const IMG_A = "C:/Users/zhaod/.workbuddy/clipboard-images/clipboard-2026-09-11T17-15-15-154Z-e052e6ba.jpg"; // 图1 背景
const IMG_B = "C:/Users/zhaod/.workbuddy/clipboard-images/clipboard-2026-09-11T17-15-15-156Z-860f2a0e.jpg"; // 图2 人物
const SHOT = "D:/AIcode/生产实习/outputs/e2e_park_agent.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
  page.setDefaultTimeout(60000);
  try {
    await page.goto(BASE + "/agent", { waitUntil: "load" });
    await page.setInputFiles("#fileInput", [IMG_A, IMG_B]);
    await page.waitForFunction(
      () => document.querySelectorAll("#pending .chip").length === 2,
      { timeout: 60000 });
    const chips = await page.$$eval("#pending .chip",
      els => els.map(e => e.innerText.replace(/\s+/g, " ").trim()));
    console.log("[chips]", JSON.stringify(chips));
    await page.fill("#instr", "图一是背景图，图二的背景用图一替换");
    await page.click("#sendBtn");
    await page.waitForSelector(".msg.agent .final", { timeout: 480000 });
    await page.waitForTimeout(1500);
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
