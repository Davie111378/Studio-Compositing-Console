// e2e_humanmatting.js — /agent 对话前端 E2E: 上传普通人像 → "用MODNet人像语义抠图"
// 验收: T01 以 humanmatting-hrnet_w18 引擎执行并出片
const { chromium } = require("playwright-core");

const BASE = "http://127.0.0.1:8765";
const IMG = "D:/AIcode/生产实习/data/ai_generated/green_fg/fg_03_glasses.png";
const SHOT = "D:/AIcode/生产实习/outputs/e2e_humanmatting_agent.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
  try {
    await page.goto(BASE + "/agent", { waitUntil: "load" });
    await page.setInputFiles("#fileInput", [IMG]);
    await page.waitForFunction(
      () => document.querySelectorAll("#pending .chip").length === 1, { timeout: 60000 });
    await page.fill("#instr", "用MODNet人像语义抠图抠出这张图里的人物");
    await page.click("#sendBtn");
    await page.waitForSelector(".msg.agent .final", { timeout: 480000 });
    await page.waitForTimeout(1500);
    const rows = await page.$$eval(".msg.agent .nodes .row",
      els => els.map(e => e.innerText.replace(/\s+/g, " ").trim()));
    console.log("[dag]"); rows.forEach(r => console.log("  ", r));
    const brief = await page.$$eval(".msg.agent .bubble > div:last-child",
      els => (els[els.length - 1] || {}).innerText || "");
    console.log("[report]", brief.trim().slice(0, 200));
    await page.screenshot({ path: SHOT, fullPage: true });
    const engineHit = rows.join("\n").includes("humanmatting");
    console.log("[shot]", SHOT, "| engine_hit:", engineHit);
    console.log(engineHit ? "E2E_PASS" : "E2E_FAIL_engine");
    if (!engineHit) process.exitCode = 1;
  } catch (e) {
    console.error("E2E_FAIL:", (e && e.message) || e);
    try { await page.screenshot({ path: SHOT, fullPage: true }); } catch (_) {}
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
