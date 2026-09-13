// e2e_refiner_real.js — /agent 对话前端 E2E: 真实人像 + 精修档 → T01 应挂 refiner_real
// 前置: active_refiner.txt 已切 refiner_real 且服务已重启
const { chromium } = require("playwright-core");

const BASE = "http://127.0.0.1:8765";
const IMG = "D:/AIcode/生产实习/data/matting_real/test/hm_1803281005-00000001.jpg";
const SHOT = "D:/AIcode/生产实习/outputs/e2e_refiner_real_agent.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
  try {
    await page.goto(BASE + "/agent", { waitUntil: "load" });
    // 切到精修档 (quality=fine 才触发 AlphaRefiner)
    await page.click("#modeBtn");
    await page.evaluate(() => {
      const items = Array.from(document.querySelectorAll("#modeMenu div"));
      const fine = items.find(d => d.innerText.includes("精修"));
      if (!fine) throw new Error("modeMenu 未找到精修项: " + items.map(d => d.innerText).join(","));
      fine.click();
    });
    const modeLabel = await page.textContent("#modeLabel");
    await page.setInputFiles("#fileInput", [IMG]);
    await page.waitForFunction(
      () => document.querySelectorAll("#pending .chip").length === 1, { timeout: 60000 });
    await page.fill("#instr", "把这张真实人像抠出来，发丝边缘要干净");
    await page.click("#sendBtn");
    await page.waitForSelector(".msg.agent .final", { timeout: 480000 });
    await page.waitForTimeout(1500);
    const rows = await page.$$eval(".msg.agent .nodes .row",
      els => els.map(e => e.innerText.replace(/\s+/g, " ").trim()));
    console.log("[mode]", modeLabel);
    console.log("[dag]"); rows.forEach(r => console.log("  ", r));
    const brief = await page.$$eval(".msg.agent .bubble > div:last-child",
      els => (els[els.length - 1] || {}).innerText || "");
    console.log("[report]", brief.trim().slice(0, 200));
    await page.screenshot({ path: SHOT, fullPage: true });
    const hit = rows.join("\n").includes("refiner_real");
    console.log("[shot]", SHOT, "| refiner_real_hit:", hit);
    console.log(hit ? "E2E_PASS" : "E2E_FAIL_engine");
    if (!hit) process.exitCode = 1;
  } catch (e) {
    console.error("E2E_FAIL:", (e && e.message) || e);
    try { await page.screenshot({ path: SHOT, fullPage: true }); } catch (_) {}
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
