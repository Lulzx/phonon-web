import { chromium } from "playwright-core";
const browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true, args: ["--enable-unsafe-webgpu", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    "--use-file-for-fake-audio-capture=" + process.cwd() + "/docs/test/long.wav", "--autoplay-policy=no-user-gesture-required"] });
const ctx = await browser.newContext({ permissions: ["microphone"] });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) console.log("[console]", m.text()); });
await page.goto("http://127.0.0.1:8765/docs/index.html");
await page.click("#load");
await page.waitForFunction(() => !document.getElementById("rec").disabled, null, { timeout: 120000 });
await page.click("#rec");
const t0 = Date.now();
for (let i = 0; i < 26; i++) {
  await page.waitForTimeout(5000);
  const o = await page.evaluate(() => { const s = document.querySelectorAll("#out span");
    return { c: s[0]?.textContent || "", p: s[1]?.textContent || "" }; });
  if (i % 5 == 4) console.log(`t=${((Date.now() - t0) / 1000).toFixed(0)}s | ${await page.textContent("#status")}\n  committed(${o.c.length}): ...${o.c.slice(-90)}\n  pending: ${o.p}`);
}
await page.click("#rec");
await page.waitForFunction(() => document.getElementById("status").textContent === "Ready.", null, { timeout: 60000 });
console.log("FINAL:", await page.textContent("#out"));
console.log("ERR:", await page.textContent("#err"));
await browser.close();
