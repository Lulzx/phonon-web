import { chromium } from "playwright-core";
const browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => { if (m.type() === "error") console.log("[console]", m.text()); });
await page.goto(process.env.URL || "http://127.0.0.1:8765/docs/index.html");
await page.click("#load");
await page.waitForFunction(() => !document.getElementById("pick").disabled || document.getElementById("err").textContent, null, { timeout: 120000 });
console.log("status:", await page.textContent("#status"));
for (const f of ["test/long.wav", "test/2.wav"]) {
  await page.setInputFiles("#file", "docs/" + f);
  await page.waitForFunction(() => document.getElementById("stats").textContent || document.getElementById("err").textContent, null, { timeout: 120000 });
  console.log(f, "|", (await page.textContent("#out")).slice(0, 160), "|", await page.textContent("#stats"), "|", await page.textContent("#err"));
  await page.check("#ts"); console.log("ts:", (await page.textContent("#out")).slice(0, 120)); await page.uncheck("#ts");
  await page.evaluate(() => { document.getElementById("stats").textContent = ""; });
}
await browser.close();
