import { chromium } from "playwright-core";
const url = process.argv[2] || "http://127.0.0.1:8765/docs/test.html";
const browser = await chromium.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: process.env.HEADFUL ? false : true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=metal", "--disable-dawn-features=disallow_unsafe_apis"],
});
const page = await browser.newPage();
page.on("console", (m) => console.log("[page]", m.text()));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(url);
const t = Date.now();
while (Date.now() - t < (+process.env.TIMEOUT || 600000)) {
  const r = await page.evaluate(() => window.__results);
  if (r) { console.log("DONE", JSON.stringify(r).slice(0, 300)); break; }
  await new Promise((r) => setTimeout(r, 1000));
}
await browser.close();
