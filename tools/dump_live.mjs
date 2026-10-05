import { chromium } from "playwright-core";
import fs from "fs";
const browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true, args: ["--enable-unsafe-webgpu", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    "--use-file-for-fake-audio-capture=" + process.cwd() + "/docs/test/long.wav"] });
const page = await (await browser.newContext({ permissions: ["microphone"] })).newPage();
await page.goto("http://127.0.0.1:8765/docs/index.html");
await page.click("#load");
await page.waitForFunction(() => !document.getElementById("rec").disabled, null, { timeout: 120000 });
await page.click("#rec");
await page.waitForTimeout(40000);
const a = await page.evaluate(() => { const r = window.__live.rs; return [r.ctxRate = window.__live.ctx.sampleRate, Array.from(r.out.subarray(0, r.m))]; });
console.log("rate", a[0], "samples", a[1].length);
fs.writeFileSync("/tmp/live.f32", Buffer.from(new Float32Array(a[1]).buffer));
await browser.close();
