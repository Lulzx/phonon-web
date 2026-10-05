// Fetch the model: either a single .phon file or a manifest.json listing parts (GitHub Pages caps files at 100 MB).
// onProgress(bytesSoFar, totalBytes).  The assembled file is cached with the Cache API under the given URL.
export const CACHE = "phonon-model-v1";

async function fetchBytes(url, onChunk) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: ${url} HTTP ${res.status}`);
  const reader = res.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onChunk(value.length);
  }
  return chunks;
}

export async function cachedModel(url) {
  try {
    const hit = await (await caches.open(CACHE)).match(url);
    return hit ? await hit.arrayBuffer() : null;
  } catch { return null; }
}

export async function fetchModel(url, onProgress = () => {}) {
  let parts = [url], total = 0;
  if (url.endsWith(".json")) {
    const m = await (await fetch(url)).json();
    parts = m.parts.map((p) => new URL(p, url).href);
    total = m.bytes;
  }
  let got = 0;
  const chunks = [];
  for (const p of parts) {
    // retry each part a few times: big downloads over flaky connections (or HTTP/3 hiccups) do fail mid-stream
    for (let attempt = 1; ; attempt++) {
      let partGot = 0;
      try {
        chunks.push(...await fetchBytes(p, (n) => { partGot += n; onProgress(got + partGot, total); }));
        got += partGot;
        break;
      } catch (e) {
        if (attempt >= 4) throw e;
        onProgress(got, total);
        await new Promise((r) => setTimeout(r, 500 * attempt));
      }
    }
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { await (await caches.open(CACHE)).put(url, new Response(buf)); } catch { /* no Cache API or quota */ }
  return buf.buffer;
}
