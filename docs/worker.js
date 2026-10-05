// Runs the engine off the main thread.  Messages in: {type:"load", url} | {type:"transcribe", id, pcm}
// Messages out: {type:"progress", phase, frac} | {type:"ready", info} | {type:"result", id, ...} | {type:"error", msg}
import { Phonon } from "./engine.js";
import { fetchModel, cachedModel } from "./model.js";

let eng = null;

onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === "load") {
      const t0 = performance.now();
      let buf = await cachedModel(m.url);
      if (buf) postMessage({ type: "progress", phase: "download", frac: 1, cached: true });
      else buf = await fetchModel(m.url, (bytes, total) =>
        postMessage({ type: "progress", phase: "download", frac: total ? bytes / total : 0, bytes, total }));
      const t1 = performance.now();
      eng = await Phonon.create(buf, { onProgress: (f) => postMessage({ type: "progress", phase: "init", frac: f }) });
      eng.onLayer = (i, n) => postMessage({ type: "progress", phase: "encode", frac: i / n });
      postMessage({ type: "ready", info: { bytes: buf.byteLength, download_ms: t1 - t0, init_ms: performance.now() - t1,
        adapter: eng.adapterInfo } });
    } else if (m.type === "transcribe") {
      const r = await eng.transcribe(m.pcm);
      postMessage({ type: "result", id: m.id, ...r });
    }
  } catch (e) {
    postMessage({ type: "error", id: m.id, msg: String(e && e.stack || e) });
  }
};
