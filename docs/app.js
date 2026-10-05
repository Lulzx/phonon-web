import { decodeAudio } from "./audio.js";
import { Live } from "./live.js";

const MODEL_URL = new URLSearchParams(location.search).get("model") || "./model/manifest.json";
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let ready = false, busy = false, last = null, nextId = 1;
const pending = new Map();

const setBar = (f) => { $("bar").style.width = `${Math.round(f * 100)}%`; };
const status = (s) => { $("status").textContent = s; };
const mb = (b) => `${(b / 1e6).toFixed(1)} MB`;

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "progress") {
    setBar(m.frac);
    if (m.phase === "download") status(m.cached ? "Loading model from cache…" :
      `Downloading model… ${m.total ? `${mb(m.bytes)} / ${mb(m.total)}` : mb(m.bytes || 0)}`);
    else if (m.phase === "init") status("Decompressing weights and uploading to the GPU…");
    else if (m.phase === "encode") status(`Encoding… layer ${Math.round(m.frac * 24)} / 24`);
  } else if (m.type === "ready") {
    ready = true;
    setBar(1);
    status(`Ready — ${mb(m.info.bytes)} model, initialised in ${(m.info.init_ms / 1000).toFixed(1)} s` +
      (m.info.adapter ? ` on ${m.info.adapter}` : ""));
    $("load").textContent = "Model loaded";
    $("pick").disabled = $("rec").disabled = false;
  } else if (m.type === "result" || m.type === "error") {
    const p = pending.get(m.id);
    if (p) { pending.delete(m.id); m.type === "error" ? p.reject(new Error(m.msg)) : p.resolve(m); }
    else if (m.type === "error") showErr(m.msg);
  }
};

function showErr(s) { $("err").textContent = s; $("load").disabled = false; }

function transcribePcm(pcm) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "transcribe", id, pcm }, [pcm.buffer]);
  });
}

// split very long audio at quiet points so the full-attention encoder stays within GPU memory
function segments(pcm, maxSec = 600) {
  const max = maxSec * 16000;
  if (pcm.length <= max) return [[0, pcm.length]];
  const out = [];
  let s = 0;
  while (pcm.length - s > max) {
    const lo = s + max - 30 * 16000, hi = s + max;
    let best = hi, be = Infinity;
    for (let c = lo; c + 3200 <= hi; c += 1600) {
      let e = 0;
      for (let i = c; i < c + 3200; i++) e += pcm[i] * pcm[i];
      if (e < be) { be = e; best = c + 1600; }
    }
    out.push([s, best]);
    s = best;
  }
  out.push([s, pcm.length]);
  return out;
}

const fmtT = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

function render() {
  if (!last) return;
  const out = $("out");
  out.textContent = "";
  if (!$("ts").checked) { out.textContent = last.text; return; }
  for (const seg of last.sentences) {
    const d = document.createElement("span");
    d.className = "seg";
    const b = document.createElement("b");
    b.textContent = fmtT(seg.start);
    d.append(b, seg.text);
    out.append(d);
  }
}

// group tokens into sentences with start times (encoder frames are 80 ms)
function sentences(vocabText, toks, offset) {
  const res = [];
  let cur = "", start = null;
  for (const t of toks) {
    if (start === null) start = offset + t.frame * 0.08;
    cur += t.piece;
    if (/[.?!]$/.test(t.piece)) {
      res.push({ start, text: cur.replace(/▁/g, " ").trim() });
      cur = ""; start = null;
    }
  }
  if (cur.trim()) res.push({ start: start ?? offset, text: cur.replace(/▁/g, " ").trim() });
  return res;
}

async function handleAudio(buf, label) {
  if (!ready || busy) return;
  busy = true;
  $("err").textContent = "";
  $("out").textContent = "";
  $("stats").textContent = "";
  try {
    status(`Decoding ${label}…`);
    const pcm = await decodeAudio(buf);
    const segs = segments(pcm);
    const parts = [];
    let enc = 0, dec = 0, feat = 0;
    const t0 = performance.now();
    for (const [a, b] of segs) {
      const r = await transcribePcm(pcm.slice(a, b));
      enc += r.timing.encoder; dec += r.timing.decoder; feat += r.timing.features;
      parts.push({ r, offset: a / 16000 });
    }
    const wall = (performance.now() - t0) / 1000;
    const dur = pcm.length / 16000;
    const text = parts.map((p) => p.r.text).join(" ").trim();
    const sents = parts.flatMap((p) => sentences(null, p.r.pieces.map((piece, i) => ({ piece, frame: p.r.frames[i] })),
      p.offset));
    last = { text, sentences: sents };
    render();
    $("copy").disabled = false;
    $("stats").textContent = `${fmtT(dur)} of audio in ${wall.toFixed(2)} s (${(dur / wall).toFixed(0)}× real time) · ` +
      `features ${feat.toFixed(0)} ms · encoder (WebGPU) ${enc.toFixed(0)} ms · decoder (WASM) ${dec.toFixed(0)} ms`;
    status("Ready.");
  } catch (e) {
    showErr(String(e.stack || e));
    status("Failed.");
  } finally {
    busy = false;
  }
}

$("load").onclick = () => {
  $("load").disabled = true;
  status("Starting…");
  worker.postMessage({ type: "load", url: new URL(MODEL_URL, location.href).href });
};
$("pick").onclick = () => $("file").click();
$("file").onchange = async () => {
  const f = $("file").files[0];
  if (f) handleAudio(await f.arrayBuffer(), f.name);
  $("file").value = "";
};
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = async (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const f = e.dataTransfer.files[0];
  if (f) handleAudio(await f.arrayBuffer(), f.name);
};
$("ts").onchange = render;
$("copy").onclick = () => navigator.clipboard.writeText(last ? last.text : "");

let live = null;
function showLive(committed, pending) {
  const out = $("out");
  out.textContent = "";
  const c = document.createElement("span");
  c.textContent = committed;
  const p = document.createElement("span");
  p.className = "pending";
  p.textContent = (committed && pending ? " " : "") + pending;
  out.append(c, p);
}
$("rec").onclick = async () => {
  if (live) {
    const l = live;
    live = null;
    $("rec").disabled = true;
    status("Finishing…");
    const text = await l.stop();
    clearInterval(l.meter);
    last = { text, sentences: [{ start: 0, text }] };
    showLive(text, "");
    $("copy").disabled = !text;
    $("rec").textContent = "● Record";
    $("rec").classList.remove("rec");
    $("rec").disabled = $("pick").disabled = false;
    $("ts").disabled = false;
    busy = false;
    status("Ready.");
    return;
  }
  if (!ready || busy) return;
  busy = true;
  $("err").textContent = "";
  $("stats").textContent = "";
  $("out").textContent = "";
  let passMs = 0, passSec = 0;
  const l = new Live(async (pcm) => {
    const r = await transcribePcm(pcm);
    passMs = r.timing.features + r.timing.encoder + r.timing.decoder;
    passSec = r.timing.audio;
    return r;
  }, showLive);
  try {
    await l.begin();
  } catch (e) {
    busy = false;
    showErr("Microphone unavailable: " + e.message);
    return;
  }
  live = l;
  $("ts").checked = false;
  $("ts").disabled = $("pick").disabled = true;
  $("rec").textContent = "■ Stop";
  $("rec").classList.add("rec");
  l.meter = setInterval(() => {
    const db = 20 * Math.log10(l.level() + 1e-6);
    status(`Listening… level ${db.toFixed(0)} dB` +
      (passMs ? ` · last pass ${passSec.toFixed(1)} s of audio in ${passMs.toFixed(0)} ms` : ""));
  }, 250);
};

if (!navigator.gpu) {
  showErr("This browser does not expose WebGPU. Try a recent Chrome, Edge or Safari.");
  $("load").disabled = true;
}
// start automatically when the model is already cached
caches?.open("phonon-model-v1").then((c) => c.match(new URL(MODEL_URL, location.href).href))
  .then((hit) => { if (hit && navigator.gpu) $("load").click(); }).catch(() => {});
