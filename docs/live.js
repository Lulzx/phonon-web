// Live microphone transcription.  The mic is captured at the device rate with an AudioWorklet, resampled to 16 kHz,
// and the not-yet-committed tail of the recording is re-transcribed whenever the engine is free.  Once the tail is
// longer than COMMIT_SEC, text up to a pause (or word boundary) a few seconds back is committed and the audio
// window moves past it, so each pass stays short no matter how long you talk.

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) { this.port.postMessage(this.buf); this.buf = new Float32Array(2048); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor("tap", Tap);`;

const FRAME = 0.08;        // encoder frame, seconds
const COMMIT_SEC = 10;     // commit once the live window is longer than this
const FORCE_SEC = 20;      // ...but only cut mid-sentence (at the widest word gap) beyond this
const STALL_SEC = 3;       // speech this long after the last emitted word means the decoder stalled
const KEEP_SEC = 4;        // ...keeping at least this much audio live
const MIN_SEC = 0.5;       // don't run on less audio than this
const STEP_SEC = 0.4;      // run again after at least this much new audio

// windowed-sinc resampler, streaming over a growing source buffer
class Resampler {
  constructor(srcRate) {
    this.r = srcRate / 16000;
    this.src = new Float32Array(srcRate * 60);
    this.n = 0;
    this.out = new Float32Array(16000 * 60);
    this.m = 0;
    const fc = Math.min(1, 1 / this.r) * 0.92;  // cutoff relative to the source Nyquist
    this.fc = fc;
    this.half = Math.ceil(8 * this.r);
  }
  push(x) {
    if (this.n + x.length > this.src.length) {
      const b = new Float32Array(this.src.length * 2); b.set(this.src.subarray(0, this.n)); this.src = b;
    }
    this.src.set(x, this.n);
    this.n += x.length;
    const { r, fc, half } = this;
    while (true) {
      const p = this.m * r;
      const c = Math.floor(p);
      if (c + half >= this.n) break;
      let s = 0, ws = 0;
      for (let k = c - half + 1; k <= c + half; k++) {
        if (k < 0) continue;
        const d = k - p;
        const x = Math.PI * fc * d;
        const sinc = d === 0 ? 1 : Math.sin(x) / x;
        const w = 0.5 + 0.5 * Math.cos(Math.PI * d / (half + 1));
        s += this.src[k] * sinc * w;
        ws += sinc * w;
      }
      if (this.m >= this.out.length) {
        const b = new Float32Array(this.out.length * 2); b.set(this.out); this.out = b;
      }
      this.out[this.m++] = s / ws;
    }
  }
}

export class Live {
  /** transcribe(pcm Float32Array 16 kHz) -> Promise<{tokens, pieces, frames}>;  onText(committed, pending) */
  constructor(transcribe, onText) {
    this.transcribe = transcribe;
    this.onText = onText;
    this.committed = "";
    this.pending = "";
    this.start = 0;          // first 16 kHz sample of the live window
    this.lastRun = 0;        // sample count at the last pass
    this.running = false;
    this.stopped = false;
  }

  async begin() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true,
      noiseSuppression: false, autoGainControl: true } });
    this.ctx = new AudioContext();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await this.ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    this.rs = new Resampler(this.ctx.sampleRate);
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "tap");
    this.node.port.onmessage = (e) => { this.rs.push(e.data); this.kick(); };
    src.connect(this.node);
    // keep the graph pulled without making sound
    const g = this.ctx.createGain(); g.gain.value = 0;
    this.node.connect(g).connect(this.ctx.destination);
  }

  level() { // recent RMS for a meter
    const n = this.rs ? this.rs.m : 0, a = Math.max(0, n - 1600);
    let s = 0; for (let i = a; i < n; i++) s += this.rs.out[i] ** 2;
    return n > a ? Math.sqrt(s / (n - a)) : 0;
  }

  kick() {
    if (this.running || this.stopped) return;
    const n = this.rs.m;
    if (n - this.start < MIN_SEC * 16000 || n - this.lastRun < STEP_SEC * 16000) return;
    this.pass(false);
  }

  async pass(final) {
    this.running = true;
    try {
      const end = this.rs.m;
      this.lastRun = end;
      const pcm = this.rs.out.slice(this.start, end);
      if (pcm.length < 0.3 * 16000) { if (final) this.commitAll(""); return; }
      const dur = pcm.length / 16000;
      const env = rmsEnvelope(pcm);   // computed before pcm is transferred to the worker
      const r = await this.transcribe(pcm);
      const words = toWords(r);
      if (final) { this.commitAll(words.map((w) => w.text).join("")); return; }
      const cut = this.chooseCut(words, dur, env);
      if (cut) {
        this.committed += words.slice(0, cut.i).map((w) => w.text).join("");
        this.start += Math.round(Math.max(0, cut.t) * 16000);
        this.pending = words.slice(cut.i).map((w) => w.text).join("");
      } else {
        this.pending = words.map((w) => w.text).join("");
      }
      this.emit();
    } finally {
      this.running = false;
      if (!final) this.kick();
    }
  }

  // where to commit: returns {i: words before the cut, t: cut time in the window} or null
  chooseCut(words, dur, env) {
    const at = (i) => (i < words.length ? words[i].start : dur);
    // a word's tokens are emitted near its onset, so cut just before the next word's first token
    const between = (i) => Math.max(words[i - 1].last + FRAME, at(i) - 1.5 * FRAME);
    // stall: speech continues well past the last emitted word -> restart the window right after it
    if (words.length) {
      const lastEnd = words[words.length - 1].last + FRAME;
      if (dur - lastEnd > STALL_SEC && speechIn(env, lastEnd + 0.2, dur)) {
        return { i: words.length, t: lastEnd + 0.4 };
      }
    }
    if (dur <= COMMIT_SEC || words.length < 2) return null;
    // prefer the last sentence end that leaves KEEP_SEC live; else the widest word gap
    let best = null;
    for (let i = 1; i < words.length; i++) {
      if (at(i) < 2 || at(i) > dur - KEEP_SEC) continue;
      if (/[.?!]$/.test(words[i - 1].text)) best = { i, t: between(i), sent: true };
    }
    if (best) return best;
    if (dur <= FORCE_SEC) return null;  // no sentence end yet: wait, mid-sentence cuts cost accuracy
    let gap = -1;
    for (let i = 1; i < words.length; i++) {
      if (at(i) < 2 || at(i) > dur - KEEP_SEC) continue;
      const g = at(i) - words[i - 1].last;
      if (g > gap) { gap = g; best = { i, t: between(i) }; }
    }
    return best;
  }

  commitAll(text) {
    this.committed += text;
    this.pending = "";
    this.emit();
  }

  emit() { this.onText(tidy(this.committed), tidy(this.pending), !!this.committed); }

  async stop() {
    this.stopped = true;
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close();
    while (this.running) await new Promise((r) => setTimeout(r, 30));
    await this.pass(true);
    return tidy(this.committed);
  }
}

// tokens -> words (a piece starting with U+2581 opens a new word); each word keeps its first and last frame time
function toWords(r) {
  const words = [];
  r.pieces.forEach((p, i) => {
    const t = r.frames[i] * FRAME;
    if (!words.length || p.startsWith("▁")) words.push({ text: "", start: t, last: t });
    const w = words[words.length - 1];
    w.text += p.replace(/▁/g, " ");
    w.last = t;
  });
  if (words.length && !words[0].text.startsWith(" ")) words[0].text = " " + words[0].text;
  return words;
}

// RMS per 100 ms
function rmsEnvelope(pcm) {
  const n = Math.floor(pcm.length / 1600), e = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = k * 1600; i < (k + 1) * 1600; i++) s += pcm[i] * pcm[i];
    e[k] = Math.sqrt(s / 1600);
  }
  return e;
}

// is there sustained speech in [a, b) seconds?  (energy well above the window's quiet floor for >= 1 s)
function speechIn(env, a, b) {
  if (!env.length) return false;
  const sorted = Array.from(env).sort((x, y) => x - y);
  const floor = sorted[Math.floor(sorted.length * 0.1)], loud = sorted[Math.floor(sorted.length * 0.9)];
  const thr = Math.max(floor * 4, loud * 0.25, 3e-3);
  let n = 0;
  for (let k = Math.floor(a * 10); k < Math.min(env.length, Math.floor(b * 10)); k++) if (env[k] > thr) n++;
  return n >= 10;
}

const tidy = (s) => s.replace(/\s+/g, " ").trim();
