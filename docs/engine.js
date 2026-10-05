// Phonon-2 in the browser: WebGPU encoder + WASM (weight decompression, log-mel, TDT decoder).
import init, * as wasm from "./pkg/phonon_wasm.js";
import { matmulWGSL, layerNormWGSL, scoresWGSL, softmaxWGSL, pvWGSL, gluWGSL, dwconvWGSL, conv0WGSL, dw2dWGSL }
  from "./kernels.js";

const D = 1024, NL = 24, P = 640;
const U = GPUBufferUsage;

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export function parseContainer(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "PHON") throw new Error("not a .phon file");
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hlen)));
  const base = 12 + hlen + ((4 - ((12 + hlen) % 4)) % 4);
  const bytes = (off, len) => new Uint8Array(buf, base + off, len);
  return { header, bytes };
}

export class Phonon {
  static async create(modelBuf, { onProgress = () => {} } = {}) {
    const e = new Phonon();
    await init();
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    e.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(L.maxStorageBuffersPerShaderStage, 10),
      },
    });
    e.maxBind = L.maxStorageBufferBindingSize;
    const ai = adapter.info || {};
    e.adapterInfo = [ai.vendor, ai.architecture, ai.description].filter(Boolean).join(" ");
    e.device.lost.then((i) => console.error("WebGPU device lost:", i.message));
    e.pipes = {};
    const mk = (name, code) => {
      e.pipes[name] = e.device.createComputePipeline({
        layout: "auto",
        compute: { module: e.device.createShaderModule({ code }), entryPoint: "main" },
      });
    };
    mk("mm5", matmulWGSL(true)); mk("mmd", matmulWGSL(false)); mk("ln", layerNormWGSL); mk("scores", scoresWGSL);
    mk("softmax", softmaxWGSL); mk("pv", pvWGSL); mk("glu", gluWGSL); mk("dwconv", dwconvWGSL);
    mk("conv0", conv0WGSL); mk("dw2d", dw2dWGSL);
    await e.loadWeights(modelBuf, onProgress);
    return e;
  }

  buf(data, usage = U.STORAGE) {
    const b = this.device.createBuffer({ size: Math.max(16, (data.byteLength + 3) & ~3), usage: usage | U.COPY_DST,
      mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    b.unmap();
    return b;
  }

  empty(bytes, extra = 0) {
    return this.device.createBuffer({ size: Math.max(16, bytes), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | extra });
  }

  async loadWeights(modelBuf, onProgress) {
    const { header, bytes } = parseContainer(modelBuf);
    this.vocab = header.vocab;
    const T = header.tensors;
    const names = Object.keys(T);
    const f32 = {};
    const get = (n) => {
      if (f32[n]) return f32[n];
      const t = T[n];
      let v;
      if (t.kind === "f16") v = f16ToF32(new Uint16Array(bytes(t.off, t.len).slice().buffer));
      else if (t.kind === "dq") {
        const sc = f16ToF32(new Uint16Array(bytes(t.soff, t.slen).slice().buffer));
        v = wasm.decode_dense(bytes(t.off, t.len), t.shape.reduce((a, b) => a * b, 1), t.bits, sc, t.group);
      } else throw new Error("dense get on " + t.kind);
      f32[n] = v;
      return v;
    };
    this.W = {};
    let done = 0;
    const total = names.length;
    for (const n of names) {
      const t = T[n];
      if (t.kind === "fv") {
        const [O, I] = t.shape;
        const planes = wasm.decode_five(bytes(t.off, t.len), O, I);
        const lh = f16ToF32(new Uint16Array(bytes(t.loff, t.llen).slice().buffer));
        this.W[n] = { five: true, O, I, w: this.buf(planes), lh: this.buf(lh) };
      }
      done++;
      if (done % 8 === 0) { onProgress(done / total); await new Promise((r) => setTimeout(r, 0)); }
    }
    const dense = (n, wName, bName, O, I, permute) => {
      let w = get(wName);
      if (permute) w = permute(w);
      this.W[n] = { five: false, O, I, w: this.buf(w), lh: this.buf(new Float32Array(4)),
        b: bName ? this.buf(get(bName)) : null };
    };
    // subsampling
    const s = "encoder.subsampling.";
    // linear input is (c * 16 + f) in torch; our channels-last layout gives (f * 256 + c)
    dense("sub.linear", s + "linear.weight", s + "linear.bias", D, 4096, (w) => {
      const o = new Float32Array(w.length);
      for (let r = 0; r < D; r++) for (let c = 0; c < 256; c++) for (let f = 0; f < 16; f++)
        o[r * 4096 + f * 256 + c] = w[r * 4096 + c * 16 + f];
      return o;
    });
    dense("sub.pw3", s + "layers.3.weight", s + "layers.3.bias", 256, 256);
    dense("sub.pw6", s + "layers.6.weight", s + "layers.6.bias", 256, 256);
    this.sub = {
      w0: this.buf(get(s + "layers.0.weight")), b0: this.buf(get(s + "layers.0.bias")),
      w2: this.buf(get(s + "layers.2.weight")), b2: this.buf(get(s + "layers.2.bias")),
      w5: this.buf(get(s + "layers.5.weight")), b5: this.buf(get(s + "layers.5.bias")),
    };
    dense("proj", "encoder_projector.weight", "encoder_projector.bias", P, D);
    this.L = [];
    for (let i = 0; i < NL; i++) {
      const q = `encoder.layers.${i}.`;
      const cat = (a, b) => { const o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
      const g = (n) => this.buf(cat(get(q + n + ".weight"), get(q + n + ".bias")));
      // fold batch norm into the depthwise conv
      const dw = get(q + "conv.depthwise_conv.weight");
      const bn = (k) => get(q + "conv.norm." + k);
      const gam = bn("weight"), bet = bn("bias"), mu = bn("running_mean"), va = bn("running_var");
      const w2 = new Float32Array(D * 9), b2 = new Float32Array(D);
      for (let c = 0; c < D; c++) {
        const sc = gam[c] / Math.sqrt(va[c] + 1e-5);
        for (let k = 0; k < 9; k++) w2[c * 9 + k] = dw[c * 9 + k] * sc;
        b2[c] = bet[c] - mu[c] * sc;
      }
      this.L.push({
        nff1: g("norm_feed_forward1"), natt: g("norm_self_att"), nconv: g("norm_conv"),
        nff2: g("norm_feed_forward2"), nout: g("norm_out"),
        buv: this.buf(cat(get(q + "self_attn.bias_u"), get(q + "self_attn.bias_v"))),
        dww: this.buf(w2), dwb: this.buf(b2),
      });
    }
    // CPU decoder: int8 codes + per-row scales straight from the container
    const qm = (n) => {
      const t = T[n];
      if (t.kind !== "dq" || t.group !== t.shape[1]) throw new Error("decoder table must be per-row dq: " + n);
      return [wasm.decode_dense_q(bytes(t.off, t.len), t.shape[0] * t.shape[1], t.bits),
        f16ToF32(new Uint16Array(bytes(t.soff, t.slen).slice().buffer))];
    };
    const lsum = (l) => { const a = get(`decoder.lstm.bias_ih_l${l}`), b = get(`decoder.lstm.bias_hh_l${l}`);
      const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i]; return o; };
    this.dec = new wasm.Decoder(get("decoder.embedding.weight"),
      ...qm("decoder.lstm.weight_ih_l0"), ...qm("decoder.lstm.weight_hh_l0"), lsum(0),
      ...qm("decoder.lstm.weight_ih_l1"), ...qm("decoder.lstm.weight_hh_l1"), lsum(1),
      ...qm("decoder.decoder_projector.weight"), get("decoder.decoder_projector.bias"),
      ...qm("joint.head.weight"), get("joint.head.bias"));
    this.zeroBias = this.buf(new Float32Array(4096));
    onProgress(1);
  }

  // ---------------------------------------------------------------------------------------------- dispatch helpers
  uni(arr) {
    const b = this.device.createBuffer({ size: Math.max(16, arr.byteLength), usage: U.UNIFORM | U.COPY_DST,
      mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(arr.buffer));
    b.unmap();
    this.tmp.push(b);
    return b;
  }

  run(pass, name, uniform, bufs, x, y = 1, z = 1) {
    const pipe = this.pipes[name];
    const entries = [{ binding: 0, resource: { buffer: uniform } }];
    bufs.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(x, y, z);
  }

  grid1(n) { // flatten n workgroups of 256 threads into (x, y)
    const wg = Math.ceil(n / 256);
    return wg <= 65535 ? [wg, 1] : [65535, Math.ceil(wg / 65535)];
  }

  // Y[yOff + t] (= | silu | += alpha* | relu) X[xOff + t] @ W^T + b,  t < T
  mm(pass, W, X, Y, T, { mode = 0, alpha = 1, xOff = 0, yOff = 0 } = {}) {
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u[0] = T; u[1] = W.I; u[2] = W.O; u[3] = mode; f[4] = alpha; u[5] = W.b ? 1 : 0; u[6] = xOff; u[7] = yOff;
    this.run(pass, W.five ? "mm5" : "mmd", this.uni(u), [X, W.w, W.lh, W.b || this.zeroBias, Y],
      W.O / 64, Math.ceil(T / 64));
  }

  ln(pass, G, X, Y, T) {
    const n = T;
    this.run(pass, "ln", this.uni(new Uint32Array([T, D, 0, 0])), [X, G, Y], Math.min(n, 65535), Math.ceil(n / 65535));
  }

  // --------------------------------------------------------------------------------------------------- encoder
  async encode(feats, T) {
    const dev = this.device;
    this.tmp = [];
    const T1 = Math.floor((T - 1) / 2) + 1, T2 = Math.floor((T1 - 1) / 2) + 1, T3 = Math.floor((T2 - 1) / 2) + 1;
    const fb = this.buf(feats);
    const x = this.empty(T3 * D * 4);
    // subsampling, in chunks of output frames so the 256 x 64 conv0 map stays small
    const CH = 256;
    const maxN1 = 4 * CH + 4, maxN2 = 2 * CH + 2;
    const b1 = this.empty(maxN1 * 64 * 256 * 4), b2 = this.empty(maxN2 * 32 * 256 * 4);
    const b2p = this.empty(maxN2 * 32 * 256 * 4), b3 = this.empty(CH * 16 * 256 * 4), b3p = this.empty(CH * 16 * 256 * 4);
    let enc = dev.createCommandEncoder();
    for (let a = 0; a < T3; a += CH) {
      const b = Math.min(T3, a + CH);
      const r2s = Math.max(0, 2 * a - 1), r2e = Math.min(T2, 2 * b);
      const r1s = Math.max(0, 2 * r2s - 1), r1e = Math.min(T1, 2 * (r2e - 1) + 2);
      const pass = enc.beginComputePass();
      let g = this.grid1((r1e - r1s) * 64 * 256);
      this.run(pass, "conv0", this.uni(new Uint32Array([T, r1s, r1e - r1s, 0])), [fb, this.sub.w0, this.sub.b0, b1], g[0], g[1]);
      g = this.grid1((r2e - r2s) * 32 * 256);
      this.run(pass, "dw2d", this.uni(new Uint32Array([T1, r1s, r2s, r2e - r2s, 64, 0, 0, 0])),
        [b1, this.sub.w2, this.sub.b2, b2], g[0], g[1]);
      this.mm(pass, this.W["sub.pw3"], b2, b2p, (r2e - r2s) * 32, { mode: 3 });
      g = this.grid1((b - a) * 16 * 256);
      this.run(pass, "dw2d", this.uni(new Uint32Array([T2, r2s, a, b - a, 32, 0, 0, 0])),
        [b2p, this.sub.w5, this.sub.b5, b3], g[0], g[1]);
      this.mm(pass, this.W["sub.pw6"], b3, b3p, (b - a) * 16, { mode: 3 });
      this.mm(pass, this.W["sub.linear"], b3p, x, b - a, { yOff: a });
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    [b1, b2, b2p, b3, b3p, fb].forEach((b) => b.destroy());

    // relative positional embeddings (computed in f64 on the CPU, like torch's f32 sin/cos of f32 products)
    const NP = 2 * T3 - 1;
    const pe = new Float32Array(NP * D);
    const inv = new Float32Array(D / 2);
    for (let m = 0; m < D / 2; m++) inv[m] = 1 / Math.pow(10000, (2 * m) / D);
    for (let p = 0; p < NP; p++) {
      const pos = T3 - 1 - p;
      for (let m = 0; m < D / 2; m++) {
        const a = Math.fround(pos * inv[m]);
        pe[p * D + 2 * m] = Math.sin(a);
        pe[p * D + 2 * m + 1] = Math.cos(a);
      }
    }
    const peb = this.buf(pe);
    // work buffers
    const RC = Math.max(64, Math.min(T3, Math.floor(this.maxBind / (4096 * 4) / 64) * 64, 4096));
    const hid = this.empty(Math.min(T3, RC) * 4096 * 4);
    const h = this.empty(T3 * D * 4), q = this.empty(T3 * D * 4), k = this.empty(T3 * D * 4), v = this.empty(T3 * D * 4);
    const o = this.empty(T3 * D * 4), pk = this.empty(NP * D * 4);
    let Qc = Math.floor(Math.min(this.maxBind, 256 * 2 ** 20) / (8 * T3 * 4) / 32) * 32;
    Qc = Math.max(32, Math.min(Qc, Math.ceil(T3 / 32) * 32));
    const S = this.empty(8 * Qc * T3 * 4);
    const scale = 1 / Math.sqrt(128);
    for (let i = 0; i < NL; i++) {
      const p = `encoder.layers.${i}.`;
      const Lw = this.L[i];
      const W = (n) => this.W[p + n + ".weight"];
      enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      // FF1 (half-step residual)
      this.ln(pass, Lw.nff1, x, h, T3);
      for (let r = 0; r < T3; r += RC) {
        const n = Math.min(RC, T3 - r);
        this.mm(pass, W("feed_forward1.linear1"), h, hid, n, { mode: 1, xOff: r });
        this.mm(pass, W("feed_forward1.linear2"), hid, x, n, { mode: 2, alpha: 0.5, yOff: r });
      }
      // MHSA with relative positions
      this.ln(pass, Lw.natt, x, h, T3);
      this.mm(pass, W("self_attn.q_proj"), h, q, T3);
      this.mm(pass, W("self_attn.k_proj"), h, k, T3);
      this.mm(pass, W("self_attn.v_proj"), h, v, T3);
      this.mm(pass, W("self_attn.relative_k_proj"), peb, pk, NP);
      for (let q0 = 0; q0 < T3; q0 += Qc) {
        const n = Math.min(Qc, T3 - q0);
        const su = new Uint32Array(4); new Float32Array(su.buffer)[3] = scale; su[0] = T3; su[1] = q0; su[2] = n;
        this.run(pass, "scores", this.uni(su), [q, k, pk, Lw.buv, S], Math.ceil(T3 / 32), Math.ceil(n / 32), 8);
        const rows = 8 * n;
        this.run(pass, "softmax", this.uni(new Uint32Array([T3, rows, 0, 0])), [S], Math.min(rows, 65535),
          Math.ceil(rows / 65535));
        this.run(pass, "pv", this.uni(new Uint32Array([T3, q0, n, 0])), [S, v, o], 2, Math.ceil(n / 64), 8);
      }
      this.mm(pass, W("self_attn.o_proj"), o, x, T3, { mode: 2 });
      // convolution module
      this.ln(pass, Lw.nconv, x, h, T3);
      for (let r = 0; r < T3; r += RC) {
        const n = Math.min(RC, T3 - r);
        this.mm(pass, W("conv.pointwise_conv1"), h, hid, n, { xOff: r });
        const g = this.grid1(n * D);
        this.run(pass, "glu", this.uni(new Uint32Array([n * D, D, r, 0])), [hid, o], g[0], g[1]);
      }
      { const g = this.grid1(T3 * D);
        this.run(pass, "dwconv", this.uni(new Uint32Array([T3, D, 0, 0])), [o, Lw.dww, Lw.dwb, h], g[0], g[1]); }
      this.mm(pass, W("conv.pointwise_conv2"), h, x, T3, { mode: 2 });
      // FF2
      this.ln(pass, Lw.nff2, x, h, T3);
      for (let r = 0; r < T3; r += RC) {
        const n = Math.min(RC, T3 - r);
        this.mm(pass, W("feed_forward2.linear1"), h, hid, n, { mode: 1, xOff: r });
        this.mm(pass, W("feed_forward2.linear2"), hid, x, n, { mode: 2, alpha: 0.5, yOff: r });
      }
      this.ln(pass, Lw.nout, x, h, T3);
      pass.end();
      // swap: the block output is in h; copy back into x
      enc.copyBufferToBuffer(h, 0, x, 0, T3 * D * 4);
      dev.queue.submit([enc.finish()]);
      if (i % 4 === 3) await dev.queue.onSubmittedWorkDone();
      if (this.onLayer) this.onLayer(i + 1, NL);
    }
    // joint encoder projection
    const e = this.empty(T3 * P * 4);
    enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.mm(pass, this.W.proj, x, e, T3);
    pass.end();
    const rb = dev.createBuffer({ size: T3 * P * 4, usage: U.MAP_READ | U.COPY_DST });
    enc.copyBufferToBuffer(e, 0, rb, 0, T3 * P * 4);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    [rb, e, x, hid, h, q, k, v, o, pk, S, peb, ...this.tmp].forEach((b) => b.destroy());
    this.tmp = [];
    return { enc: out, T: T3 };
  }

  detok(ids) {
    let s = "";
    for (const i of ids) s += this.vocab[i] || "";
    return s.replace(/▁/g, " ").trim();
  }

  // pcm: Float32Array, 16 kHz mono
  async transcribe(pcm) {
    const t0 = performance.now();
    const feats = wasm.features(pcm);
    const T = Math.floor(pcm.length / 160);
    const t1 = performance.now();
    const { enc, T: T3 } = await this.encode(feats, T);
    const t2 = performance.now();
    const r = this.dec.greedy(enc, T3);
    const n = r[0];
    const toks = Array.from(r.subarray(1, 1 + n));
    const frames = Array.from(r.subarray(1 + n, 1 + 2 * n));
    const durs = Array.from(r.subarray(1 + 2 * n, 1 + 3 * n));
    const t3 = performance.now();
    return {
      text: this.detok(toks), tokens: toks, pieces: toks.map((i) => this.vocab[i] || ""), frames, durs,
      timing: { features: t1 - t0, encoder: t2 - t1, decoder: t3 - t2, audio: pcm.length / 16000 },
    };
  }
}
