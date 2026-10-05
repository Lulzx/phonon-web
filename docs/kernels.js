// WGSL kernels for the Phonon-2 (FastConformer) encoder.  All activations f32, row-major [rows][cols].

// Y[t, o] = epi( sum_i X[t, i] * W[o, i] + bias[o] )
// W is either five-value packed (plane A: 2-bit codes 0/+/-, 16 per u32, [O][I/16]; plane B: hi bit, 32 per u32,
// [O][I/32]; per-row lo/hi magnitudes) or dense f32 [O][I].
// mode 0: Y = v   1: Y = silu(v)   2: Y += alpha * v   3: Y = relu(v)
export function matmulWGSL(five) {
  return /* wgsl */ `
struct P { T: u32, I: u32, O: u32, mode: u32, alpha: f32, hasBias: u32, xOff: u32, yOff: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
${five ? `@group(0) @binding(2) var<storage, read> WP: array<u32>;
@group(0) @binding(3) var<storage, read> LH: array<f32>;   // [lo(O) | hi(O)]`
         : `@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> LH: array<f32>;   // unused`}
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

const BM = 64u; const BN = 64u; const BK = 32u;
var<workgroup> Xs: array<f32, 2048>;  // [BK][BM]
var<workgroup> Ws: array<f32, 2048>;  // [BK][BN]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t0 = wg.y * BM;
  let o0 = wg.x * BN;
  let tx = lid % 16u;
  let ty = lid / 16u;
  var acc: array<array<f32, 4>, 4>;
  for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] = 0.0; } }
  let I = p.I;
  _ = LH[0];
  for (var k0 = 0u; k0 < I; k0 += BK) {
    for (var r = 0u; r < 8u; r++) {
      let e = lid + 256u * r;
      let m = e / BK;
      let k = e % BK;
      let t = t0 + m;
      var v = 0.0;
      if (t < p.T) { v = X[(p.xOff + t) * I + k0 + k]; }
      Xs[k * BM + m] = v;
    }
${five ? `    {
      let n = lid / 4u;
      let q = lid % 4u;
      let o = o0 + n;
      let a = WP[o * (I / 16u) + k0 / 16u + q / 2u];
      let bw = WP[p.O * (I / 16u) + o * (I / 32u) + k0 / 32u];
      let lo = LH[o];
      let hi = LH[p.O + o];
      for (var j = 0u; j < 8u; j++) {
        let kk = q * 8u + j;
        let code = (a >> (2u * (kk % 16u))) & 3u;
        let hb = (bw >> kk) & 1u;
        var mag = lo;
        if (hb == 1u) { mag = hi; }
        var v = 0.0;
        if (code == 1u) { v = mag; } else if (code == 2u) { v = -mag; }
        Ws[kk * BN + n] = v;
      }
    }`
       : `    for (var r = 0u; r < 8u; r++) {
      let e = lid + 256u * r;
      let n = e / BK;
      let k = e % BK;
      Ws[k * BN + n] = W[(o0 + n) * I + k0 + k];
    }`}
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = Xs[k * BM + ty + 16u * i]; }
      for (var j = 0u; j < 4u; j++) { b[j] = Ws[k * BN + tx + 16u * j]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] = fma(a[i], b[j], acc[i][j]); } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let t = t0 + ty + 16u * i;
    if (t >= p.T) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let o = o0 + tx + 16u * j;
      var v = acc[i][j];
      if (p.hasBias == 1u) { v += B[o]; }
      let idx = (p.yOff + t) * p.O + o;
      if (p.mode == 0u) { Y[idx] = v; }
      else if (p.mode == 1u) { Y[idx] = v / (1.0 + exp(-v)); }
      else if (p.mode == 2u) { Y[idx] = Y[idx] + p.alpha * v; }
      else { Y[idx] = max(v, 0.0); }
    }
  }
}`;
}

// LayerNorm over 1024 channels, one workgroup per row.
export const layerNormWGSL = /* wgsl */ `
struct P { T: u32, C: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;  // [w(C) | b(C)]
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x + wg.y * 65535u;
  if (t >= p.T) { return; }
  let base = t * p.C;
  var s = 0.0;
  for (var c = lid; c < p.C; c += 256u) { s += X[base + c]; }
  red[lid] = s;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid < w) { red[lid] += red[lid + w]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.C);
  workgroupBarrier();
  var v = 0.0;
  for (var c = lid; c < p.C; c += 256u) { let d = X[base + c] - mean; v += d * d; }
  red[lid] = v;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid < w) { red[lid] += red[lid + w]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.C) + 1e-5);
  for (var c = lid; c < p.C; c += 256u) { Y[base + c] = (X[base + c] - mean) * inv * G[c] + G[p.C + c]; }
}`;

// Relative-position attention scores for queries [q0, q0+Qc):
// S[h][i-q0][j] = ((q_i + bu_h) . k_j + (q_i + bv_h) . pk_{T-1-i+j}) / sqrt(128)
export const scoresWGSL = /* wgsl */ `
struct P { T: u32, q0: u32, Qc: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> PK: array<f32>;
@group(0) @binding(4) var<storage, read> BUV: array<f32>;  // [bu(1024) | bv(1024)]
@group(0) @binding(5) var<storage, read_write> S: array<f32>;
var<workgroup> qu: array<f32, 512>;  // [16][32]
var<workgroup> qv: array<f32, 512>;
var<workgroup> ks: array<f32, 512>;
var<workgroup> pks: array<f32, 1024>; // [16][64]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let h = wg.z;
  let i0 = p.q0 + wg.y * 32u;
  let j0 = wg.x * 32u;
  let tx = lid % 16u;
  let ty = lid / 16u;
  var acc: array<array<f32, 2>, 2>;
  acc[0][0] = 0.0; acc[0][1] = 0.0; acc[1][0] = 0.0; acc[1][1] = 0.0;
  // pk rows needed: T-1-i+j for i in [i0,i0+32), j in [j0,j0+32) -> base + (jj - ii + 31), base = T-1-i0+j0-31
  let pbase = i32(p.T) - 1 - i32(i0) + i32(j0) - 31;
  let NP = 2u * p.T - 1u;
  for (var d0 = 0u; d0 < 128u; d0 += 16u) {
    for (var r = 0u; r < 2u; r++) {
      let e = lid + 256u * r;     // 0..511
      let m = e / 16u;            // 0..31
      let d = e % 16u;
      let col = h * 128u + d0 + d;
      let i = i0 + m;
      var qq = 0.0;
      if (i < p.T) { qq = Q[i * 1024u + col]; }
      qu[d * 32u + m] = qq + BUV[col];
      qv[d * 32u + m] = qq + BUV[1024u + col];
      let j = j0 + m;
      var kv = 0.0;
      if (j < p.T) { kv = K[j * 1024u + col]; }
      ks[d * 32u + m] = kv;
    }
    for (var r = 0u; r < 4u; r++) {
      let e = lid + 256u * r;     // 0..1023
      let m = e / 16u;            // 0..63
      let d = e % 16u;
      let row = pbase + i32(m);
      var v = 0.0;
      if (m < 63u && row >= 0 && u32(row) < NP) { v = PK[u32(row) * 1024u + h * 128u + d0 + d]; }
      pks[d * 64u + m] = v;
    }
    workgroupBarrier();
    for (var d = 0u; d < 16u; d++) {
      for (var a = 0u; a < 2u; a++) {
        let ii = ty + 16u * a;
        let u = qu[d * 32u + ii];
        let w = qv[d * 32u + ii];
        for (var b = 0u; b < 2u; b++) {
          let jj = tx + 16u * b;
          acc[a][b] += u * ks[d * 32u + jj] + w * pks[d * 64u + jj + 31u - ii];
        }
      }
    }
    workgroupBarrier();
  }
  for (var a = 0u; a < 2u; a++) {
    let i = i0 + ty + 16u * a;
    if (i >= p.T || i >= p.q0 + p.Qc) { continue; }
    for (var b = 0u; b < 2u; b++) {
      let j = j0 + tx + 16u * b;
      if (j >= p.T) { continue; }
      S[(h * p.Qc + (i - p.q0)) * p.T + j] = acc[a][b] * p.scale;
    }
  }
}`;

// Row softmax in place over S[row][0..T), rows = H*Qc
export const softmaxWGSL = /* wgsl */ `
struct P { T: u32, rows: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.rows) { return; }
  let base = row * p.T;
  var m = -3.0e38;
  for (var j = lid; j < p.T; j += 256u) { m = max(m, S[base + j]); }
  red[lid] = m;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid < w) { red[lid] = max(red[lid], red[lid + w]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var j = lid; j < p.T; j += 256u) { let e = exp(S[base + j] - mx); S[base + j] = e; s += e; }
  red[lid] = s;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid < w) { red[lid] += red[lid + w]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var j = lid; j < p.T; j += 256u) { S[base + j] = S[base + j] * inv; }
}`;

// O[q0+i][h*128+d] = sum_j P[h][i][j] * V[j][h*128+d]
export const pvWGSL = /* wgsl */ `
struct P { T: u32, q0: u32, Qc: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
var<workgroup> Ps: array<f32, 2048>;  // [32 j][64 i]
var<workgroup> Vs: array<f32, 2048>;  // [32 j][64 d]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let h = wg.z;
  let i0 = wg.y * 64u;      // local query index
  let d0 = wg.x * 64u;      // 0 or 64
  let tx = lid % 16u;
  let ty = lid / 16u;
  var acc: array<array<f32, 4>, 4>;
  for (var a = 0u; a < 4u; a++) { for (var b = 0u; b < 4u; b++) { acc[a][b] = 0.0; } }
  for (var j0 = 0u; j0 < p.T; j0 += 32u) {
    for (var r = 0u; r < 8u; r++) {
      let e = lid + 256u * r;
      let m = e / 32u;   // i
      let k = e % 32u;   // j
      let i = i0 + m;
      let j = j0 + k;
      var v = 0.0;
      if (i < p.Qc && j < p.T) { v = S[(h * p.Qc + i) * p.T + j]; }
      Ps[k * 64u + m] = v;
      let m2 = e % 64u;  // d
      let k2 = e / 64u;  // j
      let j2 = j0 + k2;
      var w = 0.0;
      if (j2 < p.T) { w = V[j2 * 1024u + h * 128u + d0 + m2]; }
      Vs[k2 * 64u + m2] = w;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var x = 0u; x < 4u; x++) { a[x] = Ps[k * 64u + ty + 16u * x]; }
      for (var y = 0u; y < 4u; y++) { b[y] = Vs[k * 64u + tx + 16u * y]; }
      for (var x = 0u; x < 4u; x++) { for (var y = 0u; y < 4u; y++) { acc[x][y] = fma(a[x], b[y], acc[x][y]); } }
    }
    workgroupBarrier();
  }
  for (var x = 0u; x < 4u; x++) {
    let i = i0 + ty + 16u * x;
    if (i >= p.Qc) { continue; }
    for (var y = 0u; y < 4u; y++) {
      O[(p.q0 + i) * 1024u + h * 128u + d0 + tx + 16u * y] = acc[x][y];
    }
  }
}`;

// GLU over [T][2C] -> Y rows [yOff, yOff+T) of [*][C]
export const gluWGSL = /* wgsl */ `
struct P { n: u32, C: u32, yOff: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x + g.y * 65535u * 256u;
  if (e >= p.n) { return; }
  let t = e / p.C;
  let c = e % p.C;
  let a = X[t * 2u * p.C + c];
  let b = X[t * 2u * p.C + p.C + c];
  Y[p.yOff * p.C + e] = a / (1.0 + exp(-b));
}`;

// depthwise conv1d (k=9, pad 4) with batch-norm folded into W/B, then SiLU.  X,Y [T][C]; W [C][9]; B [C]
export const dwconvWGSL = /* wgsl */ `
struct P { T: u32, C: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x + g.y * 65535u * 256u;
  if (e >= p.T * p.C) { return; }
  let t = i32(e / p.C);
  let c = e % p.C;
  var s = B[c];
  for (var k = 0; k < 9; k++) {
    let tt = t + k - 4;
    if (tt >= 0 && tt < i32(p.T)) { s += W[c * 9u + u32(k)] * X[u32(tt) * p.C + c]; }
  }
  Y[e] = s / (1.0 + exp(-s));
}`;

// Subsampling conv0: feats [T][128] -> Y [(t - oOff) * 64 + f][256] (channels last), 3x3 stride 2 pad 1, ReLU.
// Output rows cover global t1 in [oOff, oOff + n1).
export const conv0WGSL = /* wgsl */ `
struct P { T: u32, oOff: u32, n1: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;  // [256][3][3]
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x + g.y * 65535u * 256u;
  if (e >= p.n1 * 64u * 256u) { return; }
  let c = e % 256u;
  let f = (e / 256u) % 64u;
  let tl = e / (256u * 64u);
  let t = i32(p.oOff + tl);
  var s = B[c];
  for (var dt = 0; dt < 3; dt++) {
    let ti = 2 * t + dt - 1;
    if (ti < 0 || ti >= i32(p.T)) { continue; }
    for (var df = 0; df < 3; df++) {
      let fi = 2 * i32(f) + df - 1;
      if (fi < 0 || fi >= 128) { continue; }
      s += W[c * 9u + u32(dt * 3 + df)] * X[u32(ti) * 128u + u32(fi)];
    }
  }
  Y[e] = max(s, 0.0);
}`;

// Depthwise 3x3 stride-2 conv, channels last: X rows global t in [iOff, iOff+nIn) of Fi freq bins,
// Y rows global t in [oOff, oOff+nOut) of Fi/2 bins. Global length of the input sequence is Tin.
export const dw2dWGSL = /* wgsl */ `
struct P { Tin: u32, iOff: u32, oOff: u32, nOut: u32, Fi: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;  // [256][3][3]
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let Fo = p.Fi / 2u;
  let e = g.x + g.y * 65535u * 256u;
  if (e >= p.nOut * Fo * 256u) { return; }
  let c = e % 256u;
  let f = (e / 256u) % Fo;
  let tl = e / (256u * Fo);
  let t = i32(p.oOff + tl);
  var s = B[c];
  for (var dt = 0; dt < 3; dt++) {
    let ti = 2 * t + dt - 1;
    if (ti < 0 || ti >= i32(p.Tin)) { continue; }
    let tr = u32(ti) - p.iOff;
    for (var df = 0; df < 3; df++) {
      let fi = 2 * i32(f) + df - 1;
      if (fi < 0 || fi >= i32(p.Fi)) { continue; }
      s += W[c * 9u + u32(dt * 3 + df)] * X[(tr * p.Fi + u32(fi)) * 256u + c];
    }
  }
  Y[e] = s;
}`;
