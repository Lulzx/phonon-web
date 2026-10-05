//! TDT prediction network (2-layer LSTM, 640) + joint (relu(enc + pred) -> 8192 tokens + blank + 5 durations) and
//! NeMo-style greedy TDT search.  Encoder frames arrive already projected to 640 by the GPU.
//!
//! Matrices are int8 with one f32 scale per row (the container's tables are <= 8-bit symmetric per row, so this is
//! exact).  Matrix-vector products quantise the input vector to int16 with one scale and use 16-bit dot products.

pub const P: usize = 640;
pub const G: usize = 4 * P;
pub const VOCAB: usize = 8193; // 8192 tokens + blank
pub const BLANK: usize = 8192;
pub const NOUT: usize = VOCAB + 5;
pub const DURS: [usize; 5] = [0, 1, 2, 3, 4];

pub struct QMat {
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

impl QMat {
    /// out[r] = (sum_j q[r,j] * xq[j]) * s[r] * xs + b[r]
    pub fn matvec(&self, xq: &[i16], xs: f32, b: Option<&[f32]>, out: &mut [f32]) {
        let k = self.cols;
        for r in 0..self.rows {
            let d = dot_i8_i16(&self.q[r * k..(r + 1) * k], xq) as f32;
            out[r] = d * self.s[r] * xs + b.map_or(0.0, |b| b[r]);
        }
    }
}

/// quantise x to int16 with one scale; returns scale
pub fn quant_vec(x: &[f32], out: &mut [i16]) -> f32 {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    if amax == 0.0 {
        out.iter_mut().for_each(|v| *v = 0);
        return 0.0;
    }
    let s = amax / 32767.0;
    let inv = 1.0 / s;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v * inv).round() as i16;
    }
    s
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn dot_i8_i16(w: &[i8], x: &[i16]) -> i32 {
    use core::arch::wasm32::*;
    let n = w.len();
    let mut acc0 = i32x4_splat(0);
    let mut acc1 = i32x4_splat(0);
    let mut i = 0;
    unsafe {
        while i + 16 <= n {
            let wv = v128_load(w.as_ptr().add(i) as *const v128);
            let lo = i16x8_extend_low_i8x16(wv);
            let hi = i16x8_extend_high_i8x16(wv);
            let x0 = v128_load(x.as_ptr().add(i) as *const v128);
            let x1 = v128_load(x.as_ptr().add(i + 8) as *const v128);
            acc0 = i32x4_add(acc0, i32x4_dot_i16x8(lo, x0));
            acc1 = i32x4_add(acc1, i32x4_dot_i16x8(hi, x1));
            i += 16;
        }
    }
    let a = i32x4_add(acc0, acc1);
    let mut s = i32x4_extract_lane::<0>(a) + i32x4_extract_lane::<1>(a) + i32x4_extract_lane::<2>(a)
        + i32x4_extract_lane::<3>(a);
    while i < n {
        s += w[i] as i32 * x[i] as i32;
        i += 1;
    }
    s
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn dot_i8_i16(w: &[i8], x: &[i16]) -> i32 {
    let mut s = 0i32;
    for i in 0..w.len() {
        s += w[i] as i32 * x[i] as i32;
    }
    s
}

pub struct Weights {
    pub emb: Vec<f32>, // [VOCAB][P]
    pub wih: [QMat; 2],
    pub whh: [QMat; 2],
    pub b: [Vec<f32>; 2], // [G] (ih + hh)
    pub pw: QMat,
    pub pb: Vec<f32>,
    pub jw: QMat,
    pub jb: Vec<f32>,
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone)]
pub struct State {
    h: [Vec<f32>; 2],
    c: [Vec<f32>; 2],
}

impl State {
    fn zero() -> State {
        State { h: [vec![0.0; P], vec![0.0; P]], c: [vec![0.0; P], vec![0.0; P]] }
    }
}

struct Scratch {
    xq: Vec<i16>,
    g1: Vec<f32>,
    g2: Vec<f32>,
}

impl Weights {
    /// one prediction-network step: returns projected output g [P] and the new state
    fn step(&self, tok: usize, st: &State, sc: &mut Scratch) -> (Vec<f32>, State) {
        let mut x: Vec<f32> = self.emb[tok * P..(tok + 1) * P].to_vec();
        let mut ns = State::zero();
        for l in 0..2 {
            let xs = quant_vec(&x, &mut sc.xq);
            self.wih[l].matvec(&sc.xq, xs, Some(&self.b[l]), &mut sc.g1);
            let hs = quant_vec(&st.h[l], &mut sc.xq);
            self.whh[l].matvec(&sc.xq, hs, None, &mut sc.g2);
            let (g1, g2) = (&sc.g1, &sc.g2);
            for j in 0..P {
                let i = sigmoid(g1[j] + g2[j]);
                let f = sigmoid(g1[P + j] + g2[P + j]);
                let g = (g1[2 * P + j] + g2[2 * P + j]).tanh();
                let o = sigmoid(g1[3 * P + j] + g2[3 * P + j]);
                let c = f * st.c[l][j] + i * g;
                ns.c[l][j] = c;
                ns.h[l][j] = o * c.tanh();
            }
            x = ns.h[l].clone();
        }
        let mut g = vec![0f32; P];
        let xs = quant_vec(&x, &mut sc.xq);
        self.pw.matvec(&sc.xq, xs, Some(&self.pb), &mut g);
        (g, ns)
    }

    /// enc: [T][P] projected encoder frames.  Returns (token ids, frame index of each token, duration of each token).
    pub fn greedy(&self, enc: &[f32], t_len: usize, max_symbols: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut sc = Scratch { xq: vec![0; P], g1: vec![0.0; G], g2: vec![0.0; G] };
        let st0 = State::zero();
        let (mut g, mut st1) = self.step(BLANK, &st0, &mut sc);
        let mut toks = Vec::new();
        let mut frames = Vec::new();
        let mut durs = Vec::new();
        let mut hid = vec![0f32; P];
        let mut hq = vec![0i16; P];
        let mut logits = vec![0f32; NOUT];
        let mut t = 0usize;
        let mut sym = 0usize;
        while t < t_len {
            let e = &enc[t * P..(t + 1) * P];
            for j in 0..P {
                let v = e[j] + g[j];
                hid[j] = if v > 0.0 { v } else { 0.0 };
            }
            let hs = quant_vec(&hid, &mut hq);
            self.jw.matvec(&hq, hs, Some(&self.jb), &mut logits);
            let mut k = 0;
            for i in 1..VOCAB {
                if logits[i] > logits[k] {
                    k = i;
                }
            }
            let mut di = 0;
            for i in 1..5 {
                if logits[VOCAB + i] > logits[VOCAB + di] {
                    di = i;
                }
            }
            let mut d = DURS[di];
            if k != BLANK {
                toks.push(k as u32);
                frames.push(t as u32);
                durs.push(d as u32);
                let (g2, s2) = self.step(k, &st1, &mut sc);
                g = g2;
                st1 = s2;
                sym += 1;
            }
            if k == BLANK && d == 0 {
                d = 1;
            }
            if d == 0 && sym >= max_symbols {
                d = 1;
            }
            if d > 0 {
                sym = 0;
            }
            t += d;
        }
        (toks, frames, durs)
    }
}
