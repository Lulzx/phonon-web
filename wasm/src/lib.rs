//! Phonon-2 browser engine, CPU side: weight decompression, log-mel features and the TDT decoder.
//! The encoder runs in WebGPU (see web/engine.js).

pub mod decoder;
pub mod mel;
pub mod rans;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn features(pcm: &[f32]) -> Vec<f32> {
    mel::features(pcm).0
}

/// Holds the prediction / joint weights (int8 codes + per-row scales).
#[wasm_bindgen]
pub struct Decoder {
    w: decoder::Weights,
}

fn qm(q: Vec<i8>, s: Vec<f32>) -> decoder::QMat {
    let rows = s.len();
    let cols = q.len() / rows;
    decoder::QMat { rows, cols, q, s }
}

#[wasm_bindgen]
impl Decoder {
    /// emb f32 [8193*640]; matrices as (int8 codes, per-row f32 scales); b0/b1 = bias_ih + bias_hh
    #[wasm_bindgen(constructor)]
    pub fn new(emb: Vec<f32>, wih0: Vec<i8>, wih0s: Vec<f32>, whh0: Vec<i8>, whh0s: Vec<f32>, b0: Vec<f32>,
               wih1: Vec<i8>, wih1s: Vec<f32>, whh1: Vec<i8>, whh1s: Vec<f32>, b1: Vec<f32>,
               pw: Vec<i8>, pws: Vec<f32>, pb: Vec<f32>, jw: Vec<i8>, jws: Vec<f32>, jb: Vec<f32>) -> Decoder {
        Decoder { w: decoder::Weights { emb, wih: [qm(wih0, wih0s), qm(wih1, wih1s)], whh: [qm(whh0, whh0s), qm(whh1, whh1s)],
            b: [b0, b1], pw: qm(pw, pws), pb, jw: qm(jw, jws), jb } }
    }

    /// enc: projected encoder output [T*640].  Returns [n, tok_0..tok_n-1, frame_0.., dur_0..].
    pub fn greedy(&self, enc: &[f32], t_len: usize) -> Vec<u32> {
        let (t, f, d) = self.w.greedy(enc, t_len, 10);
        let mut out = Vec::with_capacity(1 + 3 * t.len());
        out.push(t.len() as u32);
        out.extend_from_slice(&t);
        out.extend_from_slice(&f);
        out.extend_from_slice(&d);
        out
    }
}

/// Decode one entropy-coded five-value matrix into GPU layout.  Returns the u32 words: plane A ([O][I/16], 2-bit
/// codes 0=zero 1=+ 2=-) followed by plane B ([O][I/32], 1 = |w| is hi).
#[wasm_bindgen]
pub fn decode_five(blob: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    rans::decode_five(blob, rows, cols)
}

/// Decode an entropy-coded integer table (symbols 0..2^bits, offset -2^(bits-1)) and dequantize with per-row (or
/// per-group) f32 scales -> f32 values.
#[wasm_bindgen]
pub fn decode_dense(blob: &[u8], n: usize, bits: u32, scales: &[f32], group: usize) -> Vec<f32> {
    rans::decode_dense(blob, n, bits, scales, group)
}

/// Decode an entropy-coded integer table to its signed integer codes (|q| <= 127).
#[wasm_bindgen]
pub fn decode_dense_q(blob: &[u8], n: usize, bits: u32) -> Vec<i8> {
    rans::decode_dense_q(blob, n, bits)
}
