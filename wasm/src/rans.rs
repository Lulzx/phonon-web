//! Static interleaved rANS (32-bit state, byte renormalisation, 4 interleaved states) over small alphabets.
//!
//! Stream: [u8 K classes] [K * A u16 freqs, each table sums to 2^PB] [class id per row, 4 bits packed, if K>1]
//!         [4 x u32 final states] [bytes...]
//! Symbols are coded row-major; the class of a symbol is the class of its row.
//! Five-value matrices are coded as triplets of 5-ary symbols (alphabet 125, iid product of the class's 5-ary
//! distribution), rows padded to a multiple of 3 with the zero symbol (2).

pub const PB: u32 = 15;
const M: u32 = 1 << PB;
const L: u32 = 1 << 23;

pub struct Table {
    pub freq: Vec<u32>,
    pub cum: Vec<u32>,
    pub lut: Vec<u8>,
}

impl Table {
    pub fn new(freq: Vec<u32>) -> Table {
        let mut cum = vec![0u32; freq.len() + 1];
        for i in 0..freq.len() {
            cum[i + 1] = cum[i] + freq[i];
        }
        assert_eq!(cum[freq.len()], M, "freq table must sum to 2^PB");
        let mut lut = vec![0u8; M as usize];
        for s in 0..freq.len() {
            for j in cum[s]..cum[s + 1] {
                lut[j as usize] = s as u8;
            }
        }
        Table { freq, cum, lut }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.b[self.p];
        self.p += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]);
        self.p += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes([self.b[self.p], self.b[self.p + 1], self.b[self.p + 2], self.b[self.p + 3]]);
        self.p += 4;
        v
    }
}

/// Generic decoder: calls `emit(row, symbol)` for `per_row` symbols on each of `rows` rows.
fn decode_stream<F: FnMut(usize, usize, u8)>(blob: &[u8], alpha: usize, rows: usize, per_row: usize, mut emit: F) {
    let mut r = Reader { b: blob, p: 0 };
    let k = r.u8() as usize;
    let mut tabs = Vec::with_capacity(k);
    for _ in 0..k {
        let f: Vec<u32> = (0..alpha).map(|_| r.u16() as u32).collect();
        tabs.push(Table::new(f));
    }
    let mut cls = vec![0u8; rows];
    if k > 1 {
        for i in 0..rows {
            if i % 2 == 0 {
                let b = r.b[r.p + i / 2];
                cls[i] = b & 15;
            } else {
                cls[i] = r.b[r.p + i / 2] >> 4;
            }
        }
        r.p += (rows + 1) / 2;
    }
    let mut x = [r.u32(), r.u32(), r.u32(), r.u32()];
    let b = r.b;
    let mut p = r.p;
    let mut n = 0usize;
    for row in 0..rows {
        let t = &tabs[cls[row] as usize];
        for j in 0..per_row {
            let st = &mut x[n & 3];
            let slot = *st & (M - 1);
            let s = t.lut[slot as usize];
            *st = t.freq[s as usize] * (*st >> PB) + slot - t.cum[s as usize];
            while *st < L {
                *st = (*st << 8) | b[p] as u32;
                p += 1;
            }
            emit(row, j, s);
            n += 1;
        }
    }
}

pub fn decode_five(blob: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    let wa = cols / 16;
    let wb = cols / 32;
    let mut out = vec![0u32; rows * (wa + wb)];
    let (pa, pb) = out.split_at_mut(rows * wa);
    let per_row = (cols + 2) / 3;
    // 5-ary symbol v: 0=-hi 1=-lo 2=0 3=+lo 4=+hi ; plane A code: 0 zero, 1 plus, 2 minus ; plane B: hi
    const CODE: [u32; 5] = [2, 2, 0, 1, 1];
    const HI: [u32; 5] = [1, 0, 0, 0, 1];
    decode_stream(blob, 125, rows, per_row, |row, j, s| {
        let mut s = s as u32;
        for k in 0..3 {
            let c = j * 3 + k;
            let v = (s % 5) as usize;
            s /= 5;
            if c < cols {
                pa[row * wa + c / 16] |= CODE[v] << ((c % 16) * 2);
                pb[row * wb + c / 32] |= HI[v] << (c % 32);
            }
        }
    });
    out
}

pub fn decode_dense(blob: &[u8], n: usize, bits: u32, scales: &[f32], group: usize) -> Vec<f32> {
    let off = 1i32 << (bits - 1);
    let mut out = vec![0f32; n];
    let alpha = 1usize << bits;
    decode_stream(blob, alpha, 1, n, |_, j, s| {
        out[j] = (s as i32 - off) as f32 * scales[j / group];
    });
    out
}

pub fn decode_dense_q(blob: &[u8], n: usize, bits: u32) -> Vec<i8> {
    let off = 1i32 << (bits - 1);
    let mut out = vec![0i8; n];
    decode_stream(blob, 1usize << bits, 1, n, |_, j, s| {
        out[j] = (s as i32 - off) as i8;
    });
    out
}

// ------------------------------------------------------------------------------------------------------- encoder
/// Quantise counts to a table summing to 2^PB with every present symbol >= 1.
pub fn normalize(counts: &[u64]) -> Vec<u32> {
    let tot: u64 = counts.iter().sum();
    let mut f: Vec<u32> = counts
        .iter()
        .map(|&c| if c == 0 { 0 } else { ((c as f64 * M as f64 / tot as f64).round() as u32).max(1) })
        .collect();
    loop {
        let s: i64 = f.iter().map(|&x| x as i64).sum();
        let d = M as i64 - s;
        if d == 0 {
            break;
        }
        // adjust the symbol where the change costs the least
        let mut best = usize::MAX;
        let mut bc = f64::INFINITY;
        for i in 0..f.len() {
            if counts[i] == 0 || (d < 0 && f[i] <= 1) {
                continue;
            }
            let nf = if d > 0 { f[i] + 1 } else { f[i] - 1 } as f64;
            let cost = counts[i] as f64 * ((f[i] as f64).ln() - nf.ln());
            if cost < bc {
                bc = cost;
                best = i;
            }
        }
        if d > 0 { f[best] += 1 } else { f[best] -= 1 }
    }
    f
}

/// syms: row-major symbols (rows * per_row), cls: class per row, tables: per class freqs (sum 2^PB).
pub fn encode_stream(syms: &[u8], rows: usize, per_row: usize, cls: &[u8], freqs: &[Vec<u32>]) -> Vec<u8> {
    let tabs: Vec<Table> = freqs.iter().map(|f| Table::new(f.clone())).collect();
    let n = rows * per_row;
    let mut x = [L; 4];
    let mut rev: Vec<u8> = Vec::with_capacity(n / 2);
    for i in (0..n).rev() {
        let row = i / per_row;
        let t = &tabs[cls[row] as usize];
        let s = syms[i] as usize;
        let f = t.freq[s];
        assert!(f > 0, "symbol with zero frequency");
        let st = &mut x[i & 3];
        let xmax = ((L >> PB) << 8) * f;
        while *st >= xmax {
            rev.push(*st as u8);
            *st >>= 8;
        }
        *st = ((*st / f) << PB) + (*st % f) + t.cum[s];
    }
    let mut out = Vec::new();
    out.push(freqs.len() as u8);
    for f in freqs {
        for &v in f {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
    if freqs.len() > 1 {
        for i in (0..rows).step_by(2) {
            let lo = cls[i] & 15;
            let hi = if i + 1 < rows { cls[i + 1] & 15 } else { 0 };
            out.push(lo | (hi << 4));
        }
    }
    for s in x {
        out.extend_from_slice(&s.to_le_bytes());
    }
    rev.reverse();
    out.extend_from_slice(&rev);
    out
}
