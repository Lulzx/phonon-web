//! stdin: u32 alpha, u32 rows, u32 per_row, u32 K, cls[rows] u8, syms[rows*per_row] u8  ->  stdout: rANS stream
use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let rd = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) as usize;
    let (alpha, rows, per_row, k) = (rd(0), rd(4), rd(8), rd(12));
    let cls = &buf[16..16 + rows];
    let syms = &buf[16 + rows..16 + rows + rows * per_row];
    let mut counts = vec![vec![0u64; alpha]; k];
    for r in 0..rows {
        for &s in &syms[r * per_row..(r + 1) * per_row] {
            counts[cls[r] as usize][s as usize] += 1;
        }
    }
    let freqs: Vec<Vec<u32>> = counts
        .iter()
        .map(|c| {
            if c.iter().sum::<u64>() == 0 {
                let mut f = vec![0u32; alpha];
                f[0] = 1 << phonon_wasm::rans::PB;
                f
            } else {
                phonon_wasm::rans::normalize(c)
            }
        })
        .collect();
    let out = phonon_wasm::rans::encode_stream(syms, rows, per_row, cls, &freqs);
    std::io::stdout().write_all(&out).unwrap();
}
