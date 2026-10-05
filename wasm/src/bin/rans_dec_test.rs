//! args: five <rows> <cols> < stream  -> prints checksum of planes; dense <n> <bits>
use std::io::Read;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let t = std::time::Instant::now();
    if a[1] == "five" {
        let (r, c): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
        let o = phonon_wasm::rans::decode_five(&buf, r, c);
        let bytes: Vec<u8> = o.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&a[4], bytes).unwrap();
    }
    eprintln!("{:?}", t.elapsed());
}
