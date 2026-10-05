use std::io::Read;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut b = Vec::new();
    std::fs::File::open(&a[1]).unwrap().read_to_end(&mut b).unwrap();
    let pcm: Vec<f32> = b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let t = std::time::Instant::now();
    let (f, _) = phonon_wasm::mel::features(&pcm);
    eprintln!("{:?}", t.elapsed());
    let o: Vec<u8> = f.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&a[2], o).unwrap();
}
