# Phonon-2 in the browser

English speech recognition that runs entirely in the browser. It uses
[Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) by Fermion Research, derived from NVIDIA Parakeet TDT 0.6B v3.
WebGPU runs the 24-layer FastConformer encoder. A 42 KB WebAssembly module handles weight decompression, log-mel
features and the TDT decoder. Audio never leaves the page.

**Live demo:** https://lulzx.github.io/phonon-web/

- Transcribes audio files, or the microphone live while you speak
- 162.6 MB model download, cached by the browser after the first visit
- Output is token-for-token identical to the PyTorch reference running the original Phonon-2 weights
- About 24× real time on an Apple M4 Pro: 2 min 7 s of audio in 5.3 s

## Model format

Phonon-2 stores each encoder weight as one of five values {0, ±lo, ±hi} (per-row lo/hi), about 2.1 bits each, plus
int6 tables for the rest. Here those symbols are re-coded losslessly with interleaved rANS: base-5 triplets,
per-row distribution classes, 4 states. The repack is 162.6 MB; the original download is 164 MB. The weights are
already close to their entropy (1.95 bits/weight).

At load time the WASM module decodes each matrix straight into a GPU layout: a 2-bit sign/zero plane plus a 1-bit
hi/lo plane, 3 bits/weight, about 230 MB of GPU memory. The matmul kernel expands these planes in workgroup memory.

The file is split into four parts in `docs/model/`, because GitHub Pages limits files to 100 MB.

## Layout

| Path | What |
|---|---|
| `docs/` | the static site (GitHub Pages): `index.html`, `app.js`, `live.js` (mic streaming), `worker.js`, `engine.js` (WebGPU encoder), `kernels.js` (WGSL), `pkg/` (WASM build) |
| `docs/test.html` | regression page: transcribes `docs/test/*.wav` and compares tokens with the Python reference |
| `wasm/` | Rust crate: `mel.rs` (features), `decoder.rs` (LSTM + joint, int8/SIMD), `rans.rs` (entropy coder); `src/bin/rans_enc.rs` is the packer's encoder |
| `py/` | reference model (`model.py`), evaluation, the packer (`pack.py`), and the GPTQ requantisation experiments |
| `tools/` | Playwright scripts that drive headless Chrome with WebGPU |

## Build

```sh
# WASM
cd wasm && wasm-pack build --release --target web --out-dir ../docs/pkg

# model: download FermionResearch/Phonon-2 into hf/ and nvidia/parakeet-tdt-0.6b-v3's tokenizer.json/config into base/,
# then build the rANS encoder (cargo build --release --bins) and run
python py/pack.py dist/phonon2-lossless.phon
```

Serve the repository root (`python3 -m http.server`) and open `/docs/`. Needs WebGPU: Chrome or Edge 113+,
Safari 26+, or Firefox 141+ on Windows.

## Live transcription

Phonon-2 is an offline model, so live mode re-transcribes the not-yet-committed tail of the recording whenever the
engine is free. Once the tail passes 10 s, text up to the last sentence end is committed. Mid-sentence cuts happen
only after 20 s without a sentence end. If speech continues more than 3 s past the last emitted word, the window
restarts there. This works around a quirk of the model, which sometimes stops emitting for windows that begin
mid-sentence.

## Licences

- Model weights: CC-BY-4.0 (Fermion Research; derived from NVIDIA Parakeet TDT 0.6B v3). See
  `LICENSE-WEIGHTS-CC-BY-4.0.txt` and `NOTICE-Phonon-2.txt`. The weights here are Phonon-2's, entropy-coded into a
  different container format; their values are unchanged.
- Code in this repository: Apache-2.0 (`LICENSE`).
- Test clips in `docs/test/`: LibriSpeech (CC-BY-4.0).
