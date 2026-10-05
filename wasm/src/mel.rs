//! Parakeet log-mel front end: pre-emphasis 0.97, 512-point STFT (hop 160, symmetric Hann 400 centred in the
//! frame, zero "constant" centre padding), power spectrum, 128 Slaney mel bands (librosa norm="slaney"),
//! log(x + 2^-24), per-band mean/std normalisation over the valid frames.

const N_FFT: usize = 512;
const HOP: usize = 160;
const WIN: usize = 400;
const N_MELS: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1;

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
}

/// librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney"), row-major [128][257].
pub fn mel_filterbank() -> Vec<f32> {
    let sr = 16000.0f64;
    let fft: Vec<f64> = (0..N_BINS).map(|i| i as f64 * (sr / 2.0) / (N_BINS - 1) as f64).collect();
    let (mmin, mmax) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut w = vec![0f32; N_MELS * N_BINS];
    for m in 0..N_MELS {
        let d0 = mel_f[m + 1] - mel_f[m];
        let d1 = mel_f[m + 2] - mel_f[m + 1];
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for k in 0..N_BINS {
            let lower = (fft[k] - mel_f[m]) / d0;
            let upper = (mel_f[m + 2] - fft[k]) / d1;
            let v = lower.min(upper).max(0.0);
            w[m * N_BINS + k] = (v * enorm) as f32;
        }
    }
    w
}

struct Fft {
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Fft {
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let cos = (0..n / 2).map(|i| (-2.0 * std::f64::consts::PI * i as f64 / n as f64).cos() as f32).collect();
        let sin = (0..n / 2).map(|i| (-2.0 * std::f64::consts::PI * i as f64 / n as f64).sin() as f32).collect();
        Fft { cos, sin, rev }
    }

    fn run(&self, re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for s in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (s + k, s + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}

/// 16 kHz mono PCM -> normalised features, row-major [T][128], T = len / 160.
pub fn features(pcm: &[f32]) -> (Vec<f32>, usize) {
    let n = pcm.len();
    let t_valid = n / HOP;
    if t_valid < 2 {
        return (vec![0.0; t_valid * N_MELS], t_valid);
    }
    // pre-emphasis
    let mut x = vec![0f32; n];
    x[0] = pcm[0];
    for i in 1..n {
        x[i] = pcm[i] - 0.97 * pcm[i - 1];
    }
    // window: symmetric Hann(400) centred in 512
    let mut win = vec![0f32; N_FFT];
    let off = (N_FFT - WIN) / 2;
    for i in 0..WIN {
        win[off + i] = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64).cos()) as f32;
    }
    let fb = mel_filterbank();
    let fft = Fft::new(N_FFT);
    let mut out = vec![0f32; t_valid * N_MELS];
    let mut re = vec![0f32; N_FFT];
    let mut im = vec![0f32; N_FFT];
    let mut pw = vec![0f32; N_BINS];
    let pad = N_FFT / 2;
    for t in 0..t_valid {
        let start = (t * HOP) as isize - pad as isize;
        for i in 0..N_FFT {
            let j = start + i as isize;
            re[i] = if j >= 0 && (j as usize) < n { x[j as usize] * win[i] } else { 0.0 };
            im[i] = 0.0;
        }
        fft.run(&mut re, &mut im);
        for k in 0..N_BINS {
            pw[k] = re[k] * re[k] + im[k] * im[k];
        }
        let row = &mut out[t * N_MELS..(t + 1) * N_MELS];
        for m in 0..N_MELS {
            let f = &fb[m * N_BINS..(m + 1) * N_BINS];
            let mut acc = 0f32;
            for k in 0..N_BINS {
                acc += f[k] * pw[k];
            }
            row[m] = (acc + 5.960464477539063e-8).ln();
        }
    }
    for m in 0..N_MELS {
        let mut mean = 0f64;
        for t in 0..t_valid {
            mean += out[t * N_MELS + m] as f64;
        }
        mean /= t_valid as f64;
        let mut var = 0f64;
        for t in 0..t_valid {
            let d = out[t * N_MELS + m] as f64 - mean;
            var += d * d;
        }
        let std = (var / (t_valid - 1) as f64).sqrt();
        let inv = 1.0 / (std + 1e-5);
        for t in 0..t_valid {
            let v = &mut out[t * N_MELS + m];
            *v = ((*v as f64 - mean) * inv) as f32;
        }
    }
    (out, t_valid)
}
