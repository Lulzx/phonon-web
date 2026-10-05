"""Minimal, self-contained Parakeet-TDT (Phonon-2) forward in torch.  Mirrors exactly what the browser engine computes:
mel features -> dw-striding subsampling -> 24 FastConformer blocks (rel-pos MHSA) -> TDT greedy decode."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

D = 1024
H = 8
DH = 128
NL = 24
BLANK = 8192
DURS = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------------------------------------- features
_mel = None


def mel_filters():
    global _mel
    if _mel is None:
        import librosa
        _mel = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000, norm="slaney"))
    return _mel


def features(wav: np.ndarray) -> torch.Tensor:
    """wav float32 [N] -> normalized log-mel [T, 128] with T = N // 160 (valid frames only)."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])
    win = torch.hann_window(400, periodic=False)
    st = torch.stft(x, 512, hop_length=160, win_length=400, window=win, return_complex=True, pad_mode="constant")
    mag = st.real ** 2 + st.imag ** 2
    m = torch.log(mel_filters() @ mag + 2 ** -24).T  # [frames, 128]
    T = len(wav) // 160
    m = m[:T]
    mean = m.mean(0, keepdim=True)
    std = ((m - mean) ** 2).sum(0, keepdim=True).div(T - 1).sqrt()
    return (m - mean) / (std + 1e-5)


# ------------------------------------------------------------------------------------------------------------ encoder
def rel_pos(T, device, dtype):
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float32)
    f = pos[:, None] * inv[None, :]
    pe = torch.stack([f.sin(), f.cos()], -1).reshape(2 * T - 1, D)
    return pe.to(device, dtype)


def ln(x, w, b):
    return F.layer_norm(x, (D,), w, b, 1e-5)


def rel_shift(x):
    b, h, q, p = x.shape
    x = F.pad(x, (1, 0)).view(b, h, -1, q)
    return x[:, :, 1:].view(b, h, q, p)


def encoder(W, feats: torch.Tensor, lin=None) -> torch.Tensor:
    """feats [T,128] -> [T', 1024]  (single utterance, no padding).  `lin(name, x)` overrides the linear layers
    (used for activation capture / quantized matmuls)."""
    if lin is None:
        def lin(name, x):
            return x @ W[name + ".weight"].T
    p = "encoder.subsampling."
    x = feats[None, None]  # [1,1,T,128]
    x = F.relu(F.conv2d(x, W[p + "layers.0.weight"], W[p + "layers.0.bias"], stride=2, padding=1))
    x = F.conv2d(x, W[p + "layers.2.weight"], W[p + "layers.2.bias"], stride=2, padding=1, groups=256)
    x = F.relu(F.conv2d(x, W[p + "layers.3.weight"], W[p + "layers.3.bias"]))
    x = F.conv2d(x, W[p + "layers.5.weight"], W[p + "layers.5.bias"], stride=2, padding=1, groups=256)
    x = F.relu(F.conv2d(x, W[p + "layers.6.weight"], W[p + "layers.6.bias"]))
    x = x.transpose(1, 2).reshape(1, x.shape[2], -1)[0]  # [T', 256*16]
    x = x @ W[p + "linear.weight"].T + W[p + "linear.bias"]
    T = x.shape[0]
    pe = rel_pos(T, x.device, x.dtype)
    for i in range(NL):
        q = f"encoder.layers.{i}."
        r = ln(x, W[q + "norm_feed_forward1.weight"], W[q + "norm_feed_forward1.bias"])
        r = lin(q + "feed_forward1.linear2", F.silu(lin(q + "feed_forward1.linear1", r)))
        x = x + 0.5 * r
        a = ln(x, W[q + "norm_self_att.weight"], W[q + "norm_self_att.bias"])
        qq = lin(q + "self_attn.q_proj", a).view(T, H, DH).transpose(0, 1)
        kk = lin(q + "self_attn.k_proj", a).view(T, H, DH).transpose(0, 1)
        vv = lin(q + "self_attn.v_proj", a).view(T, H, DH).transpose(0, 1)
        pk = lin(q + "self_attn.relative_k_proj", pe).view(-1, H, DH).permute(1, 2, 0)  # [H, DH, 2T-1]
        qu = qq + W[q + "self_attn.bias_u"][:, None]
        qv = qq + W[q + "self_attn.bias_v"][:, None]
        bd = rel_shift((qv @ pk)[None])[0][..., :T]
        s = (qu @ kk.transpose(1, 2) + bd) / math.sqrt(DH)
        o = (torch.softmax(s.float(), -1).to(x.dtype) @ vv).transpose(0, 1).reshape(T, D)
        x = x + lin(q + "self_attn.o_proj", o)
        c = ln(x, W[q + "norm_conv.weight"], W[q + "norm_conv.bias"])
        c = F.glu(lin(q + "conv.pointwise_conv1", c), dim=-1)  # [T, D]
        c = F.conv1d(c.T[None], W[q + "conv.depthwise_conv.weight"], padding=4, groups=D)[0]  # [D,T]
        bn = q + "conv.norm."
        c = (c - W[bn + "running_mean"][:, None]) / torch.sqrt(W[bn + "running_var"][:, None] + 1e-5) \
            * W[bn + "weight"][:, None] + W[bn + "bias"][:, None]
        c = lin(q + "conv.pointwise_conv2", F.silu(c).T)
        x = x + c
        r = ln(x, W[q + "norm_feed_forward2.weight"], W[q + "norm_feed_forward2.bias"])
        r = lin(q + "feed_forward2.linear2", F.silu(lin(q + "feed_forward2.linear1", r)))
        x = x + 0.5 * r
        x = ln(x, W[q + "norm_out.weight"], W[q + "norm_out.bias"])
    return x


# ------------------------------------------------------------------------------------------------------------ decoder
class Decoder:
    def __init__(self, W):
        g = lambda k: W[k].float().cpu()  # noqa: E731
        self.emb = g("decoder.embedding.weight")
        self.wih = [g(f"decoder.lstm.weight_ih_l{l}") for l in range(2)]
        self.whh = [g(f"decoder.lstm.weight_hh_l{l}") for l in range(2)]
        self.b = [g(f"decoder.lstm.bias_ih_l{l}") + g(f"decoder.lstm.bias_hh_l{l}") for l in range(2)]
        self.pw, self.pb = g("decoder.decoder_projector.weight"), g("decoder.decoder_projector.bias")
        self.ew, self.eb = g("encoder_projector.weight"), g("encoder_projector.bias")
        self.jw, self.jb = g("joint.head.weight"), g("joint.head.bias")

    def step(self, tok, h, c):
        x = self.emb[tok]
        h2, c2 = [], []
        for l in range(2):
            gates = self.wih[l] @ x + self.whh[l] @ h[l] + self.b[l]
            i, f, gg, o = gates.chunk(4)
            cc = torch.sigmoid(f) * c[l] + torch.sigmoid(i) * torch.tanh(gg)
            hh = torch.sigmoid(o) * torch.tanh(cc)
            h2.append(hh); c2.append(cc)
            x = hh
        return self.pw @ x + self.pb, h2, c2

    @torch.no_grad()
    def greedy(self, enc: torch.Tensor, max_symbols=10):
        """enc [T,1024] -> token ids (TDT greedy, NeMo semantics)."""
        e = enc.float().cpu() @ self.ew.T + self.eb  # [T, 640]
        T = e.shape[0]
        z = torch.zeros(640)
        h, c = [z, z], [z, z]
        g, h1, c1 = self.step(BLANK, h, c)  # start: blank as SOS (blank_as_pad -> zero embedding row)
        out = []
        t = 0
        sym = 0
        while t < T:
            logits = self.jw @ torch.relu(e[t] + g) + self.jb
            k = int(logits[:BLANK + 1].argmax())
            d = DURS[int(logits[BLANK + 1:].argmax())]
            if k != BLANK:
                out.append(k)
                h, c = h1, c1
                g, h1, c1 = self.step(k, h, c)
                sym += 1
            if k == BLANK and d == 0:
                d = 1
            if d == 0 and sym >= max_symbols:
                d = 1
            if d > 0:
                sym = 0
            t += d
        return out
