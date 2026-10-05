"""Shared helpers: load Phonon-2 container into a stock transformers ParakeetForTDT, LibriSpeech eval rows, WER."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hf"))
from fermion_container import read_container  # noqa: E402

BASE = os.path.join(ROOT, "base")
CONTAINER = os.path.join(ROOT, "hf", "model.fermion")


def to_state_dict(tensors: dict) -> dict:
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            sd[k] = torch.tensor(0, dtype=torch.int64)
            continue
        arr = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
        if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
            arr = arr[:, :, None]
        sd[k] = torch.from_numpy(arr)
    return sd


_model_cache = {}
_audio_cache = {}


def build_model(tensors: dict, device="cpu"):
    from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    cfg = ParakeetTDTConfig.from_pretrained(BASE)
    if "m" not in _model_cache:
        m = ParakeetForTDT(cfg)
        m.eval()
        m.generation_config = GenerationConfig.from_pretrained(BASE)
        _model_cache["m"] = m
        _model_cache["p"] = AutoProcessor.from_pretrained(BASE)
    m = _model_cache["m"]
    missing, unexpected = m.load_state_dict(to_state_dict(tensors), strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m.to(device), _model_cache["p"]


_PQ = {"test-clean": "all_test.clean_0000", "test-other": "all_test.other_0000",
       "dev-other": "all_validation.other_0000", "train-clean": "all_train.clean.100_0000",
       "train-other": "all_train.other.500_0000"}


def librispeech_rows(split: str, n: int, seed: int = 0):
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    t = pq.read_table(os.path.join(ROOT, "data", "pq", _PQ[split] + ".parquet"), columns=["id", "text", "audio"])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.permutation(t.num_rows)[:n].tolist())
    sub = t.take(idx).to_pylist()
    rows = []
    for r in sub:
        key = split + "/" + r["id"]
        if key not in _audio_cache:
            w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            assert sr == 16000
            _audio_cache[key] = w
        rows.append({"id": r["id"], "path": key, "ref": r["text"], "split": split})
    return rows


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num2words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num2words(n % 100))
    if n < 1_000_000:
        return _num2words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _num2words(n % 1000))
    return str(n)


def normalize(s: str) -> str:
    s = s.lower().replace("-", " ")
    s = re.sub(r"\d+", lambda m: " " + _num2words(int(m.group())) + " ", s)
    s = re.sub(r"[^a-z' ]", " ", s)
    s = re.sub(r"'(?!\w)|(?<!\w)'", " ", s)
    return " ".join(s.split())


def wer(refs, hyps):
    import jiwer
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    return jiwer.wer(r, h)


def load_audio(path):
    return _audio_cache[path]


def transcribe(model, processor, rows, device="cpu", batch=8):
    hyps = []
    rows_sorted = sorted(range(len(rows)), key=lambda i: len(load_audio(rows[i]["path"])))
    out = [None] * len(rows)
    for b in range(0, len(rows_sorted), batch):
        ids = rows_sorted[b:b + batch]
        wavs = [load_audio(rows[i]["path"]) for i in ids]
        inp = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            seq = model.generate(input_features=inp["input_features"].to(device),
                                 attention_mask=inp["attention_mask"].to(device))
        seq = getattr(seq, "sequences", seq)
        texts = processor.batch_decode(seq, skip_special_tokens=True)
        for i, t in zip(ids, texts):
            out[i] = t.strip()
    return out


def eval_tensors(tensors, n=200, splits=("test-clean", "test-other"), device="cpu", verbose=False):
    m, p = build_model(tensors, device)
    res = {}
    for sp in splits:
        rows = librispeech_rows(sp, n)
        t = time.time()
        hyps = transcribe(m, p, rows, device)
        res[sp] = round(100 * wer([r["ref"] for r in rows], hyps), 3)
        if verbose:
            print(sp, res[sp], f"{time.time() - t:.0f}s", flush=True)
    return res


# ------------------------------------------------------------------------------------- fast path (py/model.py based)
_tok = None


def detok(ids):
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer.json"))
    return _tok.decode(ids, skip_special_tokens=True).strip()


def to_torch(tensors, device="mps", dtype=torch.float32):
    W = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            continue
        a = torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32)))
        W[k] = a.to(device, dtype)
    return W


def fast_eval(tensors, n=200, splits=("test-clean", "test-other"), device="mps", dtype=torch.float32, lin_factory=None,
              verbose=False, return_hyps=False, W=None):
    import model as M
    if W is None:
        W = to_torch(tensors, device, dtype)
    dec = M.Decoder(W)
    lin = lin_factory(W) if lin_factory else None
    res, allh = {}, {}
    for sp in splits:
        rows = librispeech_rows(sp, n)
        t0 = time.time()
        hyps = []
        with torch.no_grad():
            for r in rows:
                f = M.features(load_audio(r["path"])).to(device, dtype)
                enc = M.encoder(W, f, lin)
                hyps.append(detok(dec.greedy(enc)))
        res[sp] = round(100 * wer([r["ref"] for r in rows], hyps), 3)
        allh[sp] = hyps
        if verbose:
            print(sp, res[sp], f"{time.time() - t0:.0f}s", flush=True)
    return (res, allh) if return_hyps else res
