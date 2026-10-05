"""Write a .phon container for the browser engine.

usage: pack.py OUT.phon [--enc q_five_X.pkl] [--dense-bits 6] [--dense-group 0] [--classes 8]

Layout: "PHON" u32 version u32 header_len, header JSON, pad to 4, blobs.  Tensor kinds:
  fv  five-value [O,I]: rANS stream of base-5 triplets with per-row classes; lo/hi as f16 [2*O]
  dq  integer table: rANS stream of (q + 2^(bits-1)), f16 scales per row (group=I) or per group
  f16 raw half floats
"""
import argparse
import json
import pickle
import struct
import sys

sys.path.insert(0, "py")
from common import *  # noqa
from pack_util import five_syms_from_raw, rans_encode, triplets


def row_classes(tr, K, alpha=125, iters=6):
    """cluster rows by symbol histogram (cross-entropy Lloyd), returns cls [O] uint8"""
    O = tr.shape[0]
    if K <= 1 or O < 64:
        return np.zeros(O, np.uint8), 1
    H = np.zeros((O, alpha))
    np.add.at(H, (np.repeat(np.arange(O), tr.shape[1]), tr.ravel()), 1)
    pz = H[:, 62] / H.sum(1)  # triplet (2,2,2) = all zero
    cls = np.minimum((np.argsort(np.argsort(pz)) * K) // O, K - 1)
    for _ in range(iters):
        C = np.stack([H[cls == k].sum(0) + 0.01 for k in range(K)])
        C = C / C.sum(1, keepdims=True)
        cost = -(H @ np.log2(C).T)  # [O, K]
        cls = cost.argmin(1)
    return cls.astype(np.uint8), K


def best_classes(tr, Kmax):
    best = None
    for K in sorted({1, 2, 4, 8, Kmax}):
        if K > Kmax:
            continue
        cls, k = row_classes(tr, K)
        b = rans_encode(tr, tr.shape[0], tr.shape[1], 125, cls, k)
        if best is None or len(b) < len(best):
            best = b
    return best


def dense_quant(w, bits, group):
    shp = w.shape
    O = shp[0]
    w2 = w.reshape(O, -1).astype(np.float32)
    g = group or w2.shape[1]
    w2 = w2.reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(w2).max(1, keepdims=True) + 1e-12
    best = None
    for f in np.linspace(1.0, 0.6, 9):
        s = (amax * f / qmax).astype(np.float16).astype(np.float32)
        q = np.clip(np.round(w2 / s), -qmax - 1, qmax)
        err = ((q * s - w2) ** 2).sum(1, keepdims=True)
        if best is None:
            best = [q, s, err]
        else:
            m = err < best[2]
            best = [np.where(m, q, best[0]), np.where(m, s, best[1]), np.where(m, err, best[2])]
    return best[0].astype(np.int32).ravel(), best[1].ravel().astype(np.float16), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--enc", default=None)
    ap.add_argument("--dense-bits", type=int, default=6)
    ap.add_argument("--dense-group", type=int, default=0)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--bits-map", default="", help="name_substring=bits,... overrides for dense tables")
    a = ap.parse_args()

    t, idx, raw = read_container(CONTAINER, with_raw=True)
    q = pickle.load(open(os.path.join(ROOT, "work", a.enc), "rb")) if a.enc else None
    overrides = [kv.split("=") for kv in a.bits_map.split(",") if kv]
    blobs, header, off = [], {}, 0

    def add(b):
        nonlocal off
        b = bytes(b)
        o = off
        blobs.append(b)
        pad = (-len(b)) % 4
        if pad:
            blobs.append(b"\0" * pad)
        off += len(b) + pad
        return o, len(b)

    stats = {"fv": 0, "dq": 0, "f16": 0}
    for e in idx:
        n, k, shape = e["n"], e["k"], e["shape"]
        if n.endswith("num_batches_tracked"):
            continue
        if k == "five_value":
            if q is not None:
                sym, lo, hi = q[n]
                sym = sym.astype(np.uint8)
            else:
                r = raw[n]
                sym, lo, hi = five_syms_from_raw(r), r["lo"].astype(np.float32), r["hi"].astype(np.float32)
            tr = triplets(sym)
            stream = best_classes(tr, a.classes)
            o, l = add(stream)
            lh = np.concatenate([lo, hi]).astype(np.float16)
            lo_, ll = add(lh.tobytes())
            header[n + ".weight"] = {"kind": "fv", "shape": shape, "off": o, "len": l, "loff": lo_, "llen": ll}
            stats["fv"] += l + ll
        elif k.startswith("int"):
            bits = a.dense_bits
            for sub, b in overrides:
                if sub in n:
                    bits = int(b)
            w = t[n].astype(np.float32)
            if bits >= 6:  # keep the stored int6 values exactly
                r = raw[n]
                qq, sc, g = r["q"].astype(np.int32).ravel(), r["scale"].astype(np.float16), int(np.prod(shape[1:]))
                bits = 6
            else:
                qq, sc, g = dense_quant(w, bits, a.dense_group)
            stream = rans_encode((qq + 2 ** (bits - 1)).astype(np.uint8), 1, qq.size, 2 ** bits)
            o, l = add(stream)
            so, sl = add(sc.tobytes())
            header[n] = {"kind": "dq", "shape": shape, "off": o, "len": l, "bits": bits, "group": g,
                         "soff": so, "slen": sl}
            stats["dq"] += l + sl
        else:
            o, l = add(t[n].astype(np.float16).tobytes())
            header[n] = {"kind": "f16", "shape": shape, "off": o, "len": l}
            stats["f16"] += l
    tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
    special = {x["id"] for x in tok["added_tokens"]}
    vocab = [""] * 8192
    for p, i in tok["model"]["vocab"].items():
        vocab[i] = "" if i in special else p
    hj = json.dumps({"format": "phonon2-web-v1", "tensors": header, "vocab": vocab,
                     "source": "FermionResearch/Phonon-2 (CC-BY-4.0, derived from nvidia/parakeet-tdt-0.6b-v3)"},
                    ensure_ascii=False, separators=(",", ":")).encode()
    with open(a.out, "wb") as f:
        f.write(b"PHON" + struct.pack("<II", 1, len(hj)) + hj)
        f.write(b"\0" * ((-(12 + len(hj))) % 4))
        for b in blobs:
            f.write(b)
    total = os.path.getsize(a.out)
    print(f"{a.out}: {total / 1e6:.2f} MB  (fv {stats['fv'] / 1e6:.2f}, dense {stats['dq'] / 1e6:.2f}, "
          f"f16 {stats['f16'] / 1e6:.2f}, header {len(hj) / 1e6:.3f})")


if __name__ == "__main__":
    main()
