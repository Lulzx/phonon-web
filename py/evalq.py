"""Evaluate a quantized encoder pickle (from gptq.py) + optional dense-table quantization.
usage: evalq.py q_five_0.01.pkl [n] [dense_bits] [group]"""
import pickle
import sys

sys.path.insert(0, "py")
from common import *  # noqa
from gptq import deq


def entropy(x):
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


DENSE_SKIP = ("num_batches_tracked",)


def quant_dense(t, idx, raw, bits, group=0):
    """re-quantize every int6 table to `bits` (symmetric, per-row or per-group absmax/MSE scale) -> (tensors, bytes)"""
    out = dict(t)
    tot = 0.0
    for e in idx:
        if not e["k"].startswith("int"):
            continue
        n = e["n"]
        w = t[n].astype(np.float32)
        shp = w.shape
        w2 = w.reshape(shp[0], -1)
        if group:
            w2 = w2.reshape(-1, group)
        qmax = 2 ** (bits - 1) - 1
        best = None
        amax = np.abs(w2).max(1, keepdims=True) + 1e-12
        for f in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            s = amax * f / qmax
            q = np.clip(np.round(w2 / s), -qmax - 1, qmax)
            err = ((q * s - w2) ** 2).sum(1, keepdims=True)
            if best is None:
                best = [q, s, err]
            else:
                m = err < best[2]
                best[0] = np.where(m, q, best[0]); best[1] = np.where(m, s, best[1]); best[2] = np.where(m, err, best[2])
        q, s, _ = best
        out[n] = (q * s).reshape(shp)
        tot += entropy(q) * q.size / 8 + s.size * 2
    return out, tot


if __name__ == "__main__":
    t, idx, raw = read_container(CONTAINER, with_raw=True)
    out = dict(t)
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if name != "none":
        q = pickle.load(open(os.path.join(ROOT, "work", name), "rb"))
        from gptq import entropy_bits
        bits = 0
        for k, (sym, lo, hi) in q.items():
            out[k + ".weight"] = deq(torch.from_numpy(sym), torch.from_numpy(lo), torch.from_numpy(hi)).numpy()
            bits += entropy_bits(sym) * sym.size
        print(name, "encoder MB (order-0 entropy)", round(bits / 8e6, 2))
    if len(sys.argv) > 3:
        db = int(sys.argv[3])
        g = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        out2, by = quant_dense(out, idx, raw, db, g)
        print("dense bits", db, "group", g, "MB", round(by / 1e6, 2))
        out = out2
    print(fast_eval(out, n=n, verbose=True), flush=True)
