"""Entropy-constrained GPTQ on Phonon-2's five-value grid {-hi,-lo,0,lo,hi} (per-row lo/hi), starting from the
five-value weights themselves.  lam=0 reproduces the input exactly; lam>0 trades rate (bits/weight, measured as the
order-0 entropy of the symbols) for output error, with Hessian-aware error feedback.  Afterwards lo/hi are refit per
row by least squares in the H metric.  mode='tern' restricts the grid to {-s,0,s}."""
import sys

sys.path.insert(0, "py")
from common import *  # noqa

DEV = "cpu"


def entropy_bits(sym, k=5):
    c = np.bincount(sym.ravel(), minlength=k).astype(np.float64)
    p = c / c.sum()
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def gptq_layer(Wt, lo, hi, H, lam, mode="five", damp=0.01, block=128, iters=2, rimp=None):
    """Wt [O,I] target weights (fp32 torch, on DEV); lo,hi [O]; H [I,I].  Returns (sym int8 [O,I] in 0..4 with 2=zero,
    lo, hi)."""
    O, I = Wt.shape
    H = H.clone().double()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += damp * torch.mean(torch.diag(H)) * torch.eye(I, dtype=H.dtype)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    U = torch.linalg.cholesky(Hinv, upper=True).float().to(DEV)
    Hd = H.float().to(DEV)
    d2 = torch.diag(U) ** 2  # [I]
    probs = torch.tensor([0.09, 0.15, 0.52, 0.15, 0.09], device=DEV)
    for it in range(iters):
        if mode == "five":
            lv = torch.stack([-hi, -lo, torch.zeros_like(lo), lo, hi], 1)  # [O,5]
        else:
            lv = torch.stack([-lo, torch.zeros_like(lo), lo], 1)
        R = -torch.log2(probs.clamp_min(1e-6))  # bits per symbol
        rw = torch.ones(O, device=DEV) if rimp is None else rimp.to(DEV) / rimp.mean()
        unit = (((Wt @ Hd) * Wt).sum(1) * rw).sum() / (O * I)  # weighted output energy per weight
        rwc = rw[:, None]
        W = Wt.clone()
        sym = torch.empty(O, I, dtype=torch.int64, device=DEV)
        for b0 in range(0, I, block):
            b1 = min(b0 + block, I)
            Wb = W[:, b0:b1].clone()
            Eb = torch.zeros_like(Wb)
            Ub = U[b0:b1, b0:b1]
            for j in range(b1 - b0):
                w = Wb[:, j]
                d = Ub[j, j]
                cost = rwc * (w[:, None] - lv) ** 2 / d ** 2 + lam * unit * R[None, :]
                s = cost.argmin(1)
                q = lv.gather(1, s[:, None])[:, 0]
                sym[:, b0 + j] = s
                e = (w - q) / d
                Wb[:, j:] -= e[:, None] * Ub[j, j:][None, :]
                Eb[:, j] = e
            W[:, b1:] -= Eb @ U[b0:b1, b1:]
        # symbol stats for the rate model
        cnt = torch.bincount(sym.ravel(), minlength=lv.shape[1]).float()
        probs = cnt / cnt.sum()
        # refit lo/hi per row: q = lo*A + hi*B  (least squares in H metric against the target)
        if mode == "five":
            sg = torch.tensor([-1., -1., 0., 1., 1.], device=DEV)[sym]
            ishi = torch.tensor([1., 0., 0., 0., 1.], device=DEV)[sym]
            A = sg * (1 - ishi)
            B = sg * ishi
            AH, BH = A @ Hd, B @ Hd
            a11 = (AH * A).sum(1); a12 = (AH * B).sum(1); a22 = (BH * B).sum(1)
            r1 = (AH * Wt).sum(1); r2 = (BH * Wt).sum(1)
            det = a11 * a22 - a12 ** 2
            ok = det.abs() > 1e-12 * (a11 * a22).abs().clamp_min(1e-30)
            nlo = torch.where(ok, (a22 * r1 - a12 * r2) / det, lo)
            nhi = torch.where(ok, (a11 * r2 - a12 * r1) / det, hi)
            lo, hi = nlo.clamp_min(1e-8), nhi.clamp_min(1e-8)
        else:
            A = torch.tensor([-1., 0., 1.], device=DEV)[sym]
            AH = A @ Hd
            lo = ((AH * Wt).sum(1) / (AH * A).sum(1).clamp_min(1e-12)).clamp_min(1e-8)
    if mode != "five":
        sym = sym * 2  # 0,1,2 -> 0,2,4 (so 2 stays zero)
        hi = lo
    return sym.to(torch.int8).cpu(), lo.cpu(), hi.cpu()


def deq(sym, lo, hi):
    lv = torch.stack([-hi, -lo, torch.zeros_like(lo), lo, hi], 1)
    return lv.gather(1, sym.long())


def hkey(name):
    for a in ("self_attn.k_proj", "self_attn.v_proj"):
        if name.endswith(a):
            return name[: -len(a)] + "self_attn.q_proj"
    return name


GROUPS = {"ff_lin1": ["feed_forward1.linear1", "feed_forward2.linear1"],
          "ff_lin2": ["feed_forward1.linear2", "feed_forward2.linear2"],
          "qkv": ["q_proj", "k_proj", "v_proj"], "o": ["o_proj"], "pw1": ["pointwise_conv1"], "pw2": ["pointwise_conv2"]}


def group_of(n):
    for g, subs in GROUPS.items():
        if any(s in n for s in subs):
            return g
    return None


def run(lam, mode="five", hess=None, mult=None, importance=True, log=False):
    """mult: {group: lam multiplier or None (= keep original)}"""
    t, idx, raw = read_container(CONTAINER, with_raw=True)
    if hess is None:
        hess = torch.load(os.path.join(ROOT, "work", "hess.pt"))
    mult = mult or {}
    out = dict(t)
    bits, nw = 0.0, 0
    q = {}
    for e in idx:
        if e["k"] != "five_value":
            continue
        n = e["n"]
        r = raw[n]
        g = group_of(n)
        m = mult.get(g, 1.0)
        Wt = torch.from_numpy(t[n + ".weight"].astype(np.float32)).to(DEV)
        lo = torch.from_numpy(r["lo"].astype(np.float32)).to(DEV)
        hi = torch.from_numpy(r["hi"].astype(np.float32)).to(DEV)
        if g is None or m is None:
            sym = torch.from_numpy(np.where(r["sign"] == 0, 2, np.where(r["sign"] < 0, np.where(r["is_hi"], 0, 1),
                                                                         np.where(r["is_hi"], 4, 3))).astype(np.int8))
            lo_, hi_ = lo.cpu(), hi.cpu()
        else:
            rimp = None
            if importance and n.endswith("linear1"):
                w2 = torch.from_numpy(t[n[:-1] + "2.weight"].astype(np.float32))
                rimp = hess[n + ".dsilu2"] * (w2 ** 2).sum(0)
            sym, lo_, hi_ = gptq_layer(Wt, lo, hi, hess[hkey(n)], lam * m, mode, rimp=rimp)
        out[n + ".weight"] = deq(sym, lo_, hi_).numpy()
        q[n] = (sym.numpy(), lo_.numpy(), hi_.numpy())
        b = entropy_bits(sym.numpy())
        if log:
            print(n, round(b, 3), flush=True)
        bits += b * sym.numel(); nw += sym.numel()
    return out, q, bits / nw, bits / 8 / 1e6


if __name__ == "__main__":
    import argparse
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("lam", type=float)
    ap.add_argument("--mode", default="five")
    ap.add_argument("--mult", default="", help="group=mult,...  (mult 'none' keeps the group original)")
    ap.add_argument("--no-importance", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    mult = {}
    for kv in filter(None, a.mult.split(",")):
        k, v = kv.split("=")
        mult[k] = None if v == "none" else float(v)
    out, q, bpw, mb = run(a.lam, a.mode, mult=mult, importance=not a.no_importance)
    print(f"lam={a.lam} mult={a.mult} bpw={bpw:.4f} enc_MB={mb:.2f}", flush=True)
    pickle.dump(q, open(os.path.join(ROOT, "work", a.out), "wb"))
    if a.eval:
        print(fast_eval(out, n=a.eval, verbose=True), flush=True)
