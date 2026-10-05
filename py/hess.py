"""Collect per-linear input Hessians H = sum x x^T over calibration audio (train sets), from a given weight set."""
import sys

sys.path.insert(0, "py")
from common import *  # noqa
import model as M

SHARED = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj"}


def collect(tensors, n=96, device="mps", out=None):
    W = to_torch(tensors, device)
    Hs, cnt = {}, {}

    def lin(name, x):
        key = name
        for a, b in SHARED.items():
            if name.endswith(a):
                key = name[: -len(a)] + b
        if key == name:
            xx = x.float()
            if key not in Hs:
                Hs[key] = torch.zeros(x.shape[1], x.shape[1], device=device)
                cnt[key] = 0
            Hs[key] += xx.T @ xx
            cnt[key] += x.shape[0]
        y = x @ W[name + ".weight"].T
        if name.endswith("linear1"):
            sg = torch.sigmoid(y.float())
            d = sg * (1 + y.float() * (1 - sg))  # silu'
            k2 = name + ".dsilu2"
            if k2 not in Hs:
                Hs[k2] = torch.zeros(y.shape[1], device=device)
                cnt[k2] = 0
            Hs[k2] += (d * d).sum(0)
            cnt[k2] += y.shape[0]
        return y

    rows = librispeech_rows("train-other", n // 2, seed=1) + librispeech_rows("train-clean", n // 2, seed=1)
    with torch.no_grad():
        for i, r in enumerate(rows):
            f = M.features(load_audio(r["path"])).to(device)
            M.encoder(W, f, lin)
    res = {k: (v / cnt[k]).cpu() for k, v in Hs.items()}
    if out:
        torch.save(res, out)
    return res


if __name__ == "__main__":
    t, _ = read_container(CONTAINER)
    collect(t, n=int(sys.argv[1]), out=os.path.join(ROOT, "work", "hess.pt"))
