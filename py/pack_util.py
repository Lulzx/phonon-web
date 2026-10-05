import subprocess, struct, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC = os.path.join(ROOT, "wasm", "target", "release", "rans_enc")

def rans_encode(syms, rows, per_row, alpha, cls=None, k=1):
    if cls is None: cls = np.zeros(rows, np.uint8)
    hdr = struct.pack("<4I", alpha, rows, per_row, k)
    data = hdr + np.asarray(cls, np.uint8).tobytes() + np.asarray(syms, np.uint8).tobytes()
    return subprocess.run([ENC], input=data, capture_output=True, check=True).stdout

def triplets(sym5):
    """sym5 [O,I] uint8 in 0..4 (2 = zero) -> [O, ceil(I/3)] in 0..124, little-endian base 5"""
    O, I = sym5.shape
    pr = (I + 2) // 3
    p = np.full((O, pr * 3), 2, np.uint8); p[:, :I] = sym5
    p = p.reshape(O, pr, 3).astype(np.uint16)
    return (p[:, :, 0] + 5 * p[:, :, 1] + 25 * p[:, :, 2]).astype(np.uint8)

def five_syms_from_raw(r):
    s, h = r["sign"], r["is_hi"]
    return np.where(s == 0, 2, np.where(s < 0, np.where(h, 0, 1), np.where(h, 4, 3))).astype(np.uint8)
