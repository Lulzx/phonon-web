import sys; sys.path.insert(0,'py')
from common import *
t, idx, raw = read_container(CONTAINER, with_raw=True)
t2 = dict(t)
for e in idx:
    if e['k']!='five_value': continue
    r=raw[e['n']]; w=t[e['n']+'.weight'].astype(np.float32); s=r['sign'].astype(np.float32)
    nz=s!=0
    sc=(np.abs(w)*nz).sum(1)/np.maximum(nz.sum(1),1)
    t2[e['n']+'.weight']=s*sc[:,None]
print('naive ternary', fast_eval(t2, n=100, verbose=True))
