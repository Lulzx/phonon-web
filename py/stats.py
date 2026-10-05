import sys, numpy as np
sys.path.insert(0,'py')
from common import *
t, idx, raw = read_container(CONTAINER, with_raw=True)
import pickle
tot=0; H=0; Hr=0
cnt=np.zeros(5)
for e in idx:
    if e['k']!='five_value': continue
    r=raw[e['n']]; s=r['sign']; h=r['is_hi']
    sym = (s+1) + 0  # 0,1,2
    sym5 = np.where(s==0, 2, np.where(s<0, np.where(h,0,1), np.where(h,4,3)))
    c=np.bincount(sym5.ravel(),minlength=5); cnt+=c
    p=c/c.sum(); h0=-(p[p>0]*np.log2(p[p>0])).sum()
    # per-row entropy
    n=s.size; tot+=n; H+=h0*n
    if 'layers.0.' in e['n'] or 'layers.12.' in e['n']:
        print(e['n'], np.round(p,3), round(h0,3), 'lo/hi ratio', np.round(np.median(r['hi']/r['lo']),2), e['b']*8/n)
print('overall p', np.round(cnt/cnt.sum(),4), 'avg H', H/tot, 'MB at H', H/8/1e6)
