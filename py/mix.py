import sys, pickle; sys.path.insert(0,'py')
from common import *
from gptq import deq, entropy_bits
t, idx, raw = read_container(CONTAINER, with_raw=True)
from pack_util import five_syms_from_raw
q = pickle.load(open('work/q_five_0.01.pkl','rb'))
base_bits = {n: entropy_bits(five_syms_from_raw(raw[n]))*raw[n]['sign'].size for n in q}
groups = {'ff_lin1':['feed_forward1.linear1','feed_forward2.linear1'], 'ff_lin2':['feed_forward1.linear2','feed_forward2.linear2'],
          'qkv':['q_proj','k_proj','v_proj'], 'o':['o_proj'], 'pw1':['pointwise_conv1'], 'pw2':['pointwise_conv2']}
sel = sys.argv[1:]
out=dict(t); saved=0
for n,(sym,lo,hi) in q.items():
    if any(any(s in n for s in groups[g]) for g in sel):
        out[n+'.weight']=deq(torch.from_numpy(sym),torch.from_numpy(lo),torch.from_numpy(hi)).numpy()
        saved += base_bits[n]-entropy_bits(sym)*sym.size
print(sel, 'saved MB', round(saved/8e6,2), fast_eval(out, n=200, splits=('test-other',)), flush=True)
