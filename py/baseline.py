import sys, time; sys.path.insert(0,'py')
from common import *
t,_=read_container(CONTAINER)
import torch
torch.set_num_threads(12)
for dev in ['cpu']:
    s=time.time(); print(dev, eval_tensors(t, n=int(sys.argv[1]), device=dev, verbose=True), time.time()-s)
