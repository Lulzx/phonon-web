import sys; sys.path.insert(0,'py')
from common import *
t,_=read_container(CONTAINER)
print(fast_eval(t, n=int(sys.argv[1]), verbose=True))
