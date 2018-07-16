
import numpy as np
from Communicator import Communicator
import time

N = 1  # number of variables
sigma = 1
maxStep = 1000

# comm = Communicator(N/2, N/2, 2)
comm = Communicator(N, N)

while(True):
    s = np.random.normal(0, sigma, N) # initial state is observation
    comm.sendInitState(s)
    step = 0

    while (True):
      p = comm.recvAction()
      r = - np.sum((p-s)**2)
      o = np.random.normal(0, sigma, N)
      s = 0.5*s + o # s.t. 20 steps BPTT captures most dependency

      if(step < maxStep): comm.sendState(o, r)
      else:
        comm.truncateSeq(o, r)
        break
      step += 1
