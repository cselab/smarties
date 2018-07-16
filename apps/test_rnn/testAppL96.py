
import numpy as np
from Communicator import Communicator
import time

N = 4  # number of variables
F = 8  # forcing
dt = 0.01
maxStep = 1000

def Lorenz96(x):
  d = np.zeros(N)
  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  for i in range(2, N-1): d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
  d = d + F # add the forcing term
  return d # return the state derivatives

# comm = Communicator(N/2, N/2, 2)
comm = Communicator(N, N, 2)

while(True):
    s = F*np.ones(N) # initial state (equilibrium)
    s = np.random.normal(1, 0.01, N) #perturb
    # o = s[0::2] #observation
    o = s #observation
    comm.sendInitState(o)
    step = 0

    while (True):
      s = s + dt * Lorenz96(s)
      # o = s[0::2]
      o = s #observation
      p = comm.recvAction()
      r = - np.sum((p-o)**2)

      if(step < maxStep): comm.sendState(o, r)
      else:
        comm.truncateSeq(o, r)
        break
      step += 1
