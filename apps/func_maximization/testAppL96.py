# objective of this problem is to guess all state variables
# by observing roughly half of them

import numpy as np
import smarties as rl
import time

N = 8  # number of variables
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

def app_main(comm):
  comm.setStateActionDims(N//2, N//2)

  while(True):
    s = np.random.normal(F, 0.1, N) # equilibrium (s=F) + perturbation
    o = s[0::2] # observation
    comm.sendInitState(o)
    step = 0

    while (True):
      p = comm.recvAction() # RL has to predict next observation
      s = s + dt * Lorenz96(s)
      o = s[0::2]           # observation is half of state variables
      r = - np.sum((p-o)**2)

      if(step < maxStep): comm.sendState(o, r)
      else:
        comm.sendLastState(o, r)
        break
      step += 1

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )