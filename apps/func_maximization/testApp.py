# objective of this problem is to learn simple time
# dependencies

import numpy as np
from smarties import Communicator
import time

N = 1  # number of variables
sigma = 1
maxStep = 1000

def app_main(comm):
  comm.set_state_action_dims(N, N)

  while(True):
    s = np.random.normal(0, sigma, N) # initial state is observation
    comm.sendInitState(s)
    step = 0

    while (True):
      p = comm.recvAction()
      r = - np.sum((p-s)**2) # action has to figure out current state

      o = np.random.normal(0, sigma, N) # observation is more noise
      s = 0.9*s + 0.1*o # state is exponential average of observations
      # learn rate is such that 20 steps BPTT captures most dependency

      if(step < maxStep): comm.sendState(o, r)
      else:
        comm.sendLastState(o, r)
        break
      step += 1

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )