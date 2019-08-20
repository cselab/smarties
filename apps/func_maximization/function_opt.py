import numpy as np
from Communicator import Communicator
from scipy.integrate import ode

# Define the environment:
class EnvironmentForRL:
  # Class initializer:
  def __init__(self):
    self.t = 0 # Iteration counter
    self.x = 0 # Location at current iteration
    self.percSuccess = 0
    self.sigma = 1
    self.noise = 0
 
  # Function to begin a new time series:
  def reset(self):
    # set initial value of x
    self.x = 10 #np.random.randn()
    self.t = 0 # reset simulation step counter
    self.percSuccess = 0
    self.sigma = 1
    self.noise = np.random.randn()
    
  def getState(self):
    # returns the current state to RL
    return np.array([self.x, self.percSuccess])
    #return np.array([self.percSuccess])

  def getReward(self):
    # given the current state, compute reward
    return 1/(1 + self.computeFunction() )
    #return self.percSuccess

  def computeFunction(self, X=None):
    # y = f(x)
    eps = 0.0
    if X is not None: return (X + 100 + 0*self.noise)**2
    else: return (self.x + 100 + 0*self.noise)**2

  def advance(self, action):
    # given action, advance to new state
    self.t = self.t + 1
    self.sigma = self.sigma * action
    # displacement = N(0, action)
    displacement = 0. + self.sigma * np.random.randn()
    newx = self.x + displacement
    accept = self.computeFunction(newx) < self.computeFunction() 
    if accept: 
      # accept displaced state
      self.x = newx
      # increase the self.percSuccess average over last 10 steps
      self.percSuccess = .9*self.percSuccess + 0.1
    else:
      # decrease the self.percSuccess average over last 10 steps
      self.percSuccess = .9*self.percSuccess
    # this function should return if time-series is over
    episodeIsOver =  self.t > 200 
    return episodeIsOver


if __name__ == '__main__':
    # state is x, v, angle, omega, cos(angle), sin(angle), action is Fx
    # This specifies states and action dimensionality
    dimState = 2 # how many state variables
    dimAction = 1 # how many action variables
    comm = Communicator(dimState, dimAction)
    # create environment defined above
    env = EnvironmentForRL()

    # Define action space:
    actionMax = 1.9*np.ones(dimAction)
    actionMin = 0.1*np.ones(dimAction)
    isActBounded = True 
    comm.set_action_scales(actionMax, actionMin, bounded=isActBounded)

    bObservable = np.ones(dimState)
    bObservable[0] = 0
    comm.set_state_observable(bObservable)

    while 1: #train loop, each new episode starts here
        env.reset() # (slightly) random initial conditions are best
        # send initial state along with info that it is an initial state:
        comm.sendInitState(env.getState());

        while 1: #simulation loop
            action = comm.recvAction();

            #advance the simulation:
            terminated = env.advance(action);

            state = env.getState();
            reward = env.getReward();

            if terminated:  #tell smarties that this is a terminal state
                comm.sendTermState(state, reward);
                break
            else: comm.sendState(state, reward); # normal state
