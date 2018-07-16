import numpy as np
from Communicator import Communicator
from scipy.integrate import ode

class CartPole:
    def __init__(self):
        self.dt = 4e-4
        self.nsteps = 50
        self.step=0
        self.u = np.asarray([0, 0, 0, 0])
        self.F=0
        self.t=0
        self.ODE = ode(self.system).set_integrator('dopri5')

    def reset(self):
        self.u = np.random.uniform(-0.05, 0.05, 4)
        self.step = 0
        self.F = 0
        self.t = 0

    def isOver(self): # is episode over
      #return step>=500 || std::fabs(u.y1)>2.4 #swingup
      return (self.step>=500 or  abs(self.u[0])>2.4 or  abs(self.u[2])>np.pi/15)

    def isTruncated(self):  # check that cause for termination is time limits
      return (self.step>=500 and abs(self.u[0])<2.4 and abs(self.u[2])<np.pi/15)

    @staticmethod
    def system(t, y, act): #dynamics function
        #print(t,y,act) sys.stdout.flush()
        mp, mc, l, g = 0.1, 1, 0.5, 9.81
        x, v, a, w = y[0], y[1], y[2], y[3]
        cosy, siny = np.cos(a), np.sin(a)
        #const double fac1 = 1./(mc + mp * siny*siny);  #swingup
        #const double fac2 = fac1/l;
        #res.y2 = fac1*(F + mp*siny*(l*w*w + g*cosy));
        #res.y4 = fac2*(-F*cosy -mp*l*w*w*cosy*siny -(mc+mp)*g*siny);
        totMass = mp + mc
        fac2 = l*(4./3. - mp*cosy*cosy/totMass)
        F1 = act + mp*l*w*w*siny
        wdot = (g*siny - F1*cosy/totMass)/fac2
        vdot = (F1 - mp*l*wdot*cosy)/totMass
        return [v, vdot, w, wdot]

    def advance(self, action):
        self.F = action[0]
        self.ODE.set_initial_value(self.u, self.t).set_f_params(self.F)
        self.u = self.ODE.integrate(self.t + self.dt*self.nsteps)
        self.t = self.t + self.dt*self.nsteps
        self.step = self.step + self.nsteps
        if self.isOver(): return 1
        else: return 0

    def getState(self):
        state = np.copy(self.u)
        state.resize(6)
        state[4] = np.cos(self.u[2])
        state[5] = np.sin(self.u[2])
        return state

    def getReward(self):
        #double angle = std::fmod(u.y3, 2*M_PI); #swingup
        #angle = angle<0 ? angle+2*M_PI : angle;
        #return fabs(angle-M_PI)<M_PI/6 ? 1 : 0;
        return 1 -(abs(self.u[2])>np.pi/15 or abs(self.u[0])>2.4 );

if __name__ == '__main__':
    # state is x, v, angle, omega, cos(angle), sin(angle), action is Fx
    comm = Communicator(6, 1)
    env = CartPole()
    #Here, the action space is in [-10, 10] and is bounded.
    comm.set_action_scales(10*np.ones(1), -10*np.ones(1), bounded=True)
    #Agent knows cos and sin of angle, angle itself is hidden
    bObservable = np.ones(6)
    bObservable[2] = 0
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

            if env.isTruncated():
                # Episodes that are truncated due to time-limits are treated
                # differently from terminal states. Without time limits, the
                # agent could continue the episode following the current policy
                # (i.e. policy did not cause a failure condition). Therefore
                # the value of the last state s_T in a truncated ep. is not 0
                # (the value of a terminal state is zero by definition) but is
                # the expected on-pol returns from s_T (ie. V^\pi (s_T) )
                comm.truncateSeq(state, reward);
                break
            elif terminated:  #tell smarties that this is a terminal state
                comm.sendTermState(state, reward);
                break
            else: comm.sendState(state, reward); # normal state
