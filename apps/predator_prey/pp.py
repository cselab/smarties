
import numpy as np
from Communicator import Communicator
import matplotlib.pyplot as plt
import time

EXTENT = 1
maxStep = 500
nQuadrants = 8
velMagnitude = 0.02*EXTENT

class Entity():
    def reset(self):
        self.x = np.random.uniform(0, 1)
        self.y = np.random.uniform(0, 1)
        self.actScal = 1
        self.background = np.random.normal(0, 1, nQuadrants)

    def advance(self, act):
        assert(act.size == 2)
        self.actScal = np.sqrt(act[0]*act[0] + act[1]*act[1]) / self.maxVel
        if self.actScal > 1:
          self.x += act[0] * self.maxVel / self.actScal
          self.y += act[1] * self.maxVel / self.actScal
          self.actScal = 1
        else:
          self.x += act[0]
          self.y += act[1]

        if (self.x >= EXTENT): self.x -= EXTENT;
        if (self.x <  0):      self.x += EXTENT;
        if (self.y >= EXTENT): self.y -= EXTENT;
        if (self.y <  0):      self.y += EXTENT;

    def getQuadrant(self, other):
        relX = other.x - self.x;
        relY = other.y - self.y;
        relA = np.arctan2(relY, relX) + np.pi; # between 0 and 2pi
        assert(relA >= 0 and relA <= 2*np.pi);
        return int(nQuadrants*relA/(2*np.pi + 2.2e-16))

    def __init__(self, maxVelFac = 1):
        self.maxVel = velMagnitude * maxVelFac

class Prey(Entity):
    def getState(self, other):
        noise = np.random.normal(0, 1, nQuadrants)
        self.background = (1-self.actScal)*self.background + self.actScal*noise
        state = self.background
        quadEnemy = self.getQuadrant(other)
        state[quadEnemy] = max(1, state[quadEnemy])
        return state

    def getReward(self, other):
        relX = other.x - self.x
        relY = other.y - self.y
        return np.sqrt(relX*relX + relY*relY);

    def __init__(self):
        Entity.__init__(self)

class Predator(Entity):
    def getState(self, other):
        state = np.zeros(nQuadrants)
        quadEnemy = self.getQuadrant(other)
        state[quadEnemy] = 1
        return state

    def getReward(self, other):
        relX = other.x - self.x
        relY = other.y - self.y
        return -np.sqrt(relX*relX + relY*relY);

    def __init__(self):
        Entity.__init__(self, maxVelFac = 0.5)

class Plotter:
    def __init__(self):
        self.PyX = np.zeros(maxStep)
        self.PyY = np.zeros(maxStep)
        self.PdX = np.zeros(maxStep)
        self.PdY = np.zeros(maxStep)
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.liney, = self.ax.plot(self.PyX, self.PyY, 'b-')
        self.lined, = self.ax.plot(self.PdX, self.PdY, 'r-')

    def update(self, step, pred, prey):
        self.PyX[step:maxStep] = prey.x
        self.PyY[step:maxStep] = prey.y
        self.PdX[step:maxStep] = pred.x
        self.PdY[step:maxStep] = pred.y
        self.liney.set_xdata(self.PyX)
        self.lined.set_xdata(self.PdX)
        self.liney.set_ydata(self.PyY)
        self.lined.set_ydata(self.PdY)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #self.ax.clear()
        #self.ax.plot(self.PyX, self.PyY, 'b-')
        #self.ax.plot(self.PdX, self.PdY, 'r-')
        #plt.draw()
        # time.sleep(0.0001)

comm = Communicator(nQuadrants, 2, 2)
pred = Predator()
prey = Prey()
plot = Plotter()

while(True):
    pred.reset()
    prey.reset()
    comm.sendInitState(pred.getState(prey), agent_id=0)
    comm.sendInitState(prey.getState(pred), agent_id=1)

    step = 0

    while (True):
      plot.update(step, pred, prey)
      pred.advance(comm.recvAction(0))
      prey.advance(comm.recvAction(1))
      if(step < maxStep):
        comm.sendState(  pred.getState(prey), pred.getReward(prey), agent_id=0);
        comm.sendState(  prey.getState(pred), prey.getReward(pred), agent_id=1);
      else:
        comm.truncateSeq(pred.getState(prey), pred.getReward(prey), agent_id=0);
        comm.truncateSeq(prey.getState(pred), prey.getReward(pred), agent_id=1);
        break
      step += 1
