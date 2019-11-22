#!/usr/bin/env python3
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import gym, sys, os, numpy as np, cv2
from gym import wrappers
cv2.ocl.setUseOpenCL(False)
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
import smarties as rl

class atariWrapper():
  def base_reset(self):
    self.buffI = 1
    self.buffer = np.zeros((self.nPool,)+self.obsShape, dtype=np.uint8)
    self.buffer[0] = self.env.reset()

  def buffIND(self):
    ret = self.buffI
    self.buffI = ( self.buffI+1 ) % self.nPool
    return ret

  def base_step(self, act):
    total_reward, done = 0.0, None
    for i in range(self.nSkip):
      self.buffer[self.buffIND()], reward, done, info = self.env.step(act)
      total_reward += reward
      if done: break
    # Note that the observation on the done=True frame doesn't matter
    return total_reward, done, info

  def noop_reset(self):
    self.base_reset();
    #if(self.env.unwrapped.get_action_meanings()[1] == 'FIRE'):
    #  self.buffer[self.buffIND()], _, done, _ = self.env.step(1)
    #  if done: self.base_reset()
    #  self.buffer[self.buffIND()], _, done, _ = self.env.step(2)
    #  if done: self.base_reset()

    for _ in range(np.random.randint(1,self.noop_max+1)): # 0 is noop action
      self.buffer[self.buffIND()], _, done, _ = self.env.step(0)
      if done: self.base_reset()

  def life_reset(self):
    if self.was_real_done: self.noop_reset()
    else: # no-op step to advance from terminal/lost life state
      self.buffer[self.buffIND()], _, _, _ = self.env.step(0)
    self.lives = self.env.unwrapped.ale.lives()

  def life_step(self, action):
    reward, done, info = self.base_step(action)
    self.was_real_done = done
    lives = self.env.unwrapped.ale.lives()
    if lives < self.lives and lives > 0:
      # for Qbert sometimes we stay in lives == 0 for a few frames
      # so its important to keep lives > 0, so that we only reset once
      # the environment advertises done.
      done = True
    self.lives = lives
    return reward, done, info

  def env_step(self, action):
    reward, done, info = self.life_step(action)
    obs = cv2.cvtColor(self.buffer.max(axis=0), cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs[:, :, None].ravel(), reward, done, info

  def env_reset(self):
    self.life_reset()
    obs = cv2.cvtColor(self.buffer.max(axis=0), cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs[:, :, None].ravel()

  def __init__(self):
    self.noop_max = 30
    self.nSkip = 4
    self.nPool = 3
    self.buffI = 0
    self.lives = 0
    self.was_real_done = True
    self.env = gym.make(sys.argv[1]+'NoFrameskip-v4')
    assert( hasattr(self.env.action_space, 'n') )
    assert( self.env.unwrapped.get_action_meanings()[0] == 'NOOP' )
    self.obsShape = self.env.observation_space.shape
    self.buffer = np.zeros((self.nPool,)+self.obsShape, dtype=np.uint8)


def app_main(comm):
  print("openAI atari environment: ", sys.argv[1])
  env = atariWrapper()
  # state is a 84*84 grayscale (valued from 0 to 255) image
  comm.setStateActionDims(84*84, 1)
  upprScale, lowrScale = 84*84 * [255.0], 84*84 * [0.0]
  comm.setStateScales(upprScale, lowrScale, 0)
  # how many action options depends on the game:
  comm.setActionOptions(env.env.action_space.n)
  # define how obersation is preprocessed by approximators:
  # 1) chain together 4 observation
  comm.setNumAppendedPastObservations(3)
  # 2) add convolutional layers:
  #    input is image of size 84 * 84 * (1 + 3)
  comm.setPreprocessingConv2d(84, 84,  4, 32, 8, 4)
  #    then some math to figure out the input dim of next layers
  comm.setPreprocessingConv2d(20, 20, 32, 64, 4, 2)
  comm.setPreprocessingConv2d( 9,  9, 64, 64, 3, 1)

  while True: #training loop
    observation = env.env_reset()
    comm.sendInitState(observation) #send initial state

    while True: # simulation loop
      buf = comm.recvAction() #receive action from smarties
      #advance the environment
      observation, reward, done, info = env.env_step(int(buf[0]))
      if done: #send the observation to smarties
        comm.sendTermState(observation.ravel().tolist(), reward)
        break
      else: comm.sendState(observation.ravel().tolist(), reward)

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )