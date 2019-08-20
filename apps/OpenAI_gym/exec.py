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
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
import smarties as rl

def getAction(comm, env):
  buf = comm.recvAction()
  if   hasattr(env.action_space, 'n'):
    action = int(buf[0])
  elif hasattr(env.action_space, 'spaces'):
    action = [int(buf[0])]
    for i in range(1, comm.nActions): action = action+[int(buf[i])]
  elif hasattr(env.action_space, 'shape'):
    action = buf
  else: assert(False)
  return action

def constructCommunicator(env):
  # first figure out dimensionality of state
  dimState = 1
  if hasattr(env.observation_space, 'shape'):
    for i in range(len(env.observation_space.shape)):
      dimState *= env.observation_space.shape[i]
  elif hasattr(env.observation_space, 'n'):
    dimState = 1
  else: assert(False)

  # then figure out action dims and details
  comm = None
  if hasattr(env.action_space, 'spaces'):
    dimAction = len(env.action_space.spaces)
    comm = Communicator(dimState, dimAction, 1) # 1 agent
    control_options = dimAction * [0] 
    for i in range(dimAction):
      control_options[i] = env.action_space.spaces[i].n
    comm.set_action_options(control_options, 0) # agent 0
  elif hasattr(env.action_space, 'n'):
    dimAction = 1
    comm = Communicator(dimState, dimAction, 1) # 1 agent
    comm.set_action_options(env.action_space.n, 0) # agent 0
  elif hasattr(env.action_space, 'shape'):
    dimAction = env.action_space.shape[0]
    comm = Communicator(dimState, dimAction, 1) # 1 agent
    upprScale = dimAction * [0.0]
    lowrScale = dimAction * [0.0]
    isBounded = dimAction * [False]
    for i in range(dimAction):
      test = env.reset()
      test_act = 0.5*(env.action_space.low + env.action_space.high)
      test_act[i] = env.action_space.high[i]+1
      try: test = env.step(test_act)
      except: isBounded[i] = True
      assert(env.action_space.high[i]< 1e6) # make sure that values
      assert(env.action_space.low[i] >-1e6) # make sense
      upprScale[i] = env.action_space.high[i]
      lowrScale[i] = env.action_space.low[i]
    comm.set_action_scales(upprScale, lowrScale, isBounded, 0)
  else: assert(False)

  return comm


def app_main(comm):
  print("openAI environment: ", sys.argv[1])
  env = gym.make(sys.argv[1])
  comm = constructCommunicator(env) # create communicator with smarties

  #fig = plt.figure()
  while True: #training loop
    observation = env.reset()
    t = 0
    comm.sendInitState(observation)
    while True: # simulation loop
      action = getAction(comm, env) #receive action from smarties
      observation, reward, done, info = env.step(action)
      #if t>0 : env.env.viewer_setup()
      #img = env.render(mode='rgb_array')
      #img = plt.imshow(img)
      #fig.savefig('frame%04d.png' % t)
      t = t + 1
      if done == True and t >= env._max_episode_steps:
        comm.sendLastState(observation, reward)
      elif done == True:
        comm.sendTermState(observation, reward)
      else: comm.sendState(observation, reward)
      if done: break

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )