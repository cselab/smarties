#!/usr/bin/env python3
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import gym, sys, os, numpy as np
#import pybullet_envs
#import pybulletgym
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
import smarties as rl
from HumanoidWrapper import HumanoidWrapper

def getAction(comm, env):
  buf = comm.recvAction()
  if   hasattr(env.action_space, 'n'):
    action = int(buf[0])
  elif hasattr(env.action_space, 'spaces'):
    action = [int(buf[0])]
    for i in range(1, comm.nActions): action = action + [int(buf[i])]
  elif hasattr(env.action_space, 'shape'):
    action = buf
  else: assert(False)
  return action

def setupSmartiesCommon(comm, task):
  env = gym.make(task)

  ## setup MDP properties:
  # first figure out dimensionality of state
  dimState = 1
  if hasattr(env.observation_space, 'shape'):
    for i in range(len(env.observation_space.shape)):
      dimState *= env.observation_space.shape[i]
  elif hasattr(env.observation_space, 'n'):
    dimState = 1
  else: assert(False)

  # then figure out action dims and details
  if hasattr(env.action_space, 'spaces'):
    dimAction = len(env.action_space.spaces)
    comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
    control_options = dimAction * [0]
    for i in range(dimAction):
      control_options[i] = env.action_space.spaces[i].n
    comm.setActionOptions(control_options, 0) # agent 0
  elif hasattr(env.action_space, 'n'):
    dimAction = 1
    comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
    comm.setActionOptions(env.action_space.n, 0) # agent 0
  elif hasattr(env.action_space, 'shape'):
    dimAction = env.action_space.shape[0]
    comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
    isBounded = dimAction * [True]
    comm.setActionScales(env.action_space.high, env.action_space.low, isBounded, 0)
  else: assert(False)

  return env

def app_main(comm):
  task = sys.argv[1]
  print("openAI environment: ", task)
  if task == 'Humanoid-v2' or task == 'HumanoidStandup-v2':
    env = HumanoidWrapper(comm, task)
  else:
    env = setupSmartiesCommon(comm, task)

  while True: #training loop
    observation = env.reset()
    t = 0
    comm.sendInitState(observation)
    while True: # simulation loop
      action = getAction(comm, env) #receive action from smarties
      observation, reward, done, info = env.step(action)
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
