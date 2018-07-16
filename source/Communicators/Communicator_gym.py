#!/usr/bin/env python
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import gym, sys, socket, os, os.path, time
from gym import wrappers
import numpy as np
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
from Communicator import Communicator

class Communicator_gym(Communicator):
    def get_env(self):
        assert(self.gym is not None)
        return self.gym

    def __init__(self):
        self.start_server()
        self.sent_stateaction_info = False
        self.discrete_actions = False
        self.number_of_agents = 1
        print("openAI environment: ", sys.argv[2])
        env = gym.make(sys.argv[2])
        nAct, nObs = 1, 1
        actVals = np.zeros([0], dtype=np.float64)
        actOpts = np.zeros([0], dtype=np.float64)
        obsBnds = np.zeros([0], dtype=np.float64)
        if hasattr(env.action_space, 'spaces'):
            nAct = len(env.action_space.spaces)
            self.discrete_actions = True
            for i in range(nAct):
                nActions_i = env.action_space.spaces[i].n
                actOpts = np.append(actOpts, [nActions_i+.1, 1])
                actVals = np.append(actVals, np.arange(0,nActions_i)+.1)
        elif hasattr(env.action_space, 'n'):
            nActions_i = env.action_space.n
            self.discrete_actions = True
            actOpts = np.append(actOpts, [nActions_i, 1])
            actVals = np.append(actVals, np.arange(0,nActions_i)+.1)
        elif hasattr(env.action_space, 'shape'):
            nAct = env.action_space.shape[0]
            for i in range(nAct):
                bounded = 0 #figure out if environment is strict about the bounds on action:
                test = env.reset()
                test_act = 0.5*(env.action_space.low + env.action_space.high)
                test_act[i] = env.action_space.high[i]+1
                try: test = env.step(test_act)
                except: bounded = 1.1
                env.reset()
                actOpts = np.append(actOpts, [2.1, bounded])
                if bounded:
                  actVals = np.append(actVals, max(env.action_space.low[i],-1e3))
                  actVals = np.append(actVals, min(env.action_space.high[i],1e3))
                else:
                  actVals = np.append(actVals, env.action_space.high[i])
                  actVals = np.append(actVals, env.action_space.low[i] )
                  #actVals = np.append(actVals, 1)
                  #actVals = np.append(actVals, -1)
        else: assert(False)

        if hasattr(env.observation_space, 'shape'):
            for i in range(len(env.observation_space.shape)):
                nObs *= env.observation_space.shape[i]

            for i in range(nObs):
                if(env.observation_space.high[i]<1e3 and env.observation_space.low[i]>-1e3):
                    obsBnds = np.append(obsBnds, env.observation_space.high[i])
                    obsBnds = np.append(obsBnds, env.observation_space.low[i])
                else: #no scaling
                    obsBnds = np.append(obsBnds, [1, -1])
        elif hasattr(env.observation_space, 'n'):
            obsBnds = np.append(obsBnds, env.observation_space.n)
            obsBnds = np.append(obsBnds, 0)
        else: assert(False)

        self.obs_in_use = np.ones(nObs, dtype=np.float64)
        self.nActions, self.nStates = nAct, nObs
        self.observation_bounds = obsBnds
        self.action_options = actOpts
        self.action_bounds = actVals
        self.send_stateaction_info()
        #if self.bRender==3:
        #    env = gym.wrappers.Monitor(env, './', force=True)
        self.gym = env
        self.seq_id, self.frame_id = 0, 0
        self.seed = sys.argv[1]
        self.actionBuffer = [np.zeros([nAct], dtype=np.float64)]


if __name__ == '__main__':
    comm = Communicator_gym() # create communicator with smarties
    env = comm.get_env() #
    #env.seed(comm.seed)

    while True: #training loop
        observation = env.reset()
        t = 0
        #send initial state
        comm.sendInitState(observation)
        #print(t, observation)

        while True: # simulation loop
            #receive action from smarties
            buf = comm.recvAction()
            if hasattr(env.action_space, 'n'):        action = int(buf[0])
            elif hasattr(env.action_space, 'shape'):  action = buf
            elif hasattr(env.action_space, 'spaces'):
                action = [int(buf[0])]
                for i in range(1, comm.nActions): action = action+[int(buf[i])]
            else: assert(False)
            #advance the environment
            observation, reward, done, info = env.step(action)
            t = t + 1
            #send the observation to smarties
            #print(t, done, env._max_episode_steps)
            sys.stdout.flush()
            if done == True and t >= env._max_episode_steps:
              comm.truncateSeq(observation, reward)
            else:
              comm.sendState(observation, reward, terminal=done)
            #print(t, observation, action, reward, done)
            if done: break
