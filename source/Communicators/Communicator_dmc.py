#!/usr/bin/env python
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import sys, socket, os, os.path, time
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
from dm_control import suite
import numpy as np
from Communicator import Communicator

class Communicator_dmc(Communicator):
    def get_env(self):
        assert(self.dmc is not None)
        return self.dmc

    def __init__(self):
        self.start_server()
        self.sent_stateaction_info = False
        self.discrete_actions = False
        self.number_of_agents = 1
        print("DeepMind Suite environment: ", sys.argv[2], "task: ", sys.argv[3])
        env = suite.load(sys.argv[2], sys.argv[3])
        act_spec, obs_spec = env.action_spec(), env.observation_spec()
        nAct, nObs = act_spec.shape[0], 0
        for component in obs_spec.values():
            if len(component.shape): nObs = nObs + component.shape[0]
            else: nObs = nObs + 1

        actVals = np.zeros([0], dtype=np.float64)
        actOpts = np.zeros([0], dtype=np.float64)
        obsBnds = np.zeros([0], dtype=np.float64)

        for i in range(nAct):
            # assume all continuous envs with act space bounded in -1 and 1
            #actOpts = np.append(actOpts, [2.1, 0])
            actOpts = np.append(actOpts, [2.1, 1]) # bounded actions
            actVals = np.append(actVals, [-1,  1])

        for i in range(nObs):
            obsBnds = np.append(obsBnds, [1,  -1])

        print(nAct, nObs, actVals, actOpts, obsBnds)
        self.obs_in_use = np.ones(nObs, dtype=np.float64)
        self.nActions, self.nStates = nAct, nObs
        self.observation_bounds = obsBnds
        self.action_options = actOpts
        self.action_bounds = actVals
        self.send_stateaction_info()
        #if self.bRender==3:
        #    env = gym.wrappers.Monitor(env, './', force=True)
        self.dmc = env
        self.seq_id, self.frame_id = 0, 0
        self.seed = sys.argv[1]
        self.actionBuffer = [np.zeros([nAct], dtype=np.float64)]


if __name__ == '__main__':
    comm = Communicator_dmc() # create communicator with smarties
    env = comm.get_env()

    while True: #training loop
        t = env.reset()
        obsVec = np.zeros([0], dtype=np.float64)
        for oi in t.observation.values(): obsVec = np.append(obsVec, oi)
        #send initial state
        comm.sendInitState(obsVec)

        while True: # simulation loop
            #receive action from smarties
            action = comm.recvAction()
            #advance the environment
            t = env.step(action)
            obs, rew, step = t.observation, t.reward, t.step_type.value
            obsVec = np.zeros([0], dtype=np.float64)
            for oi in obs.values(): obsVec = np.append(obsVec, oi)
            #send the observation to smarties
            # DMC suite does not have term condition, just truncated seqs
            comm.sendState(obsVec, rew, truncated = t.last() )
            if t.last(): break
