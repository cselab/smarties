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
from Communicator import Communicator
import cv2
cv2.ocl.setUseOpenCL(False)

class Communicator_atari(Communicator):
    def get_env(self):
        assert(self.env is not None)
        return self.env

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
        #    self.buffer[self.buffIND()], _, done, _ = self.env.step(1)
        #    if done: self.base_reset()
        #    self.buffer[self.buffIND()], _, done, _ = self.env.step(2)
        #    if done: self.base_reset()

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
        self.start_server()
        self.sent_stateaction_info = False
        self.number_of_agents = 1
        self.noop_max = 30
        self.nSkip = 4
        self.nPool = 3
        self.buffI = 0
        self.lives = 0
        self.was_real_done = True
        print("openAI environment: ", sys.argv[2])
        env = gym.make(sys.argv[2]+'NoFrameskip-v4')
        self.obsShape = env.observation_space.shape
        self.buffer = np.zeros((self.nPool,)+self.obsShape, dtype=np.uint8)
        nAct, nObs = 1, 84 * 84
        actVals = np.zeros([0], dtype=np.float64)
        actOpts = np.zeros([0], dtype=np.float64)
        obsBnds = np.zeros([0], dtype=np.float64)

        assert( hasattr(env.action_space, 'n') )
        assert( env.unwrapped.get_action_meanings()[0] == 'NOOP' )
        nActions_i = env.action_space.n
        self.discrete_actions = True
        actOpts = np.append(actOpts, [nActions_i, 1])
        actVals = np.append(actVals, np.arange(0,nActions_i)+.1)

        for i in range(nObs): obsBnds = np.append(obsBnds, [255,0])

        self.obs_in_use = np.ones(nObs, dtype=np.float64)
        self.nActions, self.nStates = nAct, nObs
        self.observation_bounds = obsBnds
        self.action_options = actOpts
        self.action_bounds = actVals
        self.send_stateaction_info()
        #if self.bRender==3:
        #    env = gym.wrappers.Monitor(env, './', force=True)
        self.env = env
        self.seq_id, self.frame_id = 0, 0
        self.seed = sys.argv[1]
        self.actionBuffer = [np.zeros([nAct], dtype=np.float64)]


if __name__ == '__main__':
    comm = Communicator_atari() # create communicator with smarties

    while True: #training loop
        observation = comm.env_reset()

        #send initial state
        comm.sendInitState(observation)

        while True: # simulation loop
            #receive action from smarties
            buf = comm.recvAction()
            #advance the environment
            observation, reward, done, info = comm.env_step(int(buf[0]))
            #send the observation to smarties
            comm.sendState(observation, reward, terminal=done)
            if done: break
