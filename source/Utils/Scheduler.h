//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Communicators/Communicator_internal.h"
#include <thread>
#include <mutex>
#include "../Learners/Learner.h"

class Master
{
private:
  const Settings& settings;
  Communicator_internal* const comm;
  const vector<Learner*> learners;
  const Environment* const env;


  const ActionInfo& aI = env->aI;
  const StateInfo&  sI = env->sI;
  const vector<Agent*>& agents = env->agents;
  const int nPerRank = env->nAgentsPerRank;
  const int bTrain = settings.bTrain;
  const int nWorkers_own = settings.nWorkers_own;
  const int nThreads = settings.nThreads;
  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;
  const Uint totNumSteps = settings.totNumSteps;

  Uint iterNum = 0; // no need to restart this one

  bool bNeedSequentialTasks = false;
  Profiler* profiler     = nullptr;

  mutable std::mutex dump_mutex;
  mutable std::vector<std::ostringstream> rewardsBuffer =
                                      std::vector<std::ostringstream>(nPerRank);

  std::atomic<Uint> bExit {0};
  std::vector<std::thread> worker_replies;

  inline Uint getMinSeqId() const {
    Uint lowest = learners[0]->nSeqsEval();
    for(size_t i=1; i<learners.size(); i++) {
      const Uint tmp = learners[i]->nSeqsEval();
      if(tmp<lowest) lowest = tmp;
    }
    // if agents share learning algo, return number of eps performed by env:
    if(learners.size() == 1) lowest /= nPerRank;
    return lowest;
  }
  inline Uint getMinStepId() const {
    Uint lowest = learners[0]->tStepsTrain();
    for(size_t i=1; i<learners.size(); i++) {
      const Uint tmp = learners[i]->tStepsTrain();
      if(tmp<lowest) lowest = tmp;
    }
    // if agents share learning algo, return number of turns performed by env:
    if(learners.size() == 1) lowest /= nPerRank;
    return lowest;
  }

  inline Learner* pickLearner(const Uint agentId, const Uint recvId)
  {
    assert(recvId < (Uint)nPerRank);
    if(learners.size()>1) assert(learners.size() == (Uint)nPerRank);

    assert( learners.size() == 1 || recvId < learners.size() );
    return learners.size()>1? learners[recvId] : learners[0];
  }

  inline bool learnersLockQueue() const
  {
    //When would a learning algo stop acquiring more data?
    //Off Policy algos:
    // - User specifies a ratio of observed trajectories to gradient steps.
    //    Comm is restarted or paused to maintain this ratio consant.
    //On Policy algos:
    // - if collected enough trajectories for current batch, then comm is paused
    //    untill gradient is applied (or nepocs are done), then comm restarts
    //    to obtain fresh on policy samples
    // Note:
    // - on policy traj. storage assumes that when agent reaches terminal state
    //    on a worker, all other agents on that worker must send their term state
    //    before sending any new initial state

    // However, no learner can stop others from getting data (vector of algos)
    bool locked = true;
    for(const auto& L : learners)
      locked = locked && L->blockDataAcquisition(); // if any wants to unlock...

    return locked;
  }

  inline bool learnersUnlockQueue() const
  {
    bool unlocked = true;
    for(const auto& L : learners)
      unlocked = unlocked && L->unblockGradStep(); // if any wants to unlock...
    return unlocked;
  }

  inline bool learnersInitialized() const
  {
    bool unlocked = true;
    for(const auto& L : learners)
      unlocked = unlocked && L->isReady4Init(); // if any wants to unlock...
    return unlocked;
  }

  void flushRewardBuffer();

  void dumpCumulativeReward(const int agent, const int worker,
    const unsigned giter, const unsigned tstep) const;

  void processWorker(const std::vector<int> workers);
  void processAgent(const int worker);

public:
  Master(Communicator_internal* const _c, const vector<Learner*> _l,
    Environment*const _e, Settings&_s);
  ~Master()
  {
    bExit = 1;
    for(size_t i=0; i<worker_replies.size(); i++) worker_replies[i].join();
    for(const auto& A : agents) A->writeBuffer(learn_rank);
    _dispose_object(env);
    _dispose_object(profiler);
    for(const auto& L : learners) _dispose_object(L);
    flushRewardBuffer();
  }

  void run();
};

class Worker
{
private:
  Communicator_internal* const comm;
  Environment* const env;
  const bool bTrain;
  vector<int> status;

public:
  Worker(Communicator_internal*const c, Environment*const e, Settings& s);
  ~Worker()
  {
    _dispose_object(env);
  }
  void run();
};

/*
class Client
{
private:
  Learner* const learner;
  Communicator* const comm;
  Environment* const env;
  vector<Agent*> agents;
  const ActionInfo aI;
  const StateInfo  sI;
  vector<int> status;
  void prepareState(int& iAgent, int& istatus, Real& reward);
  void prepareAction(const int iAgent);

public:
  Client(Learner*const l,Communicator*const c,Environment*const e,Settings&s);
  ~Client()
  {
    _dispose_object(env);
    _dispose_object(learner);
  }
  void run();
};
*/
