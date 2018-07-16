//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#pragma once
#include "StateAction.h"
#include "Settings.h"
#include "Communicators/Communicator.h"
#include <atomic>
#define OUTBUFFSIZE 65536
class Agent
{
protected:
  const StateInfo&  sInfo;
  const ActionInfo& aInfo;

public:
  State  sOld =  State(sInfo); // previous state
  State  s    =  State(sInfo); // current state
  // Action performed by agent. Updated by Learner::select and sent to Slave
  Action a    = Action(aInfo);
  Real r = 0;              // current reward
  Real cumulative_rewards = 0;
  const int ID;
  // status of agent's episode. 1: initial; 0: middle; 2: terminal; 3: truncated
  int Status = 1;
  int transitionID = 0;

  // for dumping to state-action-reward-policy binary log (writeBuffer):
  mutable float buf[OUTBUFFSIZE];
  mutable std::atomic<Uint> buffCnter {0};

  Agent(const int _ID, const StateInfo& _sInfo, const ActionInfo& _aInfo) :
    sInfo(_sInfo), aInfo(_aInfo), ID(_ID) { }

  void writeBuffer(const int rank) const
  {
    if(buffCnter == 0) return;
    char cpath[256];
    sprintf(cpath, "obs_rank%02d_agent%03d.raw", rank, ID);
    FILE * pFile = fopen (cpath, "ab");

    fwrite (buf, sizeof(float), buffCnter, pFile);
    fflush(pFile); fclose(pFile);
    buffCnter = 0;
  }

  void writeData(const int rank, const Rvec mu) const
  {
    // possible race conditions, avoided by the fact that each worker
    // (and therefore agent) can only be handled by one thread at the time
    // atomic op is to make sure that counter gets flushed to all threads
    const Uint writesize = 3 +sInfo.dim +aInfo.dim +mu.size();
    if(OUTBUFFSIZE<writesize) die("Edit compile-time OUTBUFFSIZE variable.");
    assert( buffCnter % writesize == 0 );
    if(buffCnter+writesize > OUTBUFFSIZE) writeBuffer(rank);
    Uint ind = buffCnter;
    buf[ind++] = Status + 0.1;
    buf[ind++] = transitionID + 0.1;

    for (Uint i=0; i<sInfo.dim; i++) buf[ind++] = (float) s.vals[i];
    for (Uint i=0; i<aInfo.dim; i++) buf[ind++] = (float) a.vals[i];
    buf[ind++] = r;
    for (Uint i=0; i<mu.size(); i++) buf[ind++] = (float) mu[i];

    buffCnter += writesize;
    assert(buffCnter == ind);
  }

  inline void getState(State& _s) const
  {
    _s = s;
  }

  inline void setState(State& _s)
  {
    s = _s;
  }

  inline void swapStates()
  {
    std::swap(s.vals, sOld.vals);
  }

  inline void getAction(Action& _a) const
  {
    _a = a;
  }

  inline void getOldState(State& _s) const
  {
    _s = sOld;
  }

  inline void act(Action& _a)
  {
    a = _a;
  }

  template<typename T>
  inline void act(const T action)
  {
    a.set(action);
  }

  inline void copyAct(double * const ary) const
  {
    for(Uint j=0; j<aInfo.dim; j++) ary[j] = a.vals[j];
  }

  inline int getStatus() const
  {
    return Status;
  }

  inline Real getReward() const
  {
    return r;
  }

  inline void reset()
  {
    Status = 1; transitionID=0; cumulative_rewards=0; r=0;
  }

  template<typename T>
  void update(const envInfo _i, const vector<T>& _s, const double _r)
  {
    if(_i == FAIL_COMM) {
      cumulative_rewards = 0; transitionID = 0; r = 0;
      return;
    }
    Status = _i;
    swapStates(); //swap sold and snew
    s.set(_s);
    r = _r;
    if(_i == INIT_COMM) {
      cumulative_rewards = 0;
      transitionID = 0;
    }
    else {
      cumulative_rewards += _r;
      transitionID++;
    }
  }
};
