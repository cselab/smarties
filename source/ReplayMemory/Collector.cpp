//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Collector.h"
#include "../Utils/FunctionUtilities.h"
#include "DataCoordinator.h"
#include <algorithm>

namespace smarties
{

Collector::Collector(MemoryBuffer*const RM, DataCoordinator*const C) :
replay(RM), sharing(C)
{
  inProgress.resize(distrib.nAgents);
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
void Collector::add_state(Agent&a)
{
  assert(a.ID < inProgress.size());
  //assert(replay->MDP.localID == a.localID);
  Episode & S = inProgress[a.ID];
  const Fvec storedState = a.getObservedState<Fval>();

  if(a.trackEpisodes == false) {
    // contain only one state and do not add more. to not store rewards either
    // RNNs then become automatically not supported because no time series!
    // (this is accompained by check in approximator)
    S.states  = std::vector<Fvec>{ storedState };
    S.rewards = std::vector<Real>{ (Real) 0 };
    S.actions.clear();      S.policies.clear();     S.latent_states.clear();
    S.SquaredError.clear(); S.offPolicImpW.clear(); S.KullbLeibDiv.clear();
    S.state_vals.clear();   S.action_adv.clear();   S.Q_RET.clear();

    a.agentStatus = INIT; // one state stored, lie to avoid catching asserts
    assert(S.agentID == -1 && "Untracked sequences are not tagged to agent");
    return;
  }

  // assign or check id of agent generating episode
  if (a.agentStatus == INIT) S.agentID = a.localID;
  else assert(S.agentID == a.localID);

  // if no tuples, init state. if tuples, cannot be initial state:
  assert( (S.nsteps() == 0) == (a.agentStatus == INIT) );
  #ifndef NDEBUG // check that last new state and new old state are the same
    if( S.nsteps() ) {
      bool same = true;
      const Fvec vecSold = a.getObservedOldState<Fval>();
      const Fvec memSold = S.states.back();
      static constexpr Fval fEPS = std::numeric_limits<Fval>::epsilon();
      for (Uint i=0; i<vecSold.size() && same; ++i) {
        auto D = std::max({std::fabs(memSold[i]), std::fabs(vecSold[i]), fEPS});
        same = same && std::fabs(memSold[i]-vecSold[i])/D < 100*fEPS;
      }
      //debugS("Agent %s and %s",print(vecSold).c_str(),print(memSold).c_str());
      if (!same) _die("Unexpected termination of EP a %u step %u seqT %lu\n",
        a.ID, a.timeStepInEpisode, S.nsteps());
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  //env->pickReward(a);
  S.ended = a.agentStatus == TERM;
  S.states.push_back(storedState);
  S.latent_states.push_back( a.getLatentState<Fval>() );
  S.rewards.push_back(a.reward);
  if( a.agentStatus not_eq INIT ) S.totR += a.reward;
  else assert(std::fabs(a.reward)<2.2e-16); //rew for init state must be 0
}

// Once network picked next action, call this method
void Collector::add_action(const Agent& a, const Rvec pol)
{
  assert(pol.size() == aI.dimPol());
  assert(a.agentStatus < TERM);
  if(a.trackEpisodes == false) {
    // do not store more stuff in sequence but also do not track data counter
    inProgress[a.ID].actions = std::vector<Rvec>{ a.action };
    inProgress[a.ID].policies = std::vector<Rvec>{ pol };
    return;
  }

  if(a.agentStatus not_eq INIT) nSeenTransitions_loc ++;
  inProgress[a.ID].actions.push_back( a.action );
  inProgress[a.ID].policies.push_back(pol);
}

// If the state is terminal, instead of calling `add_action`, call this:
void Collector::terminate_seq(Agent&a)
{
  assert(a.agentStatus >= TERM);
  if(a.trackEpisodes == false) return; // do not store seq
  // fill empty action and empty policy: last step of episode never has actions
  const Rvec dummyAct = Rvec(aI.dim(), 0), dummyPol = Rvec(aI.dimPol(), 0);
  a.resetActionNoise();
  a.setAction(dummyAct);
  inProgress[a.ID].actions.push_back( dummyAct );
  inProgress[a.ID].policies.push_back( dummyPol );

  push_back(a.ID);
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
void Collector::push_back(const size_t agentId)
{
  assert(agentId < inProgress.size());
  if(inProgress[agentId].nsteps() < 2) die("Seq must at least have s0 and sT");

  inProgress[agentId].finalize( nSeenEpisodes_loc.load() );

  if(sharing->bRunParameterServer)
  {
    // Check whether this agent is last agent of environment to call terminate.
    // This is only relevant in the case of learners on workers who receive
    // params from master, therefore bool can be incorrect in all other cases.
    // It only matters if multiple agents in env belong to same learner.
    // To ensure thread safety, we must use mutex and check that this agent is
    // last to reset the sequence.
    assert(distrib.bIsMaster == false);
    bool fullEnvReset = true;
    std::unique_lock<std::mutex> lock(envTerminationCheck);
    for(Uint i=0; i<inProgress.size(); ++i){
      //printf("%lu ", inProgress[i].nsteps());
      if (i == agentId or inProgress[i].agentID < 0) continue;
      fullEnvReset = fullEnvReset && inProgress[i].nsteps() == 0;
    }
    //printf("\n");
    Episode EP = std::move(inProgress[agentId]);
    assert(inProgress[agentId].nsteps() == 0);
    //Unlock with empy inProgress, such that next agent can check if it is last.
    lock.unlock();
    sharing->addComplete(EP, fullEnvReset);
  }
  else
  {
    sharing->addComplete(inProgress[agentId], true);
    assert(inProgress[agentId].nsteps() == 0);
  }

  nSeenTransitions_loc++;
  nSeenEpisodes_loc++;
}

Collector::~Collector() {}

}
