//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "RACER.h"

#ifndef ADV_QUAD
//#include "../Math/Mixture_advantage_gaus.h"
#include "../Math/Gaus_advantage.h"
#else
//#include "../Math/Mixture_advantage_quad.h"
#include "../Math/Quadratic_advantage.h"
#endif
#include "../Math/Discrete_advantage.h"
#include "../Math/Zero_advantage.h"

#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/Collector.h"

#include "RACER_common.cpp"
#include "RACER_train.cpp"

namespace smarties
{

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
select(Agent& agent)
{
  data_get->add_state(agent);
  const Approximator* const NET = networks[0];
  Sequence& EP = * data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(&EP);
  NET->load(MB, agent, 0);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    const Rvec output = NET->forward(agent);
    auto pol = prepare_policy<Policy_t>(output);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    const bool bSamplePolicy = settings.explNoise>0 && agent.trackSequence;
    auto act = pol.finalize(bSamplePolicy, &generators[nThreads+agent.ID], mu);
    const auto adv = prepare_advantage<Advantage_t>(output, &pol);
    const Real advantage = adv.computeAdvantage(pol.sampAct);
    EP.action_adv.push_back(advantage);
    EP.state_vals.push_back(output[VsID]);
    agent.act(act);
    data_get->add_action(agent, mu);

    #ifndef NDEBUG
      auto dbg = prepare_policy<Policy_t>(output);
      const Rvec & ACT = EP.actions.back(), & MU = EP.policies.back();
      dbg.prepare(ACT, MU);
      const double err = fabs(dbg.sampImpWeight-1);
      if(err>1e-10) _die("Imp W err %20.20e", err);
    #endif
  }
  else // either terminal or truncation state
  {
    if( agent.agentStatus == TRNC ) {
      Rvec output = NET->forward(agent);
      EP.state_vals.push_back(output[VsID]); // not a terminal state
    } else {
      EP.state_vals.push_back(0); //value of terminal state is 0
    }
    //whether seq is truncated or terminated, act adv is undefined:
    EP.action_adv.push_back(0);
    const Uint N = EP.nsteps();
    // compute initial Qret for whole trajectory:
    assert(N == EP.action_adv.size());
    assert(N == EP.state_vals.size());
    assert(0 == EP.Q_RET.size());
    //within Retrace, we use the Q_RET vector to write the Adv retrace values
    EP.Q_RET.resize(N, 0);
    EP.offPolicImpW.resize(N, 1);
    for(Uint i=EP.ndata(); i>0; --i) backPropRetrace(EP, i);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::setupTasks(TaskQueue& tasks)
{
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization

  auto stepInit = [&]()
  {
    // conditions to start the initialization task:
    if ( algoSubStepID >= 0 ) return; // we done with init
    if ( data->readNData() < nObsB4StartTraining ) return; // not enough data to init

    debugL("Initialize Learner");
    initializeLearner();
    algoSubStepID = 0;
  };
  tasks.add(stepInit);

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data

    debugL("Sample the replay memory and compute the gradients");
    spawnTrainTasks();
    debugL("Gather gradient estimates from each thread and Learner MPI rank");
    prepareGradient();
    debugL("Search work to do in the Replay Memory");
    processMemoryBuffer(); // find old eps, update avg quantities ...
    debugL("Update Retrace est. for episodes sampled in prev. grad update");
    updateRetraceEstimates();
    debugL("Compute state/rewards stats from the replay memory");
    finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev
    logStats();
    algoSubStepID = 1;
  };
  tasks.add(stepMain);

  // these are all the tasks I can do before the optimizer does an allreduce
  auto stepComplete = [&]()
  {
    if ( algoSubStepID not_eq 1 ) return;
    if ( networks[0]->ready2ApplyUpdate() == false ) return;

    debugL("Apply SGD update after reduction of gradients");
    applyGradient();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
  };
  tasks.add(stepComplete);
}

////////////////////////////////////////////////////////////////////////////

template class RACER<Discrete_advantage, Discrete_policy, Uint>;
template class RACER<Param_advantage, Gaussian_policy, Rvec>;
template class RACER<Zero_advantage, Gaussian_policy, Rvec>;
//template class RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;

}
