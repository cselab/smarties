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
  Episode& EP = data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(EP);
  NET->load(MB, agent, 0);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    const Rvec output = NET->forward(agent);
    const Policy_t pol(pol_start, aInfo, output);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto action = pol.selectAction(agent, distrib.bTrain);

    const Advantage_t adv(adv_start, aInfo, output, &pol);
    const Real advantage = adv.computeAdvantage(action);
    EP.action_adv.push_back(advantage);
    EP.state_vals.push_back(output[VsID]);
    agent.setAction(action);
    data_get->add_action(agent, mu);

    #ifndef NDEBUG
      const Policy_t dbg(pol_start, aInfo, output);
      Real impW = dbg.importanceWeight(EP.actions.back(), EP.policies.back());
      Real dkl = dbg.KLDivergence(EP.policies.back());
      if(std::fabs(impW-1)>nnEPS || dkl>nnEPS) _die("ImpW:%e DKL:%e",impW,dkl);
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
    assert(N == EP.action_adv.size() && N == EP.state_vals.size());
    assert(0 == EP.Q_RET.size());
    //within Retrace, we use the Q_RET vector to write the Adv retrace values
    EP.Q_RET.resize(N, 0);
    EP.offPolicImpW.resize(N, 1);
    for(Uint i=EP.ndata(); i>0; --i)
        EP.propagateRetrace(i, gamma, data->scaledReward(EP, i));

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
    profiler->start("DATA");
  };
  tasks.add(stepInit);

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data

    profiler->stop();
    debugL("Sample the replay memory and compute the gradients");
    spawnTrainTasks();
    if(ESpopSize>1) {
      debugL("Compute objective function for CMA optimizer.");
      prepareCMALoss();
    }
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
    profiler->start("MPI");
  };
  tasks.add(stepMain);

  // these are all the tasks I can do before the optimizer does an allreduce
  auto stepComplete = [&]()
  {
    if ( algoSubStepID not_eq 1 ) return;
    if ( networks[0]->ready2ApplyUpdate() == false ) return;

    profiler->stop();
    debugL("Apply SGD update after reduction of gradients");
    applyGradient();
    globalGradCounterUpdate(); // step ++
    algoSubStepID = 0; // rinse and repeat
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

////////////////////////////////////////////////////////////////////////////

template class RACER<Discrete_advantage, Discrete_policy, Uint>;
template class RACER<Param_advantage, Continuous_policy, Rvec>;
template class RACER<Zero_advantage, Continuous_policy, Rvec>;
//template class RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;

}
