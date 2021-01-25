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

#include "RACER_common.cpp"
#include "RACER_train.cpp"

namespace smarties
{

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
selectAction(const MiniBatch& MB, Agent& agent)
{
  networks[0]->load(MB, agent, 0);
  //Compute policy and value on most recent element of the sequence.
  const Rvec output = networks[0]->forward(agent);
  const Policy_t pol(pol_start, aInfo, output);

  // if explNoise is 0, we just act according to policy
  // since explNoise is initial value of diagonal std vectors
  // this should only be used for evaluating a learned policy
  auto action = pol.selectAction(agent, distrib.bTrain);
  const Advantage_t adv(adv_start, aInfo, output, &pol);
  const Real V = scaleNet2V(output[VsID]);
  MB.appendValues(V, V + adv.computeAdvantage(action));
  agent.setAction(action, pol.getVector());
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
processTerminal(const MiniBatch& MB, Agent& agent)
{
  //whether episode is truncated or terminated, action advantage is 0
  if( agent.agentStatus == LAST ) {
    networks[0]->load(MB, agent, 0);
    Rvec output = networks[0]->forward(agent);
    MB.appendValues(scaleNet2V(output[VsID])); // not a terminal state
  } else MB.appendValues(0); //value of terminal state is 0
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
    if ( data->nStoredSteps() < nObsB4StartTraining ) return; // not enough data to init

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
    spawnTrainTasks();
    debugL("Search work to do in the Replay Memory");
    processMemoryBuffer(); // find old eps, update avg quantities ...
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
