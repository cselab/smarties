//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PPO.h"

#define PPO_learnDKLt
#define PPO_PENALKL
#define PPO_CLIPPED

#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_policy.h"

#include "../Network/Builder.h"
#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/Collector.h"
#include "../Utils/SstreamUtilities.h"
#include "../ReplayMemory/MemoryProcessing.h"

#include "PPO_common.cpp"
#include "PPO_train.cpp"

namespace smarties
{

template class PPO<Discrete_policy, Uint>;
template class PPO<Gaussian_policy, Rvec>;

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = * data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(&EP);
  for (const auto & net : networks ) net->load(MB, agent, 0);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    auto POL = prepare_policy(actor->forward(agent));
    const Rvec sval = critc->forward(agent);
    EP.state_vals.push_back(sval[0]); // not a terminal state
    Rvec MU = POL.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    const bool bSamplePolicy = settings.explNoise>0 && agent.trackSequence;
    auto act = POL.finalize(bSamplePolicy, &generators[nThreads+agent.ID], MU);
    agent.act(act);
    data_get->add_action(agent, MU);
  }
  else if( agent.agentStatus == TRNC )
  {
    const Rvec sval = critc->forward(agent);
    EP.state_vals.push_back(sval[0]); // not a terminal state
  }
  else // TERM state
    EP.state_vals.push_back(0); //value of terminal state is 0

  updateGAE(EP);

  //advance counters of available data for training
  if(agent.agentStatus >= TERM) data_get->terminate_seq(agent);
}

template<typename Policy_t, typename Action_t>
bool PPO<Policy_t, Action_t>::blockDataAcquisition() const
{
  // block data if we have observed too many observations
  // here we add one to concurrently gather data and compute gradients
  // 'freeze if there is too much data for the  next gradient step'
  return data->readNData() >= nHorizon + cntKept;
}

template<typename Policy_t, typename Action_t>
bool PPO<Policy_t, Action_t>::blockGradientUpdates() const
{
  //_warn("readNSeen:%ld nDataGatheredB4Start:%ld gradSteps:%ld obsPerStep:%f",
  //data->readNSeen_loc(), nDataGatheredB4Startup, _nGradSteps.load(), obsPerStep_loc);
  // almost the same of the function before
  // 'freeze if there is too little data for the current gradient step'
  return data->readNData() < nHorizon;
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::setupTasks(TaskQueue& tasks)
{
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization

  auto stepInit = [&]()
  {
    // conditions to start the initialization task:
    if ( algoSubStepID >= 0 ) return; // we done with init
    if ( data->readNData() < nHorizon ) return; // not enough data to init

    debugL("Initialize Learner");
    initializeLearner();
    initializeGAE(); // rescale GAE with learner rewards scale
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
    updatePenalizationCoef();
    data_proc->finalize();
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
    advanceEpochCounters();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
  };
  tasks.add(stepComplete);
}

}
