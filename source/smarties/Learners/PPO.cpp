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

#include "../Math/Continuous_policy.h"
#include "../Math/Discrete_policy.h"

#include "../Network/Approximator.h"
#include "../Network/Builder.h"
#include "../Utils/StatsTracker.h"
#include "../Utils/SstreamUtilities.h"
#include "../ReplayMemory/MemoryProcessing.h"

#include "PPO_common.cpp"
#include "PPO_train.cpp"

namespace smarties
{

template class PPO<Discrete_policy, Uint>;
template class PPO<Continuous_policy, Rvec>;

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::selectAction(const MiniBatch& MB, Agent& agent)
{
  for (const auto & net : networks ) net->load(MB, agent, 0);
  //Compute policy and value on most recent element of the sequence.
  Policy_t POL(pol_indices, aInfo, actor->forward(agent));
  MB.appendValues(critc->forward(agent)[0]);

  // if explNoise is 0, we just act according to policy
  // since explNoise is initial value of diagonal std vectors
  // this should only be used for evaluating a learned policy
  auto action = POL.selectAction(agent, settings.explNoise>0);
  agent.setAction(action, POL.getVector());
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::processTerminal(const MiniBatch& MB, Agent& agent)
{
  if( agent.agentStatus == LAST ) {
    critc->load(MB, agent, 0);
    MB.appendValues(critc->forward(agent)[0]); // not a terminal state
  } else MB.appendValues(0); // terminal state has value 0
}

template<typename Policy_t, typename Action_t>
bool PPO<Policy_t, Action_t>::blockDataAcquisition() const
{
  // block data if we have observed too many observations
  // here we add one to concurrently gather data and compute gradients
  // 'freeze if there is too much data for the  next gradient step'
  return data->nStoredSteps() >= nHorizon + cntKept;
}

template<typename Policy_t, typename Action_t>
bool PPO<Policy_t, Action_t>::blockGradientUpdates() const
{
  //_warn("readNSeen:%ld nDataGatheredB4Start:%ld gradSteps:%ld obsPerStep:%f",
  //data->readNSeen_loc(), nDataGatheredB4Startup, _nGradSteps.load(), obsPerStep_loc);
  // almost the same of the function before
  // 'freeze if there is too little data for the current gradient step'
  return data->nStoredSteps() < nHorizon;
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
    if ( data->nStoredSteps() < nHorizon ) return; // not enough data to init

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
    updatePenalizationCoef();
    MemoryProcessing::updateTrainingStatistics(* data.get());
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
    MemoryProcessing::updateCounters(* data.get());

    debugL("shift counters of epochs over the stored data");
    cntBatch += settings.batchSize;
    if(cntBatch >= nHorizon) {
      MemoryProcessing::updateRewardsStats(* data.get());
      cntBatch = 0;
      cntEpoch++;
    }

    if(cntEpoch >= nEpochs) {
      debugL("finished epochs,: clear buffer to gather new samples");
      logStats(true);
      if(0) // keep nearly on policy data
        cntKept = data->clearOffPol(CmaxPol, 0.05);
      else {
        data->clearAll();
        cntKept = 0;
      }
      //reset batch learning counters
      cntEpoch = 0;
      cntBatch = 0;
    }

    globalGradCounterUpdate(); // step ++
    algoSubStepID = 0; // rinse and repeat
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

}
