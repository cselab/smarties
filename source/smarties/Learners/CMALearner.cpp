//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMALearner.h"

#include "../Utils/StatsTracker.h"
//#include "../Utils/SstreamUtilities.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/MemoryProcessing.h"

#ifdef SMARTIES_EXTRACT_COVAR
#undef SMARTIES_EXTRACT_COVAR
#endif

#include "../Math/Continuous_policy.h"
#include "../Math/Discrete_policy.h"

#include <unistd.h> // usleep
#include <iostream>

namespace smarties
{

template<typename Action_t> void CMALearner<Action_t>::
assignWeightID(const Agent& agent)
{
  if (curNumStartedPerEnv[agent.workerID]++ == 0) {
    assert(curNumEndedPerEnv[agent.workerID] == 0 &&
      "CMA: cannot start new EP unless all other agents have terminated");

    std::lock_guard<std::mutex> lock(workload_mutex);
    weightIDs[agent.workerID] = (lastWorkLoadStarted++) % ESpopSize;
  }
}

template<typename Action_t> void CMALearner<Action_t>::
selectAction(const MiniBatch& MB, Agent& agent)
{
  if(agent.agentStatus == INIT and curNumEndedPerEnv[agent.workerID]>0)
    die("CMA: cannot start new EP unless all other agents have terminated");

  if(agent.agentStatus == INIT) assignWeightID(agent);

  //Compute policy and value on most recent element of the sequence:
  networks[0]->load(MB, agent, weightIDs[agent.workerID]);
  computeAction( agent, networks[0]->forward(agent) );
}

template<typename Action_t> void CMALearner<Action_t>::
processTerminal(const MiniBatch& MB, Agent& agent)
{
  R [agent.workerID][weightIDs[agent.workerID]] += agent.cumulativeRewards;
  Ns[agent.workerID][weightIDs[agent.workerID]] ++;

  data->terminateCurrentEpisode(agent);

  if (curNumEndedPerEnv[agent.workerID] == 0) {
    assert(curNumStartedPerEnv[agent.workerID] == nOwnAgentsPerEnv);
    curNumStartedPerEnv[agent.workerID] = 0; // now both counters are reset
  }
  curNumEndedPerEnv[agent.workerID]++; // one more agent of workerID has finished
  if (curNumEndedPerEnv[agent.workerID] == nOwnAgentsPerEnv) {
    curNumEndedPerEnv[agent.workerID] = 0; // reset counter: all agents ended
  }
}

template<typename Action_t> void CMALearner<Action_t>::prepareCMALoss()
{
  profiler->start("LOSS");
  std::vector<Uint> allN(ESpopSize, 0);

  for (Uint w=0; w<ESpopSize; ++w) {
    for (Uint b=0; b<nOwnEnvs; ++b) {
      networks[0]->ESloss(w) -= R[b][w];
      allN[w] += Ns[b][w];
    }
    //std::cout<<w<<" "<<networks[0]->ESloss(w)<<" "<<allN[w]<<" "
    //         <<networks[0]->ESloss(w) / allN[w]<<std::endl;
    networks[0]->ESloss(w) = networks[0]->ESloss(w) / allN[w];
  }
  // reset cumulative rewards:
  R = std::vector<Rvec>(nOwnEnvs, Rvec(ESpopSize, 0));
  Ns= std::vector<std::vector<Uint>>(nOwnEnvs, std::vector<Uint>(ESpopSize, 0));

  {
    const Uint workLoadSize = settings.batchSize_local * ESpopSize;
    std::lock_guard<std::mutex> lock(workload_mutex);
    lastWorkLoadStarted = lastWorkLoadStarted % workLoadSize;
  }

  networks[0]->nAddedGradients = settings.batchSize_local * ESpopSize;

  debugL("shift counters of epochs over the stored data");
  profiler->stop_start("PRE");
  MemoryProcessing::updateRewardsStats(* data.get());
  profiler->stop();
}

template<typename Action_t>
void CMALearner<Action_t>::setupTasks(TaskQueue& tasks)
{
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = 0; // no need to initialiaze
  profiler->start("DATA");

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data
    profiler->stop();
    debugL("Gather gradient estimates from each thread and Learner MPI rank");
    prepareCMALoss();
    profiler->start("ADDW");
    debugL("Reduce gradient estimates");
    for(const auto & net : networks) {
      net->prepareUpdate();
      net->updateGradStats(learner_name, nGradSteps());
    }
    profiler->stop();

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
    debugL("Apply SGD update after reduction of gradients");
    applyGradient();
    data->clearAll();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

template<typename Action_t>
bool CMALearner<Action_t>::blockDataAcquisition() const
{
  const long nSeqPerStep = nOwnAgentsPerEnv * settings.batchSize_local * ESpopSize;
  return data->nStoredEps() >= nSeqPerStep;
}

template<typename Action_t>
bool CMALearner<Action_t>::blockGradientUpdates() const
{
  const long nSeqPerStep = nOwnAgentsPerEnv * settings.batchSize_local * ESpopSize;
  return data->nStoredEps() < nSeqPerStep;
}

template<> void CMALearner<Uint>::
computeAction(Agent& agent, const Rvec netOutput) const
{
  Discrete_policy POL({0}, aInfo, netOutput);
  Uint act = POL.selectAction(agent, settings.explNoise>0);
  agent.setAction(act, POL.getVector());
}

template<> void CMALearner<Rvec>::
computeAction(Agent& agent, const Rvec netOutput) const
{
  Continuous_policy POL({0, aInfo.dim()}, aInfo, netOutput);
  const bool bSamplePol = settings.explNoise>0 && agent.trackEpisodes;
  Rvec act = bSamplePol? POL.selectAction(agent, bSamplePol) : POL.getVector();
  agent.setAction(act, POL.getVector());
}

template<>
std::vector<Uint> CMALearner<Uint>::count_pol_outputs(const ActionInfo*const aI)
{
  return std::vector<Uint>{ aI->dimDiscrete() };
}
template<>
std::vector<Uint> CMALearner<Uint>::count_pol_starts(const ActionInfo*const aI)
{
  return std::vector<Uint>{0};
}
template<>
Uint CMALearner<Uint>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->dimDiscrete();
}

template<>
std::vector<Uint> CMALearner<Rvec>::count_pol_outputs(const ActionInfo*const aI)
{
  return std::vector<Uint>{aI->dim()};
}
template<>
std::vector<Uint> CMALearner<Rvec>::count_pol_starts(const ActionInfo*const aI)
{
  return std::vector<Uint>{0};
}
template<>
Uint CMALearner<Rvec>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->dim();
}

template<> CMALearner<Rvec>::
CMALearner(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_) :
Learner_approximator(MDP_, S_, D_)
{
  freqPrint = 1;
  createEncoder();
  assert(networks.size() <= 1);
  if(networks.size()>0) {
    networks[0]->rename("net"); // not preprocessing, is is the main&only net
  } else {
    networks.push_back(new Approximator("net", settings, distrib, data.get()));
  }
  networks[0]->buildFromSettings(aInfo.dim());
  if(settings.explNoise>0) {
    Rvec stdParam = Continuous_policy::initial_Stdev(aInfo, settings.explNoise);
    networks[0]->getBuilder().addParamLayer(aInfo.dim(), "Linear", stdParam);
  }
  networks[0]->initializeNetwork();
  if(nOwnEnvs == 0) die("CMA learner does not support workerless masters");
  if(ESpopSize<=1) die("CMA learner requires ESpopSize>1");
}

template<> CMALearner<Uint>::
CMALearner(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_) :
Learner_approximator(MDP_, S_, D_)
{
  freqPrint = 1;
  createEncoder();
  assert(networks.size() <= 1);
  if(networks.size()>0) {
    networks[0]->rename("net"); // not preprocessing, is is the main&only net
  } else {
    networks.push_back(new Approximator("net", settings, distrib, data.get()));
  }
  networks[0]->buildFromSettings(MDP.maxActionLabel);
  networks[0]->initializeNetwork();
  if(nOwnEnvs == 0) die("CMA learner does not support workerless masters");
  if(ESpopSize<=1) die("CMA learner requires ESpopSize>1");
}

template class CMALearner<Uint>;
template class CMALearner<Rvec>;

template<typename Action_t>
void CMALearner<Action_t>::
Train(const MiniBatch&MB,const Uint wID,const Uint bID) const
{}

}
