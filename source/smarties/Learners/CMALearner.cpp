//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMALearner.h"

#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/MemoryProcessing.h"

#ifdef SMARTIES_EXTRACT_COVAR
#undef SMARTIES_EXTRACT_COVAR
#endif

#include "../Math/Continuous_policy.h"
#include "../Math/Discrete_policy.h"

#include <unistd.h> // usleep

namespace smarties
{

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
  Rvec pol = netOutput; // will store representation of policy
  Rvec act = netOutput; // will store action sent to agent
  const Uint nA = aInfo.dim();
  const bool bSamplePol = settings.explNoise>0 && agent.trackEpisodes;
  if(bSamplePol) {
    assert(pol.size() == 2 * nA);
    Continuous_policy POL({0, nA}, aInfo, netOutput);
    act = POL.selectAction(agent, settings.explNoise>0);
  }
  //printf("%s\n", print(pol).c_str());
  agent.setAction(act, pol);
}

template<typename Action_t> Uint CMALearner<Action_t>::
weightID(const Agent& agent) const
{
  // to complete a step you need to do one episode per
  // 1) fraction of the total batchSize
  // 2) sample of the es population
  // 3) agent contained in each environment and owned by the learner
  // then each owned worker does that number divided by
  const Uint envID = agent.workerID;
  const Uint workID = envID + indexEndedPerEnv[envID] * nOwnEnvs;
  return workID / batchSize_local;
}

template<typename Action_t> void CMALearner<Action_t>::
selectAction(const MiniBatch& MB, Agent& agent)
{
  if(agent.agentStatus == INIT and curNumEndedPerEnv[agent.workerID]>0)
   die("Cannot start new EP for agent unless all other agents have terminated");

  //const auto W = F[0]->net->sampled_weights[weightID];
  //std::vector<nnReal> WW(W->params, W->params + W->nParams);
  //printf("Using weight %u on worker %u %s\n",weightID,wrkr,print(WW).c_str());

  //Compute policy and value on most recent element of the sequence:
  networks[0]->load(MB, agent, weightID(agent));
  computeAction( agent, networks[0]->forward(agent) );
}

template<typename Action_t> void CMALearner<Action_t>::
processTerminal(const MiniBatch& MB, Agent& agent)
{
  R[agent.workerID][ weightID(agent) ] += agent.cumulativeRewards;
  const auto myStep = nGradSteps();
  data->terminateCurrentEpisode(agent);

  //_warn("%u %u %f",wrkr, weightID, R[wrkr][weightID]);
  curNumEndedPerEnv[agent.workerID]++; // one more agent of workerID has finished
  if(curNumEndedPerEnv[agent.workerID] == nOwnAgentsPerEnv) {
    curNumEndedPerEnv[agent.workerID] = 0; // reset counter
    //printf("Simulation %ld ended for worker %u\n", WnEnded[workerID], wrkr);
    indexEndedPerEnv[agent.workerID]++; // start new workload on that env
  }
  // now we may have increased worload counter, let's see if we are done
  if( weightID(agent) >= ESpopSize ) {
    while( myStep == nGradSteps() ) usleep(1);
    //_warn("worker %u done wait", wrkr);
    indexEndedPerEnv[agent.workerID] = 0; // reset workload counter for next iter
  }
}

template<typename Action_t> void CMALearner<Action_t>::prepareCMALoss()
{
  profiler->start("LOSS");
  #pragma omp parallel for schedule(static)
  for (Uint w=0; w<ESpopSize; ++w) {
    for (Uint b=0; b<nOwnEnvs; ++b) networks[0]->ESloss(w) -= R[b][w];
    //std::cout << F[0]->losses[w] << std::endl;
  }
  // reset cumulative rewards:
  R = std::vector<Rvec>(nOwnEnvs, Rvec(ESpopSize, 0) );
  networks[0]->nAddedGradients = batchSize_local * ESpopSize;

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
  return data->nStoredEps() >= nSeqPerStep;
}

template<typename Action_t>
bool CMALearner<Action_t>::blockGradientUpdates() const
{
  return data->nStoredEps() < nSeqPerStep;
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
