//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMALearner.h"
#include "../Utils/StatsTracker.h"
#include "../ReplayMemory/Collector.h"
#include "../ReplayMemory/MemoryProcessing.h"
#ifdef SMARTIES_EXTRACT_COVAR
#undef SMARTIES_EXTRACT_COVAR
#endif
#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_policy.h"
#include "../Network/Approximator.h"

#include <unistd.h> // usleep

namespace smarties
{

template<> void CMALearner<Uint>::
computeAction(Agent& agent, const Rvec netOutput) const
{
  Discrete_policy POL({0}, & aInfo, netOutput);
  Rvec MU = POL.getVector();
  const bool bSamplePol = settings.explNoise>0 && agent.trackSequence;
  Uint act = POL.finalize(bSamplePol, &generators[nThreads+agent.ID], MU);
  agent.act(act);
  data_get->add_action(agent, MU);
}

template<> void CMALearner<Rvec>::
computeAction(Agent& agent, const Rvec netOutput) const
{
  Rvec pol = netOutput; // will store representation of policy
  Rvec act = netOutput; // will store action sent to agent
  const Uint nA = aInfo.dim();
  const bool bSamplePol = settings.explNoise>0 && agent.trackSequence;
  if(bSamplePol)
  {
    assert(pol.size() == 2 * nA);
    act.resize(nA);
    std::normal_distribution<Real> D(0, 1);
    for(Uint i=0; i<nA; ++i) {
      // map policy output into pos-definite stdev:
      pol[i+nA] = Gaussian_policy::extract_stdev(pol[i+nA]);
      act[i] += pol[i+nA] * D(generators[nThreads + agent.ID]);
    }
  }
  //printf("%s\n", print(pol).c_str());
  agent.act(act);
  data_get->add_action(agent, pol);
}

template<typename Action_t> void CMALearner<Action_t>::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = * data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(&EP);
  const Uint envID = agent.workerID;

  if(agent.agentStatus == INIT and curNumEndedPerEnv[envID]>0)
   die("Cannot start new EP for agent unless all other agents have terminated");

  // to complete a step you need to do one episode per
  // 1) fraction of the total batchSize
  // 2) sample of the es population
  // 3) agent contained in each environment and owned by the learner
  // then each owned worker does that number divided by
  const auto weightID = [&]() {
    const Uint workID = envID + indexEndedPerEnv[envID] * nOwnEnvs;
    return workID / batchSize_local;
  };

  //const auto W = F[0]->net->sampled_weights[weightID];
  //std::vector<nnReal> WW(W->params, W->params + W->nParams);
  //printf("Using weight %u on worker %u %s\n",weightID,wrkr,print(WW).c_str());

  if( agent.agentStatus <  TERM ) //non terminal state
  {
    //Compute policy and value on most recent element of the sequence:
    networks[0]->load(MB, agent, weightID());
    computeAction( agent, networks[0]->forward(agent) );
  }
  else
  {
    R[envID][ weightID() ] += agent.cumulativeRewards;
    const auto myStep = nGradSteps();
    data_get->terminate_seq(agent);

    //_warn("%u %u %f",wrkr, weightID, R[wrkr][weightID]);
    curNumEndedPerEnv[envID]++; // one more agent of envID has finished
    if(curNumEndedPerEnv[envID] == nOwnAgentsPerEnv) {
      curNumEndedPerEnv[envID] = 0; // reset counter
      //printf("Simulation %ld ended for worker %u\n", WnEnded[envID], wrkr);
      indexEndedPerEnv[envID]++; // start new workload on that env
    }
    // now we may have increased worload counter, let's see if we are done
    if( weightID() >= ESpopSize ) {
      while( myStep == nGradSteps() ) usleep(1);
      //_warn("worker %u done wait", wrkr);
      indexEndedPerEnv[envID] = 0; // reset workload counter for next iter
    }
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
  data_proc->updateRewardsStats(0.001, 0.001);
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
  return data->readNSeq() >= nSeqPerStep;
}

template<typename Action_t>
bool CMALearner<Action_t>::blockGradientUpdates() const
{
  return data->readNSeq() < nSeqPerStep;
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
CMALearner(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_) :
Learner_approximator(MDP_, S_, D_)
{
  if(D_.world_rank == 0) {
  printf(
  "==========================================================================\n"
  "           Continuous-valued CMA : Covariance Matrix Adaptation           \n"
  "==========================================================================\n"
  ); }

  createEncoder();
  assert(networks.size() <= 1);
  if(networks.size()>0) {
    networks[0]->rename("net"); // not preprocessing, is is the main&only net
  } else {
    networks.push_back(new Approximator("net", settings, distrib, data.get()));
  }
  networks[0]->buildFromSettings(aInfo.dim());
  if(settings.explNoise>0) {
    Rvec stdParam = Gaussian_policy::initial_Stdev(&aInfo, settings.explNoise);
    networks[0]->getBuilder().addParamLayer(aInfo.dim(), "Linear", stdParam);
  }
  networks[0]->initializeNetwork();
  if(nOwnEnvs == 0) die("CMA learner does not support workerless masters");
  if(ESpopSize<=1) die("CMA learner requires ESpopSize>1");
}

template<> CMALearner<Uint>::
CMALearner(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_) :
Learner_approximator(MDP_, S_, D_)
{
  if(D_.world_rank == 0) {
  printf(
  "==========================================================================\n"
  "            Discrete-action CMA : Covariance Matrix Adaptation            \n"
  "==========================================================================\n"
  ); }

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
