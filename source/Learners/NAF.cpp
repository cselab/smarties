//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "NAF.h"
#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/Collector.h"
#include "../Utils/FunctionUtilities.h"

#ifdef NAF_ADV_GAUS
#include "../Math/Gaus_advantage.h"
#define Param_advantage Gaussian_advantage
#else
#include "../Math/Quadratic_advantage.h"
#define Param_advantage Quadratic_advantage
#endif

namespace smarties
{

static inline Param_advantage prepare_advantage(const Rvec&O,
  const ActionInfo& aI, const std::vector<Uint>& net_inds)
{
  return Param_advantage(std::vector<Uint>{net_inds[1], net_inds[2]}, aI, O);
}

NAF::NAF(MDPdescriptor& MDP_, Settings& S, DistributionInfo& D):
  Learner_approximator(MDP_, S, D), nL( Param_advantage::compute_nL(aInfo) ),
  stdParam(Continuous_policy::initial_Stdev(aInfo, S.explNoise)[0])
{
  createEncoder();
  assert(networks.size() <= 1);
  if(networks.size()>0) {
    networks[0]->rename("net"); // not preprocessing, is is the main&only net
  } else {
    networks.push_back(new Approximator("net", settings, distrib, data.get()));
  }

  networks[0]->setUseTargetNetworks();
  const Uint nOutp = 1 + aInfo.dim() + Param_advantage::compute_nL(aInfo);
  assert(nOutp == net_outputs[0] + net_outputs[1] + net_outputs[2]);
  networks[0]->buildFromSettings(nOutp);
  networks[0]->initializeNetwork();

  trainInfo = new TrainData("NAF", distrib, 0, "| beta | avgW ", 2);
}

void NAF::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(EP);

  if( agent.agentStatus < TERM ) // not last of a sequence
  {
    networks[0]->load(MB, agent);
    //Compute policy and value on most recent element of the sequence.
    const Rvec output = networks[0]->forward(agent);
    Rvec polvec = Rvec(&output[net_indices[2]], &output[net_indices[2]] + nA);
    // add stdev to the policy vector representation:
    polvec.resize(2*nA, stdParam);
    Continuous_policy POL({0, nA}, aInfo, polvec);

    Rvec MU = POL.getVector();
    //cout << print(MU) << endl;
    const bool bSample = settings.explNoise>0;
    Rvec act = OrUhDecay<=0? POL.selectAction(agent, bSample) :
        POL.selectAction_OrnsteinUhlenbeck(agent, bSample, OrUhState[agent.ID]);
    agent.setAction(act);
    data_get->add_action(agent, MU);
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data_get->terminate_seq(agent);
  }
}

void NAF::setupTasks(TaskQueue& tasks)
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
    debugL("Gather gradient estimates from each thread and Learner MPI rank");
    prepareGradient();
    debugL("Search work to do in the Replay Memory");
    processMemoryBuffer();
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

void NAF::Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->start("FWD");

  const Rvec output = networks[0]->forward(bID, t);

  if(thrID==0) profiler->stop_start("CMP");
  // prepare advantage and policy
  const auto ADV = prepare_advantage(output, aInfo, net_indices);
  Rvec polvec = ADV.getMean();           assert(polvec.size() == 1 * nA);
  polvec.resize(policyVecDim, stdParam); assert(polvec.size() == 2 * nA);
  Continuous_policy POL({0, nA}, aInfo, polvec);
  const Real RHO = POL.importanceWeight(MB.action(bID,t), MB.mu(bID,t));
  const Real DKL = POL.KLDivergence(MB.mu(bID,t));
  //cout << POL.sampImpWeight << " " << POL.sampKLdiv << " " << CmaxRet << endl;

  const Real Qs = output[net_indices[0]] + ADV.computeAdvantage(MB.action(bID,t));
  const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

  Real target = MB.reward(bID, t);
  if (not MB.isTerminal(bID, t+1) && not isOff)
    target += gamma * networks[0]->forward_tgt(bID, t+1) [net_indices[0]];
  const Real error = isOff? 0 : target - Qs;
  Rvec grad(networks[0]->nOutputs());
  grad[net_indices[0]] = error;
  ADV.grad(MB.action(bID,t), error, grad);
  if(CmaxRet>1 && beta<1) { // then ReFER
    const Rvec penG = POL.KLDivGradient(MB.mu(bID,t), -1);
    for(Uint i=0; i<nA; ++i)
      grad[net_indices[2]+i] = beta*grad[net_indices[2]+i] + (1-beta)*penG[i];
  }

  trainInfo->log(Qs, error, {beta, RHO}, thrID);
  MB.setMseDklImpw(bID, t, error*error, DKL, RHO, CmaxRet, CinvRet);

  networks[0]->setGradient(grad, bID, t);
}

}
