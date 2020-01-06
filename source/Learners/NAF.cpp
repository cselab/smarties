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
  stdParam(Gaussian_policy::initial_Stdev(aInfo, S.explNoise)[0])
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

  {
    Rvec out(networks[0]->nOutputs()), act(aInfo.dim());
    std::uniform_real_distribution<Real> out_dis(-.5,.5);
    std::uniform_real_distribution<Real> act_dis(-.5,.5);
    const int thrID = omp_get_thread_num();
    for(Uint i = 0; i<aInfo.dim(); ++i) act[i] = act_dis(generators[thrID]);
    for(Uint i = 0; i<nOutp; ++i) out[i] = out_dis(generators[thrID]);
    Param_advantage A = prepare_advantage(out, aInfo, net_indices);
    A.test(act, &generators[thrID]);
  }
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
    Gaussian_policy POL({0, nA}, aInfo, polvec);

    Rvec MU = POL.getVector();
    //cout << print(MU) << endl;
    Rvec act = POL.selectAction(agent, MU, settings.explNoise>0);

    if(OrUhDecay>0)
      act = POL.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);

    agent.act(act);
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
  Gaussian_policy POL({0, nA}, aInfo, polvec);
  POL.prepare(MB.action(bID,t), MB.mu(bID,t));
  const Real DKL = POL.sampKLdiv, RHO = POL.sampImpWeight;
  //cout << POL.sampImpWeight << " " << POL.sampKLdiv << " " << CmaxRet << endl;

  const Real Qs = output[net_indices[0]] + ADV.computeAdvantage(POL.sampAct);
  const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

  Real target = MB.reward(bID, t);
  if (not MB.isTerminal(bID, t+1) && not isOff)
    target += gamma * networks[0]->forward_tgt(bID, t+1) [net_indices[0]];
  const Real error = isOff? 0 : target - Qs;
  Rvec grad(networks[0]->nOutputs());
  grad[net_indices[0]] = error;
  ADV.grad(POL.sampAct, error, grad);
  if(CmaxRet>1 && beta<1) { // then ReFER
    const Rvec penG = POL.div_kl_grad(MB.mu(bID,t), -1);
    for(Uint i=0; i<nA; ++i)
      grad[net_indices[2]+i] = beta*grad[net_indices[2]+i] + (1-beta)*penG[i];
  }

  trainInfo->log(Qs, error, {beta, RHO}, thrID);
  MB.setMseDklImpw(bID, t, error*error, DKL, RHO, CmaxRet, CinvRet);

  networks[0]->setGradient(grad, bID, t);
}

}
