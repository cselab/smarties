//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "NAF.h"

#include "../Utils/FunctionUtilities.h"
#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"

#ifdef NAF_ADV_GAUS
#include "../Math/Gaus_advantage.h"
#define Param_advantage Gaussian_advantage
#else
#include "../Math/Quadratic_advantage.h"
#define Param_advantage Quadratic_advantage
#endif

namespace smarties
{

static inline Param_advantage prepare_advantage(const Rvec& O,
  const ActionInfo& aI, const std::vector<Uint>& net_inds)
{
  return Param_advantage(std::vector<Uint>{net_inds[1], net_inds[2]}, aI, O);
}

NAF::NAF(MDPdescriptor& MDP_, HyperParameters& S, ExecutionInfo& D):
  Learner_approximator(MDP_, S, D), nL(Param_advantage::compute_nL(aInfo))
{
  if(settings.clipImpWeight > 0)
    printf("Using ReF-ER with clipping parameter C=%f, tolerance D=%f "
           "and annealing E=%f\n", S.clipImpWeight, S.penalTol, S.epsAnneal);
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
  Rvec stdParam = Continuous_policy::initial_Stdev(aInfo, settings.explNoise);
  networks[0]->getBuilder().addParamLayer(nA, "Linear", stdParam);
  networks[0]->initializeNetwork();
}

void NAF::selectAction(const MiniBatch& MB, Agent& agent)
{
  networks[0]->load(MB, agent, 0);
  const Rvec output = networks[0]->forward(agent);
  const Continuous_policy POL({net_indices[2], net_indices[3]}, aInfo, output);
  const auto ADV = prepare_advantage(output, aInfo, net_indices);

  const bool bSample = settings.explNoise>0;
  Rvec act = OrUhDecay<=0? POL.selectAction(agent, bSample) :
      POL.selectAction_OrnsteinUhlenbeck(agent, bSample, OrUhState[agent.ID]);
  agent.setAction(act, POL.getVector());

  const Real V = output[0], Q = V + ADV.computeAdvantage(agent.action);
  MB.appendValues(V, Q);
}

void NAF::processTerminal(const MiniBatch& MB, Agent& agent)
{
  if( agent.agentStatus == LAST ) {
    networks[0]->load(MB, agent, 0);
    MB.appendValues(networks[0]->forward(agent)[0]); // not a terminal state
  } else MB.appendValues(0); // value of terminal state is 0
  OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
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
    if ( data->nStoredSteps() < nObsB4StartTraining ) return; // not enough data to init

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
    debugL("Search work to do in the Replay Memory");
    processMemoryBuffer();
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
  const auto ADV = prepare_advantage(output, aInfo, net_indices);
  const Continuous_policy POL({net_indices[2], net_indices[3]}, aInfo, output);
  const Real RHO = POL.importanceWeight(MB.action(bID,t), MB.mu(bID,t));
  const Real DKL = POL.KLDivergence(MB.mu(bID,t));
  //cout << POL.sampImpWeight << " " << POL.sampKLdiv << " " << CmaxRet << endl;

  const Real Qs = output[net_indices[0]] + ADV.computeAdvantage(MB.action(bID,t));
  const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

  Real target = 0;
  if(settings.returnsEstimator not_eq "none") {
    if (MB.isTruncated(bID, t+1))
      MB.setValues(bID, t+1, networks[0]->forward(bID, t+1)[net_indices[0]]);
    target = MB.returnEstimate(bID, t);
  } else {
    target = MB.reward(bID, t);
    if (not MB.isTerminal(bID, t+1) && not isOff) {
      const Rvec vNext = networks[0]->forward_tgt(bID, t+1);
      MB.setValues(bID, t+1, vNext[net_indices[0]]);
      target += gamma * vNext[net_indices[0]];
    }
  }

  const Real error = isOff? 0 : target - Qs;
  Rvec grad(networks[0]->nOutputs());
  grad[net_indices[0]] = error;
  ADV.grad(MB.action(bID,t), error, grad);
  if(CmaxRet>1 && beta<1) { // then ReFER
    const Rvec penG = POL.KLDivGradient(MB.mu(bID,t), -1);
    for(Uint i=0; i<nA; ++i)
      grad[net_indices[2]+i] = beta*grad[net_indices[2]+i] + (1-beta)*penG[i];
  }
  const Rvec fixGrad = POL.fixExplorationGrad(settings.explNoise);
  for(Uint i=0; i<nA; ++i) grad[net_indices[3]+i] = fixGrad[nA+i];

  MB.setMseDklImpw(bID, t, error, DKL, RHO, CmaxRet, CinvRet);
  MB.setValues(bID, t, output[net_indices[0]], Qs);
  networks[0]->setGradient(grad, bID, t);
}

}
