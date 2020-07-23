//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "DQN.h"

#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"
#include "../Utils/FunctionUtilities.h"

#define DQN_USE_POLICY

#ifdef DQN_USE_POLICY
#include "../Math/Discrete_policy.h"
#endif

namespace smarties
{

DQN::DQN(MDPdescriptor& M, HyperParameters& S, ExecutionInfo& D):
  Learner_approximator(M, S, D)
{
  #ifdef DQN_USE_POLICY
    if(settings.clipImpWeight > 0)
    printf("Using ReF-ER with clipping parameter C=%f, tolerance D=%f "
           "and annealing E=%f\n", S.clipImpWeight, S.penalTol, S.epsAnneal);
  #endif
  createEncoder();
  assert(networks.size() <= 1);
  if(networks.size()>0) {
    networks[0]->rename("Q"); // not preprocessing, is is the main&only net
  } else {
    networks.push_back(new Approximator("Q", settings, distrib, data.get()));
  }
  networks[0]->setUseTargetNetworks();
  networks[0]->buildFromSettings(nA);
  networks[0]->initializeNetwork();
}

void DQN::selectAction(const MiniBatch& MB, Agent& agent)
{
  networks[0]->load(MB, agent, 0);
  //Compute policy and value on most recent element of the sequence. If RNN
  // recurrent connection from last call from same agent will be reused
  auto outVec = networks[0]->forward(agent);

  #ifdef DQN_USE_POLICY
    Discrete_policy_t<Exp> POL({0}, aInfo, outVec);
    Uint act = POL.selectAction(agent, settings.explNoise>0);
    agent.setAction(act, POL.getVector());
  #else // from paper : annealed epsilon-greedy
    const Real anneal = annealingFactor(), explNoise = settings.explNoise;
    const Real annealedEps = bTrain? anneal +(1-anneal)*explNoise : explNoise;
    const Uint greedyAct = Utilities::maxInd(outVec);
    Rvec MU(policyVecDim, annealedEps/nA);
    MU[greedyAct] += 1-annealedEps;

    std::uniform_real_distribution<Real> dis(0.0, 1.0);
    if(dis(agent.generator) < annealedEps)
      agent.setAction(nA * dis(agent.generator), MU);
    else agent.setAction(greedyAct, MU);
  #endif
}

void DQN::processTerminal(const MiniBatch& MB, Agent& agent)
{
}

void DQN::setupTasks(TaskQueue& tasks)
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
    profiler->start("MPI");

    algoSubStepID = 1;
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
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

static inline Real expectedValue(const Rvec& Qhats, const Rvec& Qtildes,
                                 const ActionInfo& aI)
{
  assert( aI.dimDiscrete() == Qhats.size() );
  assert( aI.dimDiscrete() == Qhats.size() );
  #ifdef DQN_USE_POLICY
    Discrete_policy_t<Exp> pol({0}, aI, Qhats);
    Real ret = 0;
    for(Uint i=0; i<aI.dimDiscrete(); ++i) ret += pol.probs[i] * Qtildes[i];
    return ret;
  #else
    return Qtildes[ Utilities::maxInd(Qhats) ];
  #endif
}

void DQN::Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->start("FWD");

  const Rvec Qs = networks[0]->forward(bID, t);
  const Uint actt = aInfo.actionMessage2label(MB.action(bID,t));
  assert(actt < Qs.size()); // enough to store advantages and value

  Real Vsnew = MB.reward(bID, t);
  if (not MB.isTerminal(bID, t+1)) {
    // find best action for sNew with moving wghts, evaluate it with tgt wgths:
    // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
    const Rvec Qhats = networks[0]->forward(bID, t+1);
    const Rvec Qtildes = settings.targetDelay <= 0 ? Qhats // no target nets
                         : networks[0]->forward_tgt(bID, t+1);
    //v_s = r + gamma * Q(greedy action) :
    Vsnew += gamma * expectedValue(Qhats, Qtildes, aInfo);
  }
  const Real ERR = Vsnew - Qs[actt];

  Rvec gradient(nA, 0);
  gradient[actt] = ERR;

  #ifdef DQN_USE_POLICY
    Discrete_policy_t<Exp> POL({0}, aInfo, Qs);
    const Real RHO = POL.importanceWeight(MB.action(bID,t), MB.mu(bID,t));
    const Real DKL = POL.KLDivergence(MB.mu(bID,t));
    const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

    if(CmaxRet>1) { // then refer
      if(isOff) gradient = Rvec(nA, 0); // grad clipping as if pol gradient
      const Rvec penGrad = POL.KLDivGradient(MB.mu(bID,t), -1);
      for(Uint i=0; i<nA; ++i)
        gradient[i] = beta * gradient[i] + (1-beta) * penGrad[i];
    }
    MB.setMseDklImpw(bID, t, ERR, DKL, RHO, CmaxRet, CinvRet);
  #else
    MB.setMseDklImpw(bID, t, ERR, 0, 1, CmaxRet, CinvRet);
  #endif
  MB.setValues(bID, t, expectedValue(Qs, Qs, aInfo), Qs[actt]);

  networks[0]->setGradient(gradient, bID, t);
}

}
