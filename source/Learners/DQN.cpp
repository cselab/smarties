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
#include "../ReplayMemory/Collector.h"
#include "../Utils/FunctionUtilities.h"

#define DQN_USE_POLICY

#ifdef DQN_USE_POLICY
// use discrete policy but override positive definite mapping
#define PosDefMapping_f Exp
#include "../Math/Discrete_policy.h"
#endif

namespace smarties
{

DQN::DQN(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_):
  Learner_approximator(MDP_, S_, D_)
{
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
  {  // TEST FINITE DIFFERENCES:
    Rvec output(nA), mu(nA);
    std::normal_distribution<Real> dist(0, 1);

    Real sumMu = 0;
    for(Uint i=0; i<nA; ++i) {
        const Real Qi = dist(generators[0]);
        mu[i] = PosDefMapping_f::_eval(Qi);
        sumMu += mu[i];
    }
    for(Uint i=0; i<nA; ++i) mu[i] /= sumMu;
    //for(Uint i=0; i<nA; ++i) printf("mu:%e\n",mu[i]);

    for(Uint i=0; i<nA; ++i) {
      const Real Qmu = Utilities::noiseMap_inverse(mu[i]);
      output[i] = Qmu + 0.01 * dist(generators[0]);
    }

    Discrete_policy pol({0}, aInfo, output);
    Uint act = pol.sample(generators[0], mu);
    pol.prepare(aInfo.label2actionMessage(act), mu);
    pol.test(act, mu);
  }
  trainInfo = new TrainData("DQN", distrib);
}

void DQN::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(EP);

  if( agent.agentStatus < TERM )
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    networks[0]->load(MB, agent);
    auto outVec = networks[0]->forward(agent);

    #ifdef DQN_USE_POLICY
      Discrete_policy POL({0}, aInfo, outVec);
      Rvec MU = POL.getVector();
      Uint act = POL.selectAction(agent, MU, settings.explNoise>0);
      agent.act(act);
    #else
      const Real anneal = annealingFactor(), explNoise = settings.explNoise;
      const Real annealedEps = bTrain? anneal +(1-anneal)*explNoise : explNoise;
      const Uint greedyAct = Utilities::maxInd(outVec);

      std::uniform_real_distribution<Real> dis(0.0, 1.0);
      if(dis(agent.generator) < annealedEps)
        agent.act(nA * dis(agent.generator));
      else agent.act(greedyAct);

      Rvec MU(policyVecDim, annealedEps/nA);
      MU[greedyAct] += 1-annealedEps;
    #endif

    data_get->add_action(agent, MU);
  } else
    data_get->terminate_seq(agent);
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
    Discrete_policy pol({0}, aI, Qhats);
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
    Discrete_policy POL({0}, aInfo, Qs);
    POL.prepare(MB.action(bID,t), MB.mu(bID,t));
    const Real DKL = POL.sampKLdiv, RHO = POL.sampImpWeight;
    const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

    if(CmaxRet>1) { // then refer
      if(isOff) gradient = Rvec(nA, 0); // grad clipping as if pol gradient
      const Rvec penGrad = POL.finalize_grad(POL.div_kl_grad(MB.mu(bID,t), -1));
      for(Uint i=0; i<nA; ++i)
        gradient[i] = beta * gradient[i] + (1-beta) * penGrad[i];
    }
    MB.setMseDklImpw(bID, t, ERR*ERR, DKL, RHO, CmaxRet, CinvRet);
    trainInfo->log(Qs[actt], ERR, thrID);
  #else
    MB.setMseDklImpw(bID, t, ERR*ERR, 0, 1, CmaxRet, CinvRet);
    trainInfo->log(Qs[actt], ERR, thrID);
  #endif

  networks[0]->setGradient(gradient, bID, t);
}

}
