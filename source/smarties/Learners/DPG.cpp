//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "DPG.h"

#include "../Network/Builder.h"
#include "../Network/Approximator.h"
#include "../Utils/StatsTracker.h"
#include "../Utils/SstreamUtilities.h"
#include "../Math/Continuous_policy.h"

//#define DPG_LEARN_STDEV

namespace smarties
{

void DPG::Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->start("FWD");
  const Rvec pvec = actor->forward(bID, t); // network compute
  if(thrID==0) profiler->stop_start("CMP");
  const Continuous_policy POL(aInfo, pvec);
  const Real RHO = POL.importanceWeight(MB.action(bID,t), MB.mu(bID,t));
  const Real DKL = POL.KLDivergence(MB.mu(bID,t));
  const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

  critc->setAddedInputType(ACTION, bID, t);
  const Rvec qval = critc->forward(bID, t); // network compute
  critc->setAddedInputType(NETWORK, bID, t, -1); //-1 flags to write on separate
  const Rvec pval = critc->forward(bID, t, -1); //net alloc, with target wegiths

  Real target = 0;
  if(settings.returnsEstimator not_eq "none") {
    if( MB.isTruncated(bID, t+1) ) {
      actor->forward(bID, t+1);
      critc->setAddedInputType(NETWORK, bID, t+1); // retrace : skip tgt weights
      MB.setValues(bID, t+1, critc->forward(bID, t+1)[0]);
    }
    target = MB.returnEstimate(bID, t);
  } else {
    target = MB.reward(bID, t);
    if (not MB.isTerminal(bID, t+1) && not isOff) {
      actor->forward_tgt(bID, t+1); // policy at next step, with tgt weights
      critc->setAddedInputType(NETWORK, bID, t+1, -1);
      const Rvec v_next = critc->forward_tgt(bID, t+1); //target value s_next
      MB.setValues(bID, t+1, v_next[0]);
      target += gamma * v_next[0];
    }
  }

  //code to compute deterministic policy grad:
  Rvec polGrad = isOff? Rvec(nA,0) : critc->oneStepBackProp({1}, bID, t, -1);
  assert(polGrad.size() == nA); polGrad.resize(2*nA, 0); // space for stdev
  // In order to enable learning stdev on request, stdev is part of actor output
  #ifdef DPG_LEARN_STDEV
    const Real polGradCoef = (target - pval[0]) * std::min(CmaxRet, RHO);
    const Rvec SPG = POL.policyGradient(MB.action(bID,t), polGradCoef);
    for (Uint i=0; i<nA; ++i) polGrad[i+nA] = isOff? 0 : SPG[i+nA];
  #else
    // Next line keeps stdev at user's value, else NN penal might cause drift.
    const Rvec fixGrad = POL.fixExplorationGrad(settings.explNoise);
    for (Uint i=0; i<nA; ++i) polGrad[i+nA] = fixGrad[i+nA];
  #endif

  //if(!thrID) cout << "G "<<print(detPolG) << endl;
  const Rvec penGrad = POL.KLDivGradient(MB.mu(bID,t), -1);
  Rvec finalG = Rvec(actor->nOutputs(), 0);
  POL.makeNetworkGrad(finalG, Utilities::weightSum2Grads(polGrad,penGrad,beta));
  actor->setGradient(finalG, bID, t);

  const Rvec valueG = { isOff ? 0 : ( target - qval[0] ) };
  critc->setGradient(valueG, bID, t);

  //bookkeeping:
  MB.setMseDklImpw(bID, t, target-qval[0], DKL, RHO, CmaxRet, CinvRet);
  MB.setValues(bID, t, pval[0], qval[0]);
}

void DPG::selectAction(const MiniBatch& MB, Agent& agent)
{
  for (const auto & net : networks ) net->load(MB, agent, 0);
  //Compute policy and value on most recent element of the sequence.
  const Continuous_policy POL(aInfo, actor->forward(agent));

  // if explNoise is 0, we just act according to policy
  // since explNoise is initial value of diagonal std vectors
  // this should only be used for evaluating a learned policy
  const bool bSample = settings.explNoise>0;
  Rvec action = OrUhDecay<=0? POL.selectAction(agent, bSample) :
      POL.selectAction_OrnsteinUhlenbeck(agent, bSample, OrUhState[agent.ID]);
  agent.setAction(action, POL.getVector());

  //careful! act may be scaled to agent's action space, mean/sampAct aren't
  critc->setAddedInputType(ACTION, agent, MB.indCurrStep());
  const Rvec qval = critc->forward(agent);
  critc->setAddedInputType(NETWORK, agent, MB.indCurrStep());
  const Rvec sval = critc->forward(agent, true); // overwrite = true
  MB.appendValues(sval[0], qval[0]);
}

void DPG::processTerminal(const MiniBatch& MB, Agent& agent)
{
  // whether episode is truncated or terminated, action advantage is 0
  if( agent.agentStatus == LAST ) {
    for (const auto & net : networks ) net->load(MB, agent, 0);
    const Rvec pvec = actor->forward(agent); // grab pol mean
    critc->setAddedInputType(NETWORK, agent, MB.indCurrStep());
    MB.appendValues(critc->forward(agent)[0]); // not a terminal state
  } else MB.appendValues(0); // value of terminal state is 0
  OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
}

void DPG::setupTasks(TaskQueue& tasks)
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
    processMemoryBuffer(); // find old eps, update avg quantities ...
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
    globalGradCounterUpdate(); // step ++
    algoSubStepID = 0; // rinse and repeat
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

DPG::DPG(MDPdescriptor& M, HyperParameters& S, ExecutionInfo& D):
  Learner_approximator(M, S, D)
{
  if(settings.clipImpWeight > 0)
    printf("Using ReF-ER with clipping parameter C=%f, tolerance D=%f "
           "and annealing E=%f\n", S.clipImpWeight, S.penalTol, S.epsAnneal);
  const bool bCreatedEncorder = createEncoder();
  assert(networks.size() == bCreatedEncorder? 1 : 0);
  Approximator* const encoder = bCreatedEncorder? networks[0] : nullptr;
  if(bCreatedEncorder) encoder->initializeNetwork();

  networks.push_back(
    new Approximator("policy", settings, distrib, data.get(), encoder)
  );
  actor = networks.back();
  actor->buildFromSettings(nA);
  actor->setUseTargetNetworks();
  Rvec stdParam = Continuous_policy::initial_Stdev(aInfo, settings.explNoise);
  actor->getBuilder().addParamLayer(nA, "Linear", stdParam);
  actor->initializeNetwork();

  networks.push_back(
    new Approximator("critic", settings, distrib, data.get(), encoder, actor)
  );
  critc = networks.back();
  critc->setAddedInput(NETWORK, nA);
  critc->setUseTargetNetworks();
  // update settings that are going to be read by critic:
  settings.learnrate *= 10; // DPG wants critic faster than actor
  settings.nnLambda = 1e-4; // also wants L2 penl coef
  settings.nnOutputFunc = "Linear"; // critic must be linear
  critc->buildFromSettings(1);
  critc->initializeNetwork();
}

}
