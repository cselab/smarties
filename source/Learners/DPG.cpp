//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "DPG.h"

#include "../Network/Builder.h"
#include "../Utils/StatsTracker.h"
#include "../Math/Continuous_policy.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/Collector.h"
#include "../Utils/SstreamUtilities.h"

//#define DKL_filter
#define DPG_RETRACE_TGT
#define DPG_LEARN_STDEV

namespace smarties
{

void DPG::Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->start("FWD");
  const Rvec pvec = actor->forward(bID, t); // network compute
  const Continuous_policy POL({0, aInfo.dim()}, aInfo, pvec);
  const Real RHO = POL.importanceWeight(MB.action(bID,t), MB.mu(bID,t));
  const Real DKL = POL.KLDivergence(MB.mu(bID,t));
  const bool isOff = isFarPolicy(RHO, CmaxRet, CinvRet);

  critc->setAddedInputType(ACTION, bID, t);
  const Rvec qval = critc->forward(bID, t); // network compute
  critc->setAddedInputType(NETWORK, bID, t, -1); //-1 flags to write on separate
  const Rvec pval = critc->forward(bID, t, -1); //net alloc, with target wegiths

  #ifdef DPG_RETRACE_TGT
    if( MB.isTruncated(bID, t+1) ) {
      actor->forward(bID, t+1);
      critc->setAddedInputType(NETWORK, bID, t+1); // retrace : skip tgt weights
      const Rvec v_next = critc->forward(bID, t+1); // value with state+policy
      MB.updateRetrace(bID, t+1, 0, v_next[0], 0);
    }
    const Real target = MB.Q_RET(bID, t), advantage = qval[0] - pval[0];
    const Real dQRET = MB.updateRetrace(bID, t, advantage, pval[0], RHO);
  #else
    Real target = MB.reward(bID, t);
    if (not MB.isTerminal(bID, t+1) && not isOff) {
      actor->forward_tgt(bID, t+1); // policy at next step, with tgt weights
      critc->setAddedInputType(NETWORK, bID, t+1, -1);
      const Rvec v_next = critc->forward_tgt(bID, t+1); //target value s_next
      target += gamma * v_next[0];
    }
  #endif

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
    const Rvec fixGrad = POL.fixExplorationGrad(explNoise);
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
  MB.setMseDklImpw(bID, t, std::pow(target-qval[0],2), DKL,RHO,CmaxRet,CinvRet);
  #ifdef DPG_RETRACE_TGT
    trainInfo->log(qval[0],valueG[0], polGrad,penGrad, {beta,dQRET,RHO}, thrID);
  #else
    trainInfo->log(qval[0],valueG[0], polGrad,penGrad, {beta, RHO}, thrID);
  #endif
}

void DPG::select(Agent& agent)
{
  data_get->add_state(agent);
  Episode& EP = data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(EP);
  for (const auto & net : networks ) net->load(MB, agent, 0);
  #ifdef DPG_RETRACE_TGT
    const Uint currStep = EP.nsteps() - 1; assert(EP.nsteps()>0);
  #endif

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    const Continuous_policy POL({0, aInfo.dim()}, aInfo, actor->forward(agent));

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    Rvec action = OrUhDecay<=0? POL.selectAction(agent, settings.explNoise>0) :
        POL.selectAction_OrnsteinUhlenbeck(agent, settings.explNoise>0, OrUhState[agent.ID]);
    agent.setAction(action);
    data_get->add_action(agent, POL.getVector());

    #ifdef DPG_RETRACE_TGT
      //careful! act may be scaled to agent's action space, mean/sampAct aren't
      critc->setAddedInputType(ACTION, agent, currStep);
      const Rvec qval = critc->forward(agent);
      critc->setAddedInputType(NETWORK, agent, currStep);
      const Rvec sval = critc->forward(agent, true); // overwrite = true
      EP.action_adv.push_back(qval[0]-sval[0]);
      EP.state_vals.push_back(sval[0]);
    #endif
  }
  else // either terminal or truncation state
  {
    #ifdef DPG_RETRACE_TGT
      if( agent.agentStatus == TRNC ) {
        const Rvec pvec = actor->forward(agent); // grab pol mean
        critc->setAddedInputType(NETWORK, agent, currStep);
        const Rvec sval = critc->forward(agent);
        EP.state_vals.push_back(sval[0]); // not a terminal state
      } else {
        EP.state_vals.push_back(0); //value of terminal state is 0
      }
      //whether seq is truncated or terminated, act adv is undefined:
      EP.action_adv.push_back(0);
      const Uint N = EP.nsteps();
      // compute initial Qret for whole trajectory:
      assert(N == EP.action_adv.size() && N == EP.state_vals.size());
      assert(0 == EP.Q_RET.size());
      //within Retrace, we use the Q_RET vector to write the Adv retrace values
      EP.Q_RET.resize(N, 0);
      EP.offPolicImpW.resize(N, 1);
      for(Uint i=EP.ndata(); i>0; --i)
        EP.propagateRetrace(i, gamma, data->scaledReward(EP, i));
    #endif

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
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
    processMemoryBuffer(); // find old eps, update avg quantities ...
    #ifdef DPG_RETRACE_TGT
    debugL("Update Retrace est. for episodes sampled in prev. grad update");
    updateRetraceEstimates();
    #endif
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

DPG::DPG(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_):
  Learner_approximator(MDP_, S_, D_)
{
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
  const Rvec stdParam = Continuous_policy::initial_Stdev(aInfo, explNoise);
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

  #ifdef DPG_RETRACE_TGT
    computeQretrace = true;
    trainInfo = new TrainData("DPG", distrib, 1, "| beta | dAdv | avgW ", 3);
  #else
    trainInfo = new TrainData("DPG", distrib, 1, "| beta | avgW ", 2);
  #endif
}

}
