//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ACER.h"

#include "../Network/Builder.h"
#include "../Utils/StatsTracker.h"
#include "../Network/Approximator.h"
#include "../Math/Gaussian_policy.h"
#include "../ReplayMemory/Collector.h"
#include "../Utils/SstreamUtilities.h"

#define SEQ_CUTOFF 200

namespace smarties
{

static inline Gaussian_policy prepare_policy(const Rvec & out,
                                             const ActionInfo & aInfo,
                                             const Rvec ACT = Rvec(),
                                             const Rvec MU  = Rvec())
{
  Gaussian_policy pol({0, aInfo.dim()}, aInfo, out);
  if(ACT.size()) {
    assert(MU.size());
    pol.prepare(ACT, MU);
  }
  return pol;
}

void ACER::Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Sint ndata = MB.nDataSteps(bID), thrID = omp_get_thread_num();

  std::uniform_int_distribution<Sint> dStart(0, ndata-1);
  const Sint tst_samp = dStart(generators[thrID]);
  const Sint tstart = std::min(tst_samp, std::max(ndata-SEQ_CUTOFF, (Sint)0) );
  const Sint tend   = std::min(ndata, tstart+SEQ_CUTOFF);
  const Sint nsteps = tend - tstart;

  Rvec Vstates(nsteps, 0);
  std::vector<Rvec> policy_samples(nsteps);
  std::vector<Gaussian_policy> policies, policies_tgt;
  policies_tgt.reserve(nsteps);
  policies.reserve(nsteps);
  std::vector<Rvec> advantages(nsteps, Rvec(2+nAexpectation, 0));

  if(thrID==0) profiler->start("FWD");

  for(Sint step=tstart, i=0; step < tend; ++step, ++i)
  {
    policies.push_back( prepare_policy(actor->forward(bID, step), aInfo,
                                       MB.action(bID,step), MB.mu(bID,step)) );
    assert((Sint) policies.size() == i+1);
    policies_tgt.push_back(prepare_policy(actor->forward_tgt(bID,step), aInfo));

    Vstates[i] = value->forward(bID, step) [0];

    advtg->setAddedInput(policies[i].sampAct, bID, step, 0);
    //if(thrID==0) cout << "Action: " << print(policies[i].sampAct) << endl;
    advantages[i][0] = advtg->forward(bID, step, 0) [0]; // sample 0
    policy_samples[i] = policies[i].sample(generators[thrID]);
    //if(thrID==0) cout << "Sample: " << print(policy_samples[i]) << endl;
    advtg->setAddedInput(policy_samples[i], bID, step, 1);
    advantages[i][1] = advtg->forward(bID, step, 1) [0]; // sample 1

    for(Uint k=0, samp=2; k<nAexpectation; ++k, ++samp) {
      const Rvec extraPolSample = policies[i].sample(generators[thrID]);
      advtg->setAddedInput(extraPolSample, bID, step, samp);
      advantages[i][samp] = advtg->forward(bID, step, samp) [0]; // sample
    }
  }

  Real Q_RET = MB.reward(bID, tend);
  if (not MB.isTerminal(bID,tend)) Q_RET += gamma * value->forward(bID,tend)[0];
  Real Q_OPC = Q_RET;

  if(thrID==0)  profiler->stop_start("POL");

  for(int step=tend-1, i=nsteps-1; step>=tstart; --step, --i)
  {
    Real QTheta = Vstates[i] + advantages[i][0], APol = advantages[i][1];
    for(Uint samp = 0; samp < nAexpectation; ++samp) {
      QTheta -= advantages[i][2+samp] / nAexpectation;
      APol   -= advantages[i][2+samp] / nAexpectation;
    }

    const Real RHO = policies[i].sampImpWeight, DKL = policies[i].sampKLdiv;
    const Real W = std::min((Real)1, RHO), C = std::pow(W, acerTrickPow);
    const Real R = MB.reward(bID, step), A_OPC = Q_OPC - Vstates[i];

    const Real polProbBehavior = policies[i].evalBehavior(policy_samples[i],
                                                          MB.mu(bID, step));
    const Real polProbOnPolicy = policies[i].evalProbability(policy_samples[i]);
    const Real rho_pol = polProbOnPolicy / polProbBehavior;
    const Real gain1 = A_OPC * std::min((Real) 5, policies[i].sampImpWeight);
    const Real gain2 = APol  * std::max((Real) 0, 1 - 5/rho_pol);
    const Rvec gradAcer_1 = policies[i].policy_grad(policies[i].sampAct, gain1);
    const Rvec gradAcer_2 = policies[i].policy_grad(policy_samples[i],   gain2);
    const Rvec penal = policies[i].div_kl_grad(&policies_tgt[i], 1);
    const Rvec grad = Utilities::sum2Grads(gradAcer_1, gradAcer_2);
    const Rvec clipped = Utilities::trust_region_update(grad, penal, 2 * nA,1);
    const Rvec pGrad = policies[i].finalize_grad(clipped);

    const Real Q_err = Q_RET - QTheta, V_err = Q_err * W;
    actor->setGradient(          pGrad  , bID, step);
    value->setGradient({ V_err + Q_err }, bID, step);
    advtg->setGradient({         Q_err }, bID, step);
    for(Uint samp = 0; samp < nAexpectation; ++samp)
      advtg->setGradient({ -Q_err/nAexpectation }, bID, step, samp+2);
    //prepare Q with off policy corrections for next step:
    Q_RET = R + gamma*( C * (Q_RET - QTheta) + Vstates[i]);
    Q_OPC = R + gamma*(     (Q_OPC - QTheta) + Vstates[i]); // as paper, but bad
    //Q_OPC = R + gamma*(     (Q_OPC - QTheta) + Vstates[i]); // as ret, better
    const Rvec penalBehavior = policies[i].div_kl_grad(MB.mu(bID, step), -1);
    MB.setMseDklImpw(bID, step, Q_err*Q_err, DKL, RHO, CmaxRet, CinvRet);
    trainInfo->log(QTheta, Q_err, pGrad, penalBehavior, {RHO}, thrID);
  }
}

void ACER::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = * data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(&EP);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    if (encoder) encoder->load(MB, agent);
    actor->load(MB, agent);
    //Compute policy and value on most recent element of the sequence.
    Gaussian_policy POL = prepare_policy(actor->forward(agent), aInfo);
    Rvec MU = POL.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto act = POL.selectAction(agent, MU, settings.explNoise>0);

    agent.act(act);
    data_get->add_action(agent, MU);
  }
  else data_get->terminate_seq(agent); // either terminal or truncated episode
}

void ACER::setupTasks(TaskQueue& tasks)
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
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

ACER::ACER(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_):
  Learner_approximator(MDP_, S_, D_)
{
  const bool bCreatedEncorder = createEncoder();
  assert(networks.size() == bCreatedEncorder? 1 : 0);
  encoder = bCreatedEncorder? networks[0] : nullptr;
  if(bCreatedEncorder) encoder->initializeNetwork();

  networks.push_back(
    new Approximator("policy", settings, distrib, data.get(), encoder)
  );
  actor = networks.back();

  networks.push_back(
    new Approximator("critic", settings, distrib, data.get(), encoder)
  );
  value = networks.back();

  networks.push_back(
    new Approximator("advntg", settings, distrib, data.get(), encoder)
  );
  advtg = networks.back();

  {
    actor->buildFromSettings(nA);
    actor->setUseTargetNetworks();
    const Rvec stdParam = Gaussian_policy::initial_Stdev(aInfo, explNoise);
    actor->getBuilder().addParamLayer(nA, "Linear", stdParam);
    actor->initializeNetwork();
  }

  // update settings that are going to be read by value nets:
  settings.learnrate *= 10; // faster
  settings.targetDelay = 0; // unneeded for adv and val targets
  settings.nnOutputFunc = "Linear"; // critics must be linear

  {
    value->buildFromSettings(1);
    value->initializeNetwork();
  }

  {
    advtg->setAddedInput(VECTOR, nA);
    advtg->setNumberOfAddedSamples(nAexpectation + 1);
    advtg->buildFromSettings(1);
    advtg->initializeNetwork();

  }

  settings.learnrate /= 10; // reset just in case it's needed
  trainInfo = new TrainData("ACER", distrib, 1, "| avgW ", 1);
}

}
