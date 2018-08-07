//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../StateAction.h"
#include "../Network/Builder.h"
#include "NAF.h"

NAF::NAF(Environment*const _env, Settings & _set) :
Learner_offPolicy(_env, _set)
{
  _set.splitLayers = 0;
  F.push_back(new Approximator("net", _set, input, data));
  const Uint nOutp = 1 +aInfo.dim +Quadratic_advantage::compute_nL(&aInfo);
  assert(nOutp == net_outputs[0] + net_outputs[1] + net_outputs[2]);
  Builder build_pol = F[0]->buildFromSettings(_set, nOutp);
  F[0]->initializeNetwork(build_pol, 0);
  test();
  printf("NAF\n");
  trainInfo = new TrainData("NAF", _set, 0, "| beta | avgW ", 2);
}

void NAF::select(Agent& agent)
{
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    F[0]->prepare_agent(traj, agent);
    //Compute policy and value on most recent element of the sequence.
    const Rvec output = F[0]->forward_agent(traj, agent);
    //cout << print(output) << endl;
    Rvec polvec = Rvec(&output[net_indices[2]], &output[net_indices[2]]+nA);
    #ifndef NDEBUG
      const Quadratic_advantage advantage = prepare_advantage(output);
      Rvec polvec2 = advantage.getMean();
      assert(polvec.size() == polvec2.size());
      for(Uint i=0;i<nA;i++) assert(abs(polvec[i]-polvec2[i])<2e-16);
    #endif
    polvec.resize(policyVecDim, noiseMap_inverse(explNoise));
    assert(polvec.size() == 2 * nA);
    Gaussian_policy policy({0, nA}, &aInfo, polvec);
    const Rvec MU = policy.getVector();
    //cout << print(MU) << endl;
    Rvec act = policy.finalize(explNoise>0, &generators[nThreads+agent.ID], MU);
    if(OrUhDecay>0)
      act = policy.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);

    agent.act(act);
    data->add_action(agent, MU);
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data->terminate_seq(agent);
  }
}

void NAF::TrainBySequences(const Uint seq, const Uint thrID) const
{
  die("");
}

void NAF::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  if(thrID==0) profiler->stop_start("FWD");

  Sequence* const traj = data->Set[seq];
  F[0]->prepare_one(traj, samp, thrID);

  const Rvec output = F[0]->forward(traj, samp, thrID);

  if(thrID==0) profiler->stop_start("CMP");
  // prepare advantage and policy
  const Quadratic_advantage ADV = prepare_advantage(output);
  Rvec polvec = ADV.getMean();            assert(polvec.size() == nA);
  polvec.resize(policyVecDim, noiseMap_inverse(explNoise));
  assert(polvec.size() == 2 * nA);
  Gaussian_policy POL({0, nA}, &aInfo, polvec);
  POL.prepare(traj->tuples[samp]->a, traj->tuples[samp]->mu);
  //cout << POL.sampImpWeight << " " << POL.sampKLdiv << " " << CmaxRet << endl;

  const Real Qsold = output[net_indices[0]] + ADV.computeAdvantage(POL.sampAct);
  const bool isOff = traj->isFarPolicy(samp, POL.sampImpWeight, CmaxRet);

  Real Vsnew = data->scaledReward(traj, samp+1);
  if (not traj->isTerminal(samp+1) && not isOff) {
    const Rvec target = F[0]->forward<TGT>(traj, samp+1, thrID);
    Vsnew += gamma*target[net_indices[0]];
  }
  const Real error = isOff? 0 : Vsnew - Qsold;
  Rvec grad(F[0]->nOutputs());
  grad[net_indices[0]] = error;
  ADV.grad(POL.sampAct, error, grad);
  if(CmaxRet>1 && beta<1) { // then ReFER
    const Rvec penG = POL.div_kl_grad(traj->tuples[samp]->mu, -1);
    for(Uint i=0; i<nA; i++)
      grad[net_indices[2]+i] = beta*grad[net_indices[2]+i] + (1-beta)*penG[i];
  }

  trainInfo->log(Qsold, error, {beta, POL.sampImpWeight}, thrID);
  traj->setMseDklImpw(samp, error*error, POL.sampKLdiv, POL.sampImpWeight);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(grad, traj, samp, thrID);
  F[0]->gradient(thrID);
}
