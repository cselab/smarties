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

#include "../Math/Quadratic_advantage.h"

static inline Quadratic_advantage prepare_advantage(const Rvec&O,
  const ActionInfo*const aI, const vector<Uint>& net_inds)
{
  return Quadratic_advantage(vector<Uint>{net_inds[1], net_inds[2]}, aI, O);
}

NAF::NAF(Environment*const _env, Settings & _set) :
Learner_offPolicy(_env, _set)
{
  _set.splitLayers = 0;
  F.push_back(new Approximator("net", _set, input, data));
  const Uint nOutp = 1 +aInfo.dim +Quadratic_advantage::compute_nL(&aInfo);
  assert(nOutp == net_outputs[0] + net_outputs[1] + net_outputs[2]);
  Builder build_pol = F[0]->buildFromSettings(_set, nOutp);
  F[0]->initializeNetwork(build_pol);
  test();
  printf("NAF\n");
  trainInfo = new TrainData("NAF", _set, 0, "| beta | avgW ", 2);
}

void NAF::select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    F[0]->prepare_agent(traj, agent);
    //Compute policy and value on most recent element of the sequence.
    const Rvec output = F[0]->forward_agent(agent);
    //cout << print(output) << endl;
    Rvec polvec = Rvec(&output[net_indices[2]], &output[net_indices[2]]+nA);

    #ifndef NDEBUG
      const Quadratic_advantage advantage = prepare_advantage(output, &aInfo, net_indices);
      Rvec polvec2 = advantage.getMean();
      assert(polvec.size() == polvec2.size());
      for(Uint i=0;i<nA;i++) assert(abs(polvec[i]-polvec2[i])<2e-16);
    #endif

    polvec.resize(policyVecDim, stdParam);
    assert(polvec.size() == 2 * nA);
    Gaussian_policy policy({0, nA}, &aInfo, polvec);
    const Rvec MU = policy.getVector();
    //cout << print(MU) << endl;
    Rvec act = policy.finalize(explNoise>0, &generators[nThreads+agent.ID], MU);
    if(OrUhDecay>0)
      act = policy.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);

    agent.act(act);
    data_get->add_action(agent, MU);
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data_get->terminate_seq(agent);
  }
}

void NAF::TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
  const Uint thrID) const
{
  die("");
}

void NAF::Train(const Uint seq, const Uint samp, const Uint wID,
  const Uint bID, const Uint thrID) const
{
  if(thrID==0) profiler->stop_start("FWD");

  Sequence* const traj = data->get(seq);
  F[0]->prepare_one(traj, samp, thrID, wID);

  const Rvec output = F[0]->forward(samp, thrID);

  if(thrID==0) profiler->stop_start("CMP");
  // prepare advantage and policy
  const Quadratic_advantage ADV = prepare_advantage(output, &aInfo,net_indices);
  Rvec polvec = ADV.getMean();            assert(polvec.size() == nA);
  polvec.resize(policyVecDim, stdParam);
  assert(polvec.size() == 2 * nA);
  Gaussian_policy POL({0, nA}, &aInfo, polvec);
  POL.prepare(traj->tuples[samp]->a, traj->tuples[samp]->mu);
  //cout << POL.sampImpWeight << " " << POL.sampKLdiv << " " << CmaxRet << endl;

  const Real Qsold = output[net_indices[0]] + ADV.computeAdvantage(POL.sampAct);
  const bool isOff= dropRule==1? false :
                     traj->isFarPolicy(samp, POL.sampImpWeight,CmaxRet,CinvRet);

  Real Vsnew = data->scaledReward(traj, samp+1);
  if (not traj->isTerminal(samp+1) && not isOff) {
    const Rvec target = F[0]->forward_tgt(samp+1, thrID);
    Vsnew += gamma*target[net_indices[0]];
  }
  const Real error = isOff? 0 : Vsnew - Qsold;
  Rvec grad(F[0]->nOutputs());
  grad[net_indices[0]] = error;
  ADV.grad(POL.sampAct, error, grad);
  if(CmaxRet>1 && beta<1 && dropRule!=2) { // then ReFER
    const Rvec penG = POL.div_kl_grad(traj->tuples[samp]->mu, -1);
    for(Uint i=0; i<nA; i++)
      grad[net_indices[2]+i] = beta*grad[net_indices[2]+i] + (1-beta)*penG[i];
  }

  trainInfo->log(Qsold, error, {beta, POL.sampImpWeight}, thrID);
  traj->setMseDklImpw(samp, error*error, POL.sampKLdiv, POL.sampImpWeight, CmaxRet, CinvRet);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(grad, samp, thrID);
  F[0]->gradient(thrID);
}

void NAF::test()
{
  Rvec out(F[0]->nOutputs()), act(aInfo.dim);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  const int thrID = omp_get_thread_num();
  for(Uint i = 0; i<aInfo.dim; i++) act[i] = act_dis(generators[thrID]);
  for(Uint i = 0; i<F[0]->nOutputs(); i++) out[i] = out_dis(generators[thrID]);
  Quadratic_advantage A = prepare_advantage(out, &aInfo, net_indices);
  A.test(act, &generators[thrID]);
}
