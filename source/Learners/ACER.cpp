//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Network/Builder.h"
#include "../Network/Aggregator.h"
#include "../Math/Gaussian_policy.h"
#include "ACER.h"

static inline Gaussian_policy prepare_policy(const Rvec&O,
  const ActionInfo*const aI, const Tuple*const t = nullptr) {
  Gaussian_policy pol({0, aI->dim}, aI, O);
  if(t not_eq nullptr) pol.prepare(t->a, t->mu);
  return pol;
}

void ACER::TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
  const Uint thrID) const
{
  Sequence* const traj = data->get(seq);
  const int ndata = traj->tuples.size()-1;
  //policy : we need just 2 calls: pi pi_tilde
   F[0]->prepare_seq(traj, thrID, wID);
   F[1]->prepare_seq(traj, thrID, wID);
  relay->prepare_seq(traj, thrID, ACT);
  //advantage : 1+nAexpect [A(s,a)] + 1 [A(s,a'), same normalization] calls
   F[2]->prepare_seq(traj, thrID, 1+nAexpectation);

  Rvec Vstates(ndata, 0);
  vector<Rvec> policy_samples(ndata);
  vector<Gaussian_policy> policies, policies_tgt;
  policies_tgt.reserve(ndata); policies.reserve(ndata);
  vector<Rvec> advantages(ndata, Rvec(2+nAexpectation, 0));

  if(thrID==0) profiler->stop_start("FWD");
  for(Uint k=0; k<(Uint)ndata; k++)
  {
    const Rvec outPc = F[0]->forward_cur(k, thrID);
    policies.push_back(prepare_policy(outPc, &aInfo, traj->tuples[k]));
    assert(policies.size() == k+1);
    const Rvec outPt = F[0]->forward_tgt(k, thrID);
    policies_tgt.push_back(prepare_policy(outPt, &aInfo));
    const Rvec outVs = F[1]->forward(k, thrID);

    relay->set(policies[k].sampAct, k, thrID);
    //if(thrID) cout << "Action: " << print(policies[k].sampAct) << endl;
    const Rvec At = F[2]->forward_cur(k, thrID);
    policy_samples[k] = policies[k].sample(&generators[thrID]);
    //if(thrID) cout << "Sample: " << print(policy_samples[k]) << endl;
    relay->set(policy_samples[k], k, thrID);
    const Rvec Ap = F[2]->forward_cur<TGT>(k, thrID);
    advantages[k][0] = At[0]; advantages[k][1] = Ap[0]; Vstates[k] = outVs[0];
    for(Uint i = 0; i < nAexpectation; i++) {
      relay->set(policies[k].sample(&generators[thrID]), k, thrID);
      const Rvec A = F[2]->forward(k, thrID, 1+i);
      advantages[k][2+i] = A[0];
    }
    //cout << print(advantages[k]) << endl; fflush(0);
  }
  Real Q_RET = data->scaledReward(traj, ndata);
  Real Q_OPC = data->scaledReward(traj, ndata);
  if ( not traj->ended ) {
    const Rvec v_term = F[1]->forward(ndata, thrID);
    Q_RET += gamma*v_term[0];
    Q_OPC += gamma*v_term[0];
  }
  if(thrID==0)  profiler->stop_start("POL");
  for(int k=ndata-1; k>=0; k--)
  {
    const Tuple*const T = traj->tuples[k];
    Real QTheta = Vstates[k]+advantages[k][0], APol = advantages[k][1];
    for(Uint i = 0; i < nAexpectation; i++) {
      QTheta -= facExpect*advantages[k][2+i];
      APol -= facExpect*advantages[k][2+i];
    }
    const Real A_OPC = Q_OPC - Vstates[k], Q_err = Q_RET - QTheta;

    const Real rho = policies[k].sampImpWeight;
    const Real W = std::min((Real)1, rho);
    const Real C = std::pow(W, acerTrickPow);
    const Real dkl = policies[k].sampKLdiv;
    const Real R = data->scaledReward(traj, k);
    const Real V_err = Q_err*W;
    const Rvec pGrad = policyGradient(T, policies[k], policies_tgt[k], A_OPC,
      APol, policy_samples[k]);

    F[0]->backward(pGrad, k, thrID);
    F[1]->backward({(V_err+Q_err)}, k, thrID);
    F[2]->backward({Q_err}, k, thrID);
    for(Uint i = 0; i < nAexpectation; i++)
      F[2]->backward({-facExpect*Q_err}, k, thrID, i+1);
    //prepare Q with off policy corrections for next step:
    Q_RET = R +gamma*( C*(Q_RET-QTheta) +Vstates[k]);
    Q_OPC = R +gamma*(   (Q_OPC-QTheta) +Vstates[k]); // as paper, but bad
    //Q_OPC = R +gamma*( C*(Q_OPC-QTheta) +Vstates[k]);
    const Rvec penal = policies[k].div_kl_grad(T->mu, -1);
    traj->setMseDklImpw(k, Q_err*Q_err, dkl, rho, CmaxRet, CinvRet);
    trainInfo->log(QTheta, Q_err, pGrad, penal, {rho}, thrID);
  }

  if(thrID==0)  profiler->stop_start("BCK");
   F[0]->gradient(thrID);
   F[1]->gradient(thrID);
   F[2]->gradient(thrID);
}

void ACER::Train(const Uint seq, const Uint samp, const Uint wID,
  const Uint bID, const Uint thrID) const
{
  die("not allowed");
}

Rvec ACER::policyGradient(const Tuple*const _t, const Gaussian_policy& POL,
  const Gaussian_policy& TGT, const Real ARET, const Real APol,
  const Rvec& pol_samp) const
{
  //compute quantities needed for trunc import sampl with bias correction
  const Real polProbBehavior = POL.evalBehavior(pol_samp, _t->mu);
  const Real polProbOnPolicy = POL.evalProbability(pol_samp);
  const Real rho_pol = polProbOnPolicy / polProbBehavior;
  const Real gain1 = ARET*std::min((Real) 5, POL.sampImpWeight);
  const Real gain2 = APol*std::max((Real) 0, 1-5/rho_pol);
  const Rvec gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
  const Rvec gradAcer_2 = POL.policy_grad(pol_samp,    gain2);
  const Rvec penal = POL.div_kl_grad(&TGT, 1);
  const Rvec grad = sum2Grads(gradAcer_1, gradAcer_2);
  const Rvec trust = trust_region_update(grad, penal, 2*nA, 1);
  return POL.finalize_grad(trust);
}

void ACER::select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);

  if( agent.Status < TERM_COMM ) {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    F[0]->prepare_agent(traj, agent);
    Rvec output = F[0]->forward_agent(agent);
    Gaussian_policy pol = prepare_policy(output, &aInfo);
    Rvec mu = pol.getVector();
    const auto act=pol.finalize(explNoise>0,&generators[nThreads+agent.ID], mu);
    agent.act(act);
    data_get->add_action(agent, mu);
  }
  else data_get->terminate_seq(agent);
}

ACER::ACER(Environment*const _env, Settings&_set): Learner_offPolicy(_env,_set)
{
  _set.splitLayers = 0;
  #if 1
    createSharedEncoder();
  #endif

  relay = new Aggregator(_set, data, _env->aI.dim);
  F.push_back(new Approximator("policy", _set, input, data));
  F.push_back(new Approximator("critic", _set, input, data));
  F.push_back(new Approximator("advntg", _set, input, data, relay));

  Builder build_pol = F[0]->buildFromSettings(_set, nA);

  #ifdef EXTRACT_COVAR
    const Real stdParam = noiseMap_inverse(explNoise*explNoise);
  #else
    const Real stdParam = noiseMap_inverse(explNoise);
  #endif
  build_pol.addParamLayer(nA, "Linear", stdParam);
  Builder build_val = F[1]->buildFromSettings(_set, 1 ); // V
  Builder build_adv = F[2]->buildFromSettings(_set, 1 ); // A

  F[0]->initializeNetwork(build_pol);
  _set.learnrate *= 3;
  _set.targetDelay = 0; // unneeded for adv and val targets
  F[1]->initializeNetwork(build_val);
  F[2]->initializeNetwork(build_adv);
  _set.learnrate /= 3;
  F[2]->allocMorePerThread(nAexpectation);
  printf("ACER\n");
  trainInfo = new TrainData("acer", _set, 1, "| avgW ", 1);

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
    for(Uint i=0;  i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=nA; i<mu.size(); i++) mu[i] = std::exp(mu[i]);

    Gaussian_policy pol = prepare_policy(output, &aInfo);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}
