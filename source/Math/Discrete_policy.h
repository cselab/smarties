//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Utils.h"
#include <algorithm>

struct Discrete_policy
{
  const ActionInfo* const aInfo;
  const Uint start_prob, nA;
  const Rvec netOutputs;
  const Rvec unnorm;
  const Real normalization;
  const Rvec probs;

  Uint sampAct;
  Real sampPonPolicy=0, sampPBehavior=0, sampImpWeight=0, sampKLdiv=0;

  inline Uint map_action(const Rvec& sent) const {
    return aInfo->actionToLabel(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI) {
    assert(aI->maxLabel);
    return aI->maxLabel;
  }
  static void setInitial_Stdev(const ActionInfo*const aI, Rvec&O, const Real S)
  {
    #ifdef EXTRACT_COVAR
      for(Uint e=0; e<aI->dim; e++) O.push_back(noiseMap_inverse(S*S));
    #else
      for(Uint e=0; e<aI->dim; e++) O.push_back(noiseMap_inverse(S));
    #endif
  }

  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias) { }

  Discrete_policy(const vector<Uint>& start, const ActionInfo*const aI,
    const Rvec& out) : aInfo(aI), start_prob(start[0]), nA(aI->maxLabel), netOutputs(out), unnorm(extract_unnorm()),
    normalization(compute_norm()), probs(extract_probabilities())
    {
      //printf("Discrete_policy: %u %u %u %lu %lu %lu %lu\n",
      //start_prob,start_vals,nA,netOutputs.size(),
      //unnorm.size(),vals.size(),probs.size());
    }

 private:
  inline Rvec extract_unnorm() const
  {
    assert(netOutputs.size()>=start_prob+nA);
    Rvec ret(nA);
    for (Uint j=0; j<nA; j++) ret[j] = unbPosMap_func(netOutputs[start_prob+j]);
    return ret;
  }

  inline Real compute_norm() const
  {
    assert(unnorm.size()==nA);
    Real ret = 0;
    for (Uint j=0; j<nA; j++) { ret += unnorm[j]; assert(unnorm[j]>0); }
    return ret + nnEPS;
  }

  inline Rvec extract_probabilities() const
  {
    assert(unnorm.size()==nA);
    Rvec ret(nA);
    for (Uint j=0; j<nA; j++) ret[j] = unnorm[j]/normalization;
    return ret;
  }

 public:
  inline void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampPonPolicy = probs[sampAct];
    sampPBehavior = beta[sampAct];
    sampImpWeight = sampPonPolicy / sampPBehavior;
    sampKLdiv = kl_divergence(beta);
  }

  static inline Real evalBehavior(const Uint& act, const Rvec& beta) {
    return beta[act];
  }

  static inline Uint sample(mt19937*const gen, const Rvec& beta) {
    std::discrete_distribution<Uint> dist(beta.begin(), beta.end());
    return dist(*gen);
  }

  inline Uint sample(mt19937*const gen) const {
    std::discrete_distribution<Uint> dist(probs.begin(), probs.end());
    return dist(*gen);
  }

  inline Real evalProbability(const Uint act) const {
    return probs[act];
  }

  inline Real logProbability(const Uint act) const {
    assert(act<=nA && probs.size()==nA);
    return std::log(probs[act]);
  }

  template<typename Advantage_t>
  inline Rvec control_grad(const Advantage_t*const adv, const Real eta) const {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; j++)
      ret[j] = eta*adv->computeAdvantage(j)/normalization;
    return ret;
  }

  inline Rvec policy_grad(const Uint act, const Real factor) const {
    Rvec ret(nA);
    //for (Uint i=0; i<nA; i++) ret[i] = factor*(((i==act) ? 1 : 0) -probs[i]);
    for (Uint i=0; i<nA; i++) ret[i] = -factor/normalization;
    ret[act] += factor/unnorm[act];
    return ret;
  }

  inline Real kl_divergence(const Discrete_policy*const pol_hat) const {
    const Rvec vecTarget = pol_hat->getVector();
    return kl_divergence(vecTarget);
  }
  inline Real kl_divergence(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++)
      ret += probs[i]*std::log(probs[i]/beta[i]);
    return ret;
  }

  inline Rvec div_kl_grad(const Discrete_policy*const pol_hat, const Real fac = 1) const {
    const Rvec vecTarget = pol_hat->getVector();
    return div_kl_grad(vecTarget, fac);
  }
  inline Rvec div_kl_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; j++){
      const Real tmp = fac*(1+std::log(probs[j]/beta[j]))/normalization;
      for (Uint i=0; i<nA; i++) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }

  inline void finalize_grad(const Rvec grad, Rvec&netGradient) const
  {
    assert(netGradient.size()>=start_prob+nA && grad.size() == nA);
    for (Uint j=0; j<nA; j++)
    netGradient[start_prob+j]= grad[j]*unbPosMap_diff(netOutputs[start_prob+j]);
  }

  inline Rvec getProbs() const {
    return probs;
  }
  inline Rvec getVector() const {
    return probs;
  }

  inline Uint finalize(const bool bSample, mt19937*const gen, const Rvec& beta)
  {
    sampAct = bSample? sample(gen, beta) :
      std::distance(probs.begin(), std::max_element(probs.begin(),probs.end()));
    return sampAct; //the index of max Q
  }

  Uint updateOrUhState(Rvec& state, Rvec& beta,
    const Uint act, const Real step) {
    // not applicable to discrete action spaces
    return act;
  }

  void test(const Uint act, const Rvec& beta) const;
};
/*
 inline Real diagTerm(const Rvec& S, const Rvec& mu,
      const Rvec& a) const
  {
    assert(S.size() == nA);
    assert(a.size() == nA);
    assert(mu.size() == nA);
    Real Q = 0;
    for (Uint j=0; j<nA; j++) Q += S[j]*std::pow(mu[j]-a[j],2);
    return Q;
  }
 */
