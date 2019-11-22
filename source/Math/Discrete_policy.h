//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Discrete_policy_h
#define smarties_Discrete_policy_h

#include "../Network/Layers/Functions.h"
//#include <algorithm>

#ifndef PosDefMapping_f
#define PosDefMapping_f SoftPlus
#endif

namespace smarties
{

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

  Uint map_action(const Rvec& sent) const {
    return aInfo->actionMessage2label(sent);
  }
  static Uint compute_nA(const ActionInfo* const aI) {
    assert(aI->dimDiscrete());
    return aI->dimDiscrete();
  }
  static void setInitial_Stdev(const ActionInfo*const aI, Rvec&O, const Real S)
  {
    for(Uint e=0; e<aI->dimDiscrete(); ++e)
    #ifdef SMARTIES_EXTRACT_COVAR
        O.push_back(PosDefMapping_f::_inv(S*S));
    #else
        O.push_back(PosDefMapping_f::_inv(S));
    #endif
  }

  static Rvec initial_Stdev(const ActionInfo*const aI, const Real S)
  {
    Rvec ret; ret.reserve(aI->dimDiscrete());
    setInitial_Stdev(aI, ret, S);
    return ret;
  }

  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias) { }

  Discrete_policy(const std::vector<Uint>& start, const ActionInfo*const aI,
    const Rvec& out) : aInfo(aI), start_prob(start[0]), nA(aI->dimDiscrete()),
    netOutputs(out), unnorm(extract_unnorm()), normalization(compute_norm()),
    probs(extract_probabilities())
    {
      //printf("Discrete_policy: %u %u %u %lu %lu %lu %lu\n",
      //start_prob,start_vals,nA,netOutputs.size(),
      //unnorm.size(),vals.size(),probs.size());
    }

 private:
  Rvec extract_unnorm() const
  {
    assert(netOutputs.size()>=start_prob+nA);
    Rvec ret(nA);
    for (Uint j=0; j<nA; ++j)
      ret[j] = PosDefMapping_f::_eval(netOutputs[start_prob+j]);
    return ret;
  }

  Real compute_norm() const
  {
    assert(unnorm.size()==nA);
    Real ret = 0;
    for (Uint j=0; j<nA; ++j) { ret += unnorm[j]; assert(unnorm[j]>0); }
    return ret + nnEPS;
  }

  Rvec extract_probabilities() const
  {
    assert(unnorm.size()==nA);
    Rvec ret(nA);
    for (Uint j=0; j<nA; ++j) ret[j] = unnorm[j]/normalization;
    return ret;
  }

 public:
  void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampPonPolicy = probs[sampAct];
    sampPBehavior = beta[sampAct];
    sampImpWeight = sampPonPolicy / sampPBehavior;
    sampKLdiv = kl_divergence(beta);
  }

  static Real evalBehavior(const Uint& act, const Rvec& beta) {
    return beta[act];
  }

  static Uint sample(std::mt19937*const gen, const Rvec& beta) {
    std::discrete_distribution<Uint> dist(beta.begin(), beta.end());
    return dist(*gen);
  }

  Uint sample(std::mt19937*const gen) const {
    std::discrete_distribution<Uint> dist(probs.begin(), probs.end());
    return dist(*gen);
  }

  Real evalProbability(const Uint act) const {
    return probs[act];
  }

  Real logProbability(const Uint act) const {
    assert(act<=nA && probs.size()==nA);
    return std::log(probs[act]);
  }

  template<typename Advantage_t>
  Rvec control_grad(const Advantage_t*const adv, const Real eta) const {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; ++j)
      ret[j] = eta*adv->computeAdvantage(j)/normalization;
    return ret;
  }

  Rvec policy_grad(const Real factor) const
  {
    return policy_grad(sampAct, factor);
  }
  Rvec policy_grad(const Uint act, const Real factor) const {
    Rvec ret(nA);
    //for (Uint i=0; i<nA; ++i) ret[i] = factor*(((i==act) ? 1 : 0) -probs[i]);
    for (Uint i=0; i<nA; ++i) ret[i] = -factor/normalization;
    ret[act] += factor/unnorm[act];
    return ret;
  }

  Real kl_divergence(const Discrete_policy*const pol_hat) const {
    const Rvec vecTarget = pol_hat->getVector();
    return kl_divergence(vecTarget);
  }
  Real kl_divergence(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; ++i)
      ret += probs[i]*std::log(probs[i]/beta[i]);
    return ret;
  }

  Rvec div_kl_grad(const Discrete_policy*const pol_hat, const Real fac = 1) const {
    const Rvec vecTarget = pol_hat->getVector();
    return div_kl_grad(vecTarget, fac);
  }
  Rvec div_kl_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; ++j){
      const Real tmp = fac*(1+std::log(probs[j]/beta[j]))/normalization;
      for (Uint i=0; i<nA; ++i) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }

  void finalize_grad(const Rvec& grad, Rvec&netGradient) const
  {
    assert(netGradient.size()>=start_prob+nA && grad.size() == nA);
    for (Uint j=0, k=start_prob; j<nA; ++j, ++k)
      netGradient[k]= grad[j] * PosDefMapping_f::_evalDiff(netOutputs[k]);
  }
  Rvec finalize_grad(const Rvec& grad) const
  {
    Rvec ret(nA);
    for (Uint j=0, k=start_prob; j<nA; ++j, ++k)
      ret[j]= grad[j] * PosDefMapping_f::_evalDiff(netOutputs[k]);
    return ret;
  }

  Rvec getProbs() const {
    return probs;
  }
  Rvec getVector() const {
    return probs;
  }

  Uint finalize(const bool bSample, std::mt19937*const gen, const Rvec& beta)
  {
    sampAct = bSample? sample(gen, beta) : Utilities::maxInd(probs);
    return sampAct;
  }

  Uint updateOrUhState(Rvec& state, Rvec& beta,
    const Uint act, const Real step) {
    // not applicable to discrete action spaces
    return act;
  }

  void test(const Uint act, const Rvec& beta) const;

};

} // end namespace smarties
#endif // smarties_Discrete_policy_h
