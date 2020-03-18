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

namespace smarties
{

struct Discrete_policy
{
  using PosDefFunction = Exp;
  const ActionInfo& aInfo;
  const Uint startProbs, nO;
  const Rvec netOutputs;
  const Rvec unnorm;
  const Real normalization;
  const Rvec probs;

  static Uint compute_nA(const ActionInfo & aI) {
    assert(aI.dimDiscrete());
    return aI.dimDiscrete();
  }
  static Uint compute_nPol(const ActionInfo & aI) {
    return aI.dimDiscrete();
  }

  static void setInitial_Stdev(const ActionInfo& aI, Rvec& O, const Real S) {
    printf("Stdev is not defined for discrete policy\n");
  }
  static Rvec initial_Stdev(const ActionInfo& aI, const Real S) {
    printf("Stdev is not defined for discrete policy\n");
    return Rvec();
  }

  static void setInitial_noStdev(const ActionInfo& aI, Rvec& initBias) {
    for(Uint e=0; e<aI.dimDiscrete(); e++) initBias.push_back(0);
  }

  Discrete_policy(const std::vector<Uint>& start, const ActionInfo& aI,
    const Rvec& out) : aInfo(aI), startProbs(start[0]), nO(aI.dimDiscrete()),
    netOutputs(out), unnorm(extract_unnorm()), normalization(compute_norm()),
    probs(extract_probabilities())
    {
      //printf("Discrete_policy: %u %u %u %lu %lu %lu %lu\n",
      //start_prob,start_vals,nA,netOutputs.size(),
      //unnorm.size(),vals.size(),probs.size());
    }

  Rvec extract_unnorm() const {
    assert(netOutputs.size() >= startProbs + nO);
    Rvec ret(nO);
    for (Uint j=0; j<nO; ++j)
        ret[j] = PosDefFunction::_eval(netOutputs[startProbs + j]);
    return ret;
  }

  Real compute_norm() const {
    assert(unnorm.size() == nO);
    Real ret = 0;
    for (Uint j=0; j<nO; ++j) { ret += unnorm[j]; assert(unnorm[j]>0); }
    return std::max(ret, std::numeric_limits<Real>::epsilon() );
  }

  Rvec extract_probabilities() const {
    assert(unnorm.size() == nO);
    Rvec ret(nO);
    for (Uint j=0; j<nO; ++j) ret[j] = unnorm[j]/normalization;
    return ret;
  }

  Real importanceWeight(const Rvec& action, const Rvec& beta) const {
    const Uint option = aInfo.actionMessage2label(action);
    assert(beta.size() == nO && option < nO);
    return probs[option] / beta[option];
  }

  static Real evalBehavior(const Uint option, const Rvec& beta) {
    return beta[option];
  }
  Real evalBehavior(const Rvec action, const Rvec& beta) const {
    const Uint option = aInfo.actionMessage2label(action);
    assert(beta.size() == nO && option < nO);
    return beta[option];
  }

  Real evalProbability(const Uint option) const {
    return probs[option];
  }
  Real evalProbability(const Rvec action) const {
    const Uint option = aInfo.actionMessage2label(action);
    assert(option < nO);
    return probs[option];
  }
  Real evalLogProbability(const Uint option) const {
    return std::log(evalProbability(option));
  }
  Real evalLogProbability(const Rvec& action) const {
    return std::log(evalProbability(action));
  }

  Real KLDivergence(const Discrete_policy*const tgt_pol) const {
    const Rvec vecTarget = tgt_pol->getVector();
    return KLDivergence(vecTarget);
  }
  Real KLDivergence(const Rvec& beta) const {
    Real ret = 0;
    for (Uint i=0; i<nO; ++i) ret += probs[i]*std::log(probs[i]/beta[i]);
    return ret;
  }

  Rvec policyGradient(const Rvec& action, const Real factor) const {
    const Uint option = aInfo.actionMessage2label(action);
    return policyGradient(option, factor);
  }
  Rvec policyGradient(const Uint option, const Real factor) const {
    Rvec ret(nO);
    for (Uint i=0; i<nO; ++i) ret[i] = -factor/normalization;
    ret[option] += factor/unnorm[option];
    return ret;
  }

  Rvec KLDivGradient(const Discrete_policy*const tgt_pol, const Real fac = 1) const {
    const Rvec vecTarget = tgt_pol->getVector();
    return KLDivGradient(vecTarget, fac);
  }
  Rvec KLDivGradient(const Rvec& beta, const Real fac = 1) const {
    Rvec ret(nO, 0);
    for (Uint j=0; j<nO; ++j){
      const Real tmp = fac*(1+std::log(probs[j]/beta[j]))/normalization;
      for (Uint i=0; i<nO; ++i) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }

  void makeNetworkGrad(Rvec& netGradient, const Rvec& totPolicyG) const {
    assert(netGradient.size() >= startProbs+nO && totPolicyG.size() == nO);
    for (Uint j=0, k=startProbs; j<nO; ++j, ++k)
      netGradient[k] = totPolicyG[j] * PosDefFunction::_evalDiff(netOutputs[k]);
  }
  Rvec makeNetworkGrad(const Rvec& totPolicyGrad) const {
    Rvec ret(nO);
    makeNetworkGrad(ret, totPolicyGrad);
    return ret;
  }

  Rvec getVector() const {
    return probs;
  }

  static Uint sample(std::mt19937& gen, const Rvec& beta) {
    std::discrete_distribution<Uint> dist(beta.begin(), beta.end());
    return dist(gen);
  }
  Uint sample(std::mt19937& gen) const {
    std::discrete_distribution<Uint> dist(probs.begin(), probs.end());
    return dist(gen);
  }

  template<typename Advantage_t>
  Rvec control_grad(const Advantage_t*const adv, const Real eta) const {
    Rvec ret(nO, 0);
    for (Uint j=0; j<nO; ++j)
      ret[j] = eta * adv->computeAdvantage(j)/normalization;
    return ret;
  }

  Uint selectAction(Agent& agent, const bool bTrain) const {
    const bool bSample = bTrain && agent.trackEpisodes;
    return bSample? sample(agent.generator) : Utilities::maxInd(probs);
  }

  void test(const Uint act, const Rvec& beta) const;
};

void testDiscretePolicy(std::mt19937& gen, const ActionInfo & aI);

} // end namespace smarties
#endif // smarties_Discrete_policy_h
