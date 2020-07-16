//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Discrete_advantage_h
#define smarties_Discrete_advantage_h

#include "Discrete_policy.h"

namespace smarties
{

struct Discrete_advantage
{
  const ActionInfo& aInfo;
  const Uint start_adv, nA;
  const Rvec& netOutputs;
  const Rvec advantages;
  const Discrete_policy* const policy;

  static Uint compute_nL(const ActionInfo& aI)
  {
   assert(aI.dimDiscrete());
   return aI.dimDiscrete();
  }
  static void setInitial(const ActionInfo& aI, Rvec& initBias) { }

  Discrete_advantage(const std::vector<Uint>& starts, const ActionInfo& aI,
   const Rvec& out, const Discrete_policy*const pol = nullptr) : aInfo(aI),
   start_adv(starts[0]), nA(aI.dimDiscrete()), netOutputs(out),
   advantages(extract(out)), policy(pol) {}

protected:
  Rvec extract(const Rvec & v) const
  {
   assert(v.size() >= start_adv + nA);
   return Rvec( &(v[start_adv]), &(v[start_adv +nA]) );
  }

  Real expectedAdvantage() const
  {
   Real ret = 0;
   for (Uint j=0; j<nA; ++j) ret += policy->probs[j] * advantages[j];
   return ret;
  }

public:

  void grad(const Rvec& action, const Real Qer, Rvec& netGradient) const {
    grad(aInfo.actionMessage2label(action), Qer, netGradient);
  }
  void grad(const Uint act, const Real Qer, Rvec&netGradient) const
  {
   if(policy not_eq nullptr)
     for (Uint j=0; j<nA; ++j)
       netGradient[start_adv+j] = Qer*((j==act ? 1 : 0) - policy->probs[j]);
   else
     for (Uint j=0; j<nA; ++j)
       netGradient[start_adv+j] = Qer* (j==act ? 1 : 0);
  }

  Real computeAdvantage(const Rvec& action) const {
    return computeAdvantage(aInfo.actionMessage2label(action));
  }
  Real computeAdvantage(const Uint action) const
  {
   if(policy not_eq nullptr) //subtract expectation from advantage of action
     return advantages[action] - expectedAdvantage();
   else return advantages[action];
  }

  Real computeAdvantageNoncentral(const Rvec& action) const {
    return computeAdvantageNoncentral(aInfo.actionMessage2label(action));
  }
  Real computeAdvantageNoncentral(const Uint action) const
  {
   return advantages[action];
  }

  Rvec getParam() const {
   return advantages;
  }

  Real advantageVariance() const
  {
   assert(policy not_eq nullptr);
   if(policy == nullptr) return 0;
   const Real base = expectedAdvantage();
   Real ret = 0;
   for (Uint j=0; j<nA; ++j)
     ret += policy->probs[j] * (advantages[j]-base)*(advantages[j]-base);
   return ret;
  }

  void test(const Uint& act, std::mt19937& gen) const {}
};

void testDiscreteAdvantage(std::vector<Uint> polInds, std::vector<Uint> advInds,
  std::vector<Uint> netOuts, std::mt19937& gen, const ActionInfo & aI);

} // end namespace smarties
#endif // smarties_Discrete_advantage_h
