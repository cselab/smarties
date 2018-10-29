//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Discrete_policy.h"

 struct Discrete_advantage
 {
   const ActionInfo* const aInfo;
   const Uint start_adv, nA;
   const Rvec& netOutputs;
   const Rvec advantages;
   const Discrete_policy* const policy;

   static inline Uint compute_nL(const ActionInfo* const aI)
   {
     assert(aI->maxLabel);
     return aI->maxLabel;
   }
   static void setInitial(const ActionInfo* const aI, Rvec& initBias) { }

   Discrete_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
     const Rvec& out, const Discrete_policy*const pol = nullptr) : aInfo(aI), start_adv(starts[0]), nA(aI->maxLabel), netOutputs(out),
     advantages(extract(out)), policy(pol) {}

  protected:
   inline Rvec extract(const Rvec & v) const
   {
     assert(v.size() >= start_adv + nA);
     return Rvec( &(v[start_adv]), &(v[start_adv +nA]) );
   }
   inline Real expectedAdvantage() const
   {
     Real ret = 0;
     for (Uint j=0; j<nA; j++) ret += policy->probs[j]*advantages[j];
     return ret;
   }

  public:
   inline void grad(const Uint act, const Real Qer, Rvec&netGradient) const
   {
     if(policy not_eq nullptr)
       for (Uint j=0; j<nA; j++)
         netGradient[start_adv+j] = Qer*((j==act ? 1 : 0) - policy->probs[j]);
     else
       for (Uint j=0; j<nA; j++)
         netGradient[start_adv+j] = Qer* (j==act ? 1 : 0);
   }

   Real computeAdvantage(const Uint action) const
   {
     if(policy not_eq nullptr)
       return advantages[action]-expectedAdvantage(); //subtract expectation from advantage of action
     else return advantages[action];
   }

   Real computeAdvantageNoncentral(const Uint action) const
   {
     return advantages[action];
   }

   Rvec getParam() const {
     return advantages;
   }

   inline Real advantageVariance() const
   {
     assert(policy not_eq nullptr);
     if(policy == nullptr) return 0;
     const Real base = expectedAdvantage();
     Real ret = 0;
     for (Uint j=0; j<nA; j++)
       ret += policy->probs[j]*(advantages[j]-base)*(advantages[j]-base);
     return ret;
   }

   void test(const Uint& act, mt19937*const gen) const {}
 };
