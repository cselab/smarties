//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Quadratic_advantage_h
#define smarties_Quadratic_advantage_h

#include "Continuous_policy.h"
#include "Quadratic_term.h"

namespace smarties
{

struct Zero_advantage
{
  const ActionInfo& aInfo;

  static Uint compute_nL(const ActionInfo& aI) { return 0; }
  static void setInitial(const ActionInfo& aI, Rvec& initBias) { }

  //Rvec getParam() const {
  //  Rvec ret = matrix;
  //  ret.insert(ret.begin(), coef);
  //  return ret;
  //}

  //Normalized quadratic advantage, with own mean
  Zero_advantage(const std::vector<Uint>& starts,
                 const ActionInfo& aI,
                 const Rvec& out,
                 const Continuous_policy*const pol = nullptr) : aInfo(aI) { }

  void grad(const Rvec&act, const Real Qer, Rvec& netGradient) const { }

  Real computeAdvantage(const Rvec& action) const
  {
    return 0;
  }

  Real advantageVariance() const
  {
    return 0;
  }

  void test(const Rvec& act, std::mt19937*const gen) const { }
};

} // end namespace smarties
#endif // smarties_Quadratic_advantage_h
