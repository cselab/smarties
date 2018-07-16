//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"
#include "../Math/Quadratic_advantage.h"
#include "../Math/Discrete_advantage.h"

class ACER : public Learner_offPolicy
{
 protected:
  using Policy_t = Gaussian_policy;
  using Action_t = Rvec;
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Real acerTrickPow = 1. / std::sqrt(nA);
  //const Real acerTrickPow = 1. / nA;
  const Uint nAexpectation = 5;
  const Real facExpect = 1./nAexpectation;
  const Real alpha = 1.0;
  //const Real alpha = 0.1;
  Aggregator* relay = nullptr;

  inline Policy_t prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol({0, nA}, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override;

  void Train(const Uint seq, const Uint obs, const Uint thrID) const override;

  Rvec policyGradient(const Tuple*const _t, const Policy_t& POL,
    const Policy_t& TGT, const Real ARET, const Real APol,
    const Action_t& pol_samp) const;

 public:
  void select(Agent& agent) override;

  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  ACER(Environment*const _env, Settings&_set);
  ~ACER() { }
};
