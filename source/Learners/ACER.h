//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"
class Aggregator;
class Gaussian_policy;

class ACER : public Learner_offPolicy
{
 protected:
  const Uint nA = aInfo.dim;
  const Real acerTrickPow = 1. / std::sqrt(nA);
  //const Real acerTrickPow = 1. / nA;
  static constexpr Uint nAexpectation = 5;
  static constexpr Real facExpect = 1./nAexpectation;

  Aggregator* relay = nullptr;

  void TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
    const Uint thrID) const override;

  void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint thrID) const override;

  Rvec policyGradient(const Tuple*const _t, const Gaussian_policy& POL,
    const Gaussian_policy& TGT, const Real ARET, const Real APol,
    const Rvec& pol_samp) const;

 public:
  void select(Agent& agent) override;

  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  ACER(Environment*const _env, Settings&_set);
  ~ACER() { }
};
