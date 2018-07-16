//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"

//#define LearnStDev

class DPG : public Learner_offPolicy
{
  Aggregator* relay;
  const Uint nA = env->aI.dim;
  const Real OrUhDecay = CmaxPol<=0? .85 : 0; // as in original
  //const Real OrUhDecay = 0; // no correlated noise
  vector<Rvec> OrUhState = vector<Rvec>(nAgents,Rvec(nA,0));

  inline Gaussian_policy prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Gaussian_policy pol({0, nA}, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  DPG(Environment*const env, Settings & settings);
  void select(Agent& agent) override;
};
