//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once
#include "Learner_onPolicy.h"
#include "../Math/Lognormal_policy.h"
#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_advantage.h"

#define PPO_learnDKLt

template<typename Policy_t, typename Action_t>
class PPO : public Learner_onPolicy
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  mutable vector<long double> valPenal, cntPenal;
  const Real lambda;
  const vector<Uint> pol_outputs;
  const vector<Uint> pol_indices = count_indices(pol_outputs);
  mutable std::atomic<Real> DKL_target;

  inline Policy_t prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol(pol_indices, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  inline void updateDKL_target(const bool farPolSample, const Real DivKL) const
  {
    #ifdef PPO_learnDKLt
      //In absence of penalty term, it happens that within nEpochs most samples
      //are far-pol and therefore policy loss is 0. To keep samples on policy
      //we adapt DKL_target s.t. approx. 80% of samples are always near-Policy.
      //For most gym tasks with eta=1e-4 this results in ~0 penalty term.
      if(      farPolSample && DKL_target>DivKL) DKL_target = DKL_target*0.9995;
      else if(!farPolSample && DKL_target<DivKL) DKL_target = DKL_target*1.0001;
    #endif
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq,const Uint samp,const Uint thrID) const override;

  static vector<Uint> count_pol_outputs(const ActionInfo*const aI);
  static vector<Uint> count_pol_starts(const ActionInfo*const aI);

 public:
  PPO(Environment*const _env, Settings& _set);

  void select(Agent& agent) override;

  static inline Real annealDiscount(const Real targ, const Real orig,
    const Real t, const Real T=1e5) {
      // this function makes 1/(1-ret) linearly go from
      // 1/(1-orig) to 1/(1-targ) for t = 0:T. if t>=T it returns targ
    assert(targ>=orig);
    return t<T ? 1 -(1-orig)*(1-targ)/(1-targ +(t/T)*(targ-orig)) : targ;
  }

  void updatePPO(Sequence*const seq) const;

  void prepareGradient() override;
  void initializeLearner() override;
  static Uint getnDimPolicy(const ActionInfo*const aI);
};
