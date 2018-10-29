//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once
#include "Learner.h"

class Discrete_policy;
class Gaussian_policy;

#define PPO_learnDKLt

template<typename Policy_t, typename Action_t>
class PPO : public Learner
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  mutable LDvec valPenal = LDvec(nThreads+1,0);
  mutable LDvec cntPenal = LDvec(nThreads+1,0);
  const Real lambda = settings.lambda;
  const std::vector<Uint> pol_outputs;
  const std::vector<Uint> pol_indices = count_indices(pol_outputs);
  const Uint nHorizon = settings.maxTotObsNum;
  const Uint nEpochs = settings.batchSize/settings.obsPerStep;

  mutable Uint cntBatch = 0, cntEpoch = 0, cntKept = 0;
  mutable std::atomic<Real> DKL_target{ settings.klDivConstraint };

  inline void updateDKL_target(const bool farPolSample, const Real DivKL) const
  {
    #ifdef PPO_learnDKLt
      //In absence of penalty term, it happens that within nEpochs most samples
      //are far-pol and therefore policy loss is 0. To keep samples on policy
      //we adapt DKL_target s.t. approx. 80% of samples are always near-Policy.
      //For most gym tasks with eta=1e-4 this results in ~0 penalty term.
      if( farPolSample && DKL_target>DivKL) DKL_target = DKL_target*0.9995;
      else
      if(!farPolSample && DKL_target<DivKL) DKL_target = DKL_target*1.0001;
    #endif
  }

  void Train(const Uint seq, const Uint samp,
    const Uint wID, const Uint bID, const Uint thrID) const;

  static vector<Uint> count_pol_outputs(const ActionInfo*const aI);
  static vector<Uint> count_pol_starts(const ActionInfo*const aI);

  void updatePPO(Sequence*const seq) const;

  static inline Real annealDiscount(const Real targ, const Real orig,
    const Real t, const Real T=1e5) {
      // this function makes 1/(1-ret) linearly go from
      // 1/(1-orig) to 1/(1-targ) for t = 0:T. if t>=T it returns targ
    assert(targ>=orig);
    return t<T ? 1 -(1-orig)*(1-targ)/(1-targ +(t/T)*(targ-orig)) : targ;
  }

 public:

  PPO(Environment*const _env, Settings& _set);

  void select(Agent& agent) override;

  void prepareGradient() override;
  void initializeLearner() override;
  bool blockDataAcquisition() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  bool bNeedSequentialTrain() override;

  static Uint getnDimPolicy(const ActionInfo*const aI);
};
