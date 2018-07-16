//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"
#include "../Math/Gaussian_policy.h"

class RETPG : public Learner_offPolicy
{
 protected:
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


  inline void updateQret(Sequence*const S, const Uint t) const {
    const Real rho = S->isLast(t) ? 0 : S->offPolicImpW[t];
    updateQret(S, t, S->action_adv[t], S->state_vals[t], rho);
  }
  inline void updateQretFront(Sequence*const S, const Uint t) const {
    if(t == 0) return;
    const Real D = data->scaledReward(S,t) + gamma*S->state_vals[t];
    S->Q_RET[t-1] = D +gamma*(S->Q_RET[t]-S->action_adv[t]) -S->state_vals[t-1];
  }
  inline void updateQretBack(Sequence*const S, const Uint t) const {
    if(t == 0) return;
    const Real W=S->isLast(t)? 0:S->offPolicImpW[t], R=data->scaledReward(S,t);
    const Real delta = R +gamma*S->state_vals[t] -S->state_vals[t-1];
    S->Q_RET[t-1] = delta + gamma*(W>1? 1:W)*(S->Q_RET[t] - S->action_adv[t]);
  }
  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Gaussian_policy& pol) const {
    // shift retrace advantage with update estimate for V(s_t)
    S->setRetrace(t, S->Q_RET[t] + S->state_vals[t] -V );
    S->setStateValue(t, V); S->setAdvantage(t, A);
    //prepare Qret_{t-1} with off policy corrections for future use
    return updateQret(S, t, A, V, pol.sampImpWeight);
  }
  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Real rho) const {
    assert(rho >= 0);
    if(t == 0) return 0;
    const Real oldRet = S->Q_RET[t-1], W = rho>1 ? 1 : rho;
    const Real delta = data->scaledReward(S,t) +gamma*V - S->state_vals[t-1];
    S->setRetrace(t-1, delta + gamma*W*(S->Q_RET[t] - A) );
    return std::fabs(S->Q_RET[t-1] - oldRet);
  }

 public:
  RETPG(Environment*const _env, Settings& _set);
  ~RETPG() { }

  void select(Agent& agent) override;

  void prepareGradient() override;
  void initializeLearner() override;
};
