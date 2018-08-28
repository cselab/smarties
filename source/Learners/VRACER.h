//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"

#include "../Math/Gaussian_mixture.h"

#ifndef NEXPERTS
#define NEXPERTS 1
#warning "Using Mixture_advantage with 1 expert"
#endif

#include "../Math/Discrete_policy.h"

template<typename Policy_t, typename Action_t>
class VRACER : public Learner_offPolicy
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(&aInfo);

  // tgtFrac_param: target fraction of off-pol samples
  // alpha: weight of value-update relative to policy update. 1 means equal
  const Real alpha=1;

  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs;
  const vector<Uint> net_indices = count_indices(net_outputs);
  const vector<Uint> pol_start;
  const Uint VsID = net_indices[0];

  // used in case of temporally correlated noise
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );

  inline Policy_t prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol(pol_start, &aInfo, out);
    // pol.prepare computes various quanties that depend on behavioral policy mu
    // (such as importance weight) and stores both mu and the non-scaled action

    //policy saves pol.sampAct, which is unscaled action
    //eg. if action bounds act in [-1 1]; learning is with sampAct in (-inf inf)
    // when facing out of the learner we output act = tanh(sampAct)
    // TODO semi-bounded action spaces! eg. [0 inf): act = softplus(sampAct)
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override;

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

  Rvec compute(Sequence*const S, const Uint t, const Rvec& outVec,
    const Policy_t& pol_cur, const Uint thrID) const;

  Rvec offPolGrad(Sequence*const S, const Uint t, const Rvec output,
    const Policy_t& pol, const Uint thrID) const;

  inline Real updateVret(Sequence*const S, const Uint t, const Real V,
    const Policy_t& pol) const {
    return updateVret(S, t, V, pol.sampImpWeight);
  }

  inline Real updateVret(Sequence*const S, const Uint t, const Real V,
    const Real rho) const {
    assert(rho >= 0);
    const Real rNext = data->scaledReward(S, t+1), oldVret = S->Q_RET[t];
    const Real vNext = S->state_vals[t+1], V_RET = S->Q_RET[t+1];
    const Real delta = std::min((Real)1, rho) * (rNext +gamma*vNext -V);
    //const Real trace = gamma *.95 *std::pow(rho,retraceTrickPow) *V_RET;
    const Real trace = gamma *std::min((Real)1, rho) *V_RET;
    S->setStateValue(t, V ); S->setRetrace(t, delta + trace);
    return std::fabs(S->Q_RET[t] - oldVret);
  }

  static vector<Uint> count_outputs(const ActionInfo*const aI);
  static vector<Uint> count_pol_starts(const ActionInfo*const aI);

 public:
  VRACER(Environment*const _env, Settings& _set);
  ~VRACER() { }

  void select(Agent& agent) override;

  void writeOnPolRetrace(Sequence*const seq) const;

  void prepareGradient() override;
  void initializeLearner() override;
  static Uint getnOutputs(const ActionInfo*const aI);
  static Uint getnDimPolicy(const ActionInfo*const aI);
};
