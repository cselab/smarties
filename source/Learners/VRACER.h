//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"

class Discrete_policy;
class Gaussian_policy;

template<Uint nExperts>
class Gaussian_mixture;

class Discrete_advantage;
class Quadratic_advantage;

template<Uint nExperts>
class Mixture_advantage;

template<typename Policy_t, typename Action_t>
class VRACER : public Learner_offPolicy
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(&aInfo);

  // tgtFrac_param: target fraction of off-pol samples

  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs;
  const vector<Uint> net_indices = count_indices(net_outputs);
  const vector<Uint> pol_start;
  const Uint VsID = net_indices[0];

  // used in case of temporally correlated noise
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );
  mutable vector<Rvec> rhos = vector<Rvec>(batchSize, Rvec(ESpopSize, 0) );
  mutable vector<Rvec> dkls = vector<Rvec>(batchSize, Rvec(ESpopSize, 0) );
  mutable vector<Rvec> advs = vector<Rvec>(batchSize, Rvec(ESpopSize, 0) );

  void TrainBySequences(const Uint seq, const Uint wID,
    const Uint bID, const Uint tID) const override;

  void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint tID) const override;

  inline void updateVret(Sequence*const S, const Uint t, const Fval W) const {
    assert(W >= 0);
    const Fval R = data->scaledReward(S, t+1), G = gamma, D = S->Q_RET[t+1];
    const Fval Vc = S->state_vals[t],  Vn = S->state_vals[t+1];
    S->Q_RET[t] = std::min((Fval)1, W) * (R + G * (Vn+D) - Vc);
  }

  static vector<Uint> count_outputs(const ActionInfo*const aI);
  static vector<Uint> count_pol_starts(const ActionInfo*const aI);
  void setupNet();
 public:
  VRACER(Environment*const _env, Settings& _set);
  ~VRACER() { }

  void select(Agent& agent) override;

  void prepareGradient() override;
  static Uint getnDimPolicy(const ActionInfo*const aI);
};
