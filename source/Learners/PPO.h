//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_PPO_h
#define smarties_PPO_h

#include "Learner_approximator.h"
#include "../Utils/FunctionUtilities.h"

namespace smarties
{

struct Discrete_policy;
struct Continuous_policy;

template<typename Policy_t, typename Action_t>
class PPO : public Learner_approximator
{
 protected:
  const Uint nA = Policy_t::compute_nA(aInfo);
  const std::vector<Uint> pol_outputs;
  const std::vector<Uint> pol_indices = Utilities::count_indices(pol_outputs);
  const long nHorizon = settings.maxTotObsNum;
  const long nEpochs = settings.batchSize/settings.obsPerStep;
  const Real CmaxPol = settings.clipImpWeight;

  mutable long cntBatch = 0, cntEpoch = 0, cntKept = 0;
  mutable std::atomic<Real> DKL_target{ settings.klDivConstraint };
  mutable std::atomic<Real> penalUpdateCount{ 0 }, penalUpdateDelta{ 0 };
  Real penalCoef = 1;

  DelayedReductor<long double> penal_reduce;

  Approximator* actor;
  Approximator* critc;

  void Train(const MiniBatch& MB, const Uint, const Uint) const override;

  void updatePenalizationCoef();
  void advanceEpochCounters();

  static std::vector<Uint> count_pol_outputs(const ActionInfo*const aI);
  static std::vector<Uint> count_pol_starts(const ActionInfo*const aI);

  void updateDKL_target(const bool farPolSample, const Real DivKL) const;
  void updateGAE(Sequence& seq) const;
  void initializeGAE();
  void setupNet();

  static Real annealDiscount(const Real targ,
                             const Real orig,
                             const Real t,
                             const Real T = 1e5)
  {
    // this function makes 1/(1-ret) linearly go from
    // 1/(1-orig) to 1/(1-targ) for t = 0:T. if t>=T it returns targ
    assert(targ>=orig);
    return t<T ? 1 -(1-orig)*(1-targ)/(1-targ +(t/T)*(targ-orig)) : targ;
  }

 public:

  PPO(MDPdescriptor&, Settings&, DistributionInfo&);

  void select(Agent& agent) override;

  bool blockDataAcquisition() const override;
  bool blockGradientUpdates() const override;

  void setupTasks(TaskQueue& tasks) override;

  static Uint getnDimPolicy(const ActionInfo*const aI);
};

template<> Uint PPO<Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI);

template<> Uint PPO<Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI);

}
#endif
