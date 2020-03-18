//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_RACER_h
#define smarties_RACER_h

#include "Learner_approximator.h"
#include "../Utils/FunctionUtilities.h"

namespace smarties
{

struct Discrete_policy;
struct Continuous_policy;

//template<Uint nExperts>
//class Gaussian_mixture;

struct Discrete_advantage;
#ifndef ADV_QUAD
#define Param_advantage Gaussian_advantage
#else
#define Param_advantage Quadratic_advantage
#endif
struct Param_advantage;
struct Zero_advantage;

//#ifndef NEXPERTS
//#define NEXPERTS 1
//#endif
//template<Uint nExperts>
//class Mixture_advantage;

#define RACER_simpleSigma
#define RACER_singleNet
//#define RACER_TABC

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_approximator
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(aInfo);
  // number of parameters of advantage approximator
  const Uint nL = Advantage_t::compute_nL(aInfo);

  // tgtFrac_param: target fraction of off-pol samples

  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const std::vector<Uint> net_outputs;
  const std::vector<Uint> net_indices = Utilities::count_indices(net_outputs);
  const std::vector<Uint> pol_start, adv_start;
  const Uint VsID = net_indices[0];

  const Uint batchSize = settings.batchSize, ESpopSize = settings.ESpopSize;
  // used for CMA:
  mutable std::vector<Rvec> rhos=std::vector<Rvec>(batchSize,Rvec(ESpopSize,0));
  mutable std::vector<Rvec> dkls=std::vector<Rvec>(batchSize,Rvec(ESpopSize,0));
  mutable std::vector<Rvec> advs=std::vector<Rvec>(batchSize,Rvec(ESpopSize,0));

  void prepareCMALoss();

  //void TrainByEpisodes(const Uint seq, const Uint wID,
  //  const Uint bID, const Uint tID) const override;

  void Train(const MiniBatch&MB, const Uint wID,const Uint bID) const override;

  Rvec policyGradient(const Rvec& MU, const Rvec& ACT,
                      const Policy_t& POL, const Advantage_t& ADV,
                      const Real A_RET, const Real IMPW, const Uint thrID) const;

  //inline Rvec criticGrad(const Policy_t& POL, const Advantage_t& ADV,
  //  const Real A_RET, const Real A_critic) const {
  //  const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
  //  const Real varCritic = ADV.advantageVariance();
  //  const Real iEpsA = std::pow(A_RET-A_critic,2)/(varCritic+2.2e-16);
  //  const Real eta = anneal * safeExp( -0.5*iEpsA);
  //  return POL.control_grad(&ADV, eta);
  //}

  static std::vector<Uint> count_outputs(const ActionInfo& aI);
  static std::vector<Uint> count_pol_starts(const ActionInfo& aI);
  static std::vector<Uint> count_adv_starts(const ActionInfo& aI);
  void setupNet();
 public:
  RACER(MDPdescriptor& MDP_, Settings& S, DistributionInfo& D);

  void select(Agent& agent) override;
  void setupTasks(TaskQueue& tasks) override;
  static Uint getnOutputs(const ActionInfo& aI);
  static Uint getnDimPolicy(const ActionInfo& aI);
};

template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo& aI);

template<> Uint
RACER<Param_advantage, Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo& aI);

template<> Uint
RACER<Zero_advantage, Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo& aI);

}
#endif
