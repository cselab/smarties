//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "../Network/Builder.h"
#include "../Utils/SstreamUtilities.h"

namespace smarties
{

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::prepareCMALoss()
{
  if(ESpopSize<=1) return;

  profiler->start("LOSS");
  std::vector<Real> avgRho(batchSize, 0), avgAdv(batchSize, 0);

  #pragma omp parallel for schedule(static)
  for (Uint b=0; b<batchSize; ++b) {
    Real avgRho_ = 0, avgAdv_ = 0;
    for(Uint w=0; w<ESpopSize; ++w) {
      avgRho_ += rhos[b][w]; avgAdv_ += advs[b][w];
    }
    avgRho[b] = avgRho_ / ESpopSize; avgAdv[b] = avgAdv_ / ESpopSize;
  }

  //for(Uint b=0; b<batchSize; ++b) { aR[b] = rhos[b][0]; aA[b] = advs[b][0]; }
  const auto isFar = [&](const Real&W) {return W >= CmaxRet || W <= CinvRet;};

  #pragma omp parallel for schedule(static)
  for (Uint w=0; w<ESpopSize; ++w)
  for (Uint b=0; b<batchSize; ++b) {
    const Real clipRho = std::max(CinvRet, std::min(rhos[b][w], CmaxRet));
    const Real clipAdv = isFar(rhos[b][w]) ? avgAdv[b] : advs[b][w];
    const Real criticError = std::min((Real) 1, avgRho[b]) * clipAdv;
    const Real costAdv =    - beta  * clipRho * avgAdv[b]; // minus: minimize
    const Real costVal =      beta  * std::pow(criticError, 2);
    const Real costDkl = (1 - beta) * dkls[b][w];
    networks[0]->ESloss(w) += alpha*(costAdv + costDkl) + (1-alpha)*costVal;
  }
  networks[0]->nAddedGradients = ESpopSize * batchSize;

  profiler->stop();
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::setupNet()
{
  const std::type_info& actT = typeid(Action_t);
  const std::type_info& vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();
  std::vector<Uint> nouts = count_outputs(aInfo);
  #ifdef RACER_simpleSigma // variance not dependent on state
    const Uint varianceSize = nouts.back();
    if(isContinuous) nouts.pop_back();
  #endif

  createEncoder();
  // should have already created all hidden layers and this vec should be empty:
  assert(networks.size() <= 1);
  if(networks.size()>0) {
    networks[0]->rename("net"); // not preprocessing, is is the main&only net
  } else {
    networks.push_back(new Approximator("net", settings, distrib, data.get()));
  }

  networks[0]->buildFromSettings(nouts);
  Builder& networkBuilder = networks[0]->getBuilder();

  if(isContinuous)
  {
    Rvec  biases = Rvec(1, 0);
    const Real explNoise = settings.explNoise;
    Advantage_t::setInitial(aInfo, biases);
    Policy_t::setInitial_noStdev(aInfo, biases);

    #ifdef RACER_simpleSigma // sigma not linked to state: param output
      networkBuilder.setLastLayersBias(biases);
      const Rvec stdParam = Policy_t::initial_Stdev(aInfo, explNoise);
      networkBuilder.addParamLayer(varianceSize, "Linear", stdParam);
    #else
      Policy_t::setInitial_Stdev(aInfo, biases, explNoise);
      networkBuilder.setLastLayersBias(biases);
    #endif
  }

  // construct policy net:
  if(networks.size() > 1) die("");
  networks[0]->initializeNetwork();
  //networks[0]->opt->bAnnealLearnRate= true;
  trainInfo = new TrainData("racer", distrib, 1, "| dAdv ", 1);
  computeQretrace = true;
}

// Template specializations. From now on, nothing relevant to algorithm itself.

template<> std::vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_outputs(const ActionInfo& aI) {
  return std::vector<Uint>{1, aI.dimDiscrete(), aI.dimDiscrete()};
}
template<> std::vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_pol_starts(const ActionInfo& aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = Utilities::count_indices(sizes);
  return std::vector<Uint>{indices[2]};
}
template<> std::vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_adv_starts(const ActionInfo& aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = Utilities::count_indices(sizes);
  return std::vector<Uint>{indices[1]};
}
template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnOutputs(const ActionInfo& aI) {
  return 1 + aI.dimDiscrete() + aI.dimDiscrete();
}
template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo& aI) {
  return aI.dimDiscrete();
}

template<>
RACER<Discrete_advantage, Discrete_policy, Uint>::
RACER(MDPdescriptor& MDP_, Settings& S, DistributionInfo& D):
  Learner_approximator(MDP_, S, D), net_outputs(count_outputs(aInfo)),
  pol_start(count_pol_starts(aInfo)), adv_start(count_adv_starts(aInfo))
{
  if(D.world_rank == 0) {
    using Utilities::vec2string;
    printf(
    "    Single net with outputs: [%lu] : V(s),\n"
    "                             [%s] : policy mean and stdev,\n"
    "                             [%s] : advantage\n"
    "    Size per entry = [%s].\n", VsID, vec2string(pol_start).c_str(),
      vec2string(adv_start).c_str(), vec2string(net_outputs).c_str());
  }
  setupNet();
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template<> std::vector<Uint>
RACER<Param_advantage, Continuous_policy, Rvec>::
count_outputs(const ActionInfo& aI) {
  const Uint nL = Param_advantage::compute_nL(aI);
  return std::vector<Uint>{1, nL, aI.dim(), aI.dim()};
}
template<> std::vector<Uint>
RACER<Param_advantage, Continuous_policy, Rvec>::
count_pol_starts(const ActionInfo& aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = Utilities::count_indices(sizes);
  return std::vector<Uint>{indices[2], indices[3]};
}
template<> std::vector<Uint>
RACER<Param_advantage, Continuous_policy, Rvec>::
count_adv_starts(const ActionInfo& aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = Utilities::count_indices(sizes);
  return std::vector<Uint>{indices[1]};
}
template<> Uint
RACER<Param_advantage, Continuous_policy, Rvec>::
getnOutputs(const ActionInfo& aI) {
  const Uint nL = Param_advantage::compute_nL(aI);
  return 1 + nL + 2*aI.dim();
}
template<> Uint
RACER<Param_advantage, Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo& aI) {
  return 2*aI.dim();
}

template<>
RACER<Param_advantage, Continuous_policy, Rvec>::
RACER(MDPdescriptor& MDP_, Settings& S, DistributionInfo& D):
  Learner_approximator(MDP_, S, D), net_outputs(count_outputs(aInfo)),
  pol_start(count_pol_starts(aInfo)), adv_start(count_adv_starts(aInfo))
{
  if(D.world_rank == 0) {
    using Utilities::vec2string;
    printf(
    "    Single net with outputs: [%lu] : V(s),\n"
    "                             [%s] : policy mean and stdev,\n"
    "                             [%s] : advantage\n"
    "    Size per entry = [%s].\n", VsID, vec2string(pol_start).c_str(),
      vec2string(adv_start).c_str(), vec2string(net_outputs).c_str());
  }
  setupNet();
  testGaussianAdvantage(pol_start, adv_start, net_outputs, distrib.generators[0], aInfo);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template<> std::vector<Uint>
RACER<Zero_advantage, Continuous_policy, Rvec>::
count_outputs(const ActionInfo& aI) {
  return std::vector<Uint>{1, aI.dim(), aI.dim()};
}
template<> std::vector<Uint>
RACER<Zero_advantage, Continuous_policy, Rvec>::
count_pol_starts(const ActionInfo& aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = Utilities::count_indices(sizes);
  return std::vector<Uint>{indices[1], indices[2]};
}
template<> std::vector<Uint>
RACER<Zero_advantage, Continuous_policy, Rvec>::
count_adv_starts(const ActionInfo& aI) {
  return std::vector<Uint>();
}
template<> Uint
RACER<Zero_advantage, Continuous_policy, Rvec>::
getnOutputs(const ActionInfo& aI) {
  return 1 + 2*aI.dim();
}
template<> Uint
RACER<Zero_advantage, Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo& aI) {
  return 2*aI.dim();
}

template<>
RACER<Zero_advantage, Continuous_policy, Rvec>::
RACER(MDPdescriptor& MDP_, Settings& S, DistributionInfo& D):
  Learner_approximator(MDP_, S, D), net_outputs(count_outputs(aInfo)),
  pol_start(count_pol_starts(aInfo)), adv_start(count_adv_starts(aInfo))
{
  if(D.world_rank == 0) {
    using Utilities::vec2string;
    printf(
    "    Single net with outputs: [%lu] : V(s),\n"
    "                             [%s] : policy mean and stdev,\n"
    "    Size per entry = [%s].\n", VsID, vec2string(pol_start).c_str(),
      vec2string(net_outputs).c_str());
  }
  setupNet();
  testContinuousPolicy(distrib.generators[0], aInfo);
}


////////////////////////////////////////////////////////////////////////////////
#if 0
template<> std::vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
  return std::vector<Uint>{1, nL, NEXPERTS, NEXPERTS*aI->dim(), NEXPERTS*aI->dim()};
}
template<> std::vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = count_indices(sizes);
  return std::vector<Uint>{indices[2], indices[3], indices[4]};
}
template<> std::vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_adv_starts(const ActionInfo*const aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = count_indices(sizes);
  return std::vector<Uint>{indices[1]};
}
template<> Uint
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
getnOutputs(const ActionInfo*const aI) {
  const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
  return 1 + nL + NEXPERTS*(1 +2*aI->dim());
}
template<> Uint
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return NEXPERTS*(1 +2*aI->dim());
}

template<>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
RACER(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner_approximator(MDP_, S_, D_),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Mixture-of-experts continuous-action RACER: Built network with outputs: v:%u pol:%s adv:%s (sorted %s)\n", VsID, print(pol_start).c_str(), print(adv_start).c_str(), print(net_outputs).c_str());
  }
  setupNet();
}
#endif
////////////////////////////////////////////////////////////////////////////////

}
