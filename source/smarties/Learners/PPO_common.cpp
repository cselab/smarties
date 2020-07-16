//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


namespace smarties
{

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::
updateDKL_target(const bool farPolSample, const Real DivKL) const
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

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::getMetrics(std::ostringstream& buf) const
{
  Utilities::real2SS(buf, penalCoef, 6, 1);
  Utilities::real2SS(buf, DKL_target, 6, 1);
  Learner_approximator::getMetrics(buf);
}
template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::getHeaders(std::ostringstream& buf) const
{
  buf << "| penl |DKLtgt";
  Learner_approximator::getHeaders(buf);
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::setupNet()
{
  const std::type_info & actT = typeid(Action_t), & vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();

  const bool bCreatedEncorder = createEncoder();
  assert(networks.size() == bCreatedEncorder? 1 : 0);
  if(bCreatedEncorder) networks[0]->initializeNetwork();
  const Approximator* const encoder = bCreatedEncorder? networks[0] : nullptr;

  networks.push_back(
    new Approximator("policy", settings, distrib, data.get(), encoder)
  );
  actor = networks.back();
  //actor->setBlockGradsToPreprocessing();
  actor->buildFromSettings(nA);
  if(isContinuous) {
    const Real explNoise = settings.explNoise;
    const Rvec stdParam = Policy_t::initial_Stdev(aInfo, explNoise);
    actor->getBuilder().addParamLayer(nA, "Linear", stdParam);
  }
  actor->initializeNetwork();

  networks.push_back(
    new Approximator("critic", settings, distrib, data.get(), encoder)
  );
  critc = networks.back();
  // update settings that are going to be read by critic:
  settings.learnrate *= 3; // PPO benefits from critic faster than actor
  settings.nnOutputFunc = "Linear"; // critic must be linear
  critc->buildFromSettings(1);
  critc->initializeNetwork();
}

///////////////////////////////////////////////////////////////////////
/////////// TEMPLATE SPECIALIZATION FOR CONTINUOUS ACTIONS ////////////
///////////////////////////////////////////////////////////////////////
template<> std::vector<Uint> PPO<Continuous_policy, Rvec>::
count_pol_outputs(const ActionInfo*const aI)
{
  return std::vector<Uint>{aI->dim(), aI->dim()};
}
template<> std::vector<Uint> PPO<Continuous_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI)
{
  const std::vector<Uint> indices = Utilities::count_indices(count_pol_outputs(aI));
  return std::vector<Uint>{indices[0], indices[1]};
}
template<> Uint PPO<Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI)
{
  return 2*aI->dim(); // policy dimension is mean and diag covariance
}

template<> PPO<Continuous_policy, Rvec>::
PPO(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_):
  Learner_approximator(MDP_, S_, D_), pol_outputs(count_pol_outputs(&aInfo)), penal_reduce(D_, LDvec{0.,1.})
{
  setupNet();
}

///////////////////////////////////////////////////////////////////////
//////////// TEMPLATE SPECIALIZATION FOR DISCRETE ACTIONS /////////////
///////////////////////////////////////////////////////////////////////
template<> std::vector<Uint> PPO<Discrete_policy, Uint>::
count_pol_outputs(const ActionInfo*const aI)
{
  return std::vector<Uint>{aI->dimDiscrete()};
}
template<> std::vector<Uint> PPO<Discrete_policy, Uint>::
count_pol_starts(const ActionInfo*const aI)
{
  const std::vector<Uint> indices = Utilities::count_indices(count_pol_outputs(aI));
  return std::vector<Uint>{indices[0]};
}
template<> Uint PPO<Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI)
{
  return aI->dimDiscrete();
}

template<> PPO<Discrete_policy, Uint>::
PPO(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_):
  Learner_approximator(MDP_, S_, D_), pol_outputs(count_pol_outputs(&aInfo)), penal_reduce(D_, LDvec{0.,1.})
{
  setupNet();
}

}
