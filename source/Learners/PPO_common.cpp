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
    const Rvec stdParam = Policy_t::initial_Stdev(&aInfo, explNoise);
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

  #ifdef PPO_learnDKLt
   trainInfo = new TrainData("PPO",distrib,1,"| beta |  DKL | avgW | DKLt ",4);
  #else
   trainInfo = new TrainData("PPO",distrib,1,"| beta |  DKL | avgW ",3);
  #endif
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::updateGAE(Sequence& seq) const
{
  assert(seq.states.size()==seq.state_vals.size() && seq.states.size()>0);

  //this is only triggered by t = 0 (or truncated trajectories)
  // at t=0 we do not have a reward, and we cannot compute delta
  //(if policy was updated after prev action we treat next state as initial)
  if(seq.state_vals.size() < 2)  return;
  assert(seq.states.size() == 2+seq.Q_RET.size());
  assert(seq.states.size() == 2+seq.action_adv.size());

  const Sint N = seq.states.size();
  const Fval vSold = seq.state_vals[N-2], vSnew = seq.state_vals[N-1];
  const Fval R = data->scaledReward(seq, N-1);
  // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
  // received with transition to s_t+1, sometimes referred to as r_t)

  const Fval delta = R + (Fval)gamma * vSnew - vSold;
  seq.action_adv.push_back(0);
  seq.Q_RET.push_back(0);

  Fval fac_lambda = 1, fac_gamma = 1;
  // If user selects gamma=.995 and lambda=0.97 as in Henderson2017
  // these will start at 0.99 and 0.95 (same as original) and be quickly
  // annealed upward in the first 1e5 steps.
  //const Fval rGamma  =  gamma>.99? annealDiscount( gamma,.99,_nStep) :  gamma;
  //const Fval rLambda = lambda>.95? annealDiscount(lambda,.95,_nStep) : lambda;
  const Fval rGamma = settings.gamma, rLambda = settings.lambda;
  // reward of i=0 is 0, because before any action
  // adv(0) is also 0, V(0) = V(s_0)
  for (Sint i=N-2; i>=0; --i) { //update all rewards before current step
    //will contain MC sum of returns:
    seq.Q_RET[i] += fac_gamma * R;
    //#ifndef IGNORE_CRITIC
    seq.action_adv[i] += fac_lambda * delta;
    //#else
    //  seq.action_adv[i] += fac_gamma * R;
    //#endif
    fac_lambda *= rLambda*rGamma;
    fac_gamma *= rGamma;
  }
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::initializeGAE()
{
  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) GAE and MC cumulative rewards
  // This assumes V(s) is initialized small, so we just rescale by std(rew)
  debugL("Rescale GAE est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics

  const Uint setSize = data->readNSeq();
  const Fval invstdR = data->scaledReward(1);
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < setSize; ++i) {
    Sequence& EP = * data->get(i);
    assert(EP.ndata() >= 1 && EP.state_vals.size() == EP.ndata()+1);
    assert(EP.action_adv.size() >= EP.ndata() && EP.Q_RET.size() >= EP.ndata());
    for (Uint j=0; j<EP.action_adv.size(); ++j) EP.action_adv[j] *= invstdR;
    for (Uint j=0; j<EP.Q_RET.size();      ++j) EP.Q_RET[j]      *= invstdR;
  }

  const Uint todoSize = data_get->nInProgress();
  for(Uint i = 0; i < todoSize; ++i) {
    Sequence& EP = * data_get->get(i);
    if(EP.states.size() <= 1) continue;
    for (Uint j=0; j<EP.action_adv.size(); ++j) EP.action_adv[j] *= invstdR;
    for (Uint j=0; j<EP.Q_RET.size();      ++j) EP.Q_RET[j]      *= invstdR;
  }
}

///////////////////////////////////////////////////////////////////////
/////////// TEMPLATE SPECIALIZATION FOR CONTINUOUS ACTIONS ////////////
///////////////////////////////////////////////////////////////////////
template<> std::vector<Uint> PPO<Gaussian_policy, Rvec>::
count_pol_outputs(const ActionInfo*const aI)
{
  return std::vector<Uint>{aI->dim(), aI->dim()};
}
template<> std::vector<Uint> PPO<Gaussian_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI)
{
  const std::vector<Uint> indices = Utilities::count_indices(count_pol_outputs(aI));
  return std::vector<Uint>{indices[0], indices[1]};
}
template<> Uint PPO<Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI)
{
  return 2*aI->dim(); // policy dimension is mean and diag covariance
}

template<> PPO<Gaussian_policy, Rvec>::
PPO(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_):
  Learner_approximator(MDP_, S_, D_), pol_outputs(count_pol_outputs(&aInfo)), penal_reduce(D_, LDvec{0.,1.})
{
  if(MPICommRank(distrib.world_comm) == 0) printf(
  "==========================================================================\n"
  "                          Continuous-action PPO                           \n"
  "==========================================================================\n"
  );
  setupNet();

  #if 0
  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); ++i) output[i] = dist(generators[0]);
    for(Uint i=0;  i<mu.size(); ++i) mu[i] = dist(generators[0]);
    for(Uint i=nA; i<mu.size(); ++i) mu[i] = std::exp(mu[i]);

    Gaussian_policy pol = prepare_policy<Gaussian_policy>(output, &aInfo, pol_indices);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
  #endif
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
PPO(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_):
  Learner_approximator(MDP_, S_, D_), pol_outputs(count_pol_outputs(&aInfo)), penal_reduce(D_, LDvec{0.,1.})
{
  if(MPICommRank(distrib.world_comm) == 0) printf(
  "==========================================================================\n"
  "                           Discrete-action PPO                            \n"
  "==========================================================================\n"
  );
  setupNet();
}

}
