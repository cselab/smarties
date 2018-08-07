//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PPO.h"
#include "../Network/Builder.h"

#define PPO_PENALKL
#define PPO_CLIPPED
#define PPO_simpleSigma

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::TrainBySequences(const Uint seq, const Uint thrID) const
{
  die("not allowed");
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  if(thrID==0)  profiler->stop_start("FWD");
  Sequence* const traj = data->Set[seq];
  const Real adv_est = traj->action_adv[samp], val_tgt = traj->Q_RET[samp];
  const Rvec MU = traj->tuples[samp]->mu;

  F[0]->prepare_one(traj, samp, thrID);
  const Rvec pol_cur = F[0]->forward(traj, samp, thrID);

  if(thrID==0)  profiler->stop_start("CMP");

  const Policy_t pol = prepare_policy(pol_cur, traj->tuples[samp]);
  const Real rho_cur = pol.sampImpWeight, DivKL = pol.sampKLdiv;
  const bool isFarPol = traj->isFarPolicyPPO(samp, rho_cur, CmaxPol);

  cntPenal[thrID+1]++;
  if(DivKL < DKL_target / 1.5) valPenal[thrID+1] -= valPenal[0]/2; //half
  if(DivKL > 1.5 * DKL_target) valPenal[thrID+1] += valPenal[0]; //double

  Real gain = rho_cur*adv_est;
  #ifdef PPO_CLIPPED
    if(adv_est > 0 && rho_cur > 1+CmaxPol) gain = 0;
    if(adv_est < 0 && rho_cur < 1-CmaxPol) gain = 0;
    updateDKL_target(isFarPol, DivKL);
  #endif

  F[1]->prepare_one(traj, samp, thrID);
  const Rvec val_cur = F[1]->forward(traj, samp, thrID);

  #ifdef PPO_PENALKL //*nonZero(gain)
    const Rvec policy_grad = pol.policy_grad(pol.sampAct, gain);
    const Rvec penal_grad = pol.div_kl_grad(MU, -valPenal[0]);
    const Rvec totalPolGrad = sum2Grads(penal_grad, policy_grad);
  #else //we still learn the penal coef, for simplicity, but no effect
    const Rvec totalPolGrad = pol.policy_grad(pol.sampAct, gain);
    const Rvec policy_grad = totalPolGrad;
    const Rvec penal_grad = Rvec(policy_grad.size(), 0);
  #endif

  Rvec grad(F[0]->nOutputs(), 0);
  pol.finalize_grad(totalPolGrad, grad);

  //bookkeeping:
  const Real verr = val_tgt-val_cur[0];
  #ifdef PPO_learnDKLt
  trainInfo->log(val_cur[0], verr, policy_grad, penal_grad,
    { (Real)valPenal[0], DivKL, rho_cur, DKL_target }, thrID);
  #else
  trainInfo->log(val_cur[0], verr, policy_grad, penal_grad,
    { (Real)valPenal[0], DivKL, rho_cur }, thrID);
  #endif
  traj->setMseDklImpw(samp, verr*verr, DivKL, rho_cur);

  if(thrID==0)  profiler->stop_start("BCK");
  //if(!thrID) cout << "back pol" << endl;
  F[0]->backward(grad, traj, samp, thrID);
  //if(!thrID) cout << "back val" << endl; //*(!isFarPol)
  F[1]->backward({verr*(!isFarPol)}, traj, samp, thrID);
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::updatePPO(Sequence*const seq) const
{
  assert(seq->tuples.size());
  assert(seq->tuples.size() == seq->state_vals.size());

  //this is only triggered by t = 0 (or truncated trajectories)
  // at t=0 we do not have a reward, and we cannot compute delta
  //(if policy was updated after prev action we treat next state as initial)
  if(seq->state_vals.size() < 2)  return;
  assert(seq->tuples.size() == 2+seq->Q_RET.size());
  assert(seq->tuples.size() == 2+seq->action_adv.size());

  const Uint N = seq->tuples.size();
  const Fval vSold = seq->state_vals[N-2], vSnew = seq->state_vals[N-1];
  const Fval R = data->scaledReward(seq, N-1);
  // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
  // received with transition to s_t+1, sometimes referred to as r_t)

  const Fval delta = R +(Fval)gamma*vSnew -vSold;
  seq->action_adv.push_back(0);
  seq->Q_RET.push_back(0);

  Fval fac_lambda = 1, fac_gamma = 1;
  // If user selects gamma=.995 and lambda=0.97 as in Henderson2017
  // these will start at 0.99 and 0.95 (same as original) and be quickly
  // annealed upward in the first 1e5 steps.
  const Fval rGamma  =  gamma>0.99? annealDiscount( gamma,.99,nStep) :  gamma;
  const Fval rLambda = lambda>0.95? annealDiscount(lambda,.95,nStep) : lambda;
  // reward of i=0 is 0, because before any action
  // adv(0) is also 0, V(0) = V(s_0)
  for (int i=N-2; i>=0; i--) { //update all rewards before current step
    //will contain MC sum of returns:
    seq->Q_RET[i] += fac_gamma * R;
    //#ifndef IGNORE_CRITIC
      seq->action_adv[i] += fac_lambda * delta;
    //#else
    //  seq->action_adv[i] += fac_gamma * R;
    //#endif
    fac_lambda *= rLambda*rGamma;
    fac_gamma *= rGamma;
  }
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::prepareGradient()
{
  debugL("update lagrangian penalization coefficient");
  if(learn_rank > 0)
    die("This method does not support multiple learner ranks yet");

  cntPenal[0] = 0;
  for(Uint i=1; i<=nThreads; i++) {
    cntPenal[0] += cntPenal[i]; cntPenal[i] = 0;
  }
  if(cntPenal[0]<nnEPS) die("undefined behavior");
  const Real fac = learnR/cntPenal[0]; // learnRate*grad/N //
  cntPenal[0] = 0;
  for(Uint i=1; i<=nThreads; i++) {
      valPenal[0] += fac*valPenal[i];
      valPenal[i] = 0;
  }
  if(valPenal[0] <= nnEPS) valPenal[0] = nnEPS;

  Learner_onPolicy::prepareGradient();
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::initializeLearner()
{
  Learner_onPolicy::initializeLearner();

  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) GAE and MC cumulative rewards
  // This assumes V(s) is initialized small, so we just rescale by std(rew)
  debugL("Rescale GAE est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < data->Set.size(); i++) {
    assert(data->Set[i]->ndata()>=1);
    assert(data->Set[i]->action_adv.size() == data->Set[i]->ndata());
    assert(data->Set[i]->Q_RET.size()      == data->Set[i]->ndata());
    assert(data->Set[i]->state_vals.size() == data->Set[i]->ndata()+1);
    for (Uint j=data->Set[i]->ndata()-1; j>0; j--) {
      data->Set[i]->action_adv[j] *= data->invstd_reward;
      data->Set[i]->Q_RET[j] *= data->invstd_reward;
    }
  }

  for(Uint i = 0; i < data->inProgress.size(); i++) {
    if(data->inProgress[i]->tuples.size() <= 1) continue;
    for (Uint j=data->inProgress[i]->ndata()-1; j>0; j--) {
      data->inProgress[i]->action_adv[j] *= data->invstd_reward;
      data->inProgress[i]->Q_RET[j] *= data->invstd_reward;
    }
  }
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::select(Agent& agent)
{
  Sequence*const curr_seq = data->inProgress[agent.ID];
  data->add_state(agent);
  F[1]->prepare_agent(curr_seq, agent);

  if(agent.Status < TERM_COMM ) { //non terminal state
    //Compute policy and value on most recent element of the sequence:
    F[0]->prepare_agent(curr_seq, agent);
    const Rvec pol = F[0]->forward_agent(curr_seq, agent);
    const Rvec val = F[1]->forward_agent(curr_seq, agent);

    curr_seq->state_vals.push_back(val[0]);
    Policy_t policy = prepare_policy(pol);
    const Rvec MU = policy.getVector();
    auto act = policy.finalize(explNoise>0, &generators[nThreads+agent.ID], MU);
    agent.act(act);
    data->add_action(agent, MU);
  } else if( agent.Status == TRNC_COMM ) {
    const Rvec val = F[1]->forward_agent(curr_seq, agent);
    curr_seq->state_vals.push_back(val[0]);
  } else
    curr_seq->state_vals.push_back(0); // Assign value of term state to 0

  updatePPO(curr_seq);

  //advance counters of available data for training
  if(agent.Status >= TERM_COMM) data->terminate_seq(agent);
}

template<>
vector<Uint> PPO<Discrete_policy, Uint>::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->maxLabel};
}
template<>
vector<Uint> PPO<Discrete_policy, Uint>::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> indices = count_indices(count_pol_outputs(aI));
  return vector<Uint>{indices[0]};
}
template<>
Uint PPO<Discrete_policy, Uint>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->maxLabel;
}

template<>
vector<Uint> PPO<Gaussian_policy, Rvec>::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->dim, aI->dim};
}
template<>
vector<Uint> PPO<Gaussian_policy, Rvec>::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> indices = count_indices(count_pol_outputs(aI));
  return vector<Uint>{indices[0], indices[1]};
}
template<>
Uint PPO<Gaussian_policy, Rvec>::getnDimPolicy(const ActionInfo*const aI)
{
  return 2*aI->dim;
}

template<> PPO<Gaussian_policy, Rvec>::PPO(
  Environment*const _env, Settings & _set) : Learner_onPolicy(_env,_set),
  valPenal(nThreads+1,0), cntPenal(nThreads+1,0), lambda(_set.lambda),
  pol_outputs(count_pol_outputs(&_env->aI)), DKL_target(_set.klDivConstraint)
{
  #ifdef PPO_learnDKLt
    trainInfo = new TrainData("PPO", _set,1,"| beta |  DKL | avgW | DKLt ",4);
  #else
    trainInfo = new TrainData("PPO", _set,1,"| beta |  DKL | avgW ", 3);
  #endif
  valPenal[0] = 1;

  printf("Continuous-action PPO\n");
  #if 0 // shared input layers
    if(input->net not_eq nullptr) {
      delete input->opt; input->opt = nullptr;
      delete input->net; input->net = nullptr;
    }
    Builder input_build(_set);
    bool bInputNet = false;
    input_build.addInput( input->nOutputs() );
    bInputNet = bInputNet || env->predefinedNetwork(input_build);
    bInputNet = bInputNet || predefinedNetwork(input_build);
    if(bInputNet) {
      Network* net = input_build.build(true);
      input->initializeNetwork(net, input_build.opt);
    }
  #endif
  F.push_back(new Approximator("policy", _set, input, data));
  F[0]->blockInpGrad = true;
  F.push_back(new Approximator("critic", _set, input, data));

  Builder build_val = F[1]->buildFromSettings(_set, {1} );

  #ifndef PPO_simpleSigma
    Rvec initBias;
    Gaussian_policy::setInitial_noStdev(&aInfo, initBias);
    Gaussian_policy::setInitial_Stdev(&aInfo, initBias, explNoise);
    Builder build_pol = F[0]->buildFromSettings(_set, {2*aInfo.dim});
    build.setLastLayersBias(initBias);
  #else  //stddev params
    Builder build_pol = F[0]->buildFromSettings(_set,   {aInfo.dim});
    const Real initParam = noiseMap_inverse(explNoise);
    build_pol.addParamLayer(aInfo.dim, "Linear", initParam);
  #endif
  F[0]->initializeNetwork(build_pol);

  _set.learnrate *= 3; // for shared input layers
  F[1]->initializeNetwork(build_val);
  _set.learnrate /= 3;

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
    for(Uint i=0;  i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=nA; i<mu.size(); i++) mu[i] = std::exp(mu[i]);

    Gaussian_policy pol = prepare_policy(output);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

template<> PPO<Discrete_policy, Uint>::PPO(
  Environment*const _env, Settings & _set) : Learner_onPolicy(_env,_set),
  valPenal(nThreads+1,0), cntPenal(nThreads+1,0), lambda(_set.lambda),
  pol_outputs(count_pol_outputs(&_env->aI)), DKL_target(_set.klDivConstraint)
{
  #ifdef PPO_learnDKLt
    trainInfo = new TrainData("PPO", _set,1,"| beta |  DKL | avgW | DKLt ",4);
  #else
    trainInfo = new TrainData("PPO", _set,1,"| beta |  DKL | avgW ", 3);
  #endif
  valPenal[0] = 1;

  printf("Discrete-action PPO\n");
  F.push_back(new Approximator("policy", _set, input, data));
  F.push_back(new Approximator("critic", _set, input, data));
  Builder build_pol = F[0]->buildFromSettings(_set, aInfo.maxLabel);
  Builder build_val = F[1]->buildFromSettings(_set, 1 );

  //build_pol.addParamLayer(1,"Exp",1); //add klDiv penalty coefficient layer

  F[0]->initializeNetwork(build_pol);
  F[1]->initializeNetwork(build_val);
}

template class PPO<Discrete_policy, Uint>;
template class PPO<Gaussian_policy, Rvec>;

#if 0
// Update network from sampled observation `obs', part of sequence `seq'
void Train(const Uint seq, const Uint obs, const Uint thrID) const override
{
  const Sequence*const traj = data->Set[seq];          // fetch sampled sequence
  const Real advantage_obs  = traj->action_adv[obs+1];// observed advantage
  const Real value_obs      = traj->tuples[obs+1]->r; // observed state val
  const Rvec mu     = traj->tuples[obs]->mu;  // policy used for sample
  const Rvec action = traj->tuples[obs]->a;   // sample performed act

  // compute current policy and state-value-estimate for sampled state
  const Rvec out_policy = policyNet->forward(traj, samp, thrID);
  const Rvec out_value  =  valueNet->forward(traj, samp, thrID);

  //compute gradient of state-value est. and backpropagate value net
  const Real Vst_est  = out_value[0];           // estimated state value
  const Rvec  value_grad = {value_obs - Vst_est};
   valueNet->backward(value_grad, samp, thrID);

  //Create action & policy objects: generalize discrete, gaussian, lognorm pols
  const Policy_t pol = prepare_policy(pol_cur);//current state policy
  const Action_t act = pol.map_action(action); //map to pol space (eg. discrete)

  // compute importance sample rho = pol( a_t | s_t ) / mu( a_t | s_t )
  const Real actProbOnPolicy =       pol.evalLogProbability(act);
  const Real actProbBehavior = Policy_t::evalLogProbability(act, mu);
  const Real rho = std::exp(actProbOnPolicy-actProbBehavior);

  //compute policy gradient and backpropagate pol net
  const Real gain = rho * advantage_obs;
  const Rvec  policy_grad = pol.policy_grad(act, gain);
  policyNet->backward(policy_grad, samp, thrID);
}

#endif
//#ifdef INTEGRATEANDFIREMODEL
//  inline Lognormal_policy prepare_policy(const Rvec& out) const
//  {
//    return Lognormal_policy(net_indices[0], net_indices[1], nA, out);
//  }
//#else
/*
settings.splitLayers = 9; // all!
Builder build(settings);

build.stackSimple(vector<Uint>{nInputs}, vector<Uint>{nA, 1});
//add stddev layer
build.addParamLayer(nA, "Linear", -2*std::log(explNoise));
//add klDiv penalty coefficient layer
build.addParamLayer(1, "Exp", 0);

net = build.build();

//set initial value for klDiv penalty coefficient
Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
net->biases[penalparid] = std::log(1/settings.klDivConstraint);

finalize_network(build);

printf("PPO: Built network with outputs: %s %s\n",
  print(net_indices).c_str(),print(net_outputs).c_str());
*/
