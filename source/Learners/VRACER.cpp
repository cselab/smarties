//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Network/Builder.h"

#include "VRACER.h"

//#define dumpExtra
#ifndef DACER_SKIP
#define DACER_SKIP 1
#endif

#define DACER_BACKWARD
//#ifndef DACER_FORWARD
#define DACER_FORWARD 0
//#endif

#define DACER_simpleSigma

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::TrainBySequences(const Uint seq, const Uint thrID) const
{
  die("");
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  assert(samp+1 < traj->tuples.size());

  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_one(traj, samp, thrID); // prepare thread workspace
  const Rvec out_cur = F[0]->forward(traj, samp, thrID); // network compute

  if( traj->isTruncated(samp+1) ) {
    const Rvec nxt = F[0]->forward(traj, samp+1, thrID);
    traj->setStateValue(samp+1, nxt[VsID]);
  }

  const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isOff = traj->isFarPolicy(samp, pol.sampImpWeight, CmaxRet);

  if(thrID==0)  profiler->stop_start("CMP");
  Rvec grad;

  #if   DACER_SKIP == 1
    if(isOff) grad = offPolGrad(traj, samp, out_cur, pol, thrID);
    else
  #endif
      grad = compute(traj, samp, out_cur, pol, thrID);

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(grad, samp, thrID); // place gradient onto output layer
  F[0]->gradient(thrID);  // backprop
}

template<typename Policy_t, typename Action_t>
Rvec VRACER<Policy_t, Action_t>::compute(Sequence*const S, const Uint t, const Rvec& outVec,
  const Policy_t& pol_cur, const Uint thrID) const
{
  const Real rNext = data->scaledReward(S, t+1), V_cur = outVec[VsID];
  const Real A_RET = rNext +gamma*(S->Q_RET[t+1]+S->state_vals[t+1]) - V_cur;
  //prepare Q with off policy corrections for next step:
  const Real dAdv = updateVret(S, t, V_cur, pol_cur);
  const Real rho_cur = pol_cur.sampImpWeight, Ver = S->Q_RET[t];

  const Rvec policyG = pol_cur.policy_grad(pol_cur.sampAct, A_RET*rho_cur);
  const Rvec penalG  = pol_cur.div_kl_grad(S->tuples[t]->mu, -1);
  const Rvec finalG  = weightSum2Grads(policyG, penalG, beta);

  Rvec gradient(F[0]->nOutputs(), 0);
  gradient[VsID] = beta*alpha *Ver;
  pol_cur.finalize_grad(finalG, gradient);
  //bookkeeping:
  trainInfo->log(V_cur, Ver, policyG, penalG, {beta, dAdv, rho_cur}, thrID);
  S->setMseDklImpw(t, S->Q_RET[t]*S->Q_RET[t], pol_cur.sampKLdiv, rho_cur);
  return gradient;
}

template<typename Policy_t, typename Action_t>
Rvec VRACER<Policy_t, Action_t>::offPolGrad(Sequence*const S, const Uint t, const Rvec output,
  const Policy_t& pol, const Uint thrID) const
{
  updateVret(S, t, output[VsID], pol);
  S->setMseDklImpw(t,std::pow(S->Q_RET[t],2),pol.sampKLdiv,pol.sampImpWeight);
  // prepare penalization gradient:
  Rvec gradient(F[0]->nOutputs(), 0);
  const Rvec pg = pol.div_kl_grad(S->tuples[t]->mu, beta-1);
  pol.finalize_grad(pg, gradient);
  return gradient;
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::select(Agent& agent)
{
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);
  F[0]->prepare_agent(traj, agent);

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec output = F[0]->forward_agent(traj, agent);
    Policy_t pol = prepare_policy(output);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID], mu);

    #if 0 // add and update temporally correlated noise
      act = pol.updateOrUhState(OrUhState[agent.ID], mu, act, iter());
    #endif

    traj->state_vals.push_back(output[VsID]);
    agent.act(act);
    data->add_action(agent, mu);

    #ifndef NDEBUG
      //Policy_t dbg = prepare_policy(output);
      //dbg.prepare(traj->tuples.back()->a, traj->tuples.back()->mu);
      //const double err = fabs(dbg.sampImpWeight-1);
      //if(err>1e-10) _die("Imp W err %20.20e", err);
    #endif
  }
  else
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[0]->forward_agent(traj, agent);
      traj->state_vals.push_back(output[VsID]);
    } else
      traj->state_vals.push_back(0); //value of terminal state is 0

    writeOnPolRetrace(traj); // compute initial Qret for whole trajectory
    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data->terminate_seq(agent);
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::writeOnPolRetrace(Sequence*const seq) const
{
  assert(seq->tuples.size() == seq->state_vals.size());
  assert(seq->Q_RET.size() == 0);
  const Uint N = seq->tuples.size();
  //within Retrace, we use the state_vals vector to write the Q retrace values
  seq->Q_RET.resize(N, 0);
  //TODO extend for non-terminal trajectories: one more v_state predict

  seq->Q_RET[N-1] = 0; //both if truncated or not, delta is zero

  //update all q_ret before terminal step
  for (Uint i=N-1; i>0; i--) updateVret(seq, i-1, seq->state_vals[i-1], 1);
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::prepareGradient()
{
  Learner_offPolicy::prepareGradient();

  #ifdef RACER_BACKWARD
  if(updateToApply)
  {
    profiler->stop_start("QRET");
    debugL("Update Retrace est. for episodes samples in prev. grad update");
    // placed here because this happens right after update is computed
    // this can happen before prune and before workers are joined
    #pragma omp parallel for schedule(dynamic)
    for(Uint i = 0; i < data->Set.size(); i++)
      for(int j = data->Set[i]->just_sampled; j>=0; j--)
        updateVret(data->Set[i], j, data->Set[i]->state_vals[j], data->Set[i]->offPolicImpW[j]);
  }
  #endif
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::initializeLearner()
{
  Learner_offPolicy::initializeLearner();

  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < data->Set.size(); i++) {
    Sequence* const traj = data->Set[i];
    const int N = traj->ndata(); traj->setRetrace(N, 0);
    for(Uint j=N; j>0; j--) updateVret(traj, j-1, traj->state_vals[j-1], 1);
  }

  for(Uint i = 0; i < data->inProgress.size(); i++) {
    Sequence* const traj = data->inProgress[i];
    const int N = traj->ndata(); traj->setRetrace(N, 0);
    for(Uint j=N; j>0; j--) updateVret(traj, j-1, traj->state_vals[j-1], 1);
  }
}

template<>
vector<Uint> VRACER<Discrete_policy, Uint>::count_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{1, aI->maxLabel};
}
template<>
vector<Uint> VRACER<Discrete_policy, Uint>::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<>
Uint VRACER<Discrete_policy, Uint>::getnOutputs(const ActionInfo*const aI)
{
  return 1 + aI->maxLabel;
}
template<>
Uint VRACER<Discrete_policy, Uint>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->maxLabel;
}

template<>
vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::count_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{1, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<>
vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1], indices[2], indices[3]};
}
template<>
Uint VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::getnDimPolicy(const ActionInfo*const aI)
{
  return NEXPERTS*(1 +2*aI->dim);
}
template<>
Uint VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::getnOutputs(const ActionInfo*const aI)
{
  return 1 + getnDimPolicy(aI);
}

template<> VRACER<Discrete_policy, Uint>::VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Discrete-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());

  F.push_back(new Approximator("net", _set, input, data));
  vector<Uint> nouts{1, nA};
  Builder build = F[0]->buildFromSettings(_set, nouts);
  F[0]->initializeNetwork(build);

  trainInfo = new TrainData("v-racer", _set, 1, "| beta | dAdv | avgW ", 3);
}

template<> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Mixture-of-experts continuous-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());

  F.push_back(new Approximator("net", _set, input, data));

  vector<Uint> nouts{1, NEXPERTS, NEXPERTS * nA};
  #ifndef DACER_simpleSigma // network outputs also sigmas
    nouts.push_back(NEXPERTS * nA);
  #endif

  Builder build = F[0]->buildFromSettings(_set, nouts);

  Rvec initBias;
  initBias.push_back(0); // state value
  Gaussian_mixture<NEXPERTS>::setInitial_noStdev(&aInfo, initBias);

  #ifdef DACER_simpleSigma // sigma not linked to network: parametric output
    build.setLastLayersBias(initBias);
    #ifdef EXTRACT_COVAR
      Real initParam = noiseMap_inverse(explNoise*explNoise);
    #else
      Real initParam = noiseMap_inverse(explNoise);
    #endif
    build.addParamLayer(NEXPERTS * nA, "Linear", initParam);
  #else
    Gaussian_mixture<NEXPERTS>::setInitial_Stdev(&aInfo, initBias, explNoise);
    build.setLastLayersBias(initBias);
  #endif
  F[0]->initializeNetwork(build, STD_GRADCUT);

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
    for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
    Real norm = 0;
    for(Uint i=0; i<NEXPERTS; i++) {
      mu[i] = std::exp(mu[i]);
      norm += mu[i];
    }
    for(Uint i=0; i<NEXPERTS; i++) mu[i] = mu[i]/norm;
    for(Uint i=NEXPERTS*(1+nA);i<NEXPERTS*(1+2*nA);i++) mu[i]=std::exp(mu[i]);

    Gaussian_mixture<NEXPERTS>  pol = prepare_policy(output);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }

  trainInfo = new TrainData("v-racer", _set, 1, "| beta | dAdv | avgW ", 3);
}


template class VRACER<Discrete_policy, Uint>;
template class VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;

#if 0
class DACER_cont : public DACER<Quadratic_advantage, Gaussian_policy, Rvec >
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    const Uint nL = Quadratic_term::compute_nL(&aI);
    #if defined ACER_RELAX
      return vector<Uint>{1, nL, aI.dim, aI.dim};
    #else
      return vector<Uint>{1, nL, aI.dim, aI.dim, aI.dim};
    #endif
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    #if defined ACER_RELAX
    return vector<Uint>{indices[2], indices[3]};
    #else
    return vector<Uint>{indices[2], indices[4]};
    #endif
  }
  static vector<Uint> count_adv_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    #if defined ACER_RELAX
    return vector<Uint>{indices[1]};
    #else
    return vector<Uint>{indices[1], indices[3]};
    #endif
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    const Uint nL = Quadratic_advantage::compute_nL(aI);
    #if defined ACER_RELAX // I output V(s), P(s), pol(s), prec(s) (and variate)
      return 1 + nL + aI->dim + aI->dim;
    #else // I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
      return 1 + nL + aI->dim + aI->dim + aI->dim;
    #endif
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  DACER_cont(Environment*const _env, Settings& _set) :
  DACER(_env, _set, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action DACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", _set, input, data));
    vector<Uint> nouts{1, nL, nA};
    #ifndef DACER_simpleSigma
      nouts.push_back(nA);
    #endif
    Builder build = F[0]->buildFromSettings(_set, nouts);
    #ifdef DACER_simpleSigma
      build.addParamLayer(nA, "Linear", -2*std::log(explNoise));
    #endif
    F[0]->initializeNetwork(build);
  }
};
#endif
