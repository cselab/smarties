//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "RACER.h"
#include "../Network/Builder.h"

#ifdef ADV_GAUS
#include "../Math/Mixture_advantage_gaus.h"
#include "../Math/Gaus_advantage.h"
#else
#include "../Math/Mixture_advantage_quad.h"
#include "../Math/Quadratic_advantage.h"
#endif
#include "../Math/Discrete_advantage.h"

#ifdef DKL_filter // this is so bad let's force undefined
#undef DKL_filter
#endif
#define RACER_simpleSigma
#define RACER_singleNet

template<typename Policy_t>
static inline Policy_t prepare_policy(const Rvec& O, const ActionInfo*const aI,
  const vector<Uint>& pol_start, const Tuple*const t = nullptr) {
  Policy_t pol(pol_start, aI, O);
  if(t not_eq nullptr) pol.prepare(t->a, t->mu);
  return pol;
}

template<typename Advantage_t, typename Policy_t>
static inline Advantage_t prepare_advantage(const Rvec& out, const ActionInfo*const aI, const vector<Uint>& adv_start, const Policy_t*const pol) {
  return Advantage_t(adv_start, aI, out, pol);
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::TrainBySequences(
  const Uint seq, const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const traj = data->get(seq);
  const int ndata = traj->tuples.size()-1;
  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_seq(traj, thrID, wID);
  for (int k=0; k<ndata; k++) F[0]->forward(k, thrID);

  //if partial sequence then compute value of last state (!= R_end)
  if( traj->isTruncated(ndata) ) {
    const Rvec nxt = F[0]->forward(ndata, thrID);
    updateRetrace(traj, ndata, 0, nxt[VsID], 0);
  }

  if(thrID==0)  profiler->stop_start("POL");
  for(int k=ndata-1; k>=0; k--)
  {
    const Rvec out_cur = F[0]->get(k, thrID);
    const Policy_t pol = prepare_policy<Policy_t>(out_cur, &aInfo, pol_start, traj->tuples[k]);
    // far policy definition depends on rho (as in paper)
    const bool isOff = traj->isFarPolicy(k, pol.sampImpWeight, CmaxRet,CinvRet);
    // in case rho outside bounds, do not compute gradient
    Rvec G;
    if(isOff) {
      G = offPolCorrUpdate(traj, k, out_cur, pol, thrID);
      continue;
    } else G = compute(traj,k, out_cur, pol, thrID);
    //write gradient onto output layer:
    F[0]->backward(G, k, thrID);
  }

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::Train(const Uint seq, const Uint t,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const S = data->get(seq);
  assert(t+1 < S->tuples.size());

  if(thrID==0) profiler->stop_start("FWD");
  F[0]->prepare_one(S, t, thrID, wID); // prepare thread workspace
  const Rvec O = F[0]->forward(t, thrID); // network compute

  //Update Qret of eps' last state if sampled T-1. (and V(s_T) for truncated ep)
  if( S->isTruncated(t+1) ) {
    assert( t+1 == S->ndata() );
    const Rvec nxt = F[0]->forward(t+1, thrID);
    updateRetrace(S, t+1, 0, nxt[VsID], 0);
  }

  const Policy_t P = prepare_policy<Policy_t>(O, aI, pol_start, S->tuples[t]);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isOff = S->isFarPolicy(t, P.sampImpWeight, CmaxRet, CinvRet);

  if(thrID==0)  profiler->stop_start("CMP");
  Rvec grad;
  if(isOff) grad = offPolCorrUpdate(S, t, O, P, thrID);
  else grad = compute(S, t, O, P, thrID);

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(grad, t, thrID); // place gradient onto output layer
  F[0]->gradient(thrID);  // backprop
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
compute(Sequence*const traj, const Uint samp, const Rvec& outVec,
  const Policy_t& POL, const Uint thrID) const
{
  const Advantage_t ADV = prepare_advantage<Advantage_t,Policy_t>(
                                                  outVec, aI, adv_start, &POL);
  const Real A_cur = ADV.computeAdvantage(POL.sampAct), V_cur = outVec[VsID];
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = traj->Q_RET[samp] - V_cur;
  const Real rho = POL.sampImpWeight, dkl = POL.sampKLdiv;
  const Real Ver = std::min((Real)1, rho) * (A_RET-A_cur);
  // all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
  const Real Aer = std::min(CmaxRet, rho) * (A_RET-A_cur);
  const Rvec polG = policyGradient(traj->tuples[samp], POL, ADV, A_RET, thrID);
  const Rvec penalG  = POL.div_kl_grad(traj->tuples[samp]->mu, -1);
  //if(!thrID) cout<<dkl<<" s "<<print(traj->tuples[samp]->s)
  //  <<" pol "<<print(POL.getVector())<<" mu "<<print(traj->tuples[samp]->mu)
  //  <<" act: "<<print(traj->tuples[samp]->a)<<" pg: "<<print(polG)
  //  <<" pen: "<<print(penalG)<<" fin: "<<print(finalG)<<endl;
  //prepare Q with off policy corrections for next step:
  const Real dAdv = updateRetrace(traj, samp, A_cur, V_cur, rho);
  // compute the gradient:
  Rvec gradient = Rvec(F[0]->nOutputs(), 0);
  gradient[VsID] = beta * Ver;
  POL.finalize_grad(weightSum2Grads(polG, penalG, beta), gradient);
  ADV.grad(POL.sampAct, beta * Aer, gradient);
  traj->setMseDklImpw(samp, Ver*Ver, dkl, rho, CmaxRet, CinvRet);
  // logging for diagnostics:
  trainInfo->log(V_cur+A_cur, A_RET-A_cur, polG,penalG, {dAdv,rho}, thrID);
  return gradient;
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
offPolCorrUpdate(Sequence*const S, const Uint t, const Rvec output,
  const Policy_t& pol, const Uint thrID) const
{
  const Advantage_t adv = prepare_advantage<Advantage_t,Policy_t>(
                                  output, &aInfo, adv_start, &pol);
  const Real A_cur = adv.computeAdvantage(pol.sampAct);
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = S->Q_RET[t] - output[VsID];
  const Real Ver = std::min((Real)1, pol.sampImpWeight) * (A_RET-A_cur);
  updateRetrace(S, t, A_cur, output[VsID], pol.sampImpWeight);
  S->setMseDklImpw(t, Ver*Ver,pol.sampKLdiv,pol.sampImpWeight, CmaxRet,CinvRet);
  const Rvec pg = pol.div_kl_grad(S->tuples[t]->mu, beta-1);
  // only non zero gradient is policy penalization
  Rvec gradient = Rvec(F[0]->nOutputs(), 0);
  pol.finalize_grad(pg, gradient);
  return gradient;
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
policyGradient(const Tuple*const _t, const Policy_t& POL,
  const Advantage_t& ADV, const Real A_RET, const Uint thrID) const
{
  const Real rho_cur = POL.sampImpWeight;
  #if defined(RACER_TABC) // apply ACER's var trunc and bias corr trick
    //compute quantities needed for trunc import sampl with bias correction
    const Action_t sample = POL.sample(&generators[thrID]);
    const Real polProbOnPolicy = POL.evalLogProbability(sample);
    const Real polProbBehavior = Policy_t::evalBehavior(sample, _t->mu);
    const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
    const Real A_pol = ADV.computeAdvantage(sample);
    const Real gain1 = A_RET*std::min((Real)1, rho_cur);
    const Real gain2 = A_pol*std::max((Real)0, 1-1/rho_pol);

    const Rvec gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
    const Rvec gradAcer_2 = POL.policy_grad(sample,      gain2);
    return sum2Grads(gradAcer_1, gradAcer_2);
  #else
    // all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
    return POL.policy_grad(POL.sampAct, A_RET*std::min(CmaxRet,rho_cur));
  #endif
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);
  F[0]->prepare_agent(traj, agent);

  if( agent.Status < TERM_COMM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    Rvec output = F[0]->forward_agent(agent);
    Policy_t pol = prepare_policy<Policy_t>(output, &aInfo, pol_start);
    const Advantage_t adv = prepare_advantage<Advantage_t,Policy_t>(
                                    output, &aInfo, adv_start, &pol);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    Action_t act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID],mu);
    const Real advantage = adv.computeAdvantage(pol.sampAct);
    traj->action_adv.push_back(advantage);
    traj->state_vals.push_back(output[VsID]);
    agent.act(act);
    data_get->add_action(agent, mu);

    #ifndef NDEBUG
      Policy_t dbg = prepare_policy<Policy_t>(output, &aInfo, pol_start);
      dbg.prepare(traj->tuples.back()->a, traj->tuples.back()->mu);
      const double err = fabs(dbg.sampImpWeight-1);
      if(err>1e-10) _die("Imp W err %20.20e", err);
    #endif
  }
  else // either terminal or truncation state
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[0]->forward_agent(agent);
      traj->state_vals.push_back(output[VsID]); // not a terminal state
    } else {
      traj->state_vals.push_back(0); //value of terminal state is 0
    }
    //whether seq is truncated or terminated, act adv is undefined:
    traj->action_adv.push_back(0);
    const Uint N = traj->tuples.size();
    // compute initial Qret for whole trajectory:
    assert(N == traj->action_adv.size());
    assert(N == traj->state_vals.size());
    assert(0 == traj->Q_RET.size());
    //within Retrace, we use the Q_RET vector to write the Adv retrace values
    traj->Q_RET.resize(N, 0); traj->offPolicImpW.resize(N, 1);
    for(Uint i=traj->ndata(); i>0; i--) backPropRetrace(traj, i);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

// Template specializations. From now on, nothing relevant to algorithm itself.

template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, aI->maxLabel, aI->maxLabel};
}
template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2]};
}
template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnOutputs(const ActionInfo*const aI) {
  return 1 + aI->maxLabel + aI->maxLabel;
}
template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI) {
  return aI->maxLabel;
}

template<>
RACER<Discrete_advantage, Discrete_policy, Uint>::
RACER(Environment*const _env, Settings& _set) : Learner_offPolicy(_env,_set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Discrete-action RACER: Built network with outputs: v:%u pol:%s adv:%s\n", VsID, print(pol_start).c_str(), print(adv_start).c_str());
  }
  computeQretrace = true;
  setupNet();
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template<> vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
  return vector<Uint>{1, nL, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<> vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2], indices[3], indices[4]};
}
template<> vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<> Uint
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
getnOutputs(const ActionInfo*const aI) {
  const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
  return 1 + nL + NEXPERTS*(1 +2*aI->dim);
}
template<> Uint
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return NEXPERTS*(1 +2*aI->dim);
}

template<>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
RACER(Environment*const _env, Settings& _set) : Learner_offPolicy(_env, _set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Mixture-of-experts continuous-action RACER: Built network with outputs: v:%u pol:%s adv:%s (sorted %s)\n", VsID, print(pol_start).c_str(), print(adv_start).c_str(), print(net_outputs).c_str());
  }
  computeQretrace = true;
  setupNet();

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

    Gaussian_mixture<NEXPERTS> pol = prepare_policy<Gaussian_mixture<NEXPERTS>>(output, &aInfo, pol_start);
    Rvec act = pol.finalize(1, &generators[0], mu);
    Mixture_advantage<NEXPERTS> adv = prepare_advantage<
     Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS> > (
                                output, &aInfo, adv_start, &pol);
    adv.test(act, &generators[0]);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template<> vector<Uint>
RACER<Quadratic_advantage, Gaussian_policy, Rvec>::
count_outputs(const ActionInfo*const aI) {
  const Uint nL = Quadratic_advantage::compute_nL(aI);
  return vector<Uint>{1, nL, aI->dim, aI->dim};
}
template<> vector<Uint>
RACER<Quadratic_advantage, Gaussian_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2], indices[3]};
}
template<> vector<Uint>
RACER<Quadratic_advantage, Gaussian_policy, Rvec>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<> Uint
RACER<Quadratic_advantage, Gaussian_policy, Rvec>::
getnOutputs(const ActionInfo*const aI) {
  const Uint nL = Quadratic_advantage::compute_nL(aI);
  return 1 + nL + 2*aI->dim;
}
template<> Uint
RACER<Quadratic_advantage, Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return 2*aI->dim;
}

template<>
RACER<Quadratic_advantage, Gaussian_policy, Rvec>::
RACER(Environment*const _env, Settings& _set) : Learner_offPolicy(_env, _set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Mixture-of-experts continuous-action RACER: Built network with outputs: v:%u pol:%s adv:%s (sorted %s)\n", VsID, print(pol_start).c_str(), print(adv_start).c_str(), print(net_outputs).c_str());
  }
  computeQretrace = true;
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);

    for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=0; i<nA; i++) mu[i+nA] = std::exp(0.5*mu[i+nA] -1);

    for(Uint i=0; i<=nL; i++) output[i] = 0.5*dist(generators[0]);
    for(Uint i=0; i<nA; i++)
      output[1+nL+i] = mu[i] + dist(generators[0])*mu[i+nA];
    for(Uint i=0; i<nA; i++)
      output[1+nL+i+nA] = noiseMap_inverse(mu[i+nA]) + .1*dist(generators[0]);

    Gaussian_policy pol = prepare_policy<Gaussian_policy>(output, &aInfo, pol_start);
    Rvec act = pol.finalize(1, &generators[0], mu);
    Quadratic_advantage adv = prepare_advantage<Quadratic_advantage,
      Gaussian_policy> ( output, &aInfo, adv_start, &pol );
    adv.test(act, &generators[0]);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

////////////////////////////////////////////////////////////////////////////////


template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::setupNet()
{
  const std::type_info& actT = typeid(Action_t);
  const std::type_info& vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();

  vector<Uint> nouts = count_outputs(&aInfo);

  #ifdef RACER_singleNet // state value is approximated by an other net
    F.push_back(new Approximator("net", settings, input, data));
  #else
    nout.erase( nout.begin() );
    F.push_back(new Approximator("policy", settings, input, data));
    F[0]->blockInpGrad = true; // this line must happen b4 initialize
    F.push_back(new Approximator("critic", settings, input, data));
    // make value network:
    Builder build_val = F[1]->buildFromSettings(settings, 1);
    F[1]->initializeNetwork(build_val);
  #endif

  #ifdef RACER_simpleSigma // variance not dependent on state
    const Uint varianceSize = nouts.back();
    if(isContinuous) nouts.pop_back();
  #endif

  Builder build = F[0]->buildFromSettings(settings, nouts);

  if(isContinuous)
  {
    #ifdef RACER_singleNet
      Rvec  polBias = Rvec(1, 0);
      Rvec& valBias = polBias;
    #else
      Rvec polBias = Rvec(0, 0); // no state val here
      Rvec valBias = Rvec(1, 0); // separate bias init vector for val net
    #endif
    Advantage_t::setInitial(&aInfo, valBias);
    Policy_t::setInitial_noStdev(&aInfo, polBias);

    #ifdef RACER_simpleSigma // sigma not linked to state: param output
      build.setLastLayersBias(polBias);
      #ifdef EXTRACT_COVAR
        Real initParam = noiseMap_inverse(explNoise*explNoise);
      #else
        Real initParam = noiseMap_inverse(explNoise);
      #endif
      build.addParamLayer(varianceSize, "Linear", initParam);
    #else
      Policy_t::setInitial_Stdev(&aInfo, polBias, explNoise);
      build.setLastLayersBias(polBias);
    #endif
  }

  // construct policy net:
  if(F.size() > 1) die("");
  F[0]->initializeNetwork(build);
  F[0]->opt->bAnnealLearnRate= true;
  trainInfo = new TrainData("racer", settings, 1, "| dAdv | avgW ", 2);
}

////////////////////////////////////////////////////////////////////////////

template class RACER<Discrete_advantage, Discrete_policy, Uint>;
template class RACER<Quadratic_advantage, Gaussian_policy, Rvec>;
template class RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;
