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
#warning "Using Mixture_advantage with Gaussian advantages"
#else
#warning "Using Mixture_advantage with Quadratic advantages"
#endif

#ifdef DKL_filter
#undef DKL_filter
#endif
#define RACER_simpleSigma

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
TrainBySequences(const Uint seq, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  const int ndata = traj->tuples.size()-1;
  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_seq(traj, thrID);
  for (int k=0; k<ndata; k++) F[0]->forward(traj, k, thrID);
  //if partial sequence then compute value of last state (!= R_end)
  if( traj->isTerminal(ndata) ) updateQret(traj, ndata, 0, 0, 0);
  else if( traj->isTruncated(ndata) ) {
    const Rvec nxt = F[0]->forward(traj, ndata, thrID);
    traj->setStateValue(ndata, nxt[VsID]);
    updateQret(traj, ndata, 0, nxt[VsID], 0);
  }

  if(thrID==0)  profiler->stop_start("POL");
  for(int k=ndata-1; k>=0; k--)
  {
    const Rvec out_cur = F[0]->get(traj, k, thrID);
    const Policy_t pol = prepare_policy(out_cur, traj->tuples[k]);
    #ifdef DKL_filter
      const bool isOff = traj->distFarPolicy(k, pol.sampKLdiv, CmaxRet-1);
    #else
      const bool isOff = traj->isFarPolicy(k, pol.sampImpWeight, CmaxRet);
    #endif
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
void RACER<Advantage_t, Policy_t, Action_t>::
Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  assert(samp+1 < traj->tuples.size());

  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_one(traj, samp, thrID); // prepare thread workspace
  const Rvec out_cur = F[0]->forward(traj, samp, thrID); // network compute

  if( traj->isTerminal(samp+1) ) updateQret(traj, samp+1, 0, 0, 0);
  else if( traj->isTruncated(samp+1) ) {
    const Rvec nxt = F[0]->forward(traj, samp+1, thrID);
    traj->setStateValue(samp+1, nxt[VsID]);
    updateQret(traj, samp+1, 0, nxt[VsID], 0);
  }

  const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isOff = traj->isFarPolicy(samp, pol.sampImpWeight, CmaxRet);

  if(thrID==0)  profiler->stop_start("CMP");
  Rvec grad;
  if(isOff) grad = offPolCorrUpdate(traj, samp, out_cur, pol, thrID);
  else grad = compute(traj, samp, out_cur, pol, thrID);

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(grad, samp, thrID); // place gradient onto output layer
  F[0]->gradient(thrID);  // backprop
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
compute(Sequence*const traj, const Uint samp, const Rvec& outVec,
  const Policy_t& POL, const Uint thrID) const
{
  const Advantage_t ADV = prepare_advantage(outVec, &POL);
  const Real A_cur = ADV.computeAdvantage(POL.sampAct), V_cur = outVec[VsID];
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = traj->Q_RET[samp] +traj->state_vals[samp]-V_cur;
  const Real rho = POL.sampImpWeight, dkl = POL.sampKLdiv;
  const Real Ver = std::min((Real)1, rho) * (A_RET-A_cur);
  const Real Aer = std::min(CmaxRet, rho) * (A_RET-A_cur);
  const Rvec polG = policyGradient(traj->tuples[samp], POL,ADV,A_RET, thrID);
  const Rvec penalG  = POL.div_kl_grad(traj->tuples[samp]->mu, -1);
  const Rvec finalG  = weightSum2Grads(polG, penalG, beta);
  //if(!thrID) cout<<dkl<<" s "<<print(traj->tuples[samp]->s)
  //  <<" pol "<<print(POL.getVector())<<" mu "<<print(traj->tuples[samp]->mu)
  //  <<" act: "<<print(traj->tuples[samp]->a)<<" pg: "<<print(polG)
  //  <<" pen: "<<print(penalG)<<" fin: "<<print(finalG)<<endl;
  //prepare Q with off policy corrections for next step:
  const Real dAdv = updateQret(traj, samp, A_cur, V_cur, POL);

  Rvec gradient = Rvec(F[0]->nOutputs(), 0);
  gradient[VsID] = beta*alpha * Ver;
  POL.finalize_grad(finalG, gradient);
  ADV.grad(POL.sampAct, beta*alpha * Aer, gradient);
  trainInfo->log(V_cur+A_cur, A_RET-A_cur, polG,penalG, {beta,dAdv,rho}, thrID);
  traj->setMseDklImpw(samp, Ver*Ver, dkl, rho);
  return gradient;
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
offPolCorrUpdate(Sequence*const S, const Uint t, const Rvec output,
  const Policy_t& pol, const Uint thrID) const
{
  const Advantage_t adv = prepare_advantage(output, &pol);
  const Real A_cur = adv.computeAdvantage(pol.sampAct);
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = S->Q_RET[t] +S->state_vals[t] -output[VsID];
  const Real Ver = std::min((Real)1, pol.sampImpWeight) * (A_RET-A_cur);
  updateQret(S, t, A_cur, output[VsID], pol);
  S->setMseDklImpw(t, Ver*Ver, pol.sampKLdiv, pol.sampImpWeight);
  const Rvec pg = pol.div_kl_grad(S->tuples[t]->mu, beta-1);
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
  #if defined(RACER_TABC)
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
    // remember, all these min(CmaxRet,rho_cur) have no effect with ReFer
    return POL.policy_grad(POL.sampAct, A_RET*std::min(CmaxRet,rho_cur));
  #endif
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
select(Agent& agent)
{
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);
  F[0]->prepare_agent(traj, agent);

  if( agent.Status < TERM_COMM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    Rvec output = F[0]->forward_agent(traj, agent);
    Policy_t pol = prepare_policy(output);
    const Advantage_t adv = prepare_advantage(output, &pol);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    Action_t act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID],mu);
    const Real advantage = adv.computeAdvantage(pol.sampAct);
    traj->action_adv.push_back(advantage);
    traj->state_vals.push_back(output[VsID]);
    agent.act(act);

    #ifdef dumpExtra
      traj->add_action(agent.a->vals, mu);
      Rvec param = adv.getParam();
      assert(param.size() == nL);
      mu.insert(mu.end(), param.begin(), param.end());
      agent.writeData(learn_rank, mu);
    #else
      data->add_action(agent, mu);
    #endif

    #ifndef NDEBUG
      Policy_t dbg = prepare_policy(output);
      dbg.prepare(traj->tuples.back()->a, traj->tuples.back()->mu);
      const double err = fabs(dbg.sampImpWeight-1);
      if(err>1e-10) _die("Imp W err %20.20e", err);
    #endif
  }
  else
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[0]->forward_agent(traj, agent);
      traj->state_vals.push_back(output[VsID]); // not a terminal state
    } else {
      traj->state_vals.push_back(0); //value of terminal state is 0
    }
    //whether seq is truncated or terminated, act adv is undefined:
    traj->action_adv.push_back(0);
    // compute initial Qret for whole trajectory:
    assert(traj->tuples.size() == traj->action_adv.size());
    assert(traj->tuples.size() == traj->state_vals.size());
    assert(traj->Q_RET.size()  == 0);
    //within Retrace, we use the state_vals vector to write the Q retrace values
    traj->Q_RET.resize(traj->tuples.size(), 0);
    for(Uint i=traj->ndata(); i>0; i--) {
      updateQretFront(traj, i);
      //cout<<traj->Q_RET[i]<<" "<<traj->action_adv[i]<<" "<<traj->state_vals[i]<<endl;
    }
    //cout << traj->Q_RET[0]<<" "<<traj->action_adv[0]<<" "<<traj->state_vals[0]<<endl;

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    #ifdef dumpExtra
      agent.a->set(Rvec(nA,0));
      traj->add_action(agent.a->vals, Rvec(policyVecDim,0));
      agent.writeData(learn_rank, Rvec(policyVecDim+nL, 0));
      data->push_back(agent.ID);
    #else
      data->terminate_seq(agent);
    #endif
  }
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
prepareGradient()
{
  Learner_offPolicy::prepareGradient();

  if(updateToApply)
  {
    debugL("Update Retrace est. for episodes samples in prev. grad update");
    // placed here because this happens right after update is computed
    // this can happen before prune and before workers are joined
    profiler->stop_start("QRET");
    #pragma omp parallel for schedule(dynamic)
    for(Uint i = 0; i < data->Set.size(); i++)
      for(int j=data->Set[i]->just_sampled-1; j>0; j--)
        updateQretBack(data->Set[i], j);
  }
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
initializeLearner()
{
  Learner_offPolicy::initializeLearner();

  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  #pragma omp parallel for schedule(dynamic)
  for(Uint i=0; i<data->Set.size(); i++)
    for(Uint j=data->Set[i]->ndata(); j>0; j--) updateQretFront(data->Set[i],j);

  for(Uint i = 0; i < data->inProgress.size(); i++) {
    if(data->inProgress[i]->tuples.size() == 0) continue;
    for(Uint j=data->inProgress[i]->ndata(); j>0; j--)
      updateQretFront(data->inProgress[i],j);
  }
}

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
  return vector<Uint>{indices[1]};
}
template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2]};
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
RACER<Discrete_advantage, Discrete_policy, Uint>::
RACER(Environment*const _env, Settings& _set) : Learner_offPolicy(_env,_set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Discrete-action RACER: Built network with outputs: v:%u pol:%s adv:%s\n", VsID, print(pol_start).c_str(), print(adv_start).c_str());
  }

  F.push_back(new Approximator("net", _set, input, data));
  vector<Uint> nouts{1, nL, nA};
  Builder build = F[0]->buildFromSettings(_set, nouts);
  F[0]->initializeNetwork(build);

  trainInfo = new TrainData("racer", _set, 1, "| beta | dAdv | avgW ", 3);
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

  F.push_back(new Approximator("net", _set, input, data));

  vector<Uint> nouts{1, nL, NEXPERTS, NEXPERTS * nA};
  #ifndef RACER_simpleSigma // network outputs also sigmas
    nouts.push_back(NEXPERTS * nA);
  #endif

  Builder build = F[0]->buildFromSettings(_set, nouts);

  Rvec initBias;
  initBias.push_back(0); // state value
  Mixture_advantage<NEXPERTS>::setInitial(&aInfo, initBias);
  Gaussian_mixture<NEXPERTS>::setInitial_noStdev(&aInfo, initBias);

  #ifdef RACER_simpleSigma // sigma not linked to network: parametric output
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
  //F[0]->save(learner_name+"init");
  if(F.size() > 1) die("");
  F[0]->opt->bAnnealLearnRate= true;

  trainInfo = new TrainData("racer", _set, 1, "| beta | dAdv | avgW ", 3);

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
    Mixture_advantage<NEXPERTS> adv = prepare_advantage(output, &pol);
    adv.test(act, &generators[0]);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

template class RACER<Discrete_advantage, Discrete_policy, Uint>;
template class RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;

/*
  ## FORWARD RACER

      #if RACER_FORWARD>0 // prepare thread workspace
        F[0]->prepare(RACER_FORWARD+1, traj, samp, thrID);
      #else
      #endif
      #ifdef DKL_filter
        const Real KLdiv = pol.kl_divergence(S->tuples[t]->mu);
        const bool isOff = traj->distFarPolicy(t, KLdiv, 1+KLdiv, CmaxRet-1);
      #else

    #if RACER_FORWARD>0
      // do N steps of fwd net to obtain better estimate of Qret
      Uint N = std::min(traj->ndata()-samp, (Uint)RACER_FORWARD);
      for(Uint k = samp+1; k<=samp+N; k++)
      {
        if( traj->isTerminal(k) ) {
          assert(traj->action_adv[k] == 0 && traj->state_vals[k] == 0);
        } else if( traj->isTruncated(k) ) {
          assert(traj->action_adv[k] == 0);
          const Rvec nxt = F[0]->forward(traj, k, thrID);
          traj->setStateValue(k, nxt[VsID]);
        } else {
          const Rvec nxt = F[0]->forward(traj, k, thrID);
          const Policy_t polt = prepare_policy(nxt, traj->tuples[k]);
          const Advantage_t advt = prepare_advantage(nxt, &polt);
          //these are all race conditions:
          traj->setSquaredError(k, polt.kl_divergence(traj->tuples[k]->mu) );
          traj->setAdvantage(k, advt.computeAdvantage(polt.sampAct) );
          traj->setOffPolWeight(k, polt.sampImpWeight );
          traj->setStateValue(k, nxt[VsID] );
        }
      }
      for(Uint j = samp+N; j>samp; j--) updateQret(traj,j);
    #endif

  ## ADV DUMPING (bottom of writeOnPolRetrace)
  #if 0
    #pragma omp critical
    if(nStep>0) {
      // outbuf contains
      // - R[t] = sum_{t'=t}^{T-1} gamma^{t'-t} r_{t+1} (if seq is truncated
      //   instead of terminated, we must add V_T * gamma^(T-t) )
      // - Q^w(s_t,a_t) and Q^ret_t
      outBuf = vector<float>(4*(N-1), 0);
      for(Uint i=N-1; i>0; i--) {
        Real R = data->scaledReward(seq, i) +
          (seq->isTruncated(i)? gamma*seq->state_vals[i] : 0);
        for(Uint j = i; j>0; j--) { // add disc rew to R_t of all prev steps
          outBuf[4*(j-1) +0] += R; R *= gamma;
        }
        outBuf[4*(i-1) +1] = seq->action_adv[i-1];
        // we are actually storing A_RET in there:
        outBuf[4*(i-1) +2] = seq->Q_RET[i-1];
        outBuf[4*(i-1) +3] = seq->state_vals[i-1];
      }
      // revert scaling of rewards
      //for(Uint i=0; i<outBuf.size(); i--) outBuf[i] /= data->invstd_reward;
    }
  #endif
*/
