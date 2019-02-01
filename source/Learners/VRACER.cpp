//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Network/Builder.h"

#include "VRACER.h"

#include "../Math/Gaussian_mixture.h"
#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_policy.h"

#define DACER_simpleSigma
#define DACER_singleNet
//#define DACER_useAlpha

template<typename Policy_t>
static inline Policy_t prepare_policy(const Rvec& O, const ActionInfo*const aI,
  const vector<Uint>& pol_indices, const Tuple*const t = nullptr) {
  Policy_t pol(pol_indices, aI, O);
  if(t not_eq nullptr) pol.prepare(t->a, t->mu);
  return pol;
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::TrainBySequences(const Uint seq,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const traj = data->get(seq);
  const int ndata = traj->tuples.size()-1;
  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_seq(traj, thrID, wID);
  for (int k=0; k<ndata; k++) F[0]->forward(k, thrID);

  //if partial sequence then compute value of last state (!= R_end)
  Real Q_RET = data->scaledReward(traj, ndata);
  if( traj->isTruncated(ndata) ) {
    const Rvec nxt = F[0]->forward(ndata, thrID);
    Q_RET = gamma * nxt[VsID];
  }

  if(thrID==0)  profiler->stop_start("POL");
  for(int k=ndata-1; k>=0; k--)
  {
    const Rvec out_cur = F[0]->get(k, thrID);
    const Policy_t pol = prepare_policy<Policy_t>(out_cur, &aInfo, pol_start, traj->tuples[k]);
    // far policy definition depends on rho (as in paper)
    // in case rho outside bounds, do not compute gradient

    const Real W = pol.sampImpWeight, R = data->scaledReward(traj, k);
    const Real V_Sk = out_cur[0], A_RET = Q_RET - V_Sk;
    const Real D_RET = std::min((Real)1, W) * A_RET;
      // check whether importance weight is in 1/CmaxRet < c < CmaxRet
    const bool isOff = traj->isFarPolicy(k, W, CmaxRet, CinvRet);
    traj->setMseDklImpw(k, D_RET*D_RET, pol.sampKLdiv, W, CmaxRet, CinvRet);
    trainInfo->log(V_Sk, D_RET, { std::pow(0,2), W }, thrID);

    // compute the gradient
    Rvec G = Rvec(F[0]->nOutputs(), 0);
    if (isOff) {
      pol.finalize_grad(pol.div_kl_grad(traj->tuples[k]->mu, alpha*(beta-1)), G);
    }  else {
      const Rvec policyG = pol.policy_grad(pol.sampAct, alpha * A_RET * W);
      const Rvec penaltG = pol.div_kl_grad(traj->tuples[k]->mu, -alpha);
      pol.finalize_grad(weightSum2Grads(policyG, penaltG, beta), G);
      trainInfo->trackPolicy(policyG, penaltG, thrID);
    }

    // value gradient:
    assert(std::fabs(G[0])<1e-16); // make sure it was untouched
    G[0] = (1-alpha) * beta * D_RET;
    F[0]->backward(G, k, thrID);

    // update retrace for the previous step k-1:
    Q_RET = R + gamma*(V_Sk + std::min((Real)1,W)*(Q_RET - V_Sk) );
  }

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::Train(const Uint seq, const Uint t,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const S = data->get(seq);
  assert(t+1 < S->tuples.size());

  if(thrID==0) profiler->stop_start("FWD");
  F[0]->prepare_one(S, t, thrID, wID); // prepare thread workspace
  const Rvec out = F[0]->forward(t, thrID); // network compute

  #ifdef DACER_singleNet
    static constexpr int valNetID = 0;
    const Rvec& val = out;
  #else
    static constexpr int valNetID = 1;
    F[1]->prepare_one(S, t, thrID, wID); // prepare thread workspace
    const Rvec val = F[1]->forward(t, thrID); // network compute
  #endif

  if ( wID == 0 and S->isTruncated(t+1) ) {
    assert( t+1 == S->ndata() );
    const Rvec nxt = F[valNetID]->forward(t+1, thrID);
    updateRetrace(S, t+1, 0, nxt[VsID], 0);
  }

  if(thrID==0)  profiler->stop_start("CMP");
  const Policy_t P = prepare_policy<Policy_t>(out, aI, pol_start, S->tuples[t]);
  const Real W = P.sampImpWeight; // \rho = \pi / \mu
  const Real A_RET = S->Q_RET[t] - val[0], D_RET = std::min((Real)1, W) * A_RET;
    // check whether importance weight is in 1/CmaxRet < c < CmaxRet
  const bool isOff = dropRule==1? false : S->isFarPolicy(t, W, CmaxRet,CinvRet);

  if( wID == 0 )
  {
    const Real dAdv = updateRetrace(S, t, 0, val[0], W);
    S->setMseDklImpw(t, D_RET*D_RET, P.sampKLdiv, W, CmaxRet, CinvRet);
    trainInfo->log(val[0], D_RET, { std::pow(dAdv,2), W }, thrID);
  }

  if(ESpopSize>1)
  {
    advs[bID][wID] = A_RET;
    dkls[bID][wID] = P.sampKLdiv;
    rhos[bID][wID] = P.sampImpWeight;
  }
  else
  {
    const Real BETA = dropRule==2? 1 : beta;
    assert(wID == 0);
    Rvec G = Rvec(F[0]->nOutputs(), 0);
    if(isOff)
    {
      #ifdef DACER_useAlpha
        P.finalize_grad(P.div_kl_grad(S->tuples[t]->mu, alpha*(BETA-1)), G);
      #else //DACER_useAlpha
        P.finalize_grad(P.div_kl_grad(S->tuples[t]->mu,        BETA-1 ), G);
      #endif //DACER_useAlpha
    }
    else
    {
      #ifdef DACER_useAlpha
        const Rvec G1 = P.policy_grad(P.sampAct, alpha * A_RET * W);
        const Rvec G2 = P.div_kl_grad(S->tuples[t]->mu, -alpha);
      #else //DACER_useAlpha
        const Rvec G1 = P.policy_grad(P.sampAct, A_RET * W);
        const Rvec G2 = P.div_kl_grad(S->tuples[t]->mu, -1);
      #endif //DACER_useAlpha
      P.finalize_grad(weightSum2Grads(G1, G2, BETA), G);
      trainInfo->trackPolicy(G1, G2, thrID);
      #ifdef DACER_singleNet
        assert(std::fabs(G[0])<1e-16); // make sure it was untouched
        #ifdef DACER_useAlpha
          G[0] = (1-alpha) * BETA * D_RET;
        #else
          G[0] = BETA * D_RET;
        #endif
      #endif
    }

    if(thrID==0) profiler->stop_start("BCK");
    #ifndef DACER_singleNet
      F[1]->backward( Rvec(1, D_RET), t, thrID);
      F[1]->gradient(thrID);  // backprop
    #endif
    F[0]->backward(G, t, thrID);
    F[0]->gradient(thrID);  // backprop
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::select(Agent& agent)
{
  Sequence* const S = data_get->get(agent.ID);
  data_get->add_state(agent);
  F[0]->prepare_agent(S, agent);

  #ifdef DACER_singleNet
    static constexpr int valNetID = 0;
  #else
    static constexpr int valNetID = 1;
    F[1]->prepare_agent(S, agent);
  #endif

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec output = F[0]->forward_agent(agent);
    #ifdef DACER_singleNet
      const Rvec& value = output;
    #else
      Rvec value = F[1]->forward_agent(agent);
    #endif
    Policy_t pol = prepare_policy<Policy_t>(output, &aInfo, pol_start);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID], mu);

    S->state_vals.push_back(value[0]);
    agent.act(act);
    data_get->add_action(agent, mu);
  }
  else
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[valNetID]->forward_agent(agent);
      S->state_vals.push_back(output[0]);
    } else S->state_vals.push_back(0); //value of term state is 0

    const Uint N = S->tuples.size();
    // compute initial Qret for whole trajectory:
    assert(N==S->state_vals.size());
    assert(0==S->Q_RET.size() && 0==S->action_adv.size());

    // compute initial Qret for whole trajectory
    //within Retrace, we use the state_vals vector to write the Q retrace values
    //both if truncated or not, last delta is zero
    S->Q_RET.resize(N, 0);
    S->action_adv.resize(N, 0);
    S->offPolicImpW.resize(N, 1);
    for(Uint i=S->ndata(); i>0; i--) backPropRetrace(S, i);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::prepareGradient()
{
  if(updateComplete and ESpopSize>1)
  {
    profiler->stop_start("LOSS");
    std::vector<Real> aR(batchSize, 0), aA(batchSize, 0);
    #if 1
     #pragma omp parallel for schedule(static)
     for (Uint b=0; b<batchSize; b++) {
       for(Uint w=0; w<ESpopSize; w++) { aR[b]+=rhos[b][w]; aA[b]+=advs[b][w]; }
       aR[b] /= ESpopSize; aA[b] /= ESpopSize;
     }
    #else
     for(Uint b=0; b<batchSize; b++) { aR[b] = rhos[b][0]; aA[b] = advs[b][0]; }
    #endif

    const auto isFar = [&](const Real&W) {return W >= CmaxRet || W <= CinvRet;};
    #pragma omp parallel for schedule(static)
    for (Uint w=0; w<ESpopSize; w++)
    for (Uint b=0; b<batchSize; b++) {
      const Real clipR = std::max(CinvRet, std::min(rhos[b][w], CmaxRet));
      const Real clipA = isFar(rhos[b][w]) ? aA[b] : advs[b][w];
      const Real costAdv = - beta * clipR * aA[b]; //minus: to maximize pol adv
      const Real costVal = beta * std::pow(std::min((Real)1, aR[b]) * clipA, 2);
      const Real costDkl = (1-beta) * dkls[b][w];
      #ifdef DACER_singleNet
        F[0]->losses[w] += alpha * (costAdv + costDkl) + (1-alpha) * costVal;
      #else
        F[0]->losses[w] += costAdv + costDkl;
        F[1]->losses[w] += costVal;
      #endif
    }
    F[0]->nAddedGradients = ESpopSize * batchSize;
    #ifndef DACER_singleNet
      F[1]->nAddedGradients = ESpopSize * batchSize;
    #endif
  }

  Learner_offPolicy::prepareGradient();
}


///////////////////////////////////////////////////////////////////////////////

template<> vector<Uint> VRACER<Discrete_policy, Uint>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, aI->maxLabel};
}
template<> vector<Uint> VRACER<Discrete_policy, Uint>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return vector<Uint>{indices[1]};
  #else
    return vector<Uint>{indices[0]};
  #endif
}
template<> Uint VRACER<Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI) { return aI->maxLabel; }

template<> VRACER<Discrete_policy, Uint>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Discrete-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  computeQretrace = true;
  setupNet();
}

/////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

template<> vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<> vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return vector<Uint>{indices[1], indices[2], indices[3]};
  #else
    return vector<Uint>{indices[0], indices[1], indices[2]};
  #endif
}
template<> Uint VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) { return NEXPERTS*(1 +2*aI->dim); }

template<> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Mixture-of-experts continuous-action V-RACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
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
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template<> vector<Uint> VRACER<Gaussian_policy, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, aI->dim, aI->dim};
}
template<> vector<Uint> VRACER<Gaussian_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return vector<Uint>{indices[1], indices[2]};
  #else
    return vector<Uint>{indices[0], indices[1]};
  #endif
}
template<> Uint VRACER<Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI) { return 2*aI->dim; }

template<> VRACER<Gaussian_policy, Rvec>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI)) {
  printf("Gaussian continuous-action V-RACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  computeQretrace = true;
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);

    for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=0; i<nA; i++) mu[i+nA] = std::exp(0.5*mu[i+nA] -1);
    for(Uint i=0; i<nA; i++) output[1+i] = mu[i] + dist(generators[0])*mu[i+nA];
    for(Uint i=0; i<nA; i++)
      output[1+i+nA] = noiseMap_inverse(mu[i+nA]) + .1*dist(generators[0]);

    Gaussian_policy pol = prepare_policy<Gaussian_policy>(output, &aInfo, pol_start);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

///////////////////////////////////////////////////////////////////////////////

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::setupNet()
{
  const std::type_info& actT = typeid(Action_t);
  const std::type_info& vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();

  vector<Uint> nouts = count_outputs(&aInfo);

  #ifdef DACER_singleNet // state value is approximated by an other net
    F.push_back(new Approximator("net", settings, input, data));
  #else
    nouts.erase( nouts.begin() );
    F.push_back(new Approximator("policy", settings, input, data));
    F[0]->blockInpGrad = true; // this line must happen b4 initialize
    F.push_back(new Approximator("critic", settings, input, data));
    // make value network:
    Builder build_val = F[1]->buildFromSettings(settings, 1);
    F[1]->initializeNetwork(build_val);
  #endif

  #ifdef DACER_simpleSigma // variance not dependent on state
    const Uint varianceSize = nouts.back();
    if(isContinuous) nouts.pop_back();
  #endif

  Builder build = F[0]->buildFromSettings(settings, nouts);

  if(isContinuous)
  {
    #ifdef DACER_singleNet
      Rvec initBias = Rvec(1, 0);
    #else
      Rvec initBias = Rvec(0, 0); // no state val here
    #endif
    Policy_t::setInitial_noStdev(&aInfo, initBias);

    #ifdef DACER_simpleSigma // sigma not linked to state: param output
      build.setLastLayersBias(initBias);
      #ifdef EXTRACT_COVAR
        Real initParam = noiseMap_inverse(explNoise*explNoise);
      #else
        Real initParam = noiseMap_inverse(explNoise);
      #endif
      build.addParamLayer(varianceSize, "Linear", initParam);
    #else
      Policy_t::setInitial_Stdev(&aInfo, initBias, explNoise);
      build.setLastLayersBias(initBias);
    #endif
  }

  // construct policy net:
  F[0]->initializeNetwork(build);
  trainInfo = new TrainData("v-racer", settings,ESpopSize<2,"| dAdv | avgW ",2);
}

///////////////////////////////////////////////////////////////////////////////

template class VRACER<Discrete_policy, Uint>;
template class VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
template class VRACER<Gaussian_policy, Rvec>;
