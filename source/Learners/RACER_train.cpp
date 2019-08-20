//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

namespace smarties
{

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  Sequence& S = MB.getEpisode(bID);
  const Approximator& NET = * networks[0]; // racer always uses only one net
  const Uint t = MB.getTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->stop_start("FWD");
  const Rvec O = NET.forward(bID, t); // network compute

  //Update Qret of eps' last state if sampled T-1. (and V(s_T) for truncated ep)
  if( S.isTruncated(t+1) ) {
    assert( t+1 == S.ndata() );
    const Rvec nxt = NET.forward(bID, t+1);
    updateRetrace(S, t+1, 0, nxt[VsID], 0);
  }

  const auto P = prepare_policy<Policy_t>(O, MB.action(bID,t), MB.mu(bID,t));
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isOff = S.isFarPolicy(t, P.sampImpWeight, CmaxRet, CinvRet);

  if(thrID==0) profiler->stop_start("CMP");
  Rvec grad;
  if(isOff) grad = offPolCorrUpdate(S, t, O, P, thrID);
  else grad = compute(S, t, O, P, thrID);

  if(thrID==0)  profiler->stop_start("BCK");
  NET.setGradient(grad, bID, t); // place gradient onto output layer
}

/*
template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::Train(const Uint seq, const Uint t,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const S = data->get(seq);
  assert(t+1 < S->nsteps());

  if(thrID==0) profiler->stop_start("FWD");
  F[0]->prepare_one(S, t, thrID, wID); // prepare thread workspace
  const Rvec O = F[0]->forward(t, thrID); // network compute

  //Update Qret of eps' last state if sampled T-1. (and V(s_T) for truncated ep)
  if( S->isTruncated(t+1) ) {
    assert( t+1 == S->ndata() );
    const Rvec nxt = F[0]->forward(t+1, thrID);
    updateRetrace(S, t+1, 0, nxt[VsID], 0);
  }

  const auto P = prepare_policy<Policy_t>(O, S->actions[t], S->policies[t]);
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
*/

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
compute(Sequence& S, const Uint samp, const Rvec& outVec,
        const Policy_t& POL, const Uint thrID) const
{
  const auto ADV = prepare_advantage<Advantage_t>(outVec, &POL);
  const Real A_cur = ADV.computeAdvantage(POL.sampAct), V_cur = outVec[VsID];
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = S.Q_RET[samp] - V_cur;
  const Real rho = POL.sampImpWeight, dkl = POL.sampKLdiv;
  const Real Ver = std::min((Real)1, rho) * (A_RET-A_cur);
  // all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
  const Real Aer = std::min(CmaxRet, rho) * (A_RET-A_cur);
  const Rvec polG = policyGradient(S.policies[samp], POL, ADV, A_RET, thrID);
  const Rvec penalG  = POL.div_kl_grad(S.policies[samp], -1);
  //if(!thrID) cout<<dkl<<" s "<<print(S.states[samp])
  //  <<" pol "<<print(POL.getVector())<<" mu "<<MU)
  //  <<" act: "<<print(S.actions[samp])<<" pg: "<<print(polG)
  //  <<" pen: "<<print(penalG)<<" fin: "<<print(finalG)<<endl;
  //prepare Q with off policy corrections for next step:
  const Real dAdv = updateRetrace(S, samp, A_cur, V_cur, rho);
  // compute the gradient:
  Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
  gradient[VsID] = beta * Ver;
  POL.finalize_grad(Utilities::weightSum2Grads(polG, penalG, beta), gradient);
  ADV.grad(POL.sampAct, beta * Aer, gradient);
  S.setMseDklImpw(samp, Ver*Ver, dkl, rho, CmaxRet, CinvRet);
  // logging for diagnostics:
  trainInfo->log(V_cur+A_cur, A_RET-A_cur, polG,penalG, {dAdv,rho}, thrID);
  return gradient;
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
offPolCorrUpdate(Sequence& S, const Uint t, const Rvec output,
  const Policy_t& pol, const Uint thrID) const
{
  const auto adv = prepare_advantage<Advantage_t>(output, &pol);
  const Real A_cur = adv.computeAdvantage(pol.sampAct);
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = S.Q_RET[t] - output[VsID];
  const Real Ver = std::min((Real)1, pol.sampImpWeight) * (A_RET-A_cur);
  updateRetrace(S, t, A_cur, output[VsID], pol.sampImpWeight);
  S.setMseDklImpw(t, Ver*Ver,pol.sampKLdiv,pol.sampImpWeight, CmaxRet,CinvRet);
  const Rvec pg = pol.div_kl_grad(S.policies[t], beta-1);
  // only non zero gradient is policy penalization
  Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
  pol.finalize_grad(pg, gradient);
  return gradient;
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
policyGradient(const Rvec& MU, const Policy_t& POL,
  const Advantage_t& ADV, const Real A_RET, const Uint thrID) const
{
  const Real rho_cur = POL.sampImpWeight;
  #if defined(RACER_TABC) // apply ACER's var trunc and bias corr trick
    //compute quantities needed for trunc import sampl with bias correction
    const Action_t sample = POL.sample(&generators[thrID]);
    const Real polProbOnPolicy = POL.evalLogProbability(sample);
    const Real polProbBehavior = Policy_t::evalBehavior(sample, MU);
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

/*
template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::TrainBySequences(
  const Uint seq, const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const traj = data->get(seq);
  const int ndata = traj->ndata();
  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_seq(traj, thrID, wID);
  for (int k=0; k<ndata; ++k) F[0]->forward(k, thrID);

  //if partial sequence then compute value of last state (!= R_end)
  if( traj->isTruncated(ndata) ) {
    const Rvec nxt = F[0]->forward(ndata, thrID);
    updateRetrace(traj, ndata, 0, nxt[VsID], 0);
  }

  if(thrID==0)  profiler->stop_start("POL");
  for(int k=ndata-1; k>=0; k--)
  {
    const Rvec out_cur = F[0]->get(k, thrID);
    const Rvec & ACT = traj->actions[k], & MU = traj->policies[k];
    const auto pol= prepare_policy<Policy_t>(out_cur, ACT, MU);
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
*/

}
