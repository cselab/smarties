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
  const Approximator& NET = * networks[0]; // racer always uses only one net
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->start("FWD");
  const Rvec O = NET.forward(bID, t); // network compute

  //Update Qret of eps' last state if sampled T-1. (and V(s_T) for truncated ep)
  if( MB.isTruncated(bID, t+1) ) {
    assert( t+1 == MB.nDataSteps(bID) );
    MB.updateRetrace(bID, t+1, 0, NET.forward(bID, t+1)[VsID], 0);
  }

  if(thrID==0) profiler->stop_start("CMP");

  const auto POL = prepare_policy<Policy_t>(O, MB.action(bID,t), MB.mu(bID,t));
  const Real rho = POL.sampImpWeight, dkl = POL.sampKLdiv;
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isFarPol = isFarPolicy(rho, CmaxRet, CinvRet);

  const auto ADV = prepare_advantage<Advantage_t>(O, &POL);
  const Real A_cur = ADV.computeAdvantage(POL.sampAct), V_cur = O[VsID];
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = MB.Q_RET(bID, t) - V_cur;
  const Real Ver = std::min((Real)1, rho) * (A_RET-A_cur);
  // all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
  const Real Aer = std::min(CmaxRet, rho) * (A_RET-A_cur);
  const Real deltaQRET = MB.updateRetrace(bID, t, A_cur, V_cur, rho);
  //if(!thrID) std::cout<<dkl<<" s "<<print(S.states[samp])
  //  <<" pol "<<print(POL.getVector())<<" mu "<<MU)
  //  <<" act: "<<print(S.actions[samp])<<" pg: "<<print(polG)
  //  <<" pen: "<<print(penalG)<<" fin: "<<print(finalG)<<"\n";
  //prepare Q with off policy corrections for next step:

  // compute the gradient:
  Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
  gradient[VsID] = isFarPol? 0 : beta * Ver;
  const Rvec penalG  = POL.div_kl_grad(MB.mu(bID,t), -1);
  const Rvec polG = isFarPol? Rvec(penalG.size(), 0) :
                    policyGradient(MB.mu(bID,t), POL, ADV, A_RET, thrID);
  POL.finalize_grad(Utilities::weightSum2Grads(polG, penalG, beta), gradient);
  ADV.grad(POL.sampAct, isFarPol? 0 : beta * Aer, gradient);
  MB.setMseDklImpw(bID, t, Ver*Ver, dkl, rho, CmaxRet, CinvRet);
  NET.setGradient(gradient, bID, t); // place gradient onto output layer

  // logging for diagnostics:
  trainInfo->log(V_cur+A_cur, A_RET-A_cur, polG,penalG, {deltaQRET,rho}, thrID);
  if(ESpopSize > 1) {
    rhos[bID][wID] = rho;
    advs[bID][wID] = A_RET;
    dkls[bID][wID] = dkl;
  }
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
