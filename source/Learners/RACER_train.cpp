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

  const Policy_t POL(pol_start, aInfo, O);
  const auto & ACT = MB.action(bID, t), & MU = MB.mu(bID, t);
  const Real RHO = POL.importanceWeight(ACT, MU), DKL = POL.KLDivergence(MU);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isFarPol = isFarPolicy(RHO, CmaxRet, CinvRet);

  const Advantage_t ADV(adv_start, aInfo, O, &POL);
  const Real A_cur = ADV.computeAdvantage(ACT), V_cur = O[VsID];
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = MB.Q_RET(bID, t) - V_cur;
  const Real Ver = std::min((Real)1, RHO) * (A_RET-A_cur);
  // all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
  const Real Aer = std::min(CmaxRet, RHO) * (A_RET-A_cur);
  const Real deltaQRET = MB.updateRetrace(bID, t, A_cur, V_cur, RHO);
  //if(!thrID) std::cout<<DKL<<" s "<<print(S.states[samp])
  //  <<" pol "<<print(POL.getVector())<<" mu "<<MU)
  //  <<" act: "<<print(S.actions[samp])<<" pg: "<<print(polG)
  //  <<" pen: "<<print(penalG)<<" fin: "<<print(finalG)<<"\n";
  //prepare Q with off policy corrections for next step:

  // compute the gradient:
  Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
  gradient[VsID] = isFarPol? 0 : beta * Ver;
  const Rvec penalG  = POL.KLDivGradient(MU, -1);
  const Rvec polG = isFarPol? Rvec(penalG.size(), 0) :
                    policyGradient(MU, ACT, POL, ADV, A_RET, RHO, thrID);
  POL.makeNetworkGrad(gradient, Utilities::penalizeReFER(polG, penalG, beta));
  ADV.grad(MB.action(bID, t), isFarPol? 0 : beta * Aer, gradient);
  MB.setMseDklImpw(bID, t, Ver*Ver, DKL, RHO, CmaxRet, CinvRet);
  NET.setGradient(gradient, bID, t); // place gradient onto output layer

  // logging for diagnostics:
  trainInfo->log(V_cur+A_cur, A_RET-A_cur, polG,penalG, {deltaQRET}, thrID);
  if(ESpopSize > 1) {
    rhos[bID][wID] = RHO;
    advs[bID][wID] = A_RET;
    dkls[bID][wID] = DKL;
  }
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
Rvec RACER<Advantage_t, Policy_t, Action_t>::
policyGradient(const Rvec& MU, const Rvec& ACT,
               const Policy_t& POL, const Advantage_t& ADV,
               const Real A_RET, const Real IMPW, const Uint thrID) const
{
  #if defined(RACER_TABC) // apply ACER's var trunc and bias corr trick
    //compute quantities needed for trunc import sampl with bias correction
    const Action_t sample = POL.sample(generators[thrID]);
    const Real polProbOnPolicy = POL.evalLogProbability(sample);
    const Real polProbBehavior = POL.evalLogBehavior(sample, MU);
    const Real rho_pol = Utilities::safeExp(polProbOnPolicy-polProbBehavior);
    const Real A_pol = ADV.computeAdvantage(sample);
    const Real gain1 = A_RET*std::min((Real)1, IMPW);
    const Real gain2 = A_pol*std::max((Real)0, 1-1/rho_pol);
    const Rvec gradAcer_1 = POL.policyGradient(ACT, gain1);
    const Rvec gradAcer_2 = POL.policyGradient(sample, gain2);
    return Utilities::sum2Grads(gradAcer_1, gradAcer_2);
  #else
    // all these min(CmaxRet, IMPW) have no effect with ReFer enabled
    return POL.policyGradient(ACT, A_RET*std::min(CmaxRet, IMPW));
  #endif
}

}
