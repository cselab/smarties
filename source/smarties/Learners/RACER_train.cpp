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
    MB.setValues(bID, t+1, NET.forward(bID, t+1)[VsID]);
  }

  if(thrID==0) profiler->stop_start("CMP");

  const Policy_t POL(pol_start, aInfo, O);
  const auto & ACT = MB.action(bID, t), & MU = MB.mu(bID, t);
  const Real RHO = POL.importanceWeight(ACT, MU), DKL = POL.KLDivergence(MU);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const bool isFarPol = isFarPolicy(RHO, CmaxRet, CinvRet);

  const Advantage_t ADV(adv_start, aInfo, O, &POL);
  const Real Aval = ADV.computeAdvantage(ACT), Vval = O[VsID], Qval = Aval+Vval;
  // shift retrace-advantage with current V(s) estimate:
  const Real A_RET = MB.returnEstimate(bID, t) - Vval, deltaQ = A_RET - Aval;
  const Real Ver = std::min((Real)1, RHO) * deltaQ;
  // all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
  const Real Aer = std::min(CmaxRet, RHO) * deltaQ;

  // compute the gradient:
  Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
  gradient[VsID] = isFarPol? 0 : beta * Ver;
  const Rvec penalG  = POL.KLDivGradient(MU, -1);
  const Rvec polG = isFarPol? Rvec(penalG.size(), 0) :
                    POL.policyGradient(ACT, A_RET * std::min(CmaxRet, RHO));
  POL.makeNetworkGrad(gradient, Utilities::penalizeReFER(polG, penalG, beta));
  ADV.grad(MB.action(bID, t), isFarPol? 0 : beta * Aer, gradient);
  NET.setGradient(gradient, bID, t); // place gradient onto output layer

  MB.setMseDklImpw(bID, t, deltaQ, DKL, RHO, CmaxRet, CinvRet);
  MB.setValues(bID, t, Vval, Qval);

  if(ESpopSize > 1) {
    rhos[bID][wID] = RHO;
    advs[bID][wID] = A_RET;
    dkls[bID][wID] = DKL;
  }
}

}
