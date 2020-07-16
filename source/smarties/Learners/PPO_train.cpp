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
void PPO<Policy_t, Action_t>::updatePenalizationCoef()
{
  debugL("update lagrangian penalization coefficient");

  penal_reduce.update( { penalUpdateCount.load(), penalUpdateDelta.load() } );
  if(penalUpdateCount<nnEPS) die("undefined behavior");
  const LDvec penalTerms = penal_reduce.get();
  penalCoef += 1e-4 * penalTerms[1] / std::max( (long double) 1, penalTerms[0] );
  if(penalCoef <= nnEPS) penalCoef = nnEPS;
  penalUpdateCount = 0;
  penalUpdateDelta = 0;
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::
Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();
  if(thrID==0)  profiler->start("FWD");

  const Rvec pVec = actor->forward(bID, t); // network compute
  const Rvec sVal = critc->forward(bID, t); // network compute

  if(thrID==0)  profiler->stop_start("CMP");

  const Policy_t POL(pol_indices, aInfo, pVec);
  const Real RHO = POL.importanceWeight(MB.action(bID,t), MB.mu(bID,t));
  const Real DKL = POL.KLDivergence(MB.mu(bID,t));
  const bool isOff = isFarPolicyPPO(RHO, CmaxPol);

  penalUpdateCount = penalUpdateCount + 1.0;
  if(DKL < DKL_target / 1.5)
    penalUpdateDelta = penalUpdateDelta - penalCoef/2; //half
  if(DKL > 1.5 * DKL_target)
    penalUpdateDelta = penalUpdateDelta + penalCoef; //double

  Real gain = RHO * (MB.returnEstimate(bID, t) - MB.value(bID, t));
  #ifdef PPO_CLIPPED
    if(MB.returnEstimate(bID, t) > 0 && RHO > 1+CmaxPol) gain = 0;
    if(MB.returnEstimate(bID, t) < 0 && RHO < 1-CmaxPol) gain = 0;
    updateDKL_target(isOff, RHO);
  #endif

  #ifdef PPO_PENALKL //*nonZero(gain)
    const Rvec polG = POL.policyGradient(MB.action(bID,t), gain);
    const Rvec penG = POL.KLDivGradient(MB.mu(bID,t), - penalCoef);
    const Rvec totG = Utilities::weightSum2Grads(polG, penG, 1);
  #else //we still learn the penal coef, for simplicity, but no effect
    const Rvec totG = POL.policyGradient(MB.action(bID,t), gain);
    const Rvec penG = Rvec(policy_grad.size(), 0);
    const Rvec& polG = totalPolGrad;
  #endif

  assert(wID == 0);
  Rvec grad(actor->nOutputs(), 0);
  POL.makeNetworkGrad(grad, totG);

  //bookkeeping:
  const Real verr = MB.returnEstimate(bID, t) - sVal[0];
  MB.setMseDklImpw(bID, t, verr, DKL, RHO, 1+CmaxPol, 1-CmaxPol);
  MB.setValues(bID, t, sVal[0], sVal[0]);

  actor->setGradient(totG, bID, t);
  critc->setGradient({ verr * ( isOff? 1 : 0 ) }, bID, t);
}

}
