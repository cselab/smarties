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
void PPO<Policy_t, Action_t>::advanceEpochCounters()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this with this func
  debugL("shift counters of epochs over the stored data");
  cntBatch += settings.batchSize;
  if(cntBatch >= nHorizon) {
    using Utilities::annealRate;
    const Real annlLR = annealRate(settings.learnrate, currStep, epsAnneal);
    data_proc->updateRewardsStats(0.001, annlLR);
    cntBatch = 0;
    cntEpoch++;
  }

  if(cntEpoch >= nEpochs) {
    debugL("finished epochs, compute state/rew stats, clear buffer to gather new onpol samples");
    #if 0 // keep nearly on policy data
      cntKept = data->clearOffPol(CmaxPol, 0.05);
    #else
      data->clearAll();
      cntKept = 0;
    #endif
    //reset batch learning counters
    cntEpoch = 0;
    cntBatch = 0;
  }
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::
Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  Sequence& S = MB.getEpisode(bID);
  const Uint t = MB.getTstep(bID), thrID = omp_get_thread_num();
  if(thrID==0)  profiler->stop_start("FWD");

  const Rvec pVec = actor->forward(bID, t); // network compute
  const Rvec sVal = critc->forward(bID, t); // network compute

  if(thrID==0)  profiler->stop_start("CMP");

  const auto POL = prepare_policy(pVec, S.actions[t], S.policies[t]);
  const Real DKL = POL.sampKLdiv, RHO = POL.sampImpWeight;
  const bool isOff = S.isFarPolicy(t, RHO, CmaxRet, CinvRet);

  penalUpdateCount = penalUpdateCount + 1.0;
  if(DKL < DKL_target / 1.5)
    penalUpdateDelta = penalUpdateDelta - penalCoef/2; //half
  if(DKL > 1.5 * DKL_target)
    penalUpdateDelta = penalUpdateDelta + penalCoef; //double

  Real gain = RHO * S.action_adv[t];
  #ifdef PPO_CLIPPED
    if(S.action_adv[t] > 0 && RHO > 1+CmaxPol) gain = 0;
    if(S.action_adv[t] < 0 && RHO < 1-CmaxPol) gain = 0;
    updateDKL_target(isOff, RHO);
  #endif

  #ifdef PPO_PENALKL //*nonZero(gain)
    const Rvec polG = POL.policy_grad(gain);
    const Rvec penG = POL.div_kl_grad(S.policies[t], - penalCoef);
    const Rvec totG = Utilities::weightSum2Grads(polG, penG, 1);
  #else //we still learn the penal coef, for simplicity, but no effect
    const Rvec totG = POL.policy_grad(gain), penG = Rvec(policy_grad.size(), 0);
    const Rvec& polG = totalPolGrad;
  #endif

  assert(wID == 0);
  Rvec grad(actor->nOutputs(), 0);
  POL.finalize_grad(totG, grad);

  //bookkeeping:
  const Real verr = S.Q_RET[t] - sVal[0]; // Q_ret actually stores V_gae here
  #ifdef PPO_learnDKLt
  trainInfo->log(sVal[0],verr,polG,penG, {penalCoef,DKL,RHO,DKL_target}, thrID);
  #else
  trainInfo->log(sVal[0],verr,polG,penG, {penalCoef,DKL,RHO}, thrID);
  #endif
  S.setMseDklImpw(t, verr*verr, DKL, RHO, 1+CmaxPol, 1-CmaxPol);

  if(thrID==0)  profiler->stop_start("BCK");
  actor->setGradient(totG, bID, t);
  critc->setGradient({ verr * ( isOff? 1 : 0 ) }, bID, t);
}

}
