//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryProcessing.h"
#include "../Utils/SstreamUtilities.h"
#include "ExperienceRemovalAlgorithms.h"
#include "Sampling.h"
#include <algorithm>

namespace smarties
{

namespace MemoryProcessing
{

using returnsEstimator_f = std::function<Fval(const Episode& EP, const Uint t)>;
returnsEstimator_f createReturnEstimator(const MemoryBuffer & RM);

inline Fval updateReturnEstimator(
  Episode & EP, const Uint lastUpdated, const returnsEstimator_f & compute)
{
  assert(EP.stateValue.size() == EP.nsteps());
  assert(EP.offPolicImpW.size() == EP.nsteps());
  assert(EP.actionAdvantage.size() == EP.nsteps());
  assert(EP.returnEstimator.size() == EP.nsteps());
  assert(std::fabs(EP.actionAdvantage.back()) < 1e-16);
  if (EP.bReachedTermState) {
    assert(std::fabs(EP.returnEstimator.back()) < 1e-16);
    assert(std::fabs(EP.stateValue     .back()) < 1e-16);
  } else EP.returnEstimator.back() = EP.stateValue.back();

  Fval sumErr2 = 0;
  for(Sint t = lastUpdated; t>=0; --t) {
    const Fval oldEstimate = EP.returnEstimator[t];
    EP.returnEstimator[t] = compute(EP, t);
    sumErr2 += std::pow(oldEstimate - EP.returnEstimator[t], 2);
  }
  return sumErr2;
}

void updateCounters(MemoryBuffer& RM, const bool bInit)
{
  // use result from prev AllReduce to update rewards (before new reduce).
  // Assumption is that the number of off Pol trajectories does not change
  // much each step. Especially because here we update the off pol W only
  // if an obs is actually sampled. Therefore at most this fraction
  // is wrong by batchSize / nTransitions ( ~ 0 )
  // In exchange we skip an mpi implicit barrier point.

  //_warn("globalStep_reduce %ld %ld", nSeenEpisodes_loc.load(), nSeenTransitions_loc.load());
  RM.globalCounterRdx.update( { RM.nLocalSeenEps(),   RM.nLocalSeenSteps(),
                                RM.nFarPolicySteps(), RM.nStoredSteps() } );
  const std::vector<long> nDataGlobal = RM.globalCounterRdx.get(bInit);
  //_warn("nDataGlobal %ld %ld", nDataGlobal[0], nDataGlobal[1]);
  RM.counters.nSeenEpisodes = nDataGlobal[0];
  RM.counters.nSeenTransitions = nDataGlobal[1];
  assert(nDataGlobal[3]>=nDataGlobal[2]);
  Real fracOffPol = nDataGlobal[2] / (Real) std::max(nDataGlobal[3], (long) 1);

  // In the ReF-ER paper we learn ReF-ER penalization coefficient beta with the
  // network's learning rate eta (was 1e-4). In reality, beta should not depend
  // on eta. beta should reflect estimate of far-policy samples. Accuracy of
  // this estimate depends on: batch size B (bigger B increases accuracy
  // because samples importance weight rho are updated more often) and data set
  // size N (bigger N decreases accuracy because there are more samples to
  // update). We pick coef 0.1 to match learning rate chosen in original paper:
  // we had B=256 and N=2^18 and eta=1e-4. 0.1*B*N \approx 1e-4
  const long maxN = RM.settings.maxTotObsNum, BS = RM.settings.batchSize;
  const Real nDataSize = std::max((long) maxN, nDataGlobal[3]);
  const Real learnRefer = 0.1 * BS / nDataSize;
  const auto fixPointIter = [&] (const Real val, const bool goTo0) {
    if (goTo0) // fixed point iter converging to 0:
      return (1 - std::min(learnRefer, val)) * val;
    else       // fixed point iter converging to 1:
      return (1 - std::min(learnRefer, val)) * val + std::min(learnRefer, 1-val);
  };

  // if too much far policy data, increase weight of Dkl penalty
  const bool bDecreaseBeta = fracOffPol > RM.settings.penalTol;
  RM.beta = fixPointIter(RM.beta, bDecreaseBeta);

  // USED ONLY FOR CMA: how do we weight cirit cost and policy cost?
  // if we satisfy too strictly far-pol constrain, reduce weight of policy
  // TODO work in progress
  const bool bDecreaseAlpha = std::fabs(RM.settings.penalTol-fracOffPol) < 1e-3;
  RM.alpha = fixPointIter(RM.alpha, bDecreaseAlpha);
}

void updateRewardsStats(MemoryBuffer& RM, const bool bInit, const Real rRateFac)
{
  // Update the second order moment of the rewards and means and stdev of states
  // contained in the memory buffer. Used for rescaling and numerical safety.
  if(not RM.distrib.bTrain) return; //if not training, keep the stored values

  MDPdescriptor & MDP = RM.MDP;
  const Uint setSize = RM.nStoredEps(), dimS = MDP.dimStateObserved;
  const Real eta = RM.settings.learnrate, epsAnneal = RM.settings.epsAnneal;
  const Real learnR = Utilities::annealRate(eta, RM.nGradSteps(), epsAnneal);
  const Real annealLearnR = std::min((Real) 1, rRateFac * learnR);
  const Real WS = bInit? 1 : annealLearnR * (SMARTIES_OFFPOL_ADAPT_STSCALE > 0);
  const Real WR = bInit? 1 : annealLearnR;

  if(WR>0 or WS>0)
  {
    long double count = 0, newRSum = 0, newRSqSum = 0;
    std::vector<long double> newSSum(dimS, 0), newSSqSum(dimS, 0);
    #pragma omp parallel reduction(+ : count, newRSum, newRSqSum)
    {
      std::vector<long double> thNewSSum(dimS, 0), thNewSSqSum(dimS, 0);
      #pragma omp for schedule(dynamic, 1) nowait
      for(Uint i=0; i<setSize; ++i) {
        const Episode & EP = RM.get(i);
        const Uint N = EP.ndata();
        count += N;
        for(Uint j=0; j<N; ++j) {
          const long double drk = EP.rewards[j+1] - MDP.rewardsMean;
          newRSum += drk; newRSqSum += drk * drk;
          for(Uint k=0; k<dimS && WS>0; ++k) {
            const long double dsk = EP.states[j][k] - MDP.stateMean[k];
            thNewSSum[k] += dsk; thNewSSqSum[k] += dsk * dsk;
          }
        }
      }
      if(WS>0) {
        #pragma omp critical
        for(Uint k=0; k<dimS; ++k) {
          newSSum[k]   += thNewSSum[k];
          newSSqSum[k] += thNewSSqSum[k];
        }
      }
    }

    //add up gradients across nodes (masters)
    auto newSRstats = newSSum;
    assert(newSRstats.size() == dimS);
    newSRstats.insert(newSRstats.end(), newSSqSum.begin(), newSSqSum.end());
    assert(newSRstats.size() == 2*dimS);
    newSRstats.push_back(count);
    newSRstats.push_back(newRSum);
    newSRstats.push_back(newRSqSum);
    assert(newSRstats.size() == 2*dimS + 3);
    RM.StateRewRdx.update(newSRstats);
  }

  const auto newSRstats = RM.StateRewRdx.get(bInit);
  assert(newSRstats.size() == 2*dimS + 3);
  const auto count = newSRstats[2*dimS];

  // function to update {mean, std, 1/std} given:
  // - Evar = sample_mean minus old mean = E[(X-old_mean)]
  // - Evar2 = E[(X-old_mean)^2]
  const auto updateStats = [] (nnReal & mean, nnReal & stdev, nnReal & invstdev,
    const Real learnRate, const long double Evar, const long double Evar2)
  {
    // mean = (1-learnRate) * mean + learnRate * sample_mean, which becomes:
    mean += learnRate * Evar;
    // if learnRate==1 then variance is exact, otherwise update second moment
    // centered around current sample_mean (ie. var = E[(X-sample_mean)^2]):
    auto variance = Evar2 - Evar*Evar * (2*learnRate - learnRate*learnRate);
    static constexpr long double EPS = std::numeric_limits<nnReal>::epsilon();
    variance = std::max(variance, EPS); //large sum may be neg machine precision
    stdev += learnRate * (std::sqrt(variance) - stdev);
    invstdev = 1 / stdev;
  };

  if(WR>0)
  {
      updateStats(MDP.rewardsMean, MDP.rewardsStdDev, MDP.rewardsScale, WR,
                  newSRstats[2*dimS+1] / count, newSRstats[2*dimS+2] / count);
  }

  if(WS>0)
  {
    const LDvec SSum1(& newSRstats[0], & newSRstats[dimS]);
    const LDvec SSum2(& newSRstats[dimS], & newSRstats[2 * dimS]);
    for(Uint k=0; k<dimS; ++k)
      updateStats(MDP.stateMean[k], MDP.stateStdDev[k], MDP.stateScale[k], WS,
                  SSum1[k] / count, SSum2[k] / count);
  }
}

void selectEpisodeToDelete(MemoryBuffer& RM, const FORGET ALGO)
{
  const Uint nGradSteps = RM.nGradSteps() + 1;
  const bool bRecomputeProperties = ( nGradSteps % 1000) == 0;
  //shift data / gradient counters to maintain grad stepping to sample
  // collection ratio prescirbed by obsPerStep
  const Real C = RM.settings.clipImpWeight, D = RM.settings.penalTol;

  if(ALGO == BATCHRL) {
    const Real maxObsNum = RM.settings.maxTotObsNum_local;
    const Real E = RM.settings.epsAnneal;
    const Real factorUp = std::max((Real) 1, RM.nStoredSteps() / maxObsNum);
    //const Real factorDw = std::min((Real)1, maxObsNum / obsNum);
    //D *= factorUp; //CmaxRet = 1 + C * factorDw
    RM.CmaxRet = 1 + Utilities::annealRate(C, nGradSteps, E) * factorUp;
  } else {
    //CmaxRet = 1 + Utilities::annealRate(C, nGradSteps +1, settings.epsAnneal);
    RM.CmaxRet = 1 + C;
  }
  RM.CinvRet = 1 / RM.CmaxRet;
  if(RM.CmaxRet <= 1 and C > 0) die("Unallowed ReF-ER annealing values.");

  const bool bNeedsReturnEst = RM.settings.returnsEstimator not_eq "none";
  const returnsEstimator_f returnsCompute = createReturnEstimator(RM);

  MostOffPolicyEp totMostOff(D); OldestDatasetEp totFirstIn;
  MostFarPolicyEp totMostFar;    HighestAvgDklEp totHighDkl;

  Uint nOffPol = 0, nRetUpdates = 0;
  Real sumR=0, sumDKL=0, sumE2=0, sumAbsE=0, sumQ2=0, sumQ1=0, sumERet=0;
  Fval maxQ = -1e9, minQ =  1e9;
  const Uint setSize = RM.nStoredEps();
  #pragma omp parallel reduction(max: maxQ) reduction(min: minQ) reduction(+: \
      nOffPol, nRetUpdates, sumDKL, sumE2, sumAbsE, sumQ2, sumQ1, sumR, sumERet)
  {
    OldestDatasetEp locFirstIn; MostOffPolicyEp locMostOff(D);
    MostFarPolicyEp locMostFar; HighestAvgDklEp locHighDkl;

    #pragma omp for schedule(static, 1) nowait
    for (Uint i = 0; i < setSize; ++i)
    {
      Episode & EP = RM.get(i);
      if (bRecomputeProperties) {
        EP.updateCumulative(RM.CmaxRet, RM.CinvRet);
        if (bNeedsReturnEst) {
          sumERet += updateReturnEstimator(EP, EP.nsteps()-2, returnsCompute);
          nRetUpdates += EP.nsteps()-1;
        }
      } else if (bNeedsReturnEst and EP.just_sampled > 0) {
        sumERet += updateReturnEstimator(EP, EP.just_sampled-1, returnsCompute);
        nRetUpdates += EP.just_sampled;
      }

      nOffPol += EP.nFarPolicySteps();    sumDKL  += EP.sumKLDivergence;
      sumE2   += EP.sumSquaredErr;        sumAbsE += EP.sumAbsError;
      sumQ2   += EP.sumSquaredQ;          sumQ1   += EP.sumQ; sumR += EP.totR;
      maxQ     = std::max(EP.maxQ, maxQ); minQ     = std::min(EP.minQ, minQ);
      locFirstIn.compare(EP, i); locMostOff.compare(EP, i);
      locMostFar.compare(EP, i); locHighDkl.compare(EP, i);
      EP.just_sampled = -1;
    }
    #pragma omp critical
    {
      totFirstIn.compare(locFirstIn); totMostOff.compare(locMostOff);
      totMostFar.compare(locMostFar); totHighDkl.compare(locHighDkl);
    }
  }

  ReplayStats & stats = RM.stats;
  if (RM.CmaxRet<=1) nOffPol = 0; //then this counter and its effects are skipped
  const Uint nData = RM.nStoredSteps();
  stats.oldestStoredTimeStamp = totFirstIn.timestamp;
  stats.nFarPolicySteps = nOffPol;

  stats.avgKLdivergence = sumDKL / nData;
  stats.avgSquaredErr = sumE2 / nData;
  stats.avgAbsError = sumAbsE / nData;

  stats.avgReturn = sumR / setSize;
  stats.avgQ = sumQ1 / nData;
  stats.maxQ = maxQ;
  stats.minQ = minQ;
  stats.stdevQ = sumQ2 / nData  - stats.avgQ * stats.avgQ; // variance
  stats.stdevQ = std::sqrt(std::max(stats.stdevQ, 1e-16)); // anti-nan
  if (bNeedsReturnEst) {
    if(stats.countReturnsEstimateUpdates<0)
      stats.countReturnsEstimateUpdates = 0;
    stats.countReturnsEstimateUpdates += nRetUpdates;
    stats.sumReturnsEstimateErrors += sumERet;
  } else {
    stats.countReturnsEstimateUpdates = -1;
    stats.sumReturnsEstimateErrors = 0;
  }
  assert(totMostFar.ind >= 0 && totMostFar.ind < (int) setSize);
  assert(totHighDkl.ind >= 0 && totHighDkl.ind < (int) setSize);
  assert(totFirstIn.ind >= 0 && totFirstIn.ind < (int) setSize);

  //if (recompute) printf("min imp w:%e\n", totMostOff.avgClipImpW);
  stats.indexOfEpisodeToDelete = -1;
  switch (ALGO) {
    case OLDEST:      stats.indexOfEpisodeToDelete = totFirstIn(); break;
    case FARPOLFRAC: stats.indexOfEpisodeToDelete = totMostFar(); break;
    case MAXKLDIV:  stats.indexOfEpisodeToDelete = totHighDkl(); break;
    case BATCHRL:  stats.indexOfEpisodeToDelete = totMostOff(); break;
  }

  if (stats.indexOfEpisodeToDelete >= 0) {
    // prevent any race condition from causing deletion of newest data:
    const Episode & EP2delete = RM.get(stats.indexOfEpisodeToDelete);
    const Episode & EPoldest = RM.get(totFirstIn.ind);
    if (EPoldest.ID + (Sint) setSize < EP2delete.ID)
        stats.indexOfEpisodeToDelete = totFirstIn.ind;
  }
}

void prepareNextBatchAndDeleteStaleEp(MemoryBuffer & RM)
{
  ReplayStats & stats = RM.stats;
  // Here we:
  // 1) Remove episodes from the RM if needed.
  // 2) Update minibatch sampling distributions, must be done right after
  //    removal of data from RM. This is reason why we bundle these 3 steps.

  // Safety measure: we don't use as delete condition "if Nobs > maxTotObsNum",
  // We use "if Nobs - toDeleteEpisode.ndata() > maxTotObsNum".
  // This avoids bugs if any single sequence is longer than maxTotObsNum.
  // Has negligible effect if hyperparam maxTotObsNum is chosen appropriately.
  if(stats.indexOfEpisodeToDelete >= 0)
  {
    const long maxTotObs = RM.settings.maxTotObsNum_local; // for MPI-learners
    const long nDataToDelete = RM.get(stats.indexOfEpisodeToDelete).ndata();
    if(RM.nStoredSteps() - nDataToDelete > maxTotObs)
    {
      //warn("Deleting episode");
      RM.removeEpisode(stats.indexOfEpisodeToDelete);
      RM.stats.nPrunedEps ++;
    }
    RM.stats.indexOfEpisodeToDelete = -1;
  }

  RM.sampler->prepare(RM.needs_pass); // update sampling algorithm
}

void histogramImportanceWeights(const MemoryBuffer & RM)
{
  static constexpr Uint nBins = 81;
  const Real beg = std::log(1e-3), end = std::log(50.0);
  Fval bounds[nBins+1] = { 0 };
  Uint counts[nBins]   = { 0 };
  for (Uint i = 1; i < nBins; ++i)
      bounds[i] = std::exp(beg + (end-beg) * (i-1.0)/(nBins-2.0) );
  bounds[nBins] = std::numeric_limits<Fval>::max()-1e2; // -100 avoids inf later

  const Uint setSize = RM.nStoredEps();
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : counts[:nBins])
  for (Uint i = 0; i < setSize; ++i) {
    const auto & EP = RM.get(i);
    for (Uint j=0; j < EP.ndata(); ++j) {
      const auto rho = EP.offPolicImpW[j];
      for (Uint b = 0; b < nBins; ++b)
        if(rho >= bounds[b] && rho < bounds[b+1]) counts[b] ++;
    }
  }
  const auto harmonicMean = [](const Fval a, const Fval b) {
    return 2 * a * (b / (a + b));
  };
  std::ostringstream buff;
  buff<<"_____________________________________________________________________";
  buff<<"\nOFF-POLICY IMP WEIGHTS HISTOGRAMS\n";
  buff<<"weight pi/mu (harmonic mean of histogram's bounds):\n";
  for (Uint b = 0; b < nBins; ++b)
    Utilities::real2SS(buff, harmonicMean(bounds[b], bounds[b+1]), 6, 1);
  buff<<"\nfraction of dataset:\n";
  const Real dataSize = RM.nStoredSteps();
  for (Uint b = 0; b < nBins; ++b)
    Utilities::real2SS(buff, counts[b]/dataSize, 6, 1);
  buff<<"\n";
  buff<<"_____________________________________________________________________";
  printf("%s\n\n", buff.str().c_str());
}

inline Fval computeRetrace(const Episode& EP, const Uint t,
                    const Fval gamma, const Fval lambda)
{
  assert(t+1 < EP.actionAdvantage.size() and t+1 < EP.stateValue.size());
  assert(t+1 < EP.returnEstimator.size() and t+1 < EP.offPolicImpW.size());
  // From Schulman et al. 2016, https://arxiv.org/abs/1606.02647
  const Fval R = EP.scaledReward<Fval>(t+1), Q = EP.returnEstimator[t+1];
  const Fval V = EP.stateValue[t+1], A = EP.actionAdvantage[t+1];
  return R + gamma * (V + lambda * EP.clippedOffPolW<Fval>(t+1) * (Q - A - V));
}

inline Fval computeRetraceExplBonus(const Episode& EP, const Uint t,
                const Fval B, const Fval C, const Fval G, const Fval L)
{
  const Fval V = EP.stateValue[t+1], A = EP.actionAdvantage[t+1];
  const Fval E = std::fabs(EP.returnEstimator[t+1] - A - V) - B;
  return C * E + computeRetrace(EP, t, G, L);
}

inline Fval computeGAE(const Episode& EP, const Uint t,
                const Fval gamma, const Fval lambda)
{
  // From Munos et al. 2016, https://arxiv.org/pdf/1506.02438.pdf
  const Fval R = EP.scaledReward<Fval>(t+1), V = EP.stateValue[t+1];
  return R + gamma * (V + lambda * (EP.returnEstimator[t+1] - V));
}

returnsEstimator_f createReturnEstimator(const MemoryBuffer & RM)
{
  const Fval gamma = RM.settings.gamma, lambda = RM.settings.lambda;

  std::function<Fval(const Episode& EP, const Uint t)> ret;
  if(RM.settings.returnsEstimator == "retrace") {
    ret = [=](const Episode& EP, const Uint t) {
      return computeRetrace(EP, t, gamma, lambda);
    };
  }
  else
  if(RM.settings.returnsEstimator == "retraceExplore") {
    const Fval coef = (1-gamma);
    //const Fval baseline = RM.stats.avgAbsError;
    static constexpr Real EPS = std::numeric_limits<float>::epsilon();
    const Fval baseline = std::sqrt(std::max(EPS, RM.stats.avgSquaredErr));
    ret = [=](const Episode& EP, const Uint t) {
      return computeRetraceExplBonus(EP, t, baseline, coef, gamma, lambda);
    };
  }
  else
  if(RM.settings.returnsEstimator == "GAE") {
    ret = [=](const Episode& EP, const Uint t) {
      return computeGAE(EP, t, gamma, lambda);
    };
  }
  else {
    ret = [=](const Episode& EP, const Uint t) {
      return computeRetrace(EP, t, gamma, lambda);
    };
  }
  return ret;
}

void computeReturnEstimator(const MemoryBuffer & RM, Episode & EP)
{
  const Uint epLen = EP.nsteps();
  if (RM.settings.returnsEstimator == "none") return;
  const returnsEstimator_f returnsCompute = createReturnEstimator(RM);
  updateReturnEstimator(EP, epLen-2, returnsCompute);
}

void rescaleAllReturnEstimator(MemoryBuffer & RM)
{
  if (RM.settings.returnsEstimator == "none") return;
  const returnsEstimator_f returnsCompute = createReturnEstimator(RM);

  #pragma omp parallel
  {
    const Uint setSize = RM.nStoredEps();
    #pragma omp for schedule(dynamic, 1) nowait
    for(Uint i=0; i<setSize; ++i) {
      Episode& EP = RM.get(i);
      updateReturnEstimator(EP, EP.nsteps()-2, returnsCompute);
    }
    const Uint todoSize = RM.nInProgress();
    #pragma omp for schedule(dynamic, 1) nowait
    for(Uint i=0; i<todoSize; ++i) {
      Episode& EP = RM.getInProgress(i);
      if(EP.returnEstimator.size() == 0) continue;
      updateReturnEstimator(EP, EP.nsteps()-2, returnsCompute);
    }
  }
}

}
}
