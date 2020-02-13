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
static constexpr Fval EPS = std::numeric_limits<Fval>::epsilon();

MemoryProcessing::MemoryProcessing(MemoryBuffer*const _RM) : RM(_RM),
  Ssum1Rdx(distrib, LDvec(_RM->MDP.dimStateObserved, 0) ),
  Ssum2Rdx(distrib, LDvec(_RM->MDP.dimStateObserved, 1) ),
  Rsum2Rdx(distrib, LDvec(1, 1) ), Csum1Rdx(distrib, LDvec(1, 1) ),
  globalStep_reduce(distrib, std::vector<long>{0, 0}),
  ReFER_reduce(distrib, LDvec{0.,1.})
{
    globalStep_reduce.update( { nSeenSequences_loc.load(),
                                nSeenTransitions_loc.load() } );

    ReFER_reduce.update({(long double)0, (long double) settings.minTotObsNum });
}

void MemoryProcessing::updateRewardsStats(const Real WR, const Real WS, const bool bInit)
{
  // Update the second order moment of the rewards and means and stdev of states
  // contained in the memory buffer. Used for rescaling and numerical safety.

  //////////////////////////////////////////////////////////////////////////////
  //_warn("globalStep_reduce %ld %ld", nSeenSequences_loc.load(), nSeenTransitions_loc.load());
  globalStep_reduce.update( { nSeenSequences_loc.load(),
                              nSeenTransitions_loc.load() } );
  const std::vector<long> nDataGlobal = globalStep_reduce.get(bInit);
  //_warn("nDataGlobal %ld %ld", nDataGlobal[0], nDataGlobal[1]);
  nSeenSequences = nDataGlobal[0];
  nSeenTransitions = nDataGlobal[1];
  //////////////////////////////////////////////////////////////////////////////

  if(not distrib.bTrain) return; //if not training, keep the stored values
  const Uint setSize = RM->readNSeq(), dimS = MDP.dimStateObserved;

  if(WR>0 or WS>0)
  {
    long double count = 0, newstdvr = 0;
    std::vector<long double> newSSum(dimS, 0), newSSqSum(dimS, 0);
    #pragma omp parallel reduction(+ : count, newstdvr)
    {
      std::vector<long double> thNewSSum(dimS, 0), thNewSSqSum(dimS, 0);
      #pragma omp for schedule(dynamic, 1) nowait
      for(Uint i=0; i<setSize; ++i) {
        const Sequence & EP = episodes[i];
        const Uint N = EP.ndata();
        count += N;
        for(Uint j=0; j<N; ++j) {
          newstdvr += std::pow(EP.rewards[j+1], 2);
          for(Uint k=0; k<dimS && WS>0; ++k) {
            const long double sk = EP.states[j][k] - mean[k];
            thNewSSum[k] += sk; thNewSSqSum[k] += sk*sk;
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
    Ssum1Rdx.update(newSSum);
    Ssum2Rdx.update(newSSqSum);
    Csum1Rdx.update( LDvec {count});
    Rsum2Rdx.update( LDvec {newstdvr});
  }

  const long double count = Csum1Rdx.get<0>(bInit);

  if(WR>0)
  {
   long double varR = Rsum2Rdx.get<0>(bInit)/count;
   if(varR < 0) varR = 0;
   //if( settings.ESpopSize > 1e7 ) {
   //  const Real gamma = settings.gamma;
   //  const auto Rscal = (std::sqrt(varR)+EPS) * (1-gamma>EPS? 1/(1-gamma) :1);
   //  invstd_reward = (1-WR)*invstd_reward +WR/Rscal;
   //} else
   invstd_reward = (1-WR)*invstd_reward + WR / ( std::sqrt(varR) + EPS );
  }

  if(WS>0)
  {
    const LDvec SSum1 = Ssum1Rdx.get(bInit);
    const LDvec SSum2 = Ssum2Rdx.get(bInit);
    for(Uint k=0; k<dimS; ++k)
    {
      // this is the sample mean minus mean[k]:
      const long double MmM = SSum1[k]/count;
      // mean[k] = (1-WS)*mean[k] + WS * sample_mean, which becomes:
      mean[k] = mean[k] + WS * MmM;
      // if WS==1 then varS is exact, otherwise update second moment
      // centered around current mean[k] (ie. E[(Sk-mean[k])^2])
      long double varS = SSum2[k]/count - MmM*MmM*(2*WS-WS*WS);
      if(varS < 0) varS = 0;
      std[k] = (1-WS) * std[k] + WS * std::sqrt(varS);
      invstd[k] = 1/(std[k]+EPS);
    }
  }
}

void MemoryProcessing::updateReFERpenalization()
{
  // use result from prev AllReduce to update rewards (before new reduce).
  // Assumption is that the number of off Pol trajectories does not change
  // much each step. Especially because here we update the off pol W only
  // if an obs is actually sampled. Therefore at most this fraction
  // is wrong by batchSize / nTransitions ( ~ 0 )
  // In exchange we skip an mpi implicit barrier point.
  const auto dataSetSize = nTransitions.load();
  ReFER_reduce.update({(long double)nFarPolicySteps, (long double)dataSetSize});
  const LDvec nFarGlobal = ReFER_reduce.get();
  assert(nFarGlobal[1] + 1 > dataSetSize);
  const Real fracOffPol = nFarGlobal[0] / nFarGlobal[1];

  // In the ReF-ER paper we learn ReF-ER penalization coefficient beta with the
  // network's learning rate eta (was 1e-4). In reality, beta should not depend
  // on eta. beta should reflect estimate of far-policy samples. Accuracy of
  // this estimate depends on: batch size B (bigger B increases accuracy
  // because samples importance weight rho are updated more often) and data set
  // size N (bigger N decreases accuracy because there are more samples to
  // update). We pick coef 0.1 to match learning rate chosen in original paper:
  // we had B=256 and N=2^18 and eta=1e-4. 0.1*B*N \approx 1e-4
  const Real learnRefer = 0.1 * settings.batchSize / nFarGlobal[1];
  const auto fixPointIter = [&] (const Real val, const bool goTo0) {
    if (goTo0) // fixed point iter converging to 0:
      return (1 - std::min(learnRefer, val)) * val;
    else       // fixed point iter converging to 1:
      return (1 - std::min(learnRefer, val)) * val + std::min(learnRefer, 1-val);
  };

  // if too much far policy data, increase weight of Dkl penalty
  beta = fixPointIter(beta, fracOffPol > settings.penalTol);

  // USED ONLY FOR CMA: how do we weight cirit cost and policy cost?
  // if we satisfy too strictly far-pol constrain, reduce weight of policy
  alpha = fixPointIter(alpha,std::fabs(settings.penalTol - fracOffPol) < 0.001);
}

void MemoryProcessing::selectEpisodeToDelete(const FORGET ALGO)
{
  const bool bRecomputeProperties = ( (nGradSteps + 1) % 100) == 0;
  //shift data / gradient counters to maintain grad stepping to sample
  // collection ratio prescirbed by obsPerStep
  Real C = settings.clipImpWeight, E = settings.epsAnneal;

  if(ALGO == BATCHRL) {
    const Real maxObsNum = settings.maxTotObsNum_local;
    const Real factorUp = std::max((Real) 1, nTransitions.load() / maxObsNum);
    //const Real factorDw = std::min((Real)1, maxObsNum / obsNum);
    //D *= factorUp;
    //CmaxRet = 1 + C * factorDw
    CmaxRet = 1 + Utilities::annealRate(C, nGradSteps +1, E) * factorUp;
  } else {
    CmaxRet = 1 + Utilities::annealRate(C, nGradSteps +1, E);
  }

  CinvRet = 1 / CmaxRet;
  if(CmaxRet <= 1 and C > 0) die("Unallowed ReF-ER annealing values.");
  assert(CmaxRet>=1);

  MostOffPolicyEp totMostOff; OldestDatasetEp totFirstIn;
  MostFarPolicyEp totMostFar; HighestAvgDklEp totHighDkl;

  Real _avgR = 0;
  Real _totDKL = 0;
  Uint _nOffPol = 0;
  const Uint setSize = RM->readNSeq();
  #pragma omp parallel reduction(+ : _avgR, _totDKL, _nOffPol)
  {
    OldestDatasetEp locFirstIn; MostOffPolicyEp locMostOff;
    MostFarPolicyEp locMostFar; HighestAvgDklEp locHighDkl;

    #pragma omp for schedule(static, 1) nowait
    for (Uint i = 0; i < setSize; ++i)
    {
      Sequence & EP = episodes[i];
      if (bRecomputeProperties) EP.updateCumulative(CmaxRet, CinvRet);
      _nOffPol += EP.nFarPolicySteps();
      _totDKL  += EP.sumKLDivergence;
      _avgR    += EP.totR;
      locFirstIn.compare(EP, i); locMostOff.compare(EP, i);
      locMostFar.compare(EP, i); locHighDkl.compare(EP, i);
    }

    #pragma omp critical
    {
      totFirstIn.compare(locFirstIn); totMostOff.compare(locMostOff);
      totMostFar.compare(locMostFar); totHighDkl.compare(locHighDkl);
    }
  }

  if (CmaxRet<=1) nFarPolicySteps = 0; //then this counter and its effects are skipped
  avgKLdivergence = _totDKL / RM->readNData();
  nFarPolicySteps = _nOffPol;
  RM->avgCumulativeReward = _avgR / setSize;
  oldestStoresTimeStamp = totFirstIn.timestamp;

  assert(totMostFar.ind >= 0 && totMostFar.ind < (int) setSize);
  assert(totHighDkl.ind >= 0 && totHighDkl.ind < (int) setSize);
  assert(totFirstIn.ind >= 0 && totFirstIn.ind < (int) setSize);

  indexOfEpisodeToDelete = -1;
  //if (recompute) printf("min imp w:%e\n", totMostOff.avgClipImpW);
  switch (ALGO)
  {
    case OLDEST:      indexOfEpisodeToDelete = totFirstIn(); break;

    case FARPOLFRAC: indexOfEpisodeToDelete = totMostFar(); break;

    case MAXKLDIV: indexOfEpisodeToDelete = totHighDkl(); break;

    case BATCHRL: indexOfEpisodeToDelete = totMostOff(settings.penalTol); break;
  }

  if (indexOfEpisodeToDelete >= 0) {
    // prevent any race condition from causing deletion of newest data:
    const Sequence & EP2delete = episodes[indexOfEpisodeToDelete];
    if (episodes[totFirstIn.ind].ID + (Sint) setSize < EP2delete.ID)
        indexOfEpisodeToDelete = totFirstIn.ind;
  }
}

void MemoryProcessing::prepareNextBatchAndDeleteStaleEp()
{
  // Here we:
  // 1) Reset flags that signal request to update estimators. This step could
  //    be eliminated and bundled in with next minibatch sampling step.
  // 2) Remove episodes from the RM if needed.
  // 3) Update minibatch sampling distributions, must be done right after
  //    removal of data from RM. This is reason why we bundle these 3 steps.

  const std::vector<Uint>& sampled = RM->lastSampledEpisodes();
  const Uint sampledSize = sampled.size();
  for(Uint i = 0; i < sampledSize; ++i) {
    Sequence & S = RM->get(sampled[i]);
    assert(S.just_sampled >= 0);
    S.just_sampled = -1;
  }
  const long setSize = RM->readNSeq();
  for(long i=0; i<setSize; ++i) assert(RM->get(i).just_sampled < 0);

  // Safety measure: we don't use as delete condition "if Nobs > maxTotObsNum",
  // We use "if Nobs - toDeleteEpisode.ndata() > maxTotObsNum".
  // This avoids bugs if any single sequence is longer than maxTotObsNum.
  // Has negligible effect if hyperparam maxTotObsNum is chosen appropriately.
  if(indexOfEpisodeToDelete >= 0)
  {
    const Uint maxTotObs = settings.maxTotObsNum_local; // for MPI-learners
    if(RM->readNData() - episodes[indexOfEpisodeToDelete].ndata() > maxTotObs)
    {
      //warn("Deleting episode");
      RM->removeSequence(indexOfEpisodeToDelete);
      nPrunedEps ++;
    }
    indexOfEpisodeToDelete = -1;
  }

  RM->sampler->prepare(RM->needs_pass); // update sampling algorithm
}

void MemoryProcessing::getMetrics(std::ostringstream& buff)
{
  Utilities::real2SS(buff, RM->avgCumulativeReward, 9, 0);
  Utilities::real2SS(buff, 1/invstd_reward, 6, 1);
  Utilities::real2SS(buff, avgKLdivergence, 5, 1);

  buff<<" "<<std::setw(5)<<nSequences.load();
  buff<<" "<<std::setw(7)<<nTransitions.load();
  buff<<" "<<std::setw(7)<<nSeenSequences.load();
  buff<<" "<<std::setw(8)<<nSeenTransitions.load();
  //buff<<" "<<std::setw(7)<<nSeenSequences_loc.load();
  //buff<<" "<<std::setw(8)<<nSeenTransitions_loc.load();
  buff<<" "<<std::setw(7)<<oldestStoresTimeStamp;
  buff<<" "<<std::setw(4)<<nPrunedEps;
  buff<<" "<<std::setw(6)<<nFarPolicySteps;
  if(CmaxRet>1) {
    //Utilities::real2SS(buf, alpha, 6, 1);
    Utilities::real2SS(buff, beta, 6, 1);
  }
  nPrunedEps = 0;
}

void MemoryProcessing::getHeaders(std::ostringstream& buff)
{
  buff <<
  //"|  avgR  | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nFarP ";
  "|  avgR  | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nDel|nFarP ";
  if(CmaxRet>1) {
    //buf << "| alph | beta ";
    buff << "| beta ";
  }
}

void MemoryProcessing::histogramImportanceWeights()
{
  static constexpr Uint nBins = 81;
  const Real beg = std::log(1e-3), end = std::log(50.0);
  Fval bounds[nBins+1] = { 0 };
  Uint counts[nBins]   = { 0 };
  for (Uint i = 1; i < nBins; ++i)
      bounds[i] = std::exp(beg + (end-beg) * (i-1.0)/(nBins-2.0) );
  bounds[nBins] = std::numeric_limits<Fval>::max()-1e2; // -100 avoids inf later

  const Uint setSize = RM->readNSeq();
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : counts[:nBins])
  for (Uint i = 0; i < setSize; ++i) {
    const auto & EP = episodes[i];
    for (Uint j=0; j < EP.ndata(); ++j) {
      const auto rho = EP.offPolicImpW[j];
      for (Uint b = 0; b < nBins; ++b)
        if(rho >= bounds[b] && rho < bounds[b+1]) counts[b] ++;
    }
  }
  const auto harmoncMean = [](const Fval a, const Fval b) {
    return 2 * a * (b / (a + b));
  };
  std::ostringstream buff;
  buff<<"_____________________________________________________________________";
  buff<<"\nOFF-POLICY IMP WEIGHTS HISTOGRAMS\n";
  buff<<"weight pi/mu (harmonic mean of histogram's bounds):\n";
  for (Uint b = 0; b < nBins; ++b)
    Utilities::real2SS(buff, harmoncMean(bounds[b], bounds[b+1]), 6, 1);
  buff<<"\nfraction of dataset:\n";
  const Real dataSize = RM->readNData();
  for (Uint b = 0; b < nBins; ++b)
    Utilities::real2SS(buff, counts[b]/dataSize, 6, 1);
  buff<<"\n";
  buff<<"_____________________________________________________________________";
  printf("%s\n\n", buff.str().c_str());
}

}
