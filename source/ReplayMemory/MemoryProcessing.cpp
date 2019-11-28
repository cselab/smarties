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
  globalStep_reduce(distrib, std::vector<long>{0, 0})
{
    globalStep_reduce.update( { nSeenSequences_loc.load(),
                                nSeenTransitions_loc.load() } );
}

// update the second order moment of the rewards in the memory buffer
void MemoryProcessing::updateRewardsStats(const Real WR, const Real WS, const bool bInit)
{
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
      #pragma omp for schedule(dynamic) nowait
      for(Uint i=0; i<setSize; ++i) {
        const Sequence & EP = * Set[i];
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

void MemoryProcessing::prune(const FORGET ALGO, const Fval CmaxRho,
                             const bool recompute)
{
  assert(CmaxRho>=1);
  const Fval invC = 1/CmaxRho, farPolTol = settings.penalTol;

  MostOffPolicyEp totMostOff; OldestDatasetEp totFirstIn;
  MostFarPolicyEp totMostFar; HighestAvgDklEp totHighDkl;

  Real _totDKL = 0;
  Uint _nOffPol = 0;
  const Uint setSize = RM->readNSeq();
  #pragma omp parallel reduction(+ : _nOffPol, _totDKL)
  {
    OldestDatasetEp locFirstIn; MostOffPolicyEp locMostOff;
    MostFarPolicyEp locMostFar; HighestAvgDklEp locHighDkl;

    #pragma omp for schedule(static, 1) nowait
    for (Uint i = 0; i < setSize; ++i)
    {
      Sequence & EP = * Set[i];
      if (recompute) EP.updateCumulative(CmaxRho, invC);
      _nOffPol += EP.nFarPolicySteps();
      _totDKL  += EP.sumKLDivergence;
      locFirstIn.compare(EP, i); locMostOff.compare(EP, i);
      locMostFar.compare(EP, i); locHighDkl.compare(EP, i);
    }

    #pragma omp critical
    {
      totFirstIn.compare(locFirstIn); totMostOff.compare(locMostOff);
      totMostFar.compare(locMostFar); totHighDkl.compare(locHighDkl);
    }
  }

  if (CmaxRho<=1) nFarPolicySteps = 0; //then this counter and its effects are skipped
  avgKLdivergence = _totDKL / RM->readNData();
  nFarPolicySteps = _nOffPol;
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

    case BATCHRL: indexOfEpisodeToDelete = totMostOff(farPolTol); break;
  }

  if (indexOfEpisodeToDelete >= 0) {
    // prevent any race condition from causing deletion of newest data:
    const Sequence & EP2delete = * Set[indexOfEpisodeToDelete];
    if (Set[totFirstIn.ind]->ID + (Sint) setSize < EP2delete.ID)
        indexOfEpisodeToDelete = totFirstIn.ind;
  }
}

void MemoryProcessing::finalize()
{
  //std::lock_guard<std::mutex> lock(RM->dataset_mutex);
  const int nB4 = RM->readNSeq();

  // reset flags that signal request to update estimators:
  const std::vector<Uint>& sampled = RM->lastSampledEpisodes();
  const Uint sampledSize = sampled.size();
  for(Uint i = 0; i < sampledSize; ++i) {
    Sequence * const S = RM->get(sampled[i]);
    assert(S->just_sampled >= 0);
    S->just_sampled = -1;
  }
  for(int i=0; i<nB4; ++i) assert(RM->get(i)->just_sampled < 0);

  // Safety measure: we don't use as delete condition "if Nobs > maxTotObsNum",
  // We use "if Nobs - toDeleteEpisode.ndata() > maxTotObsNum".
  // This avoids bugs if any single sequence is longer than maxTotObsNum.
  // Has negligible effect if hyperparam maxTotObsNum is chosen appropriately.
  if(indexOfEpisodeToDelete >= 0)
  {
    const Uint maxTotObsNum = settings.maxTotObsNum_local; // for MPI-learners
    if(RM->readNData() - Set[indexOfEpisodeToDelete]->ndata() > maxTotObsNum) {
      //warn("Deleting episode");
      RM->removeSequence(indexOfEpisodeToDelete);
    }
    indexOfEpisodeToDelete = -1;
  }
  nPrunedEps += nB4 - RM->readNSeq();

  // update sampling algorithm:
  RM->sampler->prepare(RM->needs_pass);
}

void MemoryProcessing::getMetrics(std::ostringstream& buff)
{
  Real avgR = 0;
  const long nSeq = nSequences.load();
  for(long i=0; i<nSeq; ++i) avgR += Set[i]->totR;

  Utilities::real2SS(buff, avgR/(nSeq+1e-7), 9, 0);
  Utilities::real2SS(buff, 1/invstd_reward, 6, 1);
  Utilities::real2SS(buff, avgKLdivergence, 5, 1);

  buff<<" "<<std::setw(5)<<nSeq;
  buff<<" "<<std::setw(7)<<nTransitions.load();
  buff<<" "<<std::setw(7)<<nSeenSequences.load();
  buff<<" "<<std::setw(8)<<nSeenTransitions.load();
  //buff<<" "<<std::setw(7)<<nSeenSequences_loc.load();
  //buff<<" "<<std::setw(8)<<nSeenTransitions_loc.load();
  buff<<" "<<std::setw(7)<<oldestStoresTimeStamp;
  buff<<" "<<std::setw(4)<<nPrunedEps;
  buff<<" "<<std::setw(6)<<nFarPolicySteps;

  nPrunedEps = 0;
}

void MemoryProcessing::getHeaders(std::ostringstream& buff)
{
  buff <<
  //"|  avgR  | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nFarP ";
  "|  avgR  | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nDel|nFarP ";
}

FORGET MemoryProcessing::readERfilterAlgo(const std::string setting,
  const bool bReFER)
{
  if(setting == "oldest") {
    printf("Experience Replay storage: First In First Out.\n");
    return OLDEST;
  }
  if(setting == "farpolfrac") {
    printf("Experience Replay storage: remove most 'far policy' episode.\n");
    return FARPOLFRAC;
  }
  if(setting == "maxkldiv") {
    printf("Experience Replay storage: remove highest average DKL episode.\n");
    return MAXKLDIV;
  }
  if(setting == "batchrl") {
    printf("Experience Replay storage: remove most 'off policy' episode if and only if policy is better.\n");
    return BATCHRL;
  }
  //if(setting == "minerror")   return MINERROR; miriad ways this can go wrong
  if(setting == "default") {
    if(bReFER) {
      printf("Experience Replay storage: remove most 'off policy' episode if and only if policy is better.\n");
      return BATCHRL;
    }
    else {
      printf("Experience Replay storage: First In First Out.\n");
      return OLDEST;
    }
  }
  die("ERoldSeqFilter not recognized");
  return OLDEST; // to silence warning
}

void MemoryProcessing::histogramImportanceWeights()
{
  static constexpr Uint nBins = 81;
  static constexpr Real beg = std::log(1e-3), end = std::log(50.0);
  Fval bounds[nBins+1] = { 0 };
  Uint counts[nBins]   = { 0 };
  for (Uint i = 1; i < nBins; ++i)
      bounds[i] = std::exp(beg + (end-beg) * (i-1.0)/(nBins-2.0) );
  bounds[nBins] = std::numeric_limits<Fval>::max()-1e2; // -100 avoids inf later

  const Uint setSize = RM->readNSeq();
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : counts[:nBins])
  for (Uint i = 0; i < setSize; ++i) {
    const auto & EP = * Set[i];
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
