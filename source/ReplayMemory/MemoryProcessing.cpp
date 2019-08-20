//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryProcessing.h"
#include "../Utils/SstreamUtilities.h"
#include "Sampling.h"
#include <algorithm>

namespace smarties
{

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
        const Uint N = Set[i]->ndata();
        count += N;
        for(Uint j=0; j<N; ++j) {
          newstdvr += std::pow(Set[i]->rewards[j+1], 2);
          for(Uint k=0; k<dimS && WS>0; ++k) {
            const long double sk = Set[i]->states[j][k] - mean[k];
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

  static constexpr Real EPS = std::numeric_limits<float>::epsilon();
  const long double count = Csum1Rdx.get<0>(bInit);

  if(WR>0)
  {
   long double varR = Rsum2Rdx.get<0>(bInit)/count;
   if(varR < std::numeric_limits<long double>::epsilon()) varR = 1;
   if( settings.ESpopSize > 1e7 ) {
     const Real gamma = settings.gamma;
     const auto Rscal = (std::sqrt(varR)+EPS) * (1-gamma>EPS ? 1/(1-gamma) : 1);
     invstd_reward = (1-WR)*invstd_reward +WR/Rscal;
   } else invstd_reward = (1-WR)*invstd_reward + WR / ( std::sqrt(varR) + EPS );
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
      if(varS < std::numeric_limits<long double>::epsilon()) varS = 1;
      std[k] = (1-WS) * std[k] + WS * std::sqrt(varS);
      invstd[k] = 1/(std[k]+EPS);
    }
  }
}

void MemoryProcessing::prune(const FORGET ALGO, const Fval CmaxRho)
{
  //checkNData();
  assert(CmaxRho>=1);
  // vector indicating location of sequence to delete
  int  oldP = -1, farP = -1, dklP = -1;
  Real dklV = -1, farV = -1, oldV = 9e9;
  Real _nOffPol = 0, _totDKL = 0;
  const Uint setSize = RM->readNSeq();
  #pragma omp parallel reduction(+ : _nOffPol, _totDKL)
  {
    std::pair<int, Real> farpol{-1, -1}, maxdkl{-1, -1}, oldest{-1, 9e9};
    #pragma omp for schedule(static, 1) nowait
    for(Uint i = 0; i < setSize; ++i)
    {
      #ifndef NDEBUG
        const Fval invC = 1/CmaxRho;
        Fval dbg_nOffPol = 0, dbg_sumKLDiv = 0, dbg_sum_mse = 0;
        for(Uint j=0; j<Set[i]->ndata(); ++j) {
          const Fval W = Set[i]->offPolicImpW[j];
          dbg_sum_mse += Set[i]->SquaredError[j];
          dbg_sumKLDiv += Set[i]->KullbLeibDiv[j];
          assert( W>=0  &&  Set[i]->KullbLeibDiv[j]>=0 );
          // sequence is off policy if offPol W is out of 1/C : C
          if(W>CmaxRho || W<invC) dbg_nOffPol += 1;
        }
        const auto badErr = [&](const Fval V, const Fval R) {
            static const Fval EPS = std::numeric_limits<Fval>::epsilon();
            const Fval den = std::max({std::fabs(R), std::fabs(V), EPS});
            return std::fabs(V - R) / den > 0.1;
          };
        if( badErr(dbg_sumKLDiv, Set[i]->sumKLDiv) )
          _die("DKL %f %f", dbg_sumKLDiv, Set[i]->sumKLDiv);
        if( badErr(dbg_sum_mse, Set[i]->MSE) )
          _die("MSE %f %f", dbg_sum_mse, Set[i]->MSE);
        if(settings.epsAnneal <= 0) //else CmaxRho will change in time
          if( badErr(dbg_nOffPol, Set[i]->nOffPol) )
            _die("OFF %f %f", dbg_nOffPol, Set[i]->nOffPol);
      #endif

      const Real W_FAR = Set[i]->nOffPol /Set[i]->ndata();
      const Real W_DKL = Set[i]->sumKLDiv/Set[i]->ndata();
      _nOffPol += Set[i]->nOffPol; _totDKL += Set[i]->sumKLDiv;

      if(Set[i]->ID<oldest.second) { oldest.second=Set[i]->ID; oldest.first=i; }
      if(    W_FAR >farpol.second) { farpol.second= W_FAR;     farpol.first=i; }
      if(    W_DKL >maxdkl.second) { maxdkl.second= W_DKL;     maxdkl.first=i; }
    }
    #pragma omp critical
    {
      if(oldest.second<oldV) { oldP=oldest.first; oldV=oldest.second; }
      if(farpol.second>farV) { farP=farpol.first; farV=farpol.second; }
      if(maxdkl.second>dklV) { dklP=maxdkl.first; dklV=maxdkl.second; }
    }
  }

  if(CmaxRho<=1) _nOffPol = 0; //then this counter and its effects are skipped
  avgDKL = _totDKL / RM->readNData();
  nOffPol = _nOffPol;
  minInd = oldV;
  assert(oldP<(int)Set.size() && farP<(int)Set.size() && dklP<(int)Set.size());
  assert( oldP >=  0 && farP >=  0 && dklP >=  0 );
  switch(ALGO) {
    case OLDEST:     delPtr = oldP; break;
    case FARPOLFRAC: delPtr = farP; break;
    case MAXKLDIV:   delPtr = dklP; break;
  }
  // prevent any weird race condition from causing deletion of newest data:
  if(Set[oldP]->ID + (int)setSize < Set[delPtr]->ID) delPtr = oldP;
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

  // safety measure: do not delete trajectory if Nobs > Ntarget
  // but if N > Ntarget even if we remove the trajectory
  // done to avoid bugs if a sequence is longer than maxTotObsNum
  // negligible effect if hyperparameters are chosen wisely
  if(delPtr>=0)
  {
    const Uint maxTotObsNum_loc = settings.maxTotObsNum_local;
    if(nTransitions.load()-Set[delPtr]->ndata() > maxTotObsNum_loc)
      RM->removeSequence(delPtr);
    delPtr = -1;
  }
  nPruned += nB4 - RM->readNSeq();

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
  Utilities::real2SS(buff, avgDKL, 5, 1);

  buff<<" "<<std::setw(5)<<nSeq;
  buff<<" "<<std::setw(7)<<nTransitions.load();
  buff<<" "<<std::setw(7)<<nSeenSequences.load();
  buff<<" "<<std::setw(8)<<nSeenTransitions.load();
  //buff<<" "<<std::setw(7)<<nSeenSequences_loc.load();
  //buff<<" "<<std::setw(8)<<nSeenTransitions_loc.load();
  buff<<" "<<std::setw(7)<<minInd;
  buff<<" "<<std::setw(6)<<(int)nOffPol;

  nPruned=0;
}

void MemoryProcessing::getHeaders(std::ostringstream& buff)
{
  buff <<
  "|  avgR  | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nFarP ";
}

FORGET MemoryProcessing::readERfilterAlgo(const std::string setting,
  const bool bReFER)
{
  if(setting == "oldest")     return OLDEST;
  if(setting == "farpolfrac") return FARPOLFRAC;
  if(setting == "maxkldiv")   return MAXKLDIV;
  //if(setting == "minerror")   return MINERROR; miriad ways this can go wrong
  if(setting == "default") {
    if(bReFER) return FARPOLFRAC;
    else       return OLDEST;
  }
  die("ERoldSeqFilter not recognized");
  return OLDEST; // to silence warning
}

}


#if 0 // ndef NDEBUG
  if( settings.learner_rank == 0 ) {
   std::ofstream outf("runningAverages.dat", std::ios::app);
   outf<<count<<" "<<1/invstd_reward<<" "<<print(mean)<<" "<<print(std)<<std::endl;
   outf.flush(); outf.close();
  }
  Uint cntSamp = 0;
  for(Uint i=0; i<setSize; ++i) {
    assert(Set[i] not_eq nullptr);
    cntSamp += Set[i]->ndata();
  }
  assert(cntSamp==nTransitions.load());
  if(WS>=1)
  {
    LDvec dbgStateSum(dimS,0), dbgStateSqSum(dimS,0);
    #pragma omp parallel
    {
      LDvec thr_dbgStateSum(dimS), thr_dbgStateSqSum(dimS);
      #pragma omp for schedule(dynamic)
      for(Uint i=0; i<setSize; ++i)
        for(Uint j=0; j<Set[i]->ndata(); ++j) {
          const auto S = RM->standardize(Set[i]->states[j]);
          for(Uint k=0; k<dimS; ++k) {
            thr_dbgStateSum[k] += S[k]; thr_dbgStateSqSum[k] += S[k]*S[k];
          }
        }
      #pragma omp critical
      for(Uint k=0; k<dimS; ++k) {
        dbgStateSum[k]   += thr_dbgStateSum[k];
        dbgStateSqSum[k] += thr_dbgStateSqSum[k];
      }
    }
    for(Uint k=0; k<dimS && settings.learner_rank == 0; ++k) {
      const Real dbgMean = dbgStateSum[k]/cntSamp;
      const Real dbgVar = dbgStateSqSum[k]/cntSamp - dbgMean*dbgMean;
      if(std::fabs(dbgMean)>.001 || std::fabs(dbgVar-1)>.001)
        std::cout <<k<<" mean:"<<dbgMean<<" std:"<<dbgVar<<"\n";
    }
  }
#endif
