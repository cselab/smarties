//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryBuffer.h"
#include "Sampling.h"
#include "../Utils/FunctionUtilities.h"
#include <iterator>
#include <algorithm>
#include <unistd.h>

namespace smarties
{

MemoryBuffer::MemoryBuffer(MDPdescriptor&M_, Settings&S_, DistributionInfo&D_):
  MDP(M_), settings(S_), distrib(D_),
  sampler( MemoryBuffer::prepareSampler(this, S_, D_) )
{
  Set.reserve(settings.maxTotObsNum);
}

void MemoryBuffer::save(const std::string base, const Uint nStep, const bool bBackup)
{
  const auto write2file = [&] (FILE * wFile) {
    std::vector<double> V = std::vector<double>(mean.begin(), mean.end());
    fwrite(V.data(), sizeof(double), V.size(), wFile);
    V = std::vector<double>(invstd.begin(), invstd.end());
    fwrite(V.data(), sizeof(double), V.size(), wFile);
    V = std::vector<double>(std.begin(), std.end());
    fwrite(V.data(), sizeof(double), V.size(), wFile);
    V.resize(2); V[0] = stddev_reward; V[1] = invstd_reward;
    fwrite(V.data(), sizeof(double), 2, wFile);
  };

  FILE * wFile = fopen((base+"scaling.raw").c_str(), "wb");
  write2file(wFile); fflush(wFile); fclose(wFile);

  if(bBackup) {
    char fName[256]; sprintf(fName, "%sscaling_%09lu.raw", base.c_str(), nStep);
    wFile = fopen(fName, "wb"); write2file(wFile); fflush(wFile); fclose(wFile);
  }
}

void MemoryBuffer::restart(const std::string base)
{
  char currDirectory[512];
  getcwd(currDirectory, 512);
  chdir(distrib.initial_runDir);

  {
    FILE * wFile = fopen((base+"scaling.raw").c_str(), "rb");
    if(wFile == NULL) {
      printf("Parameters restart file %s not found.\n", (base+".raw").c_str());
      return;
    } else {
      printf("Restarting from file %s.\n", (base+"scaling.raw").c_str());
      fflush(0);
    }

    const Uint dimS = MDP.dimStateObserved; assert(mean.size() == dimS);
    std::vector<double> V(dimS);
    size_t size1 = fread(V.data(), sizeof(double), dimS, wFile);
    mean   = std::vector<nnReal>(V.begin(), V.end());
    size_t size2 = fread(V.data(), sizeof(double), dimS, wFile);
    invstd = std::vector<nnReal>(V.begin(), V.end());
    size_t size3 = fread(V.data(), sizeof(double), dimS, wFile);
    std    = std::vector<nnReal>(V.begin(), V.end());
    V.resize(2);
    size_t size4 = fread(V.data(), sizeof(double),    2, wFile);
    stddev_reward = V[0]; invstd_reward = V[1];
    fclose(wFile);
    if (size1!=dimS || size2!=dimS || size3!=dimS || size4!=2)
      _die("Mismatch in restarted file %s.", (base+"_scaling.raw").c_str());
  }
  chdir(currDirectory);
}

void MemoryBuffer::clearAll()
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  //delete already-used trajectories
  for(auto& S: Set) Utilities::dispose_object(S);

  Set.clear(); //clear trajectories used for learning
  nTransitions = 0;
  nSequences = 0;
  needs_pass = true;
}

Uint MemoryBuffer::clearOffPol(const Real C, const Real tol)
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  Uint i = 0;
  while(1) {
    if(i>=Set.size()) break;
    Uint _nOffPol = 0;
    const auto& EP = * Set[i];
    const Uint N = EP.ndata();
    for(Uint j=0; j<N; ++j)
      _nOffPol += EP.offPolicImpW[j] > 1+C || EP.offPolicImpW[j] < 1-C;
    if(_nOffPol > tol*N) {
      std::swap(Set[i], Set.back());
      nSequences   --;
      nTransitions -= N;
      Utilities::dispose_object(Set.back());
      Set.pop_back();
      assert(nSequences == (long) Set.size());
    }
    else ++i;
  }
  needs_pass = true;
  return readNData();
}

MiniBatch MemoryBuffer::sampleMinibatch(const Uint batchSize,
                                        const Uint stepID)
{
  assert(sampler);
  std::vector<Uint> sampleEID(batchSize), sampleT(batchSize);
  sampler->sample(sampleEID, sampleT);
  assert( batchSize == sampleEID.size() && batchSize == sampleT.size() );
  {
    // remember which episodes were just sampled:
    lastSampledEps = sampleEID;
    std::sort(lastSampledEps.begin(), lastSampledEps.end());
    lastSampledEps.erase( std::unique(lastSampledEps.begin(), lastSampledEps.end()), lastSampledEps.end() );
  }

  MiniBatch ret(batchSize);
  for(Uint b=0; b<batchSize; ++b)
  {
    ret.episodes[b] = Set[ sampleEID[b] ];
    ret.episodes[b]->setSampled(sampleT[b]);
    const Uint nEpSteps = ret.episodes[b]->nsteps();
    if (settings.bSampleSequences)
    {
      // check that we may have to update estimators from S_{0} to S_{T_1}
      assert( sampleT[b] == ret.episodes[b]->ndata() - 1 );
      ret.begTimeStep[b] = 0;        // prepare to compute for steps from init
      ret.endTimeStep[b] = nEpSteps; // to terminal state
      ret.sampledTimeStep[b] = 0;
    }
    else
    {
      // if t=0 always zero recurrent steps, t=1 one, and so on, up to nMaxBPTT
      const Uint nnBPTT = settings.nnBPTTseq;
      const bool bRecurrent = settings.bRecurrent || MDP.isPartiallyObservable;
      const Uint nRecur = bRecurrent? std::min(nnBPTT, sampleT[b]) : 0;
      // prepare to compute from step t-reccurrentwindow up to t+1
      // because some methods may require tnext.
      // todo: add option for n-steps ahead
      ret.begTimeStep[b] = sampleT[b] - nRecur;
      ret.endTimeStep[b] = sampleT[b] + 2;
      ret.sampledTimeStep[b] = sampleT[b];
    }
    // number of states to process ( also, see why we used sampleT[b]+2 )
    const Uint nSteps = ret.endTimeStep[b] - ret.begTimeStep[b];
    ret.resizeStep(b, nSteps);
  }
  const std::vector<Sequence*>& sampleE = ret.episodes;
  const nnReal impSampAnneal = std::min( (Real)1, stepID*settings.epsAnneal);
  const nnReal beta = 0.5 + 0.5 * impSampAnneal;
  const bool bReqImpSamp = bRequireImportanceSampling();
  #pragma omp parallel for schedule(static) // collapse(2)
  for(Uint b=0; b<batchSize; ++b)
  for(Uint t=ret.begTimeStep[b]; t<ret.endTimeStep[b]; ++t)
  {
    ret.state(b, t)  = standardizedState<nnReal>(sampleE[b], t);
    ret.set_action(b, t, sampleE[b]->actions[t] );
    ret.set_mu(b, t, sampleE[b]->policies[t] );
    ret.reward(b, t) = scaledReward(sampleE[b], t);
    if( bReqImpSamp ) {
      const nnReal impW_undef = sampleE[b]->priorityImpW[t];
      // if imp weight is 0 or less assume it was not computed and therefore
      // ep is probably a new experience that should be given high priority
      const nnReal impW_unnorm = impW_undef<=0 ? maxPriorityImpW : impW_undef;
      ret.importanceWeight(b, t) = std::pow(minPriorityImpW/impW_unnorm, beta);
    } else ret.importanceWeight(b, t) = 1;
  }

  return ret;
}

bool MemoryBuffer::bRequireImportanceSampling() const
{
  assert(sampler);
  return sampler->requireImportanceWeights();
}

MiniBatch MemoryBuffer::agentToMinibatch(Sequence* const inProgress) const
{
  MiniBatch ret(1);
  ret.episodes[0] = inProgress;
  if (settings.bSampleSequences) {
    // we may have to update estimators from S_{0} to S_{T_1}
    ret.begTimeStep[0] = 0;        // prepare to compute for steps from init
    ret.endTimeStep[0] = inProgress->nsteps(); // to current step
  } else {
    const Uint currStep = inProgress->nsteps() - 1;
    // if t=0 always zero recurrent steps, t=1 one, and so on, up to nMaxBPTT
    const Uint nnBPTT = settings.nnBPTTseq;
    const bool bRecurrent = settings.bRecurrent || MDP.isPartiallyObservable;
    const Uint nRecurr = bRecurrent? std::min(nnBPTT, currStep) : 0;
    // prepare to compute from step t-reccurrentwindow up to t
    ret.begTimeStep[0] = currStep - nRecurr;
    ret.endTimeStep[0] = currStep + 1;
  }
  ret.sampledTimeStep[0] = inProgress->nsteps() - 1;
  // number of states to process ( also, see why we used sampleT[b]+2 )
  const Uint nSteps = ret.endTimeStep[0] - ret.begTimeStep[0];
  ret.resizeStep(0, nSteps);
  for(Uint t=ret.begTimeStep[0]; t<ret.endTimeStep[0]; ++t)
  {
    ret.state(0, t) = standardizedState<nnReal>(inProgress, t);
    ret.set_action(0, t, inProgress->actions[t] );
    ret.set_mu(0, t, inProgress->policies[t] );
    ret.reward(0, t) = scaledReward(inProgress, t);
  }
  return ret;
}

void MemoryBuffer::removeSequence(const Uint ind)
{
  assert(readNSeq()>0);
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert(nTransitions >= (long) Set[ind]->ndata());
  assert(Set[ind] not_eq nullptr);
  nSequences--;
  needs_pass = true;
  nTransitions -= Set[ind]->ndata();
  std::swap(Set[ind], Set.back());
  Utilities::dispose_object(Set.back());
  Set.pop_back();
  assert(nSequences == (long) Set.size());
}

void MemoryBuffer::pushBackSequence(Sequence*const seq)
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert( readNSeq() == (long) Set.size() and seq not_eq nullptr);
  const auto ind = Set.size();
  seq->ID = nSeenSequences.load();
  seq->prefix = ind>0? Set[ind-1]->prefix +Set[ind-1]->ndata() : 0;
  Set.push_back(seq);
  nSequences++;
  nTransitions += seq->ndata();
  needs_pass = true;
  assert( readNSeq() == (long) Set.size());
}

void MemoryBuffer::initialize()
{
  { // All seqs obtained before this point should share the same time stamp
    std::lock_guard<std::mutex> lock(dataset_mutex);
    for(Uint i=0;i<Set.size();++i) Set[i]->ID = nSeenSequences.load();
  } // free mutex for sampler
  needs_pass = true;
  sampler->prepare(needs_pass);
}

MemoryBuffer::~MemoryBuffer()
{
  for(auto & S : Set) Utilities::dispose_object(S);
}

void MemoryBuffer::checkNData()
{
  #ifndef NDEBUG
    long cntSamp = 0;
    for(Uint i=0; i<Set.size(); ++i) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions);
    assert(nSequences==(long)Set.size());
  #endif
}

std::unique_ptr<Sampling> MemoryBuffer::prepareSampler(MemoryBuffer* const R,
                                                       Settings & S_,
                                                       DistributionInfo & D_)
{
  std::unique_ptr<Sampling> ret = nullptr;

  if(S_.dataSamplingAlgo == "uniform") ret = std::make_unique<Sample_uniform>(
    D_.generators, R, S_.bSampleSequences);

  if(S_.dataSamplingAlgo == "impLen")  ret = std::make_unique<Sample_impLen>(
    D_.generators, R, S_.bSampleSequences);

  if(S_.dataSamplingAlgo == "shuffle") {
    ret = std::make_unique<TSample_shuffle>(
      D_.generators, R, S_.bSampleSequences);
    if(S_.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S_.dataSamplingAlgo == "PERrank") {
    ret = std::make_unique<TSample_impRank>(
      D_.generators, R, S_.bSampleSequences);
    if(S_.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S_.dataSamplingAlgo == "PERerr") {
    ret = std::make_unique<TSample_impErr>(
      D_.generators, R, S_.bSampleSequences);
    if(S_.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S_.dataSamplingAlgo == "PERseq") ret = std::make_unique<Sample_impSeq>(
    D_.generators, R, S_.bSampleSequences);

  assert(ret not_eq nullptr);
  return ret;
}

}
