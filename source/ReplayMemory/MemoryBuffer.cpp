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
#include "../Utils/SstreamUtilities.h"
#include <iterator>
#include <algorithm>
#include <unistd.h>

namespace smarties
{

MemoryBuffer::MemoryBuffer(MDPdescriptor& M, Settings& S, DistributionInfo& D) :
  MDP(M), settings(S), distrib(D),
  sampler( Sampling::prepareSampler(this, S, D) )
{
  episodes.reserve(settings.maxTotObsNum);
}

void MemoryBuffer::restart(const std::string base)
{
  char currDirectory[512], fName[512];
  getcwd(currDirectory, 512);
  chdir(distrib.initial_runDir);

  {
    FILE * wFile = fopen((base+"_scaling.raw").c_str(), "rb");
    if(wFile == NULL) {
      printf("Parameters restart file %s not found.\n",
        (base+"_scaling.raw").c_str());
      chdir(currDirectory);
      return;
    } else {
      printf("Restarting from file %s.\n", (base+"_scaling.raw").c_str());
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

  if(distrib.bTrain == false) {
    printf("Evaluating the policy: will skip restarting the Replay Buffer from file.\n");
    return;
  }

  const Uint learn_rank = MPICommRank(distrib.learners_train_comm);
  snprintf(fName, 512, "%s_rank_%03lu_learner_status.raw",
      base.c_str(), learn_rank);
  FILE * const fstat = fopen(fName, "r");
  snprintf(fName, 512, "%s_rank_%03lu_learner_data.raw",
      base.c_str(), learn_rank);
  FILE * const fdata = fopen(fName, "rb");

  if(fstat == NULL || fdata == NULL)
  {
    if(fstat == NULL)
      printf("Learner status restart file %s not found\n", fName);
    else fclose(fstat);

    if(fdata == NULL)
      printf("Learner data restart file %s not found\n", fName);
    else fclose(fdata);

    chdir(currDirectory);
    return;
  }

  {
    Uint nStoredEps = 0, nStoredObs = 0, nLocalSeenEps = 0, nLocalSeenObs = 0;
    long nInitialData = nGatheredB4Startup, doneGradSteps = 0;
    Uint pass = 1;
    pass = pass && 1 == fscanf(fstat, "nStoredEps: %lu\n",    & nStoredEps);
    pass = pass && 1 == fscanf(fstat, "nStoredObs: %lu\n",    & nStoredObs);
    pass = pass && 1 == fscanf(fstat, "nLocalSeenEps: %lu\n", & nLocalSeenEps);
    pass = pass && 1 == fscanf(fstat, "nLocalSeenObs: %lu\n", & nLocalSeenObs);
    pass = pass && 1 == fscanf(fstat, "nInitialData: %ld\n",  & nInitialData);
    pass = pass && 1 == fscanf(fstat, "nGradSteps: %ld\n",    & doneGradSteps);
    pass = pass && 1 == fscanf(fstat, "CmaxReFER: %le\n",     & CmaxRet);
    pass = pass && 1 == fscanf(fstat, "beta: %le\n",          & beta);
    assert(doneGradSteps >= 0 && pass == 1);
    fclose(fstat);
    nSeenTransitions_loc = nLocalSeenObs; nTransitions = nStoredObs;
    nSeenSequences_loc   = nLocalSeenEps; nSequences   = nStoredEps;
    nGatheredB4Startup   = nInitialData;  nGradSteps   = doneGradSteps;
  }

  {
    episodes.resize(nSequences);
    const size_t dimS = sI.dimObs(), dimA = aI.dim(), dimP = aI.dimPol();
    for(Uint i = 0; i < episodes.size(); ++i) {
      if( episodes[i].restart(fdata, dimS, dimA, dimP) )
        _die("Unable to find sequence %u\n", i);
      episodes[i].updateCumulative(CmaxRet, CinvRet);
    }
    fclose(fdata);
  }

  chdir(currDirectory);
}

void MemoryBuffer::save(const std::string base)
{
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

    const std::string backname = base + "_scaling_backup.raw";
    FILE * wFile = fopen((backname).c_str(), "wb");
    write2file(wFile); fflush(wFile); fclose(wFile);
    Utilities::copyFile(backname, base + "_scaling.raw");
  }

  const Uint rank = MPICommRank(distrib.learners_train_comm);
  std::string fName = base + "_rank_" +Utilities::num2str(rank,3)+ "_learner_";
  FILE * const fstat = fopen((fName + "status_backup.raw").c_str(), "w");
  FILE * const fdata = fopen((fName + "data_backup.raw").c_str(), "wb");

  const long doneGradSteps = nGradSteps;
  const Uint nStoredObs = nTransitions, nLocalSeenObs = nSeenTransitions_loc;
  const Uint nStoredEps = nSequences, nLocalSeenEps = nSeenSequences_loc;
  assert(fstat != NULL);
  fprintf(fstat, "nStoredEps: %lu\n",    nStoredEps);
  fprintf(fstat, "nStoredObs: %lu\n",    nStoredObs);
  fprintf(fstat, "nLocalSeenEps: %lu\n", nLocalSeenEps);
  fprintf(fstat, "nLocalSeenObs: %lu\n", nLocalSeenObs);
  fprintf(fstat, "nInitialData: %ld\n",  nGatheredB4Startup);
  fprintf(fstat, "nGradSteps: %ld\n",    doneGradSteps);
  fprintf(fstat, "CmaxReFER: %le\n",     CmaxRet);
  fprintf(fstat, "beta: %le\n",          beta);
  fflush(fstat); fclose(fstat);

  assert(fdata != NULL);
  const size_t dimS = sI.dimObs(), dimA = aI.dim(), dimP = aI.dimPol();
  for(Uint i = 0; i <nStoredEps; ++i) episodes[i].save(fdata, dimS, dimA, dimP);
  fflush(fdata); fclose(fdata);

  Utilities::copyFile(fName + "status_backup.raw", fName + "status.raw");
  Utilities::copyFile(fName + "data_backup.raw", fName + "data.raw");
}

void MemoryBuffer::clearAll()
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  //delete already-used trajectories
  episodes.clear(); //clear trajectories used for learning
  nTransitions = 0;
  nSequences = 0;
  needs_pass = true;
}

Uint MemoryBuffer::clearOffPol(const Real C, const Real tol)
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  Uint i = 0;
  while(1) {
    if(i >= episodes.size()) break;
    Uint _nOffPol = 0;
    const auto& EP = episodes[i];
    const Uint N = EP.ndata();
    for(Uint j=0; j<N; ++j)
      _nOffPol += EP.offPolicImpW[j] > 1+C || EP.offPolicImpW[j] < 1-C;
    if(_nOffPol > tol*N) {
      std::swap(episodes[i], episodes.back());
      nSequences   --;
      nTransitions -= N;
      episodes.pop_back();
      assert(nSequences == (long) episodes.size());
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

  MiniBatch ret(batchSize, settings.gamma);

  #pragma omp parallel for schedule(static)
  for(Uint b=0; b<batchSize; ++b)
  {
    ret.episodes[b] = & episodes[ sampleEID[b] ];
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
  const nnReal annealExp = 0.5 + 0.5 * impSampAnneal; //a.k.a. beta in PER paper
  const bool bReqImpSamp = bRequireImportanceSampling();

  #pragma omp parallel for schedule(static) // collapse(2)
  for(Uint b=0; b<batchSize; ++b)
  {
    const Sequence& EP = * sampleE[b];
    for(Sint t=ret.begTimeStep[b]; t<ret.endTimeStep[b]; ++t)
    {
      ret.state(b, t)  = standardizedState<nnReal>(EP, t);
      ret.reward(b, t) = scaledReward(EP, t);
      if( bReqImpSamp ) {
        const nnReal impW_undef = EP.priorityImpW[t];
        // if imp weight is 0 or less assume it was not computed and therefore
        // ep is probably a new experience that should be given high priority
        const nnReal impWunnorm = impW_undef<=0 ? maxPriorityImpW : impW_undef;
        ret.PERweight(b, t) = std::pow(minPriorityImpW/impWunnorm, annealExp);
      } else ret.PERweight(b, t) = 1;
    }
  }

  return ret;
}

bool MemoryBuffer::bRequireImportanceSampling() const
{
  assert(sampler);
  return sampler->requireImportanceWeights();
}

MiniBatch MemoryBuffer::agentToMinibatch(Sequence & inProgress) const
{
  MiniBatch ret(1, settings.gamma);
  ret.episodes[0] = & inProgress;
  if (settings.bSampleSequences) {
    // we may have to update estimators from S_{0} to S_{T_1}
    ret.begTimeStep[0] = 0;        // prepare to compute for steps from init
    ret.endTimeStep[0] = inProgress.nsteps(); // to current step
  } else {
    const Uint currStep = inProgress.nsteps() - 1;
    // if t=0 always zero recurrent steps, t=1 one, and so on, up to nMaxBPTT
    const bool bRecurr = settings.bRecurrent || MDP.isPartiallyObservable;
    const Uint nRecurr = bRecurr? std::min(settings.nnBPTTseq, currStep) : 0;
    // prepare to compute from step t-reccurrentwindow up to t
    ret.begTimeStep[0] = currStep - nRecurr;
    ret.endTimeStep[0] = currStep + 1;
  }
  ret.sampledTimeStep[0] = inProgress.nsteps() - 1;
  // number of states to process ( also, see why we used sampleT[b]+2 )
  const Uint nSteps = ret.endTimeStep[0] - ret.begTimeStep[0];
  ret.resizeStep(0, nSteps);
  for(Sint t=ret.begTimeStep[0]; t<ret.endTimeStep[0]; ++t)
  {
    ret.state(0, t) = standardizedState<nnReal>(inProgress, t);
    ret.reward(0, t) = scaledReward(inProgress, t);
  }
  return ret;
}

void MemoryBuffer::removeSequence(const Uint ind)
{
  assert(readNSeq()>0);
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert(nTransitions >= (long) episodes[ind].ndata());
  nSequences--;
  needs_pass = true;
  nTransitions -= episodes[ind].ndata();
  std::swap(episodes[ind], episodes.back());
  episodes.pop_back();
  assert(nSequences == (long) episodes.size());
}

void MemoryBuffer::pushBackSequence(Sequence & seq)
{
  const int wrank = MPICommRank(distrib.world_comm);
  const bool logSample =  distrib.logAllSamples==1 ||
                         (distrib.logAllSamples==2 && seq.agentID==0);
  char pathRew[2048], pathObs[2048], rewArg[1024];
  snprintf(pathRew, 2048, "%s/agent_%02lu_rank%02d_cumulative_rewards.dat",
          distrib.initial_runDir, learnID, wrank);
  snprintf(pathObs, 2048, "%s/agent%03lu_rank%02d_obs.raw",
          distrib.initial_runDir, learnID, wrank);
  snprintf(rewArg, 2048, "%ld %ld %ld %lu %f", nGradSteps.load(),
          std::max(nLocTimeStepsTrain(), (long)0),
          seq.agentID, seq.nsteps(), seq.totR);
  const auto log = not logSample ? std::vector<float>(0) :
                   seq.logToFile(sI, aI, nSeenTransitions_loc.load());

  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert( readNSeq() == (long) episodes.size() );

  FILE * pFile = fopen (pathRew, "a");
  fprintf (pFile, "%s\n", rewArg); fflush (pFile); fclose (pFile);
  if(logSample) {
    pFile = fopen (pathObs, "ab");
    fwrite (log.data(), sizeof(float), log.size(), pFile);
    fflush(pFile); fclose(pFile);
  }

  const size_t ind = episodes.size(), len = seq.ndata();
  seq.ID = nSeenSequences.load();
  seq.prefix = ind>0? episodes[ind-1].prefix + episodes[ind-1].ndata() : 0;
  episodes.emplace_back(std::move(seq));
  nSequences++;
  nTransitions += len;
  needs_pass = true;
  assert( readNSeq() == (long) episodes.size());
}

void MemoryBuffer::initialize()
{
  { // All seqs obtained before this point should share the same time stamp
    std::lock_guard<std::mutex> lock(dataset_mutex);
    for(Uint i=0;i<episodes.size(); ++i) episodes[i].ID = nSeenSequences.load();
  } // free mutex for sampler
  needs_pass = true;
  sampler->prepare(needs_pass);
}

MemoryBuffer::~MemoryBuffer()
{}

void MemoryBuffer::checkNData()
{
  #ifndef NDEBUG
    long cntSamp = 0;
    for(Uint i=0; i<episodes.size(); ++i) {
      cntSamp += episodes[i].ndata();
    }
    assert(cntSamp==nTransitions);
    assert(nSequences==(long)episodes.size());
  #endif
}

}
