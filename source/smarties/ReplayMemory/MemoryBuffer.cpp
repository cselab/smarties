//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryBuffer.h"

#include "DataCoordinator.h"
#include "MemoryProcessing.h"
#include "../Utils/FunctionUtilities.h"
#include "../Utils/SstreamUtilities.h"

#include <iterator>
#include <algorithm>
#include <unistd.h>

namespace smarties
{

MemoryBuffer::MemoryBuffer(MDPdescriptor& M, HyperParameters& S, ExecutionInfo& D) :
  MDP(M), settings(S), distrib(D), sharing( new DataCoordinator(this, params) ),
  StateRewRdx(distrib, LDvec(MDP.dimStateObserved * 2 + 3, 0) ),
  globalCounterRdx(distrib, std::vector<long>{0, 0, 0, 0}),
  sampler( Sampling::prepareSampler(this, S, D) )
{
  episodes.reserve(settings.maxTotObsNum);
  inProgress.reserve(distrib.nAgents);
  for (Uint i=0; i<distrib.nAgents; ++i) inProgress.push_back(MDP);

  LDvec initGuessStateRewStats(MDP.dimStateObserved * 2 + 3, 0);
  for(Uint i=0; i<MDP.dimStateObserved; ++i)
    initGuessStateRewStats[i + MDP.dimStateObserved] = 0;
  initGuessStateRewStats[MDP.dimStateObserved*2] = 1;
  initGuessStateRewStats[MDP.dimStateObserved*2 + 2] = 1;
  StateRewRdx.update(initGuessStateRewStats);

  globalCounterRdx.update({(long)0,(long)0,(long)0,(long)settings.maxTotObsNum});

  if(settings.returnsEstimator not_eq "none")
    printf("Returns estimation method: %s.\n", S.returnsEstimator.c_str());
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
void MemoryBuffer::storeState(Agent&a)
{
  assert(a.ID < inProgress.size());
  //assert(replay->MDP.localID == a.localID);
  Episode & S = inProgress[a.ID];
  const Fvec storedState = a.getObservedState<Fval>();

  if(a.trackEpisodes == false) {
    // contain only one state and do not add more. to not store rewards either
    // RNNs then become automatically not supported because no time series!
    // (this is accompained by check in approximator)
    S.states  = std::vector<Fvec>{ storedState };
    S.rewards = std::vector<Real>{ (Real) 0 };
    S.clearNonTrackedAgent();
    a.agentStatus = INIT; // one state stored, lie to avoid catching asserts
    return;
  }

  // assign or check id of agent generating episode
  if (a.agentStatus == INIT) S.agentID = a.localID;
  else assert(S.agentID == (Sint) a.localID);

  // if no tuples, init state. if tuples, cannot be initial state:
  assert( (S.nsteps() == 0) == (a.agentStatus == INIT) );
  #ifndef NDEBUG // check that last new state and new old state are the same
    if( S.nsteps() ) {
      bool same = true;
      const Fvec vecSold = a.getObservedOldState<Fval>();
      const Fvec memSold = S.states.back();
      static constexpr Fval fEPS = std::numeric_limits<Fval>::epsilon();
      for (Uint i=0; i<vecSold.size() && same; ++i) {
        auto D = std::max({std::fabs(memSold[i]), std::fabs(vecSold[i]), fEPS});
        same = same && std::fabs(memSold[i]-vecSold[i])/D < 100*fEPS;
      }
      //debugS("Agent %s and %s",print(vecSold).c_str(),print(memSold).c_str());
      if (!same) _die("Unexpected termination of EP a %u step %u seqT %lu\n",
        a.ID, a.timeStepInEpisode, S.nsteps());
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  //env->pickReward(a);
  S.latent_states.push_back( a.getLatentState<Fval>() );
  S.bReachedTermState = a.agentStatus == TERM;
  S.states.push_back(storedState);
  S.rewards.push_back(a.reward);
  if( a.agentStatus not_eq INIT ) S.totR += a.reward;
  else assert(std::fabs(a.reward)<2.2e-16); //rew for init state must be 0
}

// Once network picked next action, call this method
void MemoryBuffer::storeAction(const Agent& a)
{
  assert(a.agentStatus < LAST);
  if(a.trackEpisodes == false) {
    // do not store more stuff in sequence but also do not track data counter
    inProgress[a.ID].actions = std::vector<Rvec>{ a.action };
    inProgress[a.ID].policies = std::vector<Rvec>{ a.policyVector };
    return;
  }

  if(a.agentStatus not_eq INIT) increaseLocalSeenSteps();
  inProgress[a.ID].actions.push_back( a.action );
  inProgress[a.ID].policies.push_back( a.policyVector );
}

// If the state is terminal, instead of calling `add_action`, call this:
void MemoryBuffer::terminateCurrentEpisode(Agent&a)
{
  //either do not store episode, or algorithm already did this (e.g. CMA):
  if(a.trackEpisodes == false or inProgress[a.ID].nsteps() == 0) return;
  assert(a.agentStatus >= LAST);
  // fill empty action and empty policy: last step of episode never has actions
  const Rvec dummyAct = Rvec(aI.dim(), 0), dummyPol = Rvec(aI.dimPol(), 0);
  a.resetActionNoise();
  a.setAction(dummyAct, dummyPol);
  inProgress[a.ID].actions.push_back( dummyAct );
  inProgress[a.ID].policies.push_back( dummyPol );

  addEpisodeToTrainingSet(a);
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
void MemoryBuffer::addEpisodeToTrainingSet(const Agent& a)
{
  assert(a.ID < inProgress.size());
  if(inProgress[a.ID].nsteps() < 2) die("Seq must at least have s0 and sT");

  const long tStamp = std::max(nLocTimeStepsTrain(), (long)0);
  inProgress[a.ID].finalize(tStamp);
  MemoryProcessing::computeReturnEstimator(* this, inProgress[a.ID]);

  if(sharing->bRunParameterServer)
  {
    // Check whether this agent is last agent of environment to call terminate.
    // This is only relevant in the case of learners on workers who receive
    // params from master, therefore bool can be incorrect in all other cases.
    // It only matters if multiple agents in env belong to same learner.
    // To ensure thread safety, we must use mutex and check that this agent is
    // last to reset the sequence.
    assert(distrib.bIsMaster == false);
    bool fullEnvReset = true;
    std::unique_lock<std::mutex> lock(envTerminationCheck);
    for(Uint i=0; i<inProgress.size(); ++i){
      //printf("%lu ", inProgress[i].nsteps());
      if (i == a.ID or inProgress[i].agentID < 0) continue;
      fullEnvReset = fullEnvReset && inProgress[i].nsteps() == 0;
    }
    //printf("\n");
    Episode EP = std::move(inProgress[a.ID]);
    assert(inProgress[a.ID].nsteps() == 0);
    //Unlock with empy inProgress, such that next agent can check if it is last.
    lock.unlock();
    sharing->addComplete(EP, fullEnvReset);
  }
  else
  {
    sharing->addComplete(inProgress[a.ID], true);
    assert(inProgress[a.ID].nsteps() == 0);
  }

  increaseLocalSeenSteps();
  increaseLocalSeenEps();
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

    const Uint dimS = MDP.dimStateObserved; assert(MDP.stateMean.size()==dimS);
    std::vector<double> V(dimS);
    size_t size1 = fread(V.data(), sizeof(double), dimS, wFile);
    MDP.stateMean = std::vector<nnReal>(V.begin(), V.end());
    size_t size2 = fread(V.data(), sizeof(double), dimS, wFile);
    MDP.stateScale = std::vector<nnReal>(V.begin(), V.end());
    size_t size3 = fread(V.data(), sizeof(double), dimS, wFile);
    MDP.stateStdDev = std::vector<nnReal>(V.begin(), V.end());
    V.resize(3);
    size_t size4 = fread(V.data(), sizeof(double),    3, wFile);
    MDP.rewardsStdDev = V[0];
    MDP.rewardsScale = V[1];
    MDP.rewardsMean = V[2];
    fclose(wFile);
    if (size1 != dimS || size2 != dimS || size3 != dimS || size4 != 3)
      _die("Mismatch in restarted file %s.", (base+"_scaling.raw").c_str());
  }

  if(distrib.bTrain == false) {
    printf("Evaluating the policy: will skip restarting the Replay Buffer from file.\n");
    chdir(currDirectory);
    return;
  }

  const Uint learn_rank = MPICommRank(distrib.learners_train_comm);
  snprintf(fName, 512, "%s_rank_%03u_learner_status.raw",
      base.c_str(), (unsigned) learn_rank);
  FILE * const fstat = fopen(fName, "r");
  snprintf(fName, 512, "%s_rank_%03u_learner_data.raw",
      base.c_str(), (unsigned) learn_rank);
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
    unsigned long nStoredEpisodes = 0, nStoredObservations = 0;
    unsigned long nLocalSeenEps = 0, nLocalSeenObs = 0;
    long nInitialData = 0, doneGradSteps = 0;
    Uint pass = 1;
    pass = pass && 1==fscanf(fstat, "nStoredEps: %lu\n", &nStoredEpisodes);
    pass = pass && 1==fscanf(fstat, "nStoredObs: %lu\n", &nStoredObservations);
    pass = pass && 1==fscanf(fstat, "nLocalSeenEps: %lu\n", &nLocalSeenEps);
    pass = pass && 1==fscanf(fstat, "nLocalSeenObs: %lu\n", &nLocalSeenObs);
    pass = pass && 1==fscanf(fstat, "nInitialData: %ld\n", &nInitialData);
    pass = pass && 1==fscanf(fstat, "nGradSteps: %ld\n", &doneGradSteps);
    pass = pass && 1==fscanf(fstat, "CmaxReFER: %le\n", &CmaxRet);
    pass = pass && 1==fscanf(fstat, "beta: %le\n", &beta);
    assert(doneGradSteps >= 0 && pass == 1);
    fclose(fstat);
    counters.nSeenTransitions_loc = nLocalSeenObs;
    counters.nSeenEpisodes_loc = nLocalSeenEps;
    counters.nTransitions = nStoredObservations;
    counters.nEpisodes = nStoredEpisodes;
    counters.nGradSteps = doneGradSteps;
    counters.nGatheredB4Startup = nInitialData;
  }

  {
    episodes.reserve(counters.nEpisodes);
    for(long i = 0; i < counters.nEpisodes; ++i) {
      episodes.push_back(MDP);
      if( episodes[i].restart(fdata) )
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
      std::vector<double> V(MDP.stateMean.begin(), MDP.stateMean.end());
      fwrite(V.data(), sizeof(double), V.size(), wFile);
      V = std::vector<double>(MDP.stateScale.begin(), MDP.stateScale.end());
      fwrite(V.data(), sizeof(double), V.size(), wFile);
      V = std::vector<double>(MDP.stateStdDev.begin(), MDP.stateStdDev.end());
      fwrite(V.data(), sizeof(double), V.size(), wFile);
      V.resize(3);
      V[0] = MDP.rewardsStdDev; V[1] = MDP.rewardsScale; V[2] = MDP.rewardsMean;
      fwrite(V.data(), sizeof(double), 3, wFile);
    };

    const std::string backname = base + "_scaling_backup.raw";
    FILE * wFile = fopen((backname).c_str(), "wb");
    write2file(wFile); fflush(wFile); fclose(wFile);
    Utilities::copyFile(backname, base + "_scaling.raw");
  }

  const Uint rank = MPICommRank(distrib.learners_train_comm);
  std::string fName = base + "_rank_" +Utilities::num2str(rank,3)+ "_learner_";

  const unsigned long nStoredEpisodes = counters.nEpisodes;
  const unsigned long nStoredObservations = counters.nTransitions;
  const unsigned long nLocalSeenEps = counters.nSeenEpisodes_loc;
  const unsigned long nLocalSeenObs = counters.nSeenTransitions_loc;
  const long nGatheredB4Startup = counters.nGatheredB4Startup;
  const long doneGradSteps = counters.nGradSteps;

  FILE * const fstat = fopen((fName + "status_backup.raw").c_str(), "w");
  assert(fstat != NULL);
  fprintf(fstat, "nStoredEps: %lu\n",    nStoredEpisodes);
  fprintf(fstat, "nStoredObs: %lu\n",    nStoredObservations);
  fprintf(fstat, "nLocalSeenEps: %lu\n", nLocalSeenEps);
  fprintf(fstat, "nLocalSeenObs: %lu\n", nLocalSeenObs);
  fprintf(fstat, "nInitialData: %ld\n",  nGatheredB4Startup);
  fprintf(fstat, "nGradSteps: %ld\n",    doneGradSteps);
  fprintf(fstat, "CmaxReFER: %le\n",     CmaxRet);
  fprintf(fstat, "beta: %le\n",          beta);
  fflush(fstat); fclose(fstat);

  FILE * const fdata = fopen((fName + "data_backup.raw").c_str(), "wb");
  assert(fdata != NULL);
  for(Uint i = 0; i <nStoredEpisodes; ++i) episodes[i].save(fdata);
  fflush(fdata); fclose(fdata);

  Utilities::copyFile(fName + "status_backup.raw", fName + "status.raw");
  Utilities::copyFile(fName + "data_backup.raw", fName + "data.raw");
}

void MemoryBuffer::clearAll()
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  //delete already-used trajectories
  episodes.clear(); //clear trajectories used for learning
  counters.nTransitions = 0;
  counters.nEpisodes = 0;
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
      counters.nEpisodes --;
      counters.nTransitions -= N;
      episodes.pop_back();
      assert(nStoredEps() == (long) episodes.size());
    }
    else ++i;
  }
  needs_pass = true;
  return nStoredSteps();
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

  #pragma omp parallel for schedule(static)
  for(Uint b=0; b<batchSize; ++b)
  {
    ret.episodes[b] = & episodes[ sampleEID[b] ];
    ret.episodes[b]->setSampled(sampleT[b]);
    const Uint nEpSteps = ret.episodes[b]->nsteps();
    if (settings.bSampleEpisodes)
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
  const std::vector<Episode*>& sampleE = ret.episodes;
  const nnReal impSampAnneal = std::min( (Real)1, stepID*settings.epsAnneal);
  const nnReal annealExp = 0.5 + 0.5 * impSampAnneal; //a.k.a. beta in PER paper
  const bool bReqImpSamp = bRequireImportanceSampling();

  #pragma omp parallel for schedule(static) // collapse(2)
  for(Uint b=0; b<batchSize; ++b)
  {
    const Episode& EP = * sampleE[b];
    for(Sint t=ret.begTimeStep[b]; t<ret.endTimeStep[b]; ++t)
    {
      ret.state(b, t)  = EP.standardizedState<nnReal>(t);
      ret.reward(b, t) = EP.scaledReward(t);
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

MiniBatch MemoryBuffer::agentToMinibatch(const Uint ID)
{
  MiniBatch ret(1);
  ret.episodes[0] = & inProgress[ID];
  if (settings.bSampleEpisodes) {
    // we may have to update estimators from S_{0} to S_{T_1}
    ret.begTimeStep[0] = 0;        // prepare to compute for steps from init
    ret.endTimeStep[0] = inProgress[ID].nsteps(); // to current step
  } else {
    const Uint currStep = inProgress[ID].nsteps() - 1;
    // if t=0 always zero recurrent steps, t=1 one, and so on, up to nMaxBPTT
    const bool bRecurr = settings.bRecurrent || MDP.isPartiallyObservable;
    const Uint nRecurr = bRecurr? std::min(settings.nnBPTTseq, currStep) : 0;
    // prepare to compute from step t-reccurrentwindow up to t
    ret.begTimeStep[0] = currStep - nRecurr;
    ret.endTimeStep[0] = currStep + 1;
  }
  ret.sampledTimeStep[0] = inProgress[ID].nsteps() - 1;
  // number of states to process ( also, see why we used sampleT[b]+2 )
  const Uint nSteps = ret.endTimeStep[0] - ret.begTimeStep[0];
  ret.resizeStep(0, nSteps);
  for(Sint t=ret.begTimeStep[0]; t<ret.endTimeStep[0]; ++t)
  {
    ret.state(0, t) = inProgress[ID].standardizedState<nnReal>(t);
    ret.reward(0, t) = inProgress[ID].scaledReward(t);
  }
  return ret;
}

void MemoryBuffer::removeEpisode(const Uint ind)
{
  assert(counters.nEpisodes > 0);
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert(counters.nTransitions >= (long) episodes[ind].ndata());
  counters.nEpisodes --;
  needs_pass = true;
  counters.nTransitions -= episodes[ind].ndata();
  std::swap(episodes[ind], episodes.back());
  episodes.pop_back();
  assert(counters.nEpisodes == (long) episodes.size());
}

void MemoryBuffer::pushBackEpisode(Episode & seq)
{
  const bool logSample =  distrib.logAllSamples==1 ||
                         (distrib.logAllSamples==2 && seq.agentID==0);
  char pathRew[2048], pathObs[2048], rewArg[1024];
  const long nGrads = nGradSteps();
  const long tStamp = std::max(nLocTimeStepsTrain(), (long)0);
  if(logSample) {
    const int wrank = MPICommRank(distrib.world_comm);
    snprintf(rewArg, 1024, "%ld %ld %d %u %f", nGrads, tStamp,
              (int) seq.agentID, (unsigned) seq.nsteps(), seq.totR);
    snprintf(pathRew, 2048, "%s/agent_%02u_rank_%03d_cumulative_rewards.dat",
              distrib.initial_runDir, (unsigned) learnID, wrank);
    snprintf(pathObs, 2048, "%s/agent_%02u_rank_%03d_obs.raw",
              distrib.initial_runDir, (unsigned) learnID, wrank);
  }

  const auto log = logSample? seq.logToFile(tStamp) : std::vector<float>(0);

  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert(counters.nEpisodes == (long) episodes.size());

  if(logSample) {
    FILE * pFile = fopen (pathRew, "a");
    fprintf (pFile, "%s\n", rewArg); fflush (pFile); fclose (pFile);
    pFile = fopen (pathObs, "ab");
    fwrite (log.data(), sizeof(float), log.size(), pFile);
    fflush(pFile); fclose(pFile);
  }

  const size_t ind = episodes.size(), len = seq.ndata();
  seq.ID = tStamp;
  seq.prefix = ind>0? episodes[ind-1].prefix + episodes[ind-1].ndata() : 0;
  episodes.emplace_back(std::move(seq));
  counters.nEpisodes ++;
  counters.nTransitions += len;
  needs_pass = true;
  assert(counters.nEpisodes == (long) episodes.size());
}

void MemoryBuffer::getMetrics(std::ostringstream& buff)
{
  Utilities::real2SS(buff, stats.avgReturn, 9, 0);
  Utilities::real2SS(buff, MDP.rewardsMean, 6, 0);
  Utilities::real2SS(buff, MDP.rewardsStdDev, 6, 1);
  Utilities::real2SS(buff, stats.avgKLdivergence, 5, 1);

  if( stats.minQ < stats.maxQ ) // else Q stats not collected
  {
    static constexpr Real EPS = std::numeric_limits<float>::epsilon();
    stats.avgSquaredErr = std::max(EPS,stats.avgSquaredErr);
    Utilities::real2SS(buff, std::sqrt(stats.avgSquaredErr), 6, 1);
    //Utilities::real2SS(buff, stats.avgAbsError, 6, 1);
    if(stats.countReturnsEstimateUpdates > 0) {
      const Sint nRet = std::max((Sint) 1, stats.countReturnsEstimateUpdates);
      const Real eRet = std::max(EPS, stats.sumReturnsEstimateErrors);
      Utilities::real2SS(buff, std::sqrt(eRet/nRet), 6, 1);
      stats.countReturnsEstimateUpdates = 0;
      stats.sumReturnsEstimateErrors = 0;
    } else {
      stats.countReturnsEstimateUpdates = -1;
      stats.sumReturnsEstimateErrors = 0;
    }
    Utilities::real2SS(buff, stats.stdevQ, 6, 1);
    Utilities::real2SS(buff, stats.avgQ, 6, 0);
    Utilities::real2SS(buff, stats.minQ, 6, 0);
    Utilities::real2SS(buff, stats.maxQ, 6, 0);
  }

  buff<<" "<<std::setw(5)<<nStoredEps();
  buff<<" "<<std::setw(7)<<nStoredSteps();
  buff<<" "<<std::setw(7)<<nSeenEps();
  buff<<" "<<std::setw(8)<<nSeenSteps();
  buff<<" "<<std::setw(7)<<stats.oldestStoredTimeStamp;
  //buff<<" "<<std::setw(4)<<stats.nPrunedEps;
  buff<<" "<<std::setw(6)<<stats.nFarPolicySteps;
  if(CmaxRet>1) {
    //Utilities::real2SS(buf, alpha, 6, 1);
    Utilities::real2SS(buff, beta, 6, 1);
  }
  stats.nPrunedEps = 0;
}

void MemoryBuffer::getHeaders(std::ostringstream& buff)
{
  buff << "|  avgR  | avgr | stdr | DKL ";
  if( stats.minQ < stats.maxQ ) { // else Q stats not collected
    if(stats.countReturnsEstimateUpdates>=0)
        buff << "| RMSE | dRet | stdQ | avgQ | minQ | maxQ ";
    else         buff << "| RMSE | stdQ | avgQ | minQ | maxQ ";
  }
  //buff << "| nEp |  nObs | totEp | totObs | oldEp |nDel|nFarP ";
  buff << "| nEp |  nObs | totEp | totObs | oldEp |nFarP ";
  if(CmaxRet>1) buff << "| beta ";
}

void MemoryBuffer::updateSampler(const bool bForce)
{
  if(bForce) needs_pass = true;
  sampler->prepare(needs_pass);
}

void MemoryBuffer::setupDataCollectionTasks(TaskQueue& tasks)
{
  sharing->setupTasks(tasks);
}

MemoryBuffer::~MemoryBuffer()
{
  delete sharing;
}

void MemoryBuffer::checkNData()
{
  #ifndef NDEBUG
    long cntSamp = 0;
    for(Uint i=0; i<episodes.size(); ++i) {
      cntSamp += episodes[i].ndata();
    }
    assert(counters.nTransitions == cntSamp);
    assert(counters.nEpisodes == (long) episodes.size());
  #endif
}

}
