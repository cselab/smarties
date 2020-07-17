//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner.h"

#include "../Utils/FunctionUtilities.h"
#include "../Utils/SstreamUtilities.h"
#include "../ReplayMemory/MemoryProcessing.h"

#include <unistd.h>

namespace smarties
{

Learner::Learner(MDPdescriptor& MD, HyperParameters& S, ExecutionInfo& D):
  distrib(D), settings(S), MDP(MD),
  ERFILTER(MemoryProcessing::readERfilterAlgo(S)),
  data(std::make_unique<MemoryBuffer>(MDP, settings, distrib)) {}

Learner::~Learner()
{
}

void Learner::select(Agent& agent)
{
  data->storeState(agent);
  const MiniBatch MB = data->agentToMinibatch(agent.ID);

  if( agent.agentStatus < LAST ) // not end of episode
  {
    selectAction(MB, agent);
    data->storeAction(agent);
  }
  else // either terminal or truncation state
  {
    processTerminal(MB, agent);
    data->terminateCurrentEpisode(agent);
  }
}

void Learner::initializeLearner()
{
  const Uint currStep = nGradSteps();

  if ( currStep > 0 ) {
    printf("Skipping initialization for restartd learner.\n");
    return;
  }

  debugL("Compute state/rewards stats from the replay memory");
  profiler->start("PRE");
  MemoryProcessing::updateCounters(* data.get(), true);
  MemoryProcessing::updateRewardsStats(* data.get(), true);
  // shift counters after initial data is gathered and sync is concluded
  data->counters.nGatheredB4Startup = data->nLocalSeenSteps();
  _nObsB4StartTraining = nObsB4StartTraining;

  data->updateSampler(true);
  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace/GAE values for all
  // experiences collected before this point.
  // This assumes V(s) is initialized small, so we just rescale by std(rew)
  debugL("Rescale Retrace est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  MemoryProcessing::rescaleAllReturnEstimator(* data.get());
  profiler->stop();
}

void Learner::processMemoryBuffer()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this
  profiler->start("FILTER");
  //if (bRecomputeProperties) printf("Using C : %f\n", C);
  MemoryProcessing::selectEpisodeToDelete(* data.get(), ERFILTER);
  if(currStep%1000==0) {
    profiler->stop_start("PRE");
    // update state mean/std with net's learning rate
    MemoryProcessing::updateRewardsStats(* data.get(), false, 10);
    //MemoryProcessing::updateRewardsStats(* data.get());
  }
  profiler->stop_start("FIND");
  MemoryProcessing::prepareNextBatchAndDeleteStaleEp(* data.get());

  if(0){ // This would act like a PID controller to keep Q mean 0 stdev 1
    const Real eta = settings.learnrate, epsAnneal = settings.epsAnneal;
    const Real learnR = Utilities::annealRate(eta,nGradSteps(),epsAnneal);
    MDP.rewardsMean += learnR/10 * data->stats.avgQ;
    MDP.rewardsStdDev += learnR/100 * (data->stats.stdevQ - 1);
    MDP.rewardsScale = 1 / MDP.rewardsStdDev;
  }

  MemoryProcessing::updateCounters(* data.get());
  profiler->stop();
}

bool Learner::blockDataAcquisition() const
{
  //if there is not enough data for training, need more data
  //_warn("readNSeen:%ld nData:%ld nDataGatheredB4Start:%ld gradSteps:%ld obsPerStep:%f",
  //data->readNSeen_loc(), data->readNData(), nDataGatheredB4Startup, _nGradSteps.load(), obsPerStep_loc);

  if( data->nStoredSteps() < _nObsB4StartTraining ) return false;

  // block data if we have observed too many observations
  // here we add one to concurrently gather data and compute gradients
  // 'freeze if there is too much data for the  next gradient step'
  return nLocTimeStepsTrain() > (nGradSteps()+1) * obsPerStep_loc;
}

bool Learner::blockGradientUpdates() const
{
  //_warn("readNSeen:%ld nDataGatheredB4Start:%ld gradSteps:%ld obsPerStep:%f",
  //data->readNSeen_loc(), nDataGatheredB4Startup, _nGradSteps.load(), obsPerStep_loc);
  // almost the same of the function before
  // 'freeze if there is too little data for the current gradient step'
  return nLocTimeStepsTrain() < nGradSteps() * obsPerStep_loc;
}

void Learner::setupDataCollectionTasks(TaskQueue& tasks)
{
  data->setupDataCollectionTasks(tasks);
}

void Learner::globalGradCounterUpdate()
{
  data->increaseGradStep();
}

void Learner::logStats(const bool bForcePrint)
{
  const Uint currStep = nGradSteps()+1;
  const Uint fProfl = freqPrint * PRFL_DMPFRQ;
  const bool bProfileAndHist = currStep % fProfl == 0 && learn_rank == 0;
  if(bProfileAndHist) {
    printf("%s\n", profiler->printStatAndReset().c_str() );
    // TODO : separate histograms frequency from profiler
    MemoryProcessing::histogramImportanceWeights(* data.get());
  }

  if(currStep % settings.saveFreq == 0 && learn_rank == 0) save();

  if(currStep % freqPrint == 0 or bForcePrint) {
    profiler->start("STAT");
    static bool bToPrintHeader = true;
    bToPrintHeader = bProfileAndHist or bToPrintHeader;
    processStats(bToPrintHeader);
    bToPrintHeader = false;
    profiler->stop();
  }
}

void Learner::processStats(const bool bPrintHeader)
{
  const unsigned currStep = nGradSteps()+1, tStamp = currStep/freqPrint;

  std::ostringstream buf;
  data->getMetrics(buf);
  getMetrics(buf);

  #ifndef PRINT_ALL_RANKS
    if(learn_rank) return;
    FILE* fout = fopen ((learner_name+"_stats.txt").c_str(),"a");
  #else
    FILE* fout = fopen ((learner_name+
      "_rank"+std::to_string(learn_rank)+"_stats.txt").c_str(), "a");
  #endif

  std::ostringstream head;
  if(bPrintHeader) {
    data->getHeaders(head);
    getHeaders(head);
    #ifdef PRINT_ALL_RANKS
      printf("ID  #/T   %s\n", head.str().c_str());
    #else
      printf("ID #/T   %s\n", head.str().c_str());
    #endif
    if(currStep==freqPrint) fprintf(fout, "ID #/T   %s\n", head.str().c_str());
  }

  const unsigned learnID = data->learnID;
  #ifdef PRINT_ALL_RANKS
    const unsigned rank = learn_rank;
    printf("%01u-%01u %05u%s\n", rank, learnID, tStamp,buf.str().c_str());
  #else
    printf(     "%02u %05u%s\n",       learnID, tStamp,buf.str().c_str());
  #endif
  fprintf(fout, "%02u %05u%s\n",       learnID, tStamp,buf.str().c_str());
  fclose(fout); fflush(0);
}

void Learner::getMetrics(std::ostringstream& buf) const {}

void Learner::getHeaders(std::ostringstream& buf) const {}

void Learner::restart()
{
  if(distrib.restart == "none") return;
  if(learn_rank==0) printf("Restarting from saved policy...\n");

  data->restart(distrib.restart+"/"+learner_name);
  //data->save("restarted_"+learner_name, 0, false);
}

void Learner::save()
{
  data->save(learner_name);
}

}
