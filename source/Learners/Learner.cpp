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
#include "../ReplayMemory/DataCoordinator.h"
#include "../ReplayMemory/Collector.h"
#include <unistd.h>

namespace smarties
{

Learner::Learner(MDPdescriptor& MD, Settings& S, DistributionInfo& D):
  distrib(D), settings(S), MDP(MD), ERFILTER(
    MemoryProcessing::readERfilterAlgo(S.ERoldSeqFilter, S.clipImpWeight>0) ),
  data_proc( new MemoryProcessing( data.get() ) ),
  data_coord( new DataCoordinator( data.get(), params ) ),
  data_get ( new Collector       ( data.get(), data_coord ) ) {}

Learner::~Learner()
{
  if(trainInfo not_eq nullptr) delete trainInfo;
  delete data_proc;
  delete data_get;
  delete data_coord;
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
  data_proc->updateRewardsStats(1, 1, true);
  // shift counters after initial data is gathered and sync is concluded
  data->nGatheredB4Startup = data->readNSeen_loc();
  _nObsB4StartTraining = nObsB4StartTraining;
  //data_proc->updateRewardsStats(1, 1e-3, true);
  if(learn_rank==0) printf("Initial reward std %e\n", 1/data->scaledReward(1));
  fflush(0);
  data->initialize();

  if( not computeQretrace ) {
    profiler->stop();
    return;
  }
  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  const Uint setSize = data->readNSeq();
  #pragma omp parallel for schedule(dynamic, 1)
  for(Uint i=0; i<setSize; ++i) {
    Sequence& SEQ = data->get(i);
    for(Uint j = SEQ.ndata(); j>0; --j)
        SEQ.propagateRetrace(j, gamma, data->scaledReward(SEQ, j));
  }
  profiler->stop();
}

void Learner::processMemoryBuffer()
{
  profiler->start("FILTER");
  //if (bRecomputeProperties) printf("Using C : %f\n", C);
  data_proc->selectEpisodeToDelete(ERFILTER);
  profiler->stop();
  data_proc->updateReFERpenalization();
}

void Learner::updateRetraceEstimates()
{
  profiler->start("QRET");
  const std::vector<Uint>& sampled = data->lastSampledEpisodes();
  const Uint setSize = sampled.size();

  #pragma omp parallel for schedule(dynamic, 1)
  for(Uint i = 0; i < setSize; ++i) {
    Sequence& SEQ = data->get(sampled[i]);
    assert(std::fabs(SEQ.Q_RET[SEQ.ndata()]) < 1e-16);
    assert(std::fabs(SEQ.action_adv[SEQ.ndata()]) < 1e-16);
    if( SEQ.isTerminal(SEQ.ndata()) )
      assert(std::fabs(SEQ.state_vals[SEQ.ndata()]) < 1e-16);
    for(Sint j = SEQ.just_sampled-1; j>0; --j)
        SEQ.propagateRetrace(j, gamma, data->scaledReward(SEQ, j));
  }
  profiler->stop();
}

void Learner::finalizeMemoryProcessing()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this
  profiler->start("FIND");
  data_proc->prepareNextBatchAndDeleteStaleEp();

  profiler->stop_start("PRE");
  if(currStep%1000==0) // update state mean/std with net's learning rate
    data_proc->updateRewardsStats(1, 1e-3*(SMARTIES_OFFPOL_ADAPT_STSCALE>0));

  profiler->stop();
}

bool Learner::blockDataAcquisition() const
{
  //if there is not enough data for training, need more data
  //_warn("readNSeen:%ld nData:%ld nDataGatheredB4Start:%ld gradSteps:%ld obsPerStep:%f",
  //data->readNSeen_loc(), data->readNData(), nDataGatheredB4Startup, _nGradSteps.load(), obsPerStep_loc);

  if( data->readNData() < _nObsB4StartTraining ) return false;

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
  data_coord->setupTasks(tasks);
}

void Learner::logStats()
{
  const Uint currStep = nGradSteps()+1;
  const Uint fProfl = freqPrint * PRFL_DMPFRQ;
  const Uint fBackup = std::ceil(settings.saveFreq / (Real)fProfl) * fProfl;

  if(currStep % fProfl == 0 && learn_rank == 0) {
    printf("%s\n", profiler->printStatAndReset().c_str() );
    // TODO : separate histograms frequency from profiler
    data_proc->histogramImportanceWeights();
  }

  if(currStep % fBackup == 0 && learn_rank == 0) save();

  if(currStep % freqPrint == 0) {
    profiler->start("STAT");
    processStats();
    profiler->stop();
  }
}

void Learner::globalGradCounterUpdate()
{
  data->nGradSteps++;
}

void Learner::processStats()
{
  const Uint currStep = nGradSteps()+1;

  std::ostringstream buf;
  data_proc->getMetrics(buf);
  getMetrics(buf);
  if(trainInfo not_eq nullptr) trainInfo->getMetrics(buf);

  #ifndef PRINT_ALL_RANKS
    if(learn_rank) return;
    FILE* fout = fopen ((learner_name+"_stats.txt").c_str(),"a");
  #else
    FILE* fout = fopen ((learner_name+
      "_rank"+std::to_string(learn_rank)+"_stats.txt").c_str(), "a");
  #endif

  std::ostringstream head;
  if( currStep%(freqPrint*PRFL_DMPFRQ)==0 || currStep==freqPrint )
  {
    data_proc->getHeaders(head);
    getHeaders(head);
    if(trainInfo not_eq nullptr) trainInfo->getHeaders(head);


    #ifdef PRINT_ALL_RANKS
      printf("ID  #/T   %s\n", head.str().c_str());
    #else
      printf("ID #/T   %s\n", head.str().c_str());
    #endif
    if(currStep==freqPrint)
      fprintf(fout, "ID #/T   %s\n", head.str().c_str());
  }
  #ifdef PRINT_ALL_RANKS
    printf("%01lu-%01lu %05u%s\n",
      learn_rank, data->learnID, currStep/freqPrint, buf.str().c_str());
  #else
    printf("%02lu %05lu%s\n", data->learnID, currStep/freqPrint, buf.str().c_str());
  #endif
  fprintf(fout,"%02lu %05lu%s\n", data->learnID, currStep/freqPrint, buf.str().c_str());
  fclose(fout);
  fflush(0);
}

void Learner::getMetrics(std::ostringstream& buf) const {}

void Learner::getHeaders(std::ostringstream& buf) const {}

void Learner::restart()
{
  if(distrib.restart == "none") return;
  if(!learn_rank) printf("Restarting from saved policy...\n");

  data->restart(distrib.restart+"/"+learner_name);
  //data->save("restarted_"+learner_name, 0, false);
}

void Learner::save()
{
  data->save(learner_name);
}

}
