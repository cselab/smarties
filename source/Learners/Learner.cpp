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
  distrib(D), settings(S), MDP(MD), params(D), ReFER_reduce(D, LDvec{0.,1.}),
  ERFILTER(MemoryProcessing::readERfilterAlgo(S.ERoldSeqFilter, CmaxPol>0)),
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

  ReFER_reduce.update({(long double)data_proc->nFarPol(),
                       (long double)data->readNData()});

  if ( currStep > 0 ) {
    warn("Skipping initialization for restartd learner.");
    return;
  }

  debugL("Compute state/rewards stats from the replay memory");
  profiler->stop_start("PRE");
  data_proc->updateRewardsStats(1, 1, true);
  // shift counters after initial data is gathered and sync is concluded
  nDataGatheredB4Startup = data->readNSeen_loc();
  _nObsB4StartTraining = nObsB4StartTraining;
  //data_proc->updateRewardsStats(1, 1e-3, true);
  if(learn_rank==0) printf("Initial reward std %e\n", 1/data->scaledReward(1));

  data->initialize();

  if( not computeQretrace ) return;
  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  const Uint setSize = data->readNSeq();
  #pragma omp parallel for schedule(dynamic, 1)
  for(Uint i=0; i<setSize; ++i) {
    Sequence& SEQ = * data->get(i);
    for(Uint j = SEQ.ndata(); j>0; --j) backPropRetrace(SEQ, j);
  }
}

void Learner::processMemoryBuffer()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this

  profiler->stop_start("PRNE");
  //shift data / gradient counters to maintain grad stepping to sample
  // collection ratio prescirbed by obsPerStep

  CmaxRet = 1 + Utilities::annealRate(CmaxPol, currStep, epsAnneal);
  CinvRet = 1 / CmaxRet;
  if(CmaxRet<=1 and CmaxPol>0)
    die("Either run lasted too long or epsAnneal is wrong.");
  data_proc->prune(ERFILTER, CmaxRet);

  // use result from prev AllReduce to update rewards (before new reduce).
  // Assumption is that the number of off Pol trajectories does not change
  // much each step. Especially because here we update the off pol W only
  // if an obs is actually sampled. Therefore at most this fraction
  // is wrong by batchSize / nTransitions ( ~ 0 )
  // In exchange we skip an mpi implicit barrier point.
  ReFER_reduce.update({(long double)data_proc->nFarPol(),
                       (long double)data->readNData()});
  const LDvec nFarGlobal = ReFER_reduce.get();
  const Real fracOffPol = nFarGlobal[0] / nFarGlobal[1];

  if(fracOffPol>ReFtol) beta = (1-1e-4)*beta; // iter converges to 0
  else beta = 1e-4 +(1-1e-4)*beta; //fixed point iter converge to 1
  if(std::fabs(ReFtol-fracOffPol)<0.001) alpha = (1-1e-4)*alpha;
  else alpha = 1e-4 + (1-1e-4)*alpha;
}

void Learner::updateRetraceEstimates()
{
  profiler->stop_start("QRET");
  const std::vector<Uint>& sampled = data->lastSampledEpisodes();
  const Uint setSize = sampled.size();
  #pragma omp parallel for schedule(dynamic, 1)
  for(Uint i = 0; i < setSize; ++i) {
    Sequence& SEQ = * data->get(sampled[i]);
    assert(std::fabs(SEQ.Q_RET[SEQ.ndata()]) < 1e-16);
    assert(std::fabs(SEQ.action_adv[SEQ.ndata()]) < 1e-16);
    if( SEQ.isTerminal(SEQ.ndata()) )
      assert(std::fabs(SEQ.state_vals[SEQ.ndata()]) < 1e-16);
    for(Sint j=SEQ.just_sampled-1; j>0; --j) backPropRetrace(SEQ, j);
  }
}

void Learner::finalizeMemoryProcessing()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this
  profiler->stop_start("FIND");
  data_proc->finalize();

  profiler->stop_start("PRE");
  if(currStep%1000==0) // update state mean/std with net's learning rate
    data_proc->updateRewardsStats(1, 1e-3*(OFFPOL_ADAPT_STSCALE>0));
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

  if(currStep % fProfl == 0 && learn_rank == 0)
    printf("%s\n", profiler->printStatAndReset().c_str() );

  if(currStep % fBackup == 0 && learn_rank == 0) save();

  if(currStep % freqPrint == 0) {
    profiler->stop_start("STAT");
    processStats();
  }
}

void Learner::globalGradCounterUpdate()
{
  _nGradSteps++;
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
    FILE* fout = fopen ((learner_name+"stats.txt").c_str(),"a");
  #else
    FILE* fout = fopen (
      (learner_name+std::to_string(learn_rank)+"stats.txt").c_str(), "a");
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
      learn_rank, learnID, currStep/freqPrint, buf.str().c_str());
  #else
    printf("%02lu %05lu%s\n", learnID, currStep/freqPrint, buf.str().c_str());
  #endif
  fprintf(fout,"%02lu %05lu%s\n", learnID,currStep/freqPrint,buf.str().c_str());
  fclose(fout);
  fflush(0);
}

void Learner::getMetrics(std::ostringstream& buf) const
{
  if(not computeQretrace) return;
  Utilities::real2SS(buf, alpha, 6, 1);
  Utilities::real2SS(buf, beta, 6, 1);
}

void Learner::getHeaders(std::ostringstream& buf) const
{
  if(not computeQretrace) return;
  buf << "| alph | beta ";
}

void Learner::restart()
{
  if(distrib.restart == "none") return;
  if(!learn_rank) printf("Restarting from saved policy...\n");

  data->restart(distrib.restart+"/"+learner_name);

  data->save("restarted_"+learner_name, 0, false);

  {
    char currDirectory[512];
    getcwd(currDirectory, 512);
    chdir(distrib.initial_runDir);

    std::ostringstream ss;
    ss<<"rank_"<<std::setfill('0')<<std::setw(3)<<learn_rank<<"_learner.raw";
    FILE * f = fopen((learner_name+ss.str()).c_str(), "rb");

    if(f == NULL) {
      _warn("Learner restart file %s not found\n",
            (learner_name+ss.str()).c_str());
      return;
    }

    Uint nSeqs;
    if(fread(&nSeqs,sizeof(Uint),1,f) != 1) die("");
    data->setNSeq(nSeqs);

    {
      Uint nObs;
      if(fread(&nObs, sizeof(Uint),1,f) != 1) die("");
      data->setNData(nObs);
    }
    {
      Uint tObs;
      if(fread(&tObs, sizeof(Uint),1,f) != 1) die("");
      data->setNSeen_loc(tObs);
    }
    {
      Uint tSeqs;
      if(fread(&tSeqs,sizeof(Uint),1,f) != 1) die("");
      data->setNSeenSeq_loc(tSeqs);
    }
    {
      long tB4;
      if(fread(&tB4, sizeof(long), 1, f) != 1) die("");
      nDataGatheredB4Startup = tB4;
    }
    {
      long int currStep;
      if(fread(&currStep, sizeof(long int), 1, f) != 1) die("");
      _nGradSteps = currStep;
    }
    if(fread(&beta, sizeof(Real), 1, f) != 1) die("");
    if(fread(&CmaxRet, sizeof(Real), 1, f) != 1) die("");

    for(Uint i = 0; i < nSeqs; ++i) {
      assert(data->get(i) == nullptr);
      Sequence* const S = new Sequence();
      if( S->restart(f, sInfo.dimObs(), aInfo.dim(), aInfo.dimPol()) )
        _die("Unable to find sequence %u\n", i);
      data->set(S, i);
    }

    chdir(currDirectory);
  }
}

void Learner::save()
{
  const Uint currStep = nGradSteps()+1;
  const bool bBackup = false; // ;
  data->save(learner_name, currStep, bBackup);

  //if(not bBackup) return;

  const std::string name = learner_name + "_learner.raw";
  const std::string backname = learner_name + "_learner_backup.raw";
  FILE * f = fopen(backname.c_str(), "wb");

  const Uint nObs=data->readNData(), tObs=data->readNSeen_loc();
  const Uint nSeqs=data->readNSeq(), tSeqs=data->readNSeenSeq_loc();
  fwrite(&nSeqs, sizeof(Uint), 1, f);
  fwrite(&nObs, sizeof(Uint), 1, f);
  fwrite(&tObs, sizeof(Uint), 1, f);
  fwrite(&tSeqs, sizeof(Uint), 1, f);
  fwrite(&nDataGatheredB4Startup, sizeof(long), 1, f);
  fwrite(&currStep, sizeof(long int), 1, f);
  fwrite(&beta, sizeof(Real), 1, f);
  fwrite(&CmaxRet, sizeof(Real), 1, f);

  for(Uint i = 0; i <nSeqs; ++i)
    data->get(i)->save(f, sInfo.dimObs(), aInfo.dim(), aInfo.dimPol() );

  Utilities::copyFile(backname, name);
}

}
