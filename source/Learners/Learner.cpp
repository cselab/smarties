//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner.h"
#include "../Network/Builder.h"
#include <chrono>

Learner::Learner(Environment*const _env, Settings& _s): settings(_s), env(_env),
totNumSteps(_s.totNumSteps), policyVecDim(_s.policyVecDim), nAgents(_s.nAgents),
batchSize(_s.batchSize), nThreads(_s.nThreads), nWorkers(_s.nWorkers),
CmaxPol(_s.clipImpWeight), ReFtol(_s.penalTol), learnR(_s.learnrate),
gamma(_s.gamma), explNoise(_s.explNoise), epsAnneal(_s.epsAnneal)
{
  if(bSampleSequences) printf("Sampling sequences.\n");
  data = new MemoryBuffer(env, _s);
  input = new Encapsulator("input", _s, data);

  Builder input_build(_s);
  input_build.addInput( input->nOutputs() );
  bool builder_used = env->predefinedNetwork(input_build);
  if(builder_used) {
    Network* net = input_build.build(true);
    Optimizer* opt = input_build.opt;
    input->initializeNetwork(net, opt);
  }
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
  data->clearFailedSim(agentOne, agentEnd);
}

void Learner::pushBackEndedSim(const int agentOne, const int agentEnd)
{
  data->pushBackEndedSim(agentOne, agentEnd);
}

void Learner::prepareGradient()
{
  if(updateToApply) die("undefined behavior");
  if(not updateComplete)
  {
    warn("prepareGradient called while waiting for workers to gather data");
    return; // there is nothing in the gradients yet
  }
  // Learner is ready for the update: send the task to the networks and
  // start preparing the next one
  updateComplete = false;
  updateToApply = true;

  profiler->stop_start("ADDW");
  debugL("Gather gradient estimates from each thread and Learner MPI rank");
  for(auto & net : F) net->prepareUpdate(batchSize);
  input->prepareUpdate(batchSize);

  for(auto & net : F) net->updateGradStats(learner_name, nStep);

  if(nSkipped >= batchSize)
    warn("Too many skipped samples caused temporary pause in resampling. " \
      "Change hyperp: probably the learn rate is too large for "  \
      "the net's combination of size/activation/batchsize.");
  nSkipped = 0;
}

void Learner::initializeLearner()
{
  data->initialize();
}

void Learner::applyGradient()
{
  if(updateComplete) die("undefined behavior");
  if(not updateToApply) {
    warn("applyGradient called while waiting for data");
    return;
  }
  updateToApply = false;

  nStep++;
  if(nStep%(1000*PRFL_DMPFRQ)==0 && learn_rank==0)
  {
    profiler->printSummary();
    profiler->reset();
    save();
  }

  if(nStep%1000 ==0)
  {
    profiler->stop_start("STAT");
    processStats();
  }

  debugL("Apply SGD update after reduction of gradients");
  profiler->stop_start("GRAD");
  for(auto & net : F) net->applyUpdate();
  input->applyUpdate();
}

void Learner::processStats()
{
  ostringstream buf;
  data->getMetrics(buf);
  if(trainInfo not_eq nullptr) trainInfo->getMetrics(buf);
  input->getMetrics(buf);
  for(auto & net : F) net->getMetrics(buf);

  if(learn_rank) return;

  FILE* fout = fopen ((learner_name+"stats.txt").c_str(),"a");

  ostringstream head;
  if( nStep%(1000*PRFL_DMPFRQ)==0 || nStep==1000 ) {
    data->getHeaders(head);
    if(trainInfo not_eq nullptr) trainInfo->getHeaders(head);
    input->getHeaders(head);
    for(auto & net : F) net->getHeaders(head);

    printf("ID #/1e3 %s\n", head.str().c_str());
    if(nStep==1000)
      fprintf(fout, "ID #/1e3 %s\n", head.str().c_str());
  }

  fprintf(fout, "%02d %05d%s\n", learnID, (int)nStep/1000, buf.str().c_str());
  printf("%02d %05d%s\n", learnID, (int)nStep/1000, buf.str().c_str());
  fclose(fout);
  fflush(0);
}

void Learner::getMetrics(ostringstream& buf) const {}
void Learner::getHeaders(ostringstream& buf) const {}

void Learner::restart()
{
  if(settings.restart == "none") return;
  if(!learn_rank) printf("Restarting from saved policy...\n");

  for(auto & net : F) net->restart(settings.restart+"/"+learner_name);
  input->restart(settings.restart+"/"+learner_name);
  data->restart(settings.restart+"/"+learner_name);

  for(auto & net : F) net->save("restarted_"+learner_name, false);
  input->save("restarted_"+learner_name, false);
}

void Learner::save()
{
  static constexpr Real freqSave = 1000*PRFL_DMPFRQ;
  const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = nStep % freqBackup == 0;
  for(auto & net : F) net->save(learner_name, bBackup);
  input->save(learner_name, bBackup);
  data->save(learner_name, nStep, bBackup);
}

bool Learner::workerHasUnfinishedSeqs(const int worker) const
{
  const Uint nAgentsPerWorker = env->nAgentsPerRank;
  for(Uint i=worker*nAgentsPerWorker; i<(worker+1)*nAgentsPerWorker; i++)
    if(data->inProgress[i]->tuples.size()) return true;
  return false;
}

//TODO: generalize!!
bool Learner::predefinedNetwork(Builder& input_net, Uint privateNum)
{
  bool ret = false; // did i add layers to input net?
  if(input_net.nOutputs > 0) {
     input_net.nOutputs = 0;
     input_net.layers.back()->bOutput = false;
     warn("Overwritten ENV's specification of CNN to add shared layers");
  }
  vector<int> sizeOrig = settings.readNetSettingsSize();
  while ( sizeOrig.size() != privateNum )
  {
    const int size = sizeOrig[0];
    sizeOrig.erase(sizeOrig.begin(), sizeOrig.begin()+1);
    const bool bOutput = sizeOrig.size() == privateNum;
    input_net.addLayer(size, settings.nnFunc, bOutput);
    ret = true;
  }
  settings.nnl1 = sizeOrig.size() > 0? sizeOrig[0] : 0;
  settings.nnl2 = sizeOrig.size() > 1? sizeOrig[1] : 0;
  settings.nnl3 = sizeOrig.size() > 2? sizeOrig[2] : 0;
  settings.nnl4 = sizeOrig.size() > 3? sizeOrig[3] : 0;
  settings.nnl5 = sizeOrig.size() > 4? sizeOrig[4] : 0;
  settings.nnl6 = sizeOrig.size() > 5? sizeOrig[5] : 0;
  return ret;
}

//bool Learner::predefinedNetwork(Builder & input_net)
//{
//  return false;
//}
