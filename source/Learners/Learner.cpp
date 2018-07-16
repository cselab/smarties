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

Learner::Learner(Environment*const _env, Settings & _s) :
mastersComm(_s.mastersComm), env(_env), bSampleSequences(_s.bSampleSequences),
bTrain(_s.bTrain), totNumSteps(_s.totNumSteps), policyVecDim(_s.policyVecDim),
batchSize(_s.batchSize), nAgents(_s.nAgents), nThreads(_s.nThreads),
nWorkers(_s.nWorkers), gamma(_s.gamma), learnR(_s.learnrate), ReFtol(_s.penalTol),
explNoise(_s.explNoise), epsAnneal(_s.epsAnneal), CmaxPol(_s.clipImpWeight),
learn_rank(_s.learner_rank), learn_size(_s.learner_size), settings(_s),
aInfo(env->aI), sInfo(env->sI), generators(_s.generators)
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

  for(auto & net : F) net->updateGradStats(nStep);

  if(nSkipped >= batchSize)
    warn("Too many skipped samples caused temporary pause in resampling. " \
      "Change hyperp: probably the learn rate is too large for "  \
      "the net's combination of size/activation/batchsize.");
  nSkipped = 0;
}

void Learner::initializeLearner()
{
  // All sequences obtained before this point should share the same time stamp
  for(Uint i=0;i<data->Set.size();i++) data->Set[i]->ID= data->readNSeenSeq()-1;
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

    for(auto & net : F) net->save(learner_name);
    input->save(learner_name);
    data->save(learner_name, nStep);
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

  for(auto & net : F) net->save("restarted_");
  input->save("restarted_");
}

bool Learner::workerHasUnfinishedSeqs(const int worker) const
{
  const Uint nAgentsPerWorker = env->nAgentsPerRank;
  for(Uint i=worker*nAgentsPerWorker; i<(worker+1)*nAgentsPerWorker; i++)
    if(data->inProgress[i]->tuples.size()) return true;
  return false;
}

//TODO: generalize!!
bool Learner::predefinedNetwork(Builder& input_net)
{
  if(settings.nnl2<=0) return false;

  if(input_net.nOutputs > 0) {
    input_net.nOutputs = 0;
    input_net.layers.back()->bOutput = false;
  }
  //                 size       function     is output (of input net)?
  input_net.addLayer(settings.nnl1, settings.nnFunc, settings.nnl3<=0);
  settings.nnl1 = settings.nnl2;
  if(settings.nnl3>0) {
    input_net.addLayer(settings.nnl2, settings.nnFunc, settings.nnl4<=0);
    settings.nnl1 = settings.nnl3;
    if(settings.nnl4>0) {
      input_net.addLayer(settings.nnl3, settings.nnFunc, settings.nnl5<=0);
      settings.nnl1 = settings.nnl4;
      if(settings.nnl5>0) {
        input_net.addLayer(settings.nnl4, settings.nnFunc, settings.nnl6<=0);
        settings.nnl1 = settings.nnl5;
        if(settings.nnl6>0) {
          input_net.addLayer(settings.nnl5, settings.nnFunc, true);
          settings.nnl1 = settings.nnl6;
        }
      }
    }
  }
  settings.nnl2 = 0; // value, adv and pol nets will be one-layer
  return true;
}

//bool Learner::predefinedNetwork(Builder & input_net)
//{
//  return false;
//}
