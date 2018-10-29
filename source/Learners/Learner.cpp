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

Learner::Learner(Environment*const E, Settings&S): settings(S), env(E)
{
  if(input->nOutputs() == 0) return;
  Builder input_build(S);
  input_build.addInput( input->nOutputs() );
  bool builder_used = env->predefinedNetwork(input_build);
  if(builder_used) {
    Network* net = input_build.build(true);
    Optimizer* opt = input_build.opt;
    input->initializeNetwork(net, opt);
  }
}

void Learner::prepareGradient()
{
  const Uint currStep = nStep()+1;
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
  for(auto & net : F) net->prepareUpdate();
  input->prepareUpdate();

  for(auto & net : F) net->updateGradStats(learner_name, currStep-1);
}

void Learner::initializeLearner()
{
  data->initialize();
  bUpdateNdata = false;
}

void Learner::applyGradient()
{
  const Uint currStep = nStep()+1;

  if(updateComplete) die("undefined behavior");
  if(not updateToApply) {
    warn("applyGradient called while waiting for data");
    return;
  }
  updateToApply = false;

  if(currStep%(tPrint*PRFL_DMPFRQ)==0 && learn_rank==0)
  {
    cout << profiler->printStatAndReset() << endl;
    save();
  }

  if(currStep%tPrint ==0)
  {
    profiler->stop_start("STAT");
    processStats();
  }

  debugL("Apply SGD update after reduction of gradients");
  profiler->stop_start("GRAD");
  for(auto & net : F) net->applyUpdate();
  input->applyUpdate();

  globalGradCounterUpdate();
}

void Learner::globalGradCounterUpdate() {
  _nStep++;
  bUpdateNdata = false;
}

void Learner::processStats()
{
  const Uint currStep = nStep()+1;

  ostringstream buf;
  data_proc->getMetrics(buf);
  getMetrics(buf);
  if(trainInfo not_eq nullptr) trainInfo->getMetrics(buf);
  input->getMetrics(buf);
  for(auto & net : F) net->getMetrics(buf);

  #ifndef PRINT_ALL_RANKS
    if(learn_rank) return;
    FILE* fout = fopen ((learner_name+"stats.txt").c_str(),"a");
  #else
    FILE* fout = fopen (
      (learner_name+std::to_string(learn_rank)+"stats.txt").c_str(), "a");
  #endif

  ostringstream head;
  if( currStep%(tPrint*PRFL_DMPFRQ)==0 || currStep==tPrint ) {
    data_proc->getHeaders(head);
    getHeaders(head);
    if(trainInfo not_eq nullptr) trainInfo->getHeaders(head);
    input->getHeaders(head);
    for(auto & net : F) net->getHeaders(head);

    #ifdef PRINT_ALL_RANKS
      printf("ID  #/T   %s\n", head.str().c_str());
    #else
      printf("ID #/T   %s\n", head.str().c_str());
    #endif
    if(currStep==tPrint)
      fprintf(fout, "ID #/T   %s\n", head.str().c_str());
  }
  #ifdef PRINT_ALL_RANKS
    printf("%01d-%01d %05u%s\n",
      learn_rank, learnID, currStep/tPrint, buf.str().c_str());
  #else
    printf("%02d %05u%s\n", learnID, currStep/tPrint, buf.str().c_str());
  #endif
  fprintf(fout, "%02d %05u%s\n", learnID, currStep/tPrint, buf.str().c_str());
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
  data->save("restarted_"+learner_name, 0, false);
}

void Learner::save()
{
  const Uint currStep = nStep()+1;
  const Real freqSave = tPrint * PRFL_DMPFRQ;
  const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = currStep % freqBackup == 0;
  for(auto & net : F) net->save(learner_name, bBackup);
  input->save(learner_name, bBackup);
  data->save(learner_name, currStep, bBackup);
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

void Learner::createSharedEncoder(const Uint privateNum)
{
  if(input->net not_eq nullptr) {
    delete input->opt; input->opt = nullptr;
    delete input->net; input->net = nullptr;
  }
  if(input->nOutputs() == 0) return;
  Builder input_build(settings);
  bool bInputNet = false;
  input_build.addInput( input->nOutputs() );
  bInputNet = bInputNet || env->predefinedNetwork(input_build);
  bInputNet = bInputNet || predefinedNetwork(input_build, privateNum);
  if(bInputNet) {
    Network* net = input_build.build(true);
    input->initializeNetwork(net, input_build.opt);
  }
}

//bool Learner::predefinedNetwork(Builder & input_net)
//{
//  return false;
//}
