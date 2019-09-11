//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_approximator.h"
#include "../Network/Approximator.h"
#include "../Network/Builder.h"
#include "../Utils/ParameterBlob.h"
#include <chrono>

namespace smarties
{

Learner_approximator::Learner_approximator(MDPdescriptor& MDP_,
                                           Settings& S_,
                                           DistributionInfo& D_) :
                                           Learner(MDP_, S_, D_)
{
  if(!settings.bSampleSequences && nObsB4StartTraining<(long)settings.batchSize)
    die("Parameter minTotObsNum is too low for given problem");
}

Learner_approximator::~Learner_approximator()
{
  for(auto & net : networks) {
    if(net not_eq nullptr) {
      delete net;
      net = nullptr;
    }
  }
}

void Learner_approximator::spawnTrainTasks()
{
  if(settings.bSampleSequences && data->readNSeq() < (long) settings.batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->start("SAMP");
  const Uint batchSize=settings.batchSize_local, ESpopSize=settings.ESpopSize;
  const Uint nThr = distrib.nThreads, CS =  batchSize / nThr;
  const MiniBatch MB = data->sampleMinibatch(batchSize, nGradSteps() );
  profiler->stop();

  if(settings.bSampleSequences)
  {
    #pragma omp parallel for collapse(2) schedule(dynamic,1) num_threads(nThr)
    for (Uint wID=0; wID<ESpopSize; ++wID)
    for (Uint bID=0; bID<batchSize; ++bID) {
      const Uint thrID = omp_get_thread_num();
      for (const auto & net : networks ) net->load(MB, bID, wID);
      Train(MB, wID, bID);
      // backprop, from last net to first for dependencies in gradients:
      if(thrID==0) profiler->stop_start("BCK");
      for (const auto & net : Utilities::reverse(networks) ) net->backProp(bID);
      if(thrID==0) profiler->stop();
    }
  }
  else
  {
    #pragma omp parallel for collapse(2) schedule(static,CS) num_threads(nThr)
    for (Uint wID=0; wID<ESpopSize; ++wID)
    for (Uint bID=0; bID<batchSize; ++bID) {
      const Uint thrID = omp_get_thread_num();
      for (const auto & net : networks ) net->load(MB, bID, wID);
      Train(MB, wID, bID);
      // backprop, from last net to first for dependencies in gradients:
      if(thrID==0) profiler->stop_start("BCK");
      for (const auto & net : Utilities::reverse(networks) ) net->backProp(bID);
      if(thrID==0) profiler->stop();
    }
  }
}

void Learner_approximator::prepareGradient()
{
  const Uint currStep = nGradSteps()+1;

  profiler->start("ADDW");
  for(const auto & net : networks) {
    net->prepareUpdate();
    net->updateGradStats(learner_name, currStep-1);
  }
  profiler->stop();
}

void Learner_approximator::applyGradient()
{
  profiler->start("GRAD");
  for(const auto & net : networks) net->applyUpdate();
  profiler->stop();
}

void Learner_approximator::getMetrics(std::ostringstream& buf) const
{
  Learner::getMetrics(buf);
  for(const auto & net : networks) net->getMetrics(buf);
}
void Learner_approximator::getHeaders(std::ostringstream& buf) const
{
  Learner::getHeaders(buf);
  for(const auto & net : networks) net->getHeaders(buf);
}

void Learner_approximator::restart()
{
  if(distrib.restart == "none") return;
  if(!learn_rank) printf("Restarting from saved policy...\n");

  for(const auto & net : networks) {
    net->restart(distrib.restart+"/"+learner_name);
    net->save("restarted_"+learner_name, false);
  }

  Learner::restart();

  for(const auto & net : networks) net->setNgradSteps(_nGradSteps);
}

void Learner_approximator::save()
{
  //const Uint currStep = nGradSteps()+1;
  //const Real freqSave = freqPrint * PRFL_DMPFRQ;
  //const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = false; // currStep % freqBackup == 0;
  for(const auto & net : networks) net->save(learner_name, bBackup);

  Learner::save();
}

// Create preprocessing network, which contains conv layers requested by env
// and some shared fully connected layers, read from vector nnLayerSizes
// from settings. The last privateLayersNum sizes of nnLayerSizes are not
// added here because we assume those sizes will parameterize the approximators
// that take the output of the preprocessor and produce policies,values, etc.
bool Learner_approximator::createEncoder()
{
  const Uint nPreProcLayers = settings.encoderLayerSizes.size();

  if ( MDP.conv2dDescriptors.size() == 0 and nPreProcLayers == 0 )
    return false; // no preprocessing

  if(networks.size()>0) warn("some network was created before preprocessing");
  networks.push_back(new Approximator("encodr", settings,distrib, data.get()));
  networks.back()->buildPreprocessing(settings.encoderLayerSizes);

  return true;
}

void Learner_approximator::initializeApproximators()
{
  for(const auto& net : networks)
  {
    net->initializeNetwork();
  }
}

void Learner_approximator::setupDataCollectionTasks(TaskQueue& tasks)
{
  params.add(MDP.stateScale.size(), MDP.stateScale.data());
  params.add(MDP.stateMean.size(), MDP.stateMean.data());
  params.add(1, & MDP.rewardsScale);
  for(const auto& net : networks)  net->gatherParameters(params);

  Learner::setupDataCollectionTasks(tasks);
}

}

/*

void Learner_approximator::createSharedEncoder(const Uint privateNum)
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
, input(new Encapsulator("input", S_, data))
if(input->nOutputs() == 0) return;
Builder input_build(S);
input_build.addInput( input->nOutputs() );
bool builder_used = env->predefinedNetwork(input_build);
if(builder_used) {
  Network* net = input_build.build(true);
  Optimizer* opt = input_build.opt;
  input->initializeNetwork(net, opt);
}
*/
