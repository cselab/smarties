//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "../ReplayMemory/MemoryBuffer.h"
#include "../ReplayMemory/Collector.h"
#include "../ReplayMemory/MemoryProcessing.h"
#include "../Network/Approximator.h"

#include <list>

class Learner
{
 protected:
  Settings & settings;
  Environment * const env;

 public:
  const MPI_Comm mastersComm = settings.mastersComm;

  const bool bSampleSequences = settings.bSampleSequences;
  const Uint nObsPerTraining = settings.minTotObsNum_loc;
  const bool bTrain = settings.bTrain;

  const Uint policyVecDim = env->aI.policyVecDim;
  const Uint nAgents = settings.nAgents;
  const Uint nThreads = settings.nThreads;

  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;

  // hyper-parameters:
  const Uint batchSize = settings.batchSize_loc;
  const Uint totNumSteps = settings.totNumSteps;
  const Uint ESpopSize = settings.ESpopSize;

  const Real learnR = settings.learnrate;
  const Real gamma = settings.gamma;
  const Real CmaxPol = settings.clipImpWeight;
  const Real ReFtol = settings.penalTol;
  const Real explNoise = settings.explNoise;
  const Real epsAnneal = settings.epsAnneal;

  const StateInfo&  sInfo = env->sI;
  const ActionInfo& aInfo = env->aI;
  const ActionInfo* const aI = &aInfo;

 protected:
  long nData_b4Startup = 0;
  mutable int percData = -5;
  std::atomic<long> _nStep{0};
  std::atomic<bool> bUpdateNdata{false};
  std::atomic<bool> bReady4Init {false};

  bool updateComplete = false;
  bool updateToApply = false;

  std::vector<std::mt19937>& generators = settings.generators;

  MemoryBuffer* const data = new MemoryBuffer(settings, env);
  Encapsulator * const input = new Encapsulator("input", settings, data);
  MemoryProcessing* const data_proc = new MemoryProcessing(settings, data);
  Collector* data_get;

  TrainData* trainInfo = nullptr;
  std::vector<Approximator*> F;
  mutable std::mutex buffer_mutex;

  virtual void processStats();
  void createSharedEncoder(const Uint privateNum = 1);
  bool predefinedNetwork(Builder& input_net, const Uint privateNum = 1);

 public:
  Profiler* profiler = nullptr;
  std::string learner_name;
  Uint learnID;
  Uint tPrint = 1000;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(data_proc);
    _dispose_object(data_get);
    _dispose_object(input);
    _dispose_object(data);
  }

  inline void setLearnerName(const std::string lName, const Uint id) {
    learner_name = lName;
    data->learnID = id;
    learnID = id;
  }

  inline unsigned tStepsTrain() const {
    return data->readNSeen_loc() - nData_b4Startup;
  }
  inline unsigned nSeqsEval() const {
    return data->readNSeenSeq_loc();
  }
  inline long int nStep() const {
    return _nStep.load();
  }
  inline Real annealingFactor() const {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.);
    const auto mynstep = nStep();
    if(mynstep*epsAnneal >= 1 || !bTrain) return 0;
    else return 1 - mynstep*epsAnneal;
  }

  virtual void select(Agent& agent) = 0;

  virtual void getMetrics(ostringstream& buff) const;
  virtual void getHeaders(ostringstream& buff) const;

  //main training loop functions:
  virtual void spawnTrainTasks_par() = 0;
  virtual void spawnTrainTasks_seq() = 0;
  virtual bool bNeedSequentialTrain() = 0;

  void globalDataCounterUpdate(const long globSeenObs, const long globSeenSeq)
  {
    data->setNSeen(globSeenObs);
    data->setNSeenSeq(globSeenSeq);
    if( bReady4Init and not blockDataAcquisition() )
      _die("? %ld %ld %ld %ld", data->readNSeen_loc(), nData_b4Startup,
           _nStep.load(), data->readNSeen());
    bReady4Init = true;
    bUpdateNdata = true;
  }
  virtual void globalGradCounterUpdate();

  bool unblockGradStep() const {
    return bUpdateNdata.load();
  }
  bool blockGradStep() const {
    return not bUpdateNdata.load();
  }

  virtual bool blockDataAcquisition() const = 0;

  inline bool isReady4Init() const {
    if(bTrain == false) return true;
    return bReady4Init.load();
  }
  inline bool checkReady4Init() const {
    if( bReady4Init.load() ) return false;
    const bool ready = data->readNData() >= nObsPerTraining;
    if(not ready && learn_rank==0) {
      std::lock_guard<std::mutex> lock(buffer_mutex);
      const int currPerc = data->readNData() * 100. / (Real) nObsPerTraining;
      if(currPerc > percData+5) {
       percData = currPerc;
       printf("\rCollected %d%% of data required to begin training. ",percData);
       fflush(0); //otherwise no show on some platforms
      }
    }
    return ready;
  }

  virtual void prepareGradient();
  virtual void applyGradient();
  virtual void initializeLearner();
  virtual void save();
  virtual void restart();
};
