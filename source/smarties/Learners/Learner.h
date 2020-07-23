//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Learner_h
#define smarties_Learner_h

#include "../Core/StateAction.h"
#include "../ReplayMemory/MemoryBuffer.h"
#include "../Utils/StatsTracker.h"
#include "../Utils/Profiler.h"
#include "../Settings/ExecutionInfo.h"
#include "../Settings/HyperParameters.h"

namespace smarties
{

class DataCoordinator;
class Collector;

class Learner
{
protected:
  Uint freqPrint = 1000;
  ExecutionInfo & distrib;
  HyperParameters settings;
  MDPdescriptor & MDP;

public:
  const MPI_Comm learnersComm = distrib.learners_train_comm;
  const Uint learn_rank = MPICommRank(learnersComm);
  const Uint learn_size = MPICommSize(learnersComm);
  const Uint nThreads = distrib.nThreads, nAgents = distrib.nAgents;

  const Uint policyVecDim = MDP.policyVecDim;
  const ActionInfo aInfo = ActionInfo(MDP);
  const StateInfo  sInfo = StateInfo(MDP);

  // training loop scheduling:
  const Real obsPerStep_loc = settings.obsPerStep_local;
  const long nObsB4StartTraining = settings.minTotObsNum_local;
  long _nObsB4StartTraining = std::numeric_limits<long>::max();
  const bool bTrain = distrib.bTrain;

  // some algorithm hyper-parameters:
  const Real gamma = settings.gamma;
  const FORGET ERFILTER;

protected:
  int algoSubStepID = -1;

  std::vector<std::mt19937>& generators = distrib.generators;

  const std::unique_ptr<MemoryBuffer> data;

  Real & alpha   = data->alpha;
  Real & beta    = data->beta;
  Real & CmaxRet = data->CmaxRet;
  Real & CinvRet = data->CinvRet;
  ParameterBlob & params = data->params;

  const std::unique_ptr<Profiler> profiler  = std::make_unique<Profiler>();

  mutable std::mutex buffer_mutex;

  virtual void processStats(const bool bPrintHeader);

public:
  std::string learner_name;

  Learner(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_);
  virtual ~Learner();

  void setLearnerName(const std::string lName, const Uint id) {
    learner_name = lName;
    data->learnID = id;
  }

  long nLocTimeStepsTrain() const {
    return data->nLocTimeStepsTrain();
  }
  long nLocTimeSteps() const {
    return data->nLocalSeenSteps();
  }
  long nSeqsEval() const {
    return data->nLocalSeenEps();
  }
  long locDataSetSize() const {
    return data->nStoredSteps();
  }
  long nGradSteps() const {
    return data->nGradSteps();
  }
  Real getAvgCumulativeReward() const {
    return data->getAvgReturn();
  }

  virtual void setupDataCollectionTasks(TaskQueue& tasks);

  virtual void setupTasks(TaskQueue& tasks) = 0;
  virtual void selectAction(const MiniBatch& MB, Agent& agent) = 0;
  virtual void processTerminal(const MiniBatch& MB, Agent& agent) = 0;

  virtual void globalGradCounterUpdate();

  virtual bool blockDataAcquisition() const;
  virtual bool blockGradientUpdates() const;

  void select(Agent& agent);
  void processMemoryBuffer();
  virtual void initializeLearner();

  virtual void logStats(const bool bForcePrint = false);

  virtual void getMetrics(std::ostringstream& buff) const;
  virtual void getHeaders(std::ostringstream& buff) const;

  virtual void save();
  virtual void restart();
};

}
#endif
