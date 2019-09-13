//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Worker_h
#define smarties_Worker_h

#include "../Utils/ParameterBlob.h"
#include "../Utils/TaskQueue.h"
#include "Environment.h"
#include "Launcher.h"
#include "../Learners/Learner.h"
#include "../Settings.h"
#include <thread>

namespace smarties
{

class Worker
{
public:
  Worker(DistributionInfo& distribinfo);
  virtual ~Worker() {}

  void synchronizeEnvironments();

  void runTraining();
  void loopSocketsToMaster();

  // may be called from application:
  void stepWorkerToMaster(Agent & agent) const;

  void run(const environment_callback_t & callback);

protected:
  DistributionInfo& distrib;
  TaskQueue dataTasks, algoTasks;

  const std::unique_ptr<Launcher> COMM;

  const MPI_Comm& master_workers_comm = distrib.master_workers_comm;
  const MPI_Comm& workerless_masters_comm = distrib.workerless_masters_comm;
  const MPI_Comm& learners_train_comm = distrib.learners_train_comm;
  const MPI_Comm& envAppComm = distrib.environment_app_comm;
  const int envMPIrank = MPICommRank(envAppComm);
  const int envMPIsize = MPICommSize(envAppComm);

  std::vector<std::unique_ptr<Learner>> learners;

  Environment& ENV;
  const std::vector<std::unique_ptr<Agent>>& agents;

  const Uint nCallingEnvs = distrib.nOwnedEnvironments;
  const int bTrain = distrib.bTrain;

  // avoid race conditions in writing cumulative rewards file:
  mutable std::mutex dump_mutex;

  // small utility functions:
  Uint getLearnerID(const Uint agentIDlocal) const;
  bool learnersBlockingDataAcquisition() const;
  void dumpCumulativeReward(const Agent&) const;

  void answerStateActionCaller(const int bufferID);

  void stepWorkerToMaster(const Uint bufferID) const;

  void answerStateAction(const Uint bufferID) const;
  void answerStateAction(Agent& agent) const;

  void sendStateRecvAction(const COMM_buffer& BUF) const;

  int getSocketID(const Uint worker) const;
  const COMM_buffer& getCommBuffer(const Uint worker) const;
};

} // end namespace smarties
#endif // smarties_Worker_h
