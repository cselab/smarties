//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Settings_h
#define smarties_Settings_h

#include "Definitions.h"
#include "../Utils/MPIUtilities.h"

#include <random>
#include <mutex>

namespace smarties
{

struct ExecutionInfo
{
  ExecutionInfo(int _argc, char ** _argv);
  ExecutionInfo(const std::vector<std::string> & args);
  ExecutionInfo(const MPI_Comm& mpi_comm, int _argc, char ** _argv);
  ~ExecutionInfo();

  const bool bOwnArgv; // whether argv needs to be deallocated
  int argc;
  char ** argv;

  void commonInit();
  int parse();

  void initialze();
  void figureOutWorkersPattern();

  char initial_runDir[1024];
  MPI_Comm world_comm;
  Uint world_rank;
  Uint world_size;

  int threadSafety = -1;
  bool bAsyncMPI;
  mutable std::mutex mpiMutex;

  Sint thisWorkerGroupID = -1;
  Uint nAgents;

  MPI_Comm master_workers_comm = MPI_COMM_NULL;
  MPI_Comm workerless_masters_comm = MPI_COMM_NULL;
  MPI_Comm learners_train_comm = MPI_COMM_NULL;
  MPI_Comm environment_app_comm = MPI_COMM_NULL;

  bool bIsMaster;
  Uint nOwnedEnvironments = 0;
  Uint nOwnedAgentsPerAlgo = 1;
  Uint nForkedProcesses2spawn = 0;
  //random number generators (one per thread)
  mutable std::vector<std::mt19937> generators;

  // Parsed. For comments look at .cpp
  Uint nThreads = 1;
  Uint nMasters = 1;
  Uint nWorkers = 1;
  Uint nEnvironments = 1;
  Uint workerProcessesPerEnv = 1;
  Uint randSeed = 0;
  Uint nTrainSteps = 10000000; // if training: total number of env time steps
  Uint nEvalEpisodes = 0; // if not training: number of episode to evaluate on

  std::string nStepPappSett = "0";
  std::string appSettings = "";
  std::string setupFolder = "";
  std::string restart = ".";

  bool bTrain = true;
  int logAllSamples = 1;
  bool learnersOnWorkers = true;
  bool forkableApplication = false;
  bool redirectAppStdoutToFile = true;
};

} // end namespace smarties
#endif // smarties_Settings_h
