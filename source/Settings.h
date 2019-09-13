//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Settings_h
#define smarties_Settings_h

#include "Utils/Definitions.h"
#include "Utils/MPIUtilities.h"

#include <random>
#include <mutex>

namespace smarties
{

struct DistributionInfo
{
  DistributionInfo(int _argc, char ** _argv);
  DistributionInfo(const std::vector<std::string> & args);
  DistributionInfo(const MPI_Comm& mpi_comm, int _argc, char ** _argv);
  ~DistributionInfo();

  const bool bOwnArgv; // whether argv needs to be deallocated
  int argc;
  char ** argv;

  void commonInit();
  int parse();

  void initialzePRNG();
  void finalizePRNG(const Uint nAgents_local);
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
  Uint totNumSteps = 10000000; // total number of env time steps

  std::string nStepPappSett = "0";
  std::string appSettings = "";
  std::string setupFolder = "";
  std::string restart = ".";

  bool bTrain = true;
  bool logAllSamples = true;
  bool learnersOnWorkers = true;
  bool forkableApplication = false;
  bool redirectAppStdoutToFile = true;
};

struct Settings
{
  Settings();
  void check();
  static std::string printArgComments();
  void initializeOpts(std::ifstream & , DistributionInfo & );
  void defineDistributedLearning(DistributionInfo&);

  //////////////////////////////////////////////////////////////////////////////
  //SETTINGS PERTAINING TO LEARNING ALGORITHM
  //////////////////////////////////////////////////////////////////////////////
  std::string learner = "VRACER";
  std::string ERoldSeqFilter = "default";
  std::string dataSamplingAlgo = "uniform";

  Real explNoise = std::sqrt(0.2);
  Real gamma = 0.995;
  Real lambda = 0.95;
  Real obsPerStep = 1;
  Real clipImpWeight = 4;
  Real penalTol = 0.1;
  Real klDivConstraint = 0.01;
  Real targetDelay = 0;
  Real epsAnneal = 0;

  Uint minTotObsNum =  65536;
  Uint maxTotObsNum = 262144;
  Uint saveFreq = 200000;

  //////////////////////////////////////////////////////////////////////////////
  //SETTINGS PERTAINING TO NETWORK
  //////////////////////////////////////////////////////////////////////////////

  std::vector<Uint> encoderLayerSizes = { 0 };
  std::vector<Uint> nnLayerSizes = { 128, 128 };

  Uint batchSize = 256;
  Uint ESpopSize = 1;
  Uint nnBPTTseq = 16;

  Real nnLambda = std::numeric_limits<float>::epsilon();
  Real learnrate = 1e-4;
  Real outWeightsPrefac = 0.1;

  std::string nnOutputFunc = "Linear";
  std::string nnFunc = "SoftSign";
  std::string nnType = "FFNN";

  ///////////////////////////////////////////////////////////////////////////////
  //SETTINGS THAT ARE NOT READ FROM FILE
  ///////////////////////////////////////////////////////////////////////////////
  // rank-local data-acquisition goals:
  Uint batchSize_local = 0;
  Real obsPerStep_local = 0;
  Uint minTotObsNum_local = 0;
  Uint maxTotObsNum_local = 0;
  // whether Recurrent network (figured out in main)
  bool bRecurrent = false;
  // whether sampling minibatch of episodes or of timesteps, determined by algo
  bool bSampleSequences = false;
};

} // end namespace smarties
#endif // smarties_Settings_h
