//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ExecutionInfo.h"

#include "../Utils/Warnings.h"
#include "../../extern/CLI.hpp"
#include "HyperParameters.h"

#include <cassert>
#include <unistd.h>

namespace smarties
{

ExecutionInfo::ExecutionInfo(const std::vector<std::string> & args) :
bOwnArgv(true)
{
  argc = (int) args.size();
  argv = new char * [argc+1];
  for(int i=0; i<argc; ++i) {
    argv[i] = new char[args[i].size() + 1];
    std::copy(args[i].begin(), args[i].end(), argv[i]);
    argv[i][args[i].size()] = '\0';
  }
  argv[argc] = nullptr;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, & threadSafety);
  world_comm = MPI_COMM_WORLD;
  MPI_Comm_set_errhandler(world_comm, MPI_ERRORS_RETURN);
  commonInit();
}

ExecutionInfo::ExecutionInfo(int _argc, char ** _argv) :
  bOwnArgv(false), argc(_argc), argv(_argv)
{
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, & threadSafety);
  world_comm = MPI_COMM_WORLD;
  MPI_Comm_set_errhandler(world_comm, MPI_ERRORS_RETURN);
  commonInit();
}

ExecutionInfo::ExecutionInfo(const MPI_Comm & initialiazed_mpi_comm,
                                   int _argc, char ** _argv) :
  bOwnArgv(false), argc(_argc), argv(_argv)
{
  world_comm = initialiazed_mpi_comm;
  MPI_Query_thread(& threadSafety);
  commonInit();
}

void ExecutionInfo::commonInit()
{
  Warnings::init_warnings();
  getcwd(initial_runDir, 1024);
  #ifdef REQUIRE_MPI_MULTIPLE
  if (threadSafety < MPI_THREAD_MULTIPLE)
  #else
  if (threadSafety < MPI_THREAD_SERIALIZED)
  #endif
    die("The MPI implementation does not have required thread support");
  // this value will determine if we can use asynchronous mpi calls:
  bAsyncMPI = threadSafety >= MPI_THREAD_MULTIPLE;
  world_size = MPICommSize(world_comm);
  world_rank = MPICommRank(world_comm);

  if (not bAsyncMPI and world_rank == 0)
    printf("MPI implementation does not support MULTIPLE thread safety!\n");
  nThreads = omp_get_max_threads();
}

ExecutionInfo::~ExecutionInfo()
{
  if(bOwnArgv) {
    for(int i=0; i<argc; ++i) delete [] argv[i];
    delete [] argv;
  }
  if(MPI_COMM_NULL not_eq     master_workers_comm)
          MPI_Comm_free(&     master_workers_comm);
  if(MPI_COMM_NULL not_eq workerless_masters_comm)
          MPI_Comm_free(& workerless_masters_comm);
  if(MPI_COMM_NULL not_eq     learners_train_comm)
          MPI_Comm_free(&     learners_train_comm);
  if(MPI_COMM_NULL not_eq    environment_app_comm)
          MPI_Comm_free(&    environment_app_comm);
  MPI_Finalize();
}

int ExecutionInfo::parse()
{
  CLI::App parser("smarties : distributed reinforcement learning framework");
  parser.allow_extras();

  parser.add_option("--nThreads", nThreads,
    "Number of threads from threaded training on each master rank."
  );
  parser.add_option("--nMasters", nMasters,
    "Number of master ranks (policy-updating ranks)."
  );
  parser.add_option("--nEnvironments", nEnvironments,
    "Number of environment processes (not necessarily ranks, may be forked)."
  );
  parser.add_option("--workerProcessesPerEnv", workerProcessesPerEnv,
    "Number of MPI ranks required by the the env application. It is 1 for "
    "serial/shared-memory solvers."
  );

  parser.add_option("--nTrainSteps", nTrainSteps,
    "Total number of time steps before end of training."
  );
  parser.add_option("--nEvalEpisodes", nEvalEpisodes,
    "Total number of episodes to evaluate training policy. "
    "If >0, training is DISABLED and network parameters frozen."
  );

  parser.add_option("--randSeed", randSeed, "Random seed." );

  parser.add_option("--nStepPappSett", nStepPappSett,
    "Number of time steps per appSettings file to use. Must be a list of "
    "positive numbers separated by semicolons. Last number will be "
    "overwritten to 0; i.e. last appSettings will be used til termination."
  );
  parser.add_option("--appSettings", appSettings,
    "Name of file containing the command line arguments for user's application."
  );
  parser.add_option("--setupFolder", setupFolder,
    "The contents of this folder are copied over into the folder where the "
    "simulation is run. It can contain additional files needed to set up "
    "the simulation such as settings files, configuration files..."
  );
  parser.add_option("--restart", restart,
    "Prefix of net save files. If 'none' then no restart."
  );

  parser.add_option("--learnersOnWorkers", learnersOnWorkers,
    "Whether to enable hosting learning algos on worker processes such that "
    "workers send training data and recv parameters from masters. If false "
    "workers only send states and recv actions from masters."
  );

  parser.add_option("--logAllSamples", logAllSamples,
    "Whether to write files recording all transitions."
  );
  parser.add_option("--redirectAppStdoutToFile", redirectAppStdoutToFile,
    "Whether to hide the screen output of the environment simulations from "
    "the terminal and print it to file."
  );

  try {
    parser.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    if(world_rank == 0) {
      parser.exit(e);
      const std::string jsonArgs = HyperParameters::printArgComments();
      printf("\nAlgorithm-specific arguments read from .json files:\n%s\n",
        jsonArgs.c_str());
      return 1;
    }
    else return 1;
  }
  MPI_Barrier(world_comm);
  return 0;
}

inline Uint notRoundedSplitting(const Uint nSplitters,
                                const Uint nToSplit,
                                const Uint splitterRank)
{
  const Uint nPerSplitter = std::ceil( nToSplit / (Real) nSplitters );
  const Uint splitBeg = std::min( splitterRank    * nPerSplitter, nToSplit);
  const Uint splitEnd = std::min((splitterRank+1) * nPerSplitter, nToSplit);
  return splitEnd - splitBeg;
}

inline Uint indxStripedMPISplitting(const Uint nSplitters,
                                    const Uint nToSplit,
                                    const Uint indexedRank)
{
  assert(indexedRank < nSplitters + nToSplit);
  for(Uint i=0, countIndex=0; i<nSplitters; ++i) {
    const Uint nInGroup = notRoundedSplitting(nSplitters, nToSplit, i);
    countIndex += nInGroup+1; // nInGroup resources + 1 handler
    if(indexedRank < countIndex) return i;
  }
  assert(false && "logic error"); return 0;
}

inline Uint rankStripedMPISplitting(const Uint nSplitters,
                                    const Uint nToSplit,
                                    const Uint indexedRank)
{
  assert(indexedRank < nSplitters + nToSplit);
  for(Uint i=0, countIndex=0; i<nSplitters; ++i) {
    const Uint nInGroup = notRoundedSplitting(nSplitters, nToSplit, i);
    if(indexedRank < countIndex + nInGroup+1)
      return indexedRank - countIndex;
    countIndex += nInGroup+1; // nInGroup resources + 1 handler
  }
  assert(false && "logic error"); return 0;
}

void ExecutionInfo::figureOutWorkersPattern()
{
  nWorkers = world_size - nMasters;
  bool bThereAreMasters = nMasters > 0;
  bool bThereAreWorkerProcesses = nWorkers > 0;
  //if(bThereAreWorkerProcesses) warn("there are worker processes");
  //else warn("there are no worker processes");
  //if(bThereAreMasters) warn("there are master processes");
  //else warn("there are no master processes");
  if(not forkableApplication && not bThereAreWorkerProcesses) {
    die("There are no processes and application needs dedicated processes. "
        "Run with more mpi processes.");
  }
  if(    forkableApplication && not bThereAreWorkerProcesses) {
    if(world_rank == 0)
      printf("Master processes to communicate via sockets.");
  }
  if(    forkableApplication &&     bThereAreWorkerProcesses) {
    nEnvironments = std::ceil(nEnvironments / (Real) nWorkers) * nWorkers;
    if(world_rank == 0)
      printf("%lu worker ranks will split %lu simulation processes.",
            nWorkers, nEnvironments);
  }
  if(not forkableApplication &&     bThereAreWorkerProcesses) {
    workerProcessesPerEnv = std::max(workerProcessesPerEnv, (Uint) 1);
    if(nWorkers not_eq nEnvironments * workerProcessesPerEnv)
      printf("%lu workers run one environment process each.", nWorkers);
    if(nWorkers % workerProcessesPerEnv not_eq 0)
      die("Mismatch between worker processes and number of ranks requested to run env application.");
    nEnvironments = nWorkers / workerProcessesPerEnv;
  }

  // the rest of this method will define (or leave untouched) these entities:
  // 1) will this process run a master (in a master-worker pattern) process
  bIsMaster = false;
  // 2) for how many environments will this process have to compute actions
  //    (principally affects how many Agents and associated memory buffers
  //     are allocated)
  nOwnedEnvironments = 0;
  // 3) does this process need to fork to create child processes which will
  //    in turn run the environment application (communication over sockets)
  nForkedProcesses2spawn = 0;
  // 4) mpi communicator to send state/actions or data/parameters from a process
  //    hosting the learning algo and other proc. handling data collection
  master_workers_comm = MPI_COMM_NULL;
  // 5) mpi communicator shared by all ranks that host the learning algorithms
  //    and perform the actual parameter update steps
  learners_train_comm = MPI_COMM_NULL;
  // 6) mpi communicator given to a group of worker processes that should pool
  //    together to run an environment app which requires distributed computing
  environment_app_comm = MPI_COMM_SELF;
  // 7) mpi communicator for masters without direct link to a worker to recv
  //    training data from other masters
  workerless_masters_comm = MPI_COMM_NULL;
  // 8) tag of group of mpi ranks running a common distributed environment
  thisWorkerGroupID = -1;


  if(bThereAreMasters)
  {
    if(bThereAreWorkerProcesses)
    {
      // then masters talk to workers, and workers own environments
      // what is the size of the mpi communicator where we have workers?
      bIsMaster = rankStripedMPISplitting(nMasters, nWorkers, world_rank) == 0;
      Uint commWorkID = indxStripedMPISplitting(nMasters, nWorkers, world_rank);

      //if(fakeMastersRanks) { // overwrite splitting if we have only fake masters
      //  bIsMaster = world_rank < nMasters;
      //  masterWorkerCommID = 0;
      //}

      MPI_Comm_split(world_comm, bIsMaster,  world_rank, & learners_train_comm);
      MPI_Comm_split(world_comm, commWorkID, world_rank, & master_workers_comm);
      _debug("Process %lu is a %s part of comm %lu.\n",
             world_rank, bIsMaster? "master" : "worker", commWorkID);

      if(bIsMaster)
      {
        nOwnedEnvironments = MPICommSize(master_workers_comm) - 1;
        _debug("master %lu owns %lu environments", world_rank, nOwnedEnvironments);
        if(nWorkers < nMasters)
             workerless_masters_comm = MPICommDup(learners_train_comm);
        else workerless_masters_comm = MPI_COMM_NULL;

        if(workerProcessesPerEnv >0) { // unblock creation of env's mpi comm
          MPI_Comm dummy; // no need to free this
          MPI_Comm_split(master_workers_comm, MPI_UNDEFINED, 0, &dummy);
        }
      }
      else // is worker
      {
        const Uint totalWorkRank = MPICommRank(learners_train_comm);
        assert(MPICommSize(learners_train_comm) == nWorkers);
        MPI_Comm_free(& learners_train_comm);
        learners_train_comm = MPI_COMM_NULL;
        nOwnedEnvironments = notRoundedSplitting(nWorkers,
                                                 nEnvironments,
                                                 totalWorkRank);
        const Uint innerWorkRank = MPICommRank(master_workers_comm);
        const Uint innerWorkSize = MPICommSize(master_workers_comm);
        assert(nOwnedEnvironments==1 && innerWorkRank>0 && innerWorkSize>1);

        if(workerProcessesPerEnv > 0)
        {
          if( (innerWorkSize-1) % workerProcessesPerEnv not_eq 0)
            _die("Number of worker ranks per master (%u) must be a multiple of "
            "the nr. of ranks that the environment app requires to run (%u).\n",
            innerWorkSize-1, workerProcessesPerEnv);

          thisWorkerGroupID = (innerWorkRank-1) / workerProcessesPerEnv;
          MPI_Comm_split(master_workers_comm, thisWorkerGroupID, innerWorkRank,
                         &environment_app_comm);
        } else {
          thisWorkerGroupID = 0;
          nForkedProcesses2spawn = nOwnedEnvironments;
        }

        _debug("worker %lu owns %lu environments, has rank %lu out of %lu. "
               "worker ID inside group %d.", world_rank, nOwnedEnvironments,
               innerWorkRank, innerWorkSize, thisWorkerGroupID);
      }
    }
    else // there are no worker processes
    {
      bIsMaster = true;
      nOwnedEnvironments = notRoundedSplitting(nMasters, nEnvironments, world_rank);
      // should also equal:
      // nWorkers/world_size + ( (nWorkers%world_size) > world_rank );
      nForkedProcesses2spawn = nOwnedEnvironments;
      learners_train_comm = world_comm;
      if(nEnvironments < nMasters) // then i need to share data
           workerless_masters_comm = MPICommDup(learners_train_comm);
      else workerless_masters_comm = MPI_COMM_NULL;
    }
  }
  else // there are no masters : workers alternate environment and learner
  {
    // only have workers, 2 cases either evaluating a policy or alternate
    // data gathering and learning algorithm iteration on same comp resources
    bIsMaster = false;
    learnersOnWorkers = true;
    if(nWorkers <= 0) die("Error in computation of world_size");
    if(nEnvironments not_eq nWorkers)
      die("Detected 0 masters : this only works if each worker "
          "also serially runs its own environment.");
    nOwnedEnvironments  = 1;
    learners_train_comm = world_comm;
    // all are workers implies all have data:
    workerless_masters_comm = MPI_COMM_NULL;

    const Uint totalWorkRank = MPICommRank(learners_train_comm);
    const Uint totalWorkSize = MPICommSize(learners_train_comm);
    if( (totalWorkSize-1) % workerProcessesPerEnv not_eq 0) {
      _die("Number of worker ranks per master (%u) must be a multiple of "
      "the nr. of ranks that the environment app requires to run (%u).\n",
      totalWorkSize-1, workerProcessesPerEnv);
    }
    thisWorkerGroupID = (totalWorkRank-1) / workerProcessesPerEnv;
    MPI_Comm_split(learners_train_comm, thisWorkerGroupID, totalWorkRank,
                   &environment_app_comm);
  }
}

void ExecutionInfo::initialze()
{
  if (nEvalEpisodes>0) bTrain = 0;
  else                 bTrain = 1;

  if(nThreads<1) die("nThreads<1");
  if(randSeed<=0) {
    std::random_device rdev; randSeed = rdev();
    MPI_Bcast(&randSeed, 1, MPI_UNSIGNED_LONG, 0, world_comm);
    if(world_rank==0) printf("Using seed %lu\n", randSeed);
  }
  randSeed += world_rank;

  generators.resize(0);
  generators.reserve(omp_get_max_threads());
  generators.push_back(std::mt19937(randSeed));
  for(int i=1; i<omp_get_max_threads(); ++i)
    generators.push_back( std::mt19937( generators[0]() ) );
}

}
