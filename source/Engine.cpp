//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Engine.h"
#include "Core/Master.h"

namespace smarties
{

Engine::Engine(int argc, char** argv) :
  distrib(new DistributionInfo(argc, argv)) { }

Engine::Engine(std::vector<std::string> args) :
  distrib(new DistributionInfo(args)) { }

Engine::Engine(MPI_Comm world, int argc, char** argv) :
  distrib(new DistributionInfo(world, argc, argv)) {}

Engine::~Engine() {
  assert(distrib not_eq nullptr);
  delete distrib;
}

int Engine::parse() {
  return distrib->parse();
}

void Engine::setNthreads(const Uint nThreads) {
  distrib->nThreads = nThreads;
}

void Engine::setNmasters(const Uint nMasters) {
  distrib->nMasters = nMasters;
}

void Engine::setNenvironments(const Uint nEnvironments) {
  distrib->nEnvironments = nEnvironments;
}

void Engine::setNworkersPerEnvironment(const Uint workerProcessesPerEnv) {
  distrib->workerProcessesPerEnv = workerProcessesPerEnv;
}

void Engine::setRandSeed(const Uint randSeed) {
  distrib->randSeed = randSeed;
}

void Engine::setTotNumTimeSteps(const Uint totNumSteps) {
  distrib->totNumSteps = totNumSteps;
}

void Engine::setSimulationArgumentsFilePath(const std::string& appSettings) {
  distrib->appSettings = appSettings;
}

void Engine::setSimulationSetupFolderPath(const std::string& setupFolder) {
  distrib->setupFolder = setupFolder;
}

void Engine::setRestartFolderPath(const std::string& restart) {
  distrib->restart = restart;
}

void Engine::setIsTraining(const bool bTrain) {
  distrib->bTrain = bTrain;
}

void Engine::setIsLoggingAllData(const bool logAllSamples) {
  distrib->logAllSamples = logAllSamples;
}

void Engine::setAreLearnersOnWorkers(const bool learnersOnWorkers) {
  distrib->learnersOnWorkers = learnersOnWorkers;
}

void Engine::setRedirectAppScreenOutput(const bool redirect) {
  distrib->redirectAppStdoutToFile = redirect;
}

void Engine::init()
{
  distrib->initialzePRNG();
  distrib->figureOutWorkersPattern();

  if(distrib->bTrain == false && distrib->restart == "none") {
   printf("Did not specify path for restart files, assumed current dir.\n");
   distrib->restart = ".";
  }

  MPI_Barrier(distrib->world_comm);
}

void Engine::run(const std::function<void(Communicator*const)> & callback)
{
  assert(distrib->workerProcessesPerEnv == 1);

  const environment_callback_t fullcallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc);
  };

  run(fullcallback);
}

void Engine::run(const std::function<void(Communicator*const,
                                          int, char **      )> & callback)
{
  assert(distrib->workerProcessesPerEnv == 1);

  const environment_callback_t fullcallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc, argc, argv);
  };

  run(fullcallback);
}

void Engine::run(const std::function<void(Communicator*const,
                                          MPI_Comm          )> & callback)
{
  const environment_callback_t fullcallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc, mc);
  };

  run(fullcallback);
}

void Engine::run(const std::function<void(Communicator*const,
                                          MPI_Comm,
                                          int, char **      )> & callback)
{
  distrib->forkableApplication = distrib->workerProcessesPerEnv == 1;
  init();
  if(distrib->bIsMaster)
  {
    if(distrib->nForkedProcesses2spawn > 0) {
      MasterSockets process(*distrib);
      process.run(callback);
    } else {
      MasterMPI process(*distrib);
      process.run();
    }
  }
  else
  {
    Worker process(*distrib);
    process.run(callback);
  }
}

}
