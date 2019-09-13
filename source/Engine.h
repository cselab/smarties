//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#ifndef smarties_Engine_h
#define smarties_Engine_h

#include "Communicator.h"


namespace smarties
{

struct DistributionInfo;

#define VISIBLE __attribute__((visibility("default")))

class Engine
{
public:
  VISIBLE Engine(int argc, char** argv);

  // designed for pybind11 interface:
  VISIBLE Engine(std::vector<std::string> args);

  VISIBLE Engine(MPI_Comm initialiazed_mpi_comm, int argc, char** argv);

  VISIBLE ~Engine();

  VISIBLE void run(const std::function<void(Communicator*const,
                                            int, char **      )> & callback);

  VISIBLE void run(const std::function<void(Communicator*const,
                                            MPI_Comm,
                                            int, char **      )> & callback);

  VISIBLE void run(const std::function<void(Communicator*const)> & callback);

  VISIBLE void run(const std::function<void(Communicator*const,
                                            MPI_Comm          )> & callback);

  VISIBLE int parse();

  VISIBLE void setNthreads(const Uint nThreads);

  VISIBLE void setNmasters(const Uint nMasters);

  VISIBLE void setNenvironments(const Uint nEnvironments);

  VISIBLE void setNworkersPerEnvironment(const Uint workerProcessesPerEnv);

  VISIBLE void setRandSeed(const Uint randSeed);

  VISIBLE void setTotNumTimeSteps(const Uint totNumSteps);

  VISIBLE void setSimulationArgumentsFilePath(const std::string& appSettings);

  VISIBLE void setSimulationSetupFolderPath(const std::string& setupFolder);

  VISIBLE void setRestartFolderPath(const std::string& restart);

  VISIBLE void setIsTraining(const bool bTrain);

  VISIBLE void setIsLoggingAllData(const bool logAllSamples);

  VISIBLE void setAreLearnersOnWorkers(const bool learnersOnWorkers);

  VISIBLE void setRedirectAppScreenOutput(const bool redirect = true);

private:
  DistributionInfo * const distrib;

  void init();
};

#undef VISIBLE

} // end namespace smarties
#endif // smarties_Engine_h
