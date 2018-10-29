//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learners/AllLearners.h"
#include "Utils/Scheduler.h"
#include "Utils/ObjectFactory.h"
using namespace std;

void runClient();
void runWorker(Settings& S);
void runMaster(Settings& S);

void runWorker(Settings& S)
{
  assert(S.workers_rank and S.workers_size>0);
  ObjectFactory factory(S);
  Environment* env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator();

  Worker simulation(&comm, env, S);
  simulation.run();
}

void runMaster(Settings& S)
{
  S.check();

  #ifdef INTERNALAPP //unblock creation of app comm if needed
    if(S.bSpawnApp) die("Unsuppored, create dedicated workers");
    MPI_Comm tmp_com;
    MPI_Comm_split(S.workersComm, MPI_UNDEFINED, 0, &tmp_com);
  #endif

  ObjectFactory factory(S);
  Environment*const env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator();

  S.finalizeSeeds(); // now i know nAgents, might need more generators

  const Uint nPols = S.bSharedPol ? 1 : env->nAgentsPerRank;
  vector<Learner*> learners(nPols, nullptr);
  for(Uint i = 0; i<nPols; i++) {
    stringstream ss; ss<<"agent_"<<std::setw(2)<<std::setfill('0')<<i;
    if(S.world_rank == 0) cout << "Learner: " << ss.str() << endl;
    learners[i] = createLearner(env, S);
    learners[i]->setLearnerName(ss.str() +"_", i);
    learners[i]->restart();
  }

  fflush(stdout); fflush(stderr); fflush(0);
  MPI_Barrier(S.mastersComm); // to avoid garbled output during run
  Master master(&comm, learners, env, S);

  master.run();
  comm.sendTerminateReq();
}

int main (int argc, char** argv)
{
  Settings S;
  vector<ArgParser::OptionStruct> opts = S.initializeOpts();

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &S.threadSafety);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  if (S.threadSafety < MPI_THREAD_SERIALIZED)
    die("The MPI implementation does not have required thread support");
  S.bAsync = S.threadSafety>=MPI_THREAD_MULTIPLE;
  MPI_Comm_rank(MPI_COMM_WORLD, &S.world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &S.world_size);
  if (! S.bAsync and S.world_rank == 0)
    std::cout << "MPI implementation does not support MULTIPLE thread safety!"<<std::endl;
  omp_set_dynamic(0);
  #pragma omp parallel
  {
    int cpu_num; GETCPU(cpu_num); //sched_getcpu()
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    //#ifndef NDEBUG
      printf("Rank %d Thread %3d  is running on CPU %3d of hose %s\n",
            S.world_rank, omp_get_thread_num(), cpu_num, hostname);
    //#endif
  }

  ArgParser::Parser parser(opts);
  parser.parse(argc, argv, S.world_rank == 0);
  MPI_Barrier(MPI_COMM_WORLD);

  if(not S.isServer) die("client.sh scripts are no longer supported");

  S.initRandomSeed();

  if(S.nMasters == S.world_size)
  {
    S.bSpawnApp = S.nWorkers > S.world_rank;
    S.mastersComm = MPI_COMM_WORLD;
    S.workersComm = MPI_COMM_NULL;
    S.workers_rank = 0;
    S.workers_size = 1;
    S.nWorkers_own =
      S.nWorkers/S.world_size + ( (S.nWorkers%S.world_size) > S.world_rank );
    runMaster(S);
  }
  else
  {
    if(S.world_size not_eq S.nMasters+S.nWorkers) die(" ");
    const int learGroupSize = std::ceil( S.world_size / (Real) S.nMasters );
    const bool bIsMaster = ( S.world_rank % learGroupSize ) == 0;
    const int workerCommInd = S.world_rank / learGroupSize;
    MPI_Comm_split(MPI_COMM_WORLD, bIsMaster,     S.world_rank, &S.mastersComm);
    MPI_Comm_split(MPI_COMM_WORLD, workerCommInd, S.world_rank, &S.workersComm);
    if(not bIsMaster) {
      MPI_Comm_free(&S.mastersComm);
      S.mastersComm = MPI_COMM_NULL;
      S.bSpawnApp = 1;
    }
    printf("Process %d is a %s part of comm %d.\n",
        S.world_rank, bIsMaster? "master" : "worker", workerCommInd);

    MPI_Comm_rank(S.workersComm, &S.workers_rank);
    MPI_Comm_size(S.workersComm, &S.workers_size);
    S.nWorkers_own = bIsMaster? S.workers_size - 1 : 1;

    MPI_Barrier(MPI_COMM_WORLD);
    if (bIsMaster) {
      runMaster(S);
      MPI_Comm_free(&S.mastersComm);
    }
    else runWorker(S);
    MPI_Comm_free(&S.workersComm);
  }

  MPI_Finalize();
  return 0;
}
