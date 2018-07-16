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
void runWorker(Settings& settings, MPI_Comm workersComm);
void runMaster(Settings& settings, MPI_Comm workersComm, MPI_Comm mastersComm);

void runWorker(Settings& settings, MPI_Comm workersComm)
{
  MPI_Comm_rank(workersComm, &settings.workers_rank);
  MPI_Comm_size(workersComm, &settings.workers_size);
  if(settings.workers_rank==0) die("Worker is master?\n");
  if(settings.workers_size<=1) die("Worker has no master?\n");
  settings.nWorkers = 1;
  ObjectFactory factory(settings);
  Environment* env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator(workersComm, settings.sockPrefix, true);

  Worker simulation(&comm, env, settings);
  simulation.run();
}

void runMaster(Settings& settings, MPI_Comm workersComm, MPI_Comm mastersComm)
{
  settings.mastersComm =  mastersComm;
  MPI_Comm_rank(workersComm, &settings.workers_rank);
  MPI_Comm_size(workersComm, &settings.workers_size);
  MPI_Comm_rank(mastersComm, &settings.learner_rank);
  MPI_Comm_size(mastersComm, &settings.learner_size);
  settings.nWorkers = settings.workers_size-1; //minus master
  assert(settings.nWorkers>=0 && settings.workers_rank == 0);

  #ifdef INTERNALAPP //unblock creation of app comm if needed
    MPI_Comm tmp_com;
    MPI_Comm_split(workersComm, MPI_UNDEFINED, 0, &tmp_com);
    //no need to free this
  #endif

  ObjectFactory factory(settings);
  Environment*const env = factory.createEnvironment();
  Communicator comm = env->create_communicator(workersComm, settings.sockPrefix, true);

  settings.finalizeSeeds(); // now i know nAgents, might need more generators
  const Real nLearners = settings.learner_size;
  // each learner computes a fraction of the batch:
  settings.batchSize    = std::ceil(settings.batchSize    / nLearners);
  // every grad step, each learner performs a fraction of the time steps:
  settings.obsPerStep   = std::ceil(settings.obsPerStep   / nLearners);
  // each learner contains a fraction of the memory buffer:
  settings.minTotObsNum = std::ceil(settings.minTotObsNum / nLearners);
  settings.maxTotObsNum = std::ceil(settings.maxTotObsNum / nLearners);

  const Uint nPols = settings.bSharedPol ? 1 : env->nAgentsPerRank;
  vector<Learner*> learners(nPols, nullptr);
  for(Uint i = 0; i<nPols; i++) {
    stringstream ss; ss<<"agent_"<<std::setw(2)<<std::setfill('0')<<i;
    cout << "Learner: " << ss.str() << endl;
    learners[i] = createLearner(env, settings);
    learners[i]->setLearnerName(ss.str() +"_", i);
    learners[i]->restart();
  }
  //#pragma omp parallel
  //printf("Rank %d Thread %3d is running on CPU %3d\n",
  //  settings.world_rank, omp_get_thread_num(), sched_getcpu());

  fflush(0);
  Master master(workersComm, learners, env, settings);
  MPI_Barrier(mastersComm); // to avoid garbled output during run

  #if 0
  if (!settings.nWorkers && !learner->nData())
  {
    printf("No workers, just dumping the policy\n");
    learner->dumpPolicy();
    abort();
  }
  #endif

  master.run();
  master.sendTerminateReq();
}

int main (int argc, char** argv)
{
  Settings settings;
  vector<ArgParser::OptionStruct> opts = settings.initializeOpts();

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &settings.threadSafety);
  if (settings.threadSafety < MPI_THREAD_SERIALIZED)
    die("The MPI implementation does not have required thread support");

  MPI_Comm_rank(MPI_COMM_WORLD, &settings.world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &settings.world_size);
  omp_set_dynamic(0);

  ArgParser::Parser parser(opts);
  parser.parse(argc, argv, settings.world_rank == 0);
  settings.check();
  MPI_Barrier(MPI_COMM_WORLD);

  if (not settings.isServer) {
    die("You should not be running the client.sh scripts");
    /*
    if (settings.sockPrefix<0)
      die("Not received a prefix for the socket\n");
    settings.generators.push_back(mt19937(settings.sockPrefix));
    printf("Launching smarties as client.\n");
    if (settings.restart == "none")
      die("smarties as client works only for evaluating policies.\n");
    settings.bTrain = 0;
    runClient();
    MPI_Finalize();
    return 0;
    */
  }

  settings.initRandomSeed();

  if(settings.world_size%settings.nMasters)
    die("Number of masters not compatible with available ranks.");
  const int workersPerMaster = settings.world_size/settings.nMasters - 1;

  MPI_Comm workersComm; //this communicator allows workers to talk to their master
  MPI_Comm mastersComm; //this communicator allows masters to talk among themselves

  int bIsMaster, workerCommInd;
  // two options: either multiple learners because they are the bottleneck
  //              or multiple workers for single master because data is expensive
  // in the second case, rank 0 will be master either away
  // in first case our objective is to maximise the spread of master ranks
  // and use processes on hyperthreaded cores to run the workers
  // in a multi socket board usually the cpus are going to be sorted
  // as (socket-core-thread): 0-0-0 0-1-0 0-2-0 ... 1-0-0 1-1-0 1-2-0 ...
  //                          0-0-1 0-1-1 0-2-1 ... 1-0-1 1-1-1 1-2-1
  // therefore if there are more than one master per node sorting changes
  // this is al very brittle. relies on my MPI implementations sorting of ranks
  if (settings.ppn > workersPerMaster+1) {
    if(settings.ppn % (workersPerMaster+1)) die("Bad number of proc per node");
    const int nMastersPerNode =  settings.ppn / (workersPerMaster+1);
    const int nodeIndx = settings.world_rank / settings.ppn;
    const int nodeRank = settings.world_rank % settings.ppn;
    // will be 1 for the first nMastersPerNode ranks of each node:
    bIsMaster = nodeRank / nMastersPerNode == 0;
    // index will be shared by every nMastersPerNode ranks:
    const int nodeMScomm = nodeRank % nMastersPerNode;
    // split communicators residing on different nodes:
    workerCommInd = nodeMScomm + nodeIndx * nMastersPerNode;
  } else {
    bIsMaster = settings.world_rank % (workersPerMaster+1) == 0;
    workerCommInd = settings.world_rank / (workersPerMaster+1);
  }

  MPI_Comm_split(MPI_COMM_WORLD, bIsMaster, settings.world_rank, &mastersComm);
  MPI_Comm_split(MPI_COMM_WORLD, workerCommInd, settings.world_rank,&workersComm);
  if (!bIsMaster) MPI_Comm_free(&mastersComm);
  printf("nRanks=%d, %d masters, %d workers per master. I'm %d: %s part of comm %d.\n",
      settings.world_size,settings.nMasters,workersPerMaster,settings.world_rank,
      bIsMaster?"master":"worker",workerCommInd);

  MPI_Barrier(MPI_COMM_WORLD);
  if (bIsMaster) runMaster(settings, workersComm, mastersComm);
  else           runWorker(settings, workersComm);

  if (bIsMaster) MPI_Comm_free(&mastersComm);
  MPI_Comm_free(&workersComm);
  MPI_Finalize();
  return 0;
}


/*
void runClient()
{
  settings.nWorkers = 1;
  ObjectFactory factory(settings);
  Environment* env = factory.createEnvironment();
  Communicator comm = env->create_communicator(MPI_COMM_NULL, settings.sockPrefix, false);

  Learner* learner = createLearner(MPI_COMM_WORLD, env, settings);
  if (settings.restart != "none") {
    learner->restart(settings.restart);
    //comm.restart(settings.restart);
  }
  Client simulation(learner, &comm, env, settings);
  simulation.run();
}
*/
