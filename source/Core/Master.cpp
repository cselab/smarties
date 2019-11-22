//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Master.h"
#include <fstream>
//#include <algorithm>
//#include <chrono>

namespace smarties
{

MasterSockets::MasterSockets(DistributionInfo& D) :
Master<MasterSockets, SOCKET_REQ>(D) { }
MasterMPI::MasterMPI(DistributionInfo& D) :
Master<MasterMPI, MPI_Request>(D) { }

#ifndef NDEBUG
inline static bool isUnfinished(const MPI_Request& req) {
  return req not_eq MPI_REQUEST_NULL;
}
inline static bool isUnfinished(const SOCKET_REQ& req) {
  return req.todo not_eq 0;
}
#endif

MasterSockets::~MasterSockets() {}
MasterMPI::~MasterMPI() {}

template<typename CommType, typename Request_t>
Master<CommType, Request_t>::Master(DistributionInfo&D) : Worker(D) {}

void MasterSockets::run(const environment_callback_t& callback)
{
  assert(distrib.nForkedProcesses2spawn > 0);
  const bool isChild = COMM->forkApplication(callback);
  if(isChild) return;
  Master<MasterSockets, SOCKET_REQ>::run();
}

void MasterMPI::run()
{
  Master<MasterMPI, MPI_Request>::run();
}

template<typename CommType, typename Request_t>
void Master<CommType, Request_t>::run()
{
  synchronizeEnvironments();
  spawnCallsHandlers(); // fills worker_replies threads
  runTraining();
}

template<typename CommType, typename Request_t>
Master<CommType, Request_t>::~Master()
{
  bExit = true;
  for(auto& thread : worker_replies) thread.join();
}

template<typename CommType, typename Request_t>
void Master<CommType, Request_t>::spawnCallsHandlers()
{
  // if workers host learning algos then no need to supply actions
  if(distrib.learnersOnWorkers && distrib.nForkedProcesses2spawn < 1) return;

  #pragma omp parallel num_threads(distrib.nThreads)
  {
    std::vector<Uint> shareWorkers;
    const Uint thrN = omp_get_num_threads();
    const Uint thrID = thrN-1 - omp_get_thread_num(); // thrN-1, thrN-2, ..., 0
    const Uint workerShare = std::ceil(nCallingEnvs / (double) thrN);
    const Uint workerBeg = thrID * workerShare;
    const Uint workerEnd = std::min(nCallingEnvs, (thrID+1)*workerShare);
    for(Uint i=workerBeg; i<workerEnd; ++i) shareWorkers.push_back(i);
    #pragma omp critical
    if (shareWorkers.size())
      worker_replies.push_back (
        std::thread( [&, shareWorkers] () {
          waitForStateActionCallers(shareWorkers); } ) );
  }
}

template<typename CommType, typename Request_t>
void Master<CommType,Request_t>::waitForStateActionCallers(const std::vector<Uint> givenWorkers)
{
  const size_t nClients = givenWorkers.size();
  std::vector<Request_t> reqs(nClients);
  // worker's rank is its index (givenWorkers[i]) plus 1 (master)
  for(size_t i=0; i<nClients; ++i) {
    const Uint callerID = givenWorkers[i]+1;
    const COMM_buffer& B = getCommBuffer( callerID );
    interface()->Irecv(B.dataStateBuf, B.sizeStateMsg, callerID, 78283, reqs[i]);
  }

  const auto sendKillMsgs = [&] (const int clientJustRecvd)
  {
    for(size_t i=0; i<nClients; ++i) {
      if( (int) i == clientJustRecvd ) {
        assert( isUnfinished(reqs[i]) == false );
        continue;
      } else assert( isUnfinished(reqs[i]) );
      interface()->WaitComm(reqs[i]);
    }
    // now all requests are completed and waiting to recv an 'action': terminate
    for(size_t i=0; i<nClients; ++i) {
      const Uint callID = givenWorkers[i], callRank = callID+1;
      const COMM_buffer& B = getCommBuffer(callRank);
      Agent::messageLearnerStatus((char*) B.dataActionBuf) = KILL;
      interface()->Send(B.dataActionBuf, B.sizeActionMsg, callRank, 22846);
    }
  };

  for(size_t i=0; ; ++i) // infinite loop : communicate until break command
  {
    const Uint j = i % nClients, callID = givenWorkers[j], callRank = callID+1;
    // communication handle is rank_of_worker := workerID + 1 (master is 0)
    const int completed = interface()->TestComm(reqs[j]);
    //Learners lock workers if they have enough data to advance step
    while (bTrain && completed && learnersBlockingDataAcquisition()) {
      if(bExit.load()>0) { // exit in case of MPI world reached max num steps
        sendKillMsgs(j);
        return;
      }
      usleep(1); // this is to avoid burning cpus when waiting learners
    }

    if(completed) {
      answerStateAction(callID);
      if(bExit.load()>0) { // exit in case this process reached max num steps
        sendKillMsgs(j);
        return;
      }
      const COMM_buffer& B = getCommBuffer(callRank);
      interface()->Send(B.dataActionBuf,B.sizeActionMsg,callRank,22846);
      interface()->Irecv(B.dataStateBuf,B.sizeStateMsg, callRank,78283,reqs[j]);
    } else {
      usleep(1); // this is to avoid burning cpus when waiting environments
    }
  }
}

}
