//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "DataCoordinator.h"
#include "../Utils/FunctionUtilities.h"
#include <unistd.h>

namespace smarties
{

DataCoordinator::DataCoordinator(MemoryBuffer*const RM, ParameterBlob & P)
  : replay(RM), params(P)
{
  episodes.reserve(distrib.nAgents);

  sharingComm = MPICommDup(distrib.workerless_masters_comm);
  sharingSize = MPICommSize(sharingComm);
  sharingRank = MPICommRank(sharingComm);

  if(distrib.learnersOnWorkers &&
     distrib.nOwnedEnvironments &&
     distrib.master_workers_comm not_eq MPI_COMM_NULL)
  {
    workerComm = MPICommDup(distrib.master_workers_comm);
    // rank>0 (worker) will always send eps to rank==0 (master)
    workerSize = MPICommSize(workerComm);
    workerRank = MPICommRank(workerComm);
    bRunParameterServer = true;
    //warn("Creating communicator to send episodes from workers to learners.");
    if(workerSize < 2) {
      warn("detected no workers in the wrong spot..."); return;
    }
    if(workerRank == 0) {
      if(not distrib.bIsMaster) die("impossible");
    } else {
      if(distrib.bIsMaster) die("impossible");
    }
  } else {
    //then either workerless or sockets or state/act loop instead of param/eps
    workerSize = 0; workerRank = 0; workerComm = MPI_COMM_NULL;
  }

  if(distrib.workerless_masters_comm not_eq MPI_COMM_NULL)
  {
    warn("Creating communicator for learners without workers to recv episodes from learners with workers.");
    bLearnersEpsSharing = true;
    sharingDest = sharingRank; // says that first full episode stays on rank
    sharingReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    sharingSeq = std::vector<Fvec>(sharingSize);
  } else {
    sharingDest = 0; sharingSize = 0; sharingRank = 0; sharingComm = MPI_COMM_NULL;
  }
}

DataCoordinator::~DataCoordinator()
{
  if(sharingComm not_eq MPI_COMM_NULL) MPI_Comm_free(&sharingComm);
  if(workerComm not_eq MPI_COMM_NULL) MPI_Comm_free(&workerComm);
}

void DataCoordinator::setupTasks(TaskQueue& tasks)
{
  allTasksPtr = & tasks;
  if(workerRank > 0) { // then we are worker
    assert(bLearnersEpsSharing == false);
    return;
  }
  if(bRunParameterServer == false && bLearnersEpsSharing == false) return;

  if (sharingSize > 0 || workerSize > 0)
  {
    auto stepDistribEps = [&] () {
      distributePendingEpisodes();
    };
    tasks.add(stepDistribEps);
  }
  if (sharingSize > 0 || workerSize > 0)
  {
    auto stepReceiveEps = [&] () {
      mastersRecvEpisodes();
    };
    tasks.add(stepReceiveEps);
  }
}

void DataCoordinator::distributePendingEpisodes()
{
  std::lock_guard<std::mutex> lockQueue(complete_mutex);
  while ( episodes.size() )
  {
    Sequence & EP = episodes.back();
    if (sharingDest == sharingRank) replay->pushBackSequence(EP);
    else {
      assert(bLearnersEpsSharing);
      const int dest = sharingDest, tag = 737283+MDPID;
      if( sharingReq[dest] not_eq MPI_REQUEST_NULL)
        MPI(Wait, & sharingReq[dest], MPI_STATUS_IGNORE);
      sharingSeq[dest] = EP.packSequence(sI.dimObs(), aI.dim(), aI.dimPol());
      MPI(Isend, sharingSeq[dest].data(), sharingSeq[dest].size(),
          SMARTIES_MPI_Fval, dest, tag, sharingComm, & sharingReq[dest]);
    }
    episodes.pop_back();
    // who's turn is next to receive an episode?
    if(sharingSize>0) sharingDest = (sharingDest+1) % sharingSize; //pick next
  }
}

void DataCoordinator::mastersRecvEpisodes()
{
  MPI_Status status;

  const auto recvEp = [&](const MPI_Comm C, MPI_Status & S)
  {
    int completed = 0;
    const int tag = 737283+MDPID;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, C, &completed, &S);
    if(completed) {
      int count = 0, source = S.MPI_SOURCE;
      MPI(Get_count, & status, SMARTIES_MPI_Fval, & count);
      Fvec EP(count, (Fval)0);
      MPI(Recv, EP.data(), EP.size(), SMARTIES_MPI_Fval, source, tag, C, &S);
      assert(EP.size());
      return EP;
    } else return Fvec();
  };

  if(sharingComm not_eq MPI_COMM_NULL)
  {
    const Fvec sharedEP = recvEp(sharingComm, status);
    if(sharedEP.size()) {
      Sequence tmp(sharedEP, sI.dimObs(), aI.dim(), aI.dimPol());
      replay->pushBackSequence(tmp);
    }
  }
  /*{ // IS THIS NEEDED???
    if( sharingReq[source] not_eq MPI_REQUEST_NULL) {
      int complete = 0;
      MPI(Test, &  sharingReq[source], &complete, MPI_STATUS_IGNORE);
      if(complete) assert(sharingReq[source] == MPI_REQUEST_NULL);
    }
  } */

  assert(allTasksPtr not_eq nullptr);
  // if all learners are locking data acquisition we do not recv eps from worker
  // such that they wait for updated parameters before gathering more data
  if(allTasksPtr->dataAcquisitionIsLocked()) return;

  if(workerComm == MPI_COMM_NULL) return;

  const Fvec workersEP = recvEp(workerComm, status);
  if(workersEP.size())
  {
    const Uint nStep = Sequence::computeTotalEpisodeNstep(
      sI.dimObs(), aI.dim(), aI.dimPol(), workersEP.size() );
    nSeenTransitions_loc += nStep - 1; // we do not count init state
    nSeenSequences_loc   += 1;

    // data sharing among masters:
    if (sharingDest == sharingRank) { // keep the episode
      Sequence tmp(workersEP, sI.dimObs(), aI.dim(), aI.dimPol());
      assert(nStep == tmp.ndata() + 1);
      //_warn("%lu storing new sequence of size %lu", MDPID,tmp->ndata());
      replay->pushBackSequence(tmp);
    } else {                          // send the episode to an other master
      const int dest = sharingDest, tag = 737283+MDPID;
      if( sharingReq[dest] not_eq MPI_REQUEST_NULL)
        MPI(Wait, & sharingReq[dest], MPI_STATUS_IGNORE);
      sharingSeq[dest] = workersEP;
      MPI(Isend, sharingSeq[dest].data(), sharingSeq[dest].size(),
          SMARTIES_MPI_Fval, dest, tag, sharingComm, & sharingReq[dest]);
    }
    if(sharingSize>0) sharingDest = (sharingDest+1) % sharingSize; //pick next

    int bSendParams = 0;
    const int tag = 275727+MDPID, dest = status.MPI_SOURCE;
    MPI(Recv, &bSendParams, 1, MPI_INT, dest, tag, workerComm, &status);

    if(bSendParams) {
      while(allTasksPtr->dataAcquisitionIsLocked()) usleep(1);
      params.send(dest, MDPID);
    }
  }
}

// called externally
void DataCoordinator::addComplete(Sequence& EP, const bool bUpdateParams)
{
  if(bLearnersEpsSharing)
  {
    assert(distrib.bIsMaster);
    std::lock_guard<std::mutex> lock(complete_mutex);
    episodes.emplace_back(std::move(EP));
  }
  else if(bRunParameterServer)
  {
    // if we created data structures for worker to send eps to master
    // this better be a worker!
    assert(workerRank>0 && workerSize>1 && not distrib.bIsMaster);
    const Fvec MSG = EP.packSequence(sI.dimObs(), aI.dim(), aI.dimPol());
    #ifndef NDEBUG
      const Sequence tmp(MSG, sI.dimObs(), aI.dim(), aI.dimPol());
      //_warn("storing new sequence of size %lu", tmp->ndata());
      assert(EP.isEqual(tmp));
    #endif
    EP.clear();

    //in theory this lock is unnecessary because all ops here are locking and
    //master processes one after the other (i.e. other threads will wait)
    //however, in case of non-multiple thread safety, it will cause deadlock
    std::lock_guard<std::mutex> send_ep_lock(complete_mutex);
    MPI(Send, MSG.data(), MSG.size(), SMARTIES_MPI_Fval, 0, 737283+MDPID, workerComm);

    const int intUpdateParams = bUpdateParams? 1 : 0;
    MPI(Send, &intUpdateParams, 1, MPI_INT, 0, 275727+MDPID, workerComm);
    if(bUpdateParams) params.recv(MDPID);
  }
  else // data stays here
  {
    //_warn("%lu stored episode of size %lu", MDPID,EP->ndata() );
    replay->pushBackSequence(EP);
  }
}

}
