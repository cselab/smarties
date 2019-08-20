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
  completed.reserve(distrib.nAgents);

  // if all masters have socketed workers, no need for the coordinator
  //if(distrib.workerless_masters_comm == MPI_COMM_NULL &&
  //   distrib.learnersOnWorkers == false) return;

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
      workerRecvSizeReq = std::vector<MPI_Request>(workerSize);
      workerRecvSeqSize = std::vector<unsigned long>(workerSize);
      workerReqParamFlag= std::vector<unsigned long>(workerSize);
      workerReqParamReq = std::vector<MPI_Request>(workerSize);
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
    sharingTurn = sharingRank; // says that first full episode stays on rank
    shareSendSizeReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareRecvSizeReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareSendSeqReq  = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareSendSeqSize = std::vector<unsigned long>(sharingSize);
    shareRecvSeqSize = std::vector<unsigned long>(sharingSize);
    shareSendSeq = std::vector<Fvec>(sharingSize);
  } else {
    sharingTurn = 0; sharingSize = 0; sharingRank = 0; sharingComm = MPI_COMM_NULL;
  }
}

DataCoordinator::~DataCoordinator()
{
  if(sharingComm not_eq MPI_COMM_NULL) MPI_Comm_free(&sharingComm);
  if(workerComm not_eq MPI_COMM_NULL) MPI_Comm_free(&workerComm);
  for(auto & S : completed) Utilities::dispose_object(S);
}

void DataCoordinator::setupTasks(TaskQueue& tasks)
{
  allTasksPtr = & tasks;
  if(workerRank > 0) { // then we are worker
    assert(bLearnersEpsSharing == false);
    return;
  }
  if(bRunParameterServer == false && bLearnersEpsSharing == false) return;

  //////////////////////////////////////////////////////////////////////
  // Waiting for workers to request parameters
  /////////////////////////////////////////////////////////////////////
  assert(workerReqParamFlag.size() == workerSize);
  assert(workerReqParamReq .size() == workerSize);
  //for(Uint i=1; i<workerSize; ++i)
  //  MPI(Irecv, & workerReqParamFlag[i], 1, MPI_UNSIGNED_LONG, i, 89,
  //             workerComm, & workerReqParamReq[i]);

  //////////////////////////////////////////////////////////////////////
  // Waiting for workers to send episodes
  /////////////////////////////////////////////////////////////////////
  for(Uint i=1; i<workerSize; ++i)
    IrecvSize(workerRecvSeqSize[i], i, workerComm, workerRecvSizeReq[i]);

  //////////////////////////////////////////////////////////////////////
  // Waiting for other masters to share episodes
  /////////////////////////////////////////////////////////////////////
  for(Uint i=0; i<sharingSize; ++i)
    if(i not_eq sharingRank)
      IrecvSize(shareRecvSeqSize[i], i, sharingComm, shareRecvSizeReq[i]);

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
  while ( completed.size() )
  {
    Sequence* const EP = completed.back();
    if (sharingTurn == sharingRank) replay->pushBackSequence(EP);
    else
    {
      assert(bLearnersEpsSharing);
      const Uint I = sharingTurn;
      if(shareSendSizeReq[I] not_eq MPI_REQUEST_NULL)
        MPI(Wait, & shareSendSizeReq[I], MPI_STATUS_IGNORE);
      if( shareSendSeqReq[I] not_eq MPI_REQUEST_NULL)
        MPI(Wait, &  shareSendSeqReq[I], MPI_STATUS_IGNORE);

      shareSendSeq[I] = EP->packSequence(sI.dimObs(),aI.dim(),aI.dimPol());
      shareSendSeqSize[I] = shareSendSeq[I].size();
      Utilities::dispose_object(EP);

      IsendSize(shareSendSeqSize[I], I, sharingComm, shareSendSizeReq[I]);
      IsendSeq(shareSendSeq[I], I, sharingComm, shareSendSeqReq[I]);
    }
    completed.pop_back();
    // who's turn is next to receive an episode?
    if(sharingSize>0) sharingTurn = (sharingTurn+1) % sharingSize; //pick next
  }
}

void DataCoordinator::mastersRecvEpisodes()
{
  for(Uint i=0; i<sharingSize; ++i)
    if (i not_eq sharingRank) {
      if (isComplete(shareRecvSizeReq[i]))
      {
        Fvec EP(shareRecvSeqSize[i], (Fval)0);
        RecvSeq(EP, i, sharingComm);
        // prepare the next one:
        IrecvSize(shareRecvSeqSize[i], i, sharingComm, shareRecvSizeReq[i]);
        Sequence * const tmp = new Sequence();
        tmp->unpackSequence(EP, sI.dimObs(), aI.dim(), aI.dimPol());
        replay->pushBackSequence(tmp);
      }
      if(shareSendSizeReq[i] not_eq MPI_REQUEST_NULL) {
        int complete = 0;
        MPI(Test, & shareSendSizeReq[i], &complete, MPI_STATUS_IGNORE);
        if(complete) assert(shareSendSizeReq[i] == MPI_REQUEST_NULL);
      }
      if( shareSendSeqReq[i] not_eq MPI_REQUEST_NULL) {
        int complete = 0;
        MPI(Test, &  shareSendSeqReq[i], &complete, MPI_STATUS_IGNORE);
        if(complete) assert(shareSendSizeReq[i] == MPI_REQUEST_NULL);
      }
    }

  assert(allTasksPtr not_eq nullptr);
  // if all learners are locking data acquisition we do not recv eps from worker
  // such that they wait for updated parameters before gathering more data
  if(allTasksPtr->dataAcquisitionIsLocked()) return;

  for(Uint i=1; i<workerSize; ++i)
    if (isComplete(workerRecvSizeReq[i]))
    {
      Fvec EP(workerRecvSeqSize[i], (Fval)0);
      const Uint nStep = Sequence::computeTotalEpisodeNstep(sI.dimObs(),
                                                      aI.dim(), aI.dimPol(),
                                                      workerRecvSeqSize[i]);
      RecvSeq(EP, i, workerComm);
      int bSendParams = 0;
      MPI(Recv, &bSendParams, 1, MPI_INT, i, 275727+MDPID, workerComm,
          MPI_STATUS_IGNORE);

      nSeenTransitions_loc += nStep - 1; // we do not count init state
      nSeenSequences_loc += 1;
      // prepare the next one:
      IrecvSize(workerRecvSeqSize[i], i, workerComm, workerRecvSizeReq[i]);

      //_warn("sending to rank %lu", sharingTurn);
      if (sharingTurn == sharingRank) {
        Sequence * const tmp = new Sequence();
        tmp->unpackSequence(EP, sI.dimObs(), aI.dim(), aI.dimPol());
        assert(nStep == tmp->ndata() + 1);
        //_warn("%lu storing new sequence of size %lu", MDPID,tmp->ndata());
        replay->pushBackSequence(tmp);
      } else {
        const Uint I = sharingTurn;
        if(shareSendSizeReq[I] not_eq MPI_REQUEST_NULL)
          MPI(Wait, & shareSendSizeReq[I], MPI_STATUS_IGNORE);
        if( shareSendSeqReq[I] not_eq MPI_REQUEST_NULL)
          MPI(Wait, &  shareSendSeqReq[I], MPI_STATUS_IGNORE);

        shareSendSeq[I] = EP;
        shareSendSeqSize[I] = shareSendSeq[I].size();

        IsendSize(shareSendSeqSize[I], I, sharingComm, shareSendSizeReq[I]);
        IsendSeq(shareSendSeq[I], I, sharingComm, shareSendSeqReq[I]);
      }

      if(bSendParams) {
        while(allTasksPtr->dataAcquisitionIsLocked()) usleep(1);
        params.send(i, MDPID);
      }
      if(sharingSize>0) sharingTurn = (sharingTurn+1) % sharingSize; //pick next
    }
}

// called externally
void DataCoordinator::addComplete(Sequence* const EP, const bool bUpdateParams)
{
  if(bLearnersEpsSharing)
  {
    std::lock_guard<std::mutex> lock(complete_mutex);
    completed.push_back(EP);
  }
  else
  if(bRunParameterServer)
  {
    // if we created data structures for worker to send eps to master
    // this better be a worker!
    assert(workerRank>0 && workerSize>1 && not distrib.bIsMaster);
    Fvec sendSq = EP->packSequence(sI.dimObs(), aI.dim(), aI.dimPol());
    #ifndef NDEBUG
      Sequence * const tmp = new Sequence();
      tmp->unpackSequence(sendSq, sI.dimObs(), aI.dim(), aI.dimPol());
      //_warn("storing new sequence of size %lu", tmp->ndata());
      assert(EP->isEqual(tmp));
      delete tmp;
    #endif
    unsigned long sendSz = sendSq.size();
    const int intUpdateParams = bUpdateParams? 1 : 0;

    //in theory this lock is unnecessary because all ops here are locking and
    //master processes one after the other (i.e. other threads will wait)
    //however, in case of non-multiple thread safety, it will cause deadlock
    std::lock_guard<std::mutex> send_ep_lock(complete_mutex);
    MPI(Send, &sendSz, 1, MPI_UNSIGNED_LONG, 0, 37536+MDPID, workerComm);
    MPI(Send, sendSq.data(), sendSz, MPI_Fval, 0, 737283+MDPID, workerComm);
    MPI(Send, &intUpdateParams, 1, MPI_INT, 0, 275727+MDPID, workerComm);
    if(bUpdateParams) {
      //_warn("%lu sent episode of size %lu and update params", MDPID,EP->ndata() );
      params.recv(MDPID);
    } //else _warn("%lu sent ep of size %lu w/o update params", MDPID,EP->ndata() );
    delete EP;
  }
  else // data stays here
  {
    //_warn("%lu stored episode of size %lu", MDPID,EP->ndata() );
    replay->pushBackSequence(EP);
  }
}

}
