//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_DataCoordinator_h
#define smarties_DataCoordinator_h

#include "MemoryBuffer.h"
#include "../Utils/TaskQueue.h"
#include "../Utils/ParameterBlob.h"
#include <thread>

namespace smarties
{

class DataCoordinator
{
  MemoryBuffer* const replay;
  ParameterBlob & params;
  const Settings & settings = replay->settings;
  const DistributionInfo & distrib = replay->distrib;

  const Uint MDPID = replay->MDP.localID;
  const StateInfo& sI = replay->sI;
  const ActionInfo& aI = replay->aI;
  std::vector<Sequence*> completed;

  // allows masters to share episodes between each others
  // each master sends the size (in floats) of the episode
  // then sends the episode itself. same goes for receiving
  MPI_Comm sharingComm = MPI_COMM_NULL;
  Uint sharingSize=0, sharingRank=0, sharingTurn=0;
  std::vector<MPI_Request> shareSendSizeReq, shareSendSeqReq, shareRecvSizeReq;
  std::vector<unsigned long> shareSendSeqSize, shareRecvSeqSize;
  std::vector<Fvec> shareSendSeq;

  MPI_Comm workerComm = MPI_COMM_NULL;
  Uint workerSize=0, workerRank=0;
  std::vector<MPI_Request> workerRecvSizeReq;
  std::vector<unsigned long> workerRecvSeqSize;


  std::vector<MPI_Request> workerReqParamReq;
  std::vector<unsigned long> workerReqParamFlag;

  std::mutex complete_mutex;

  std::atomic<long>& nSeenTransitions_loc = replay->nSeenTransitions_loc;
  std::atomic<long>& nSeenSequences_loc = replay->nSeenSequences_loc;
  const TaskQueue * allTasksPtr = nullptr;

public:
  DataCoordinator(MemoryBuffer*const RM, ParameterBlob & params);
  ~DataCoordinator();

  void setupTasks(TaskQueue& tasks);

  void distributePendingEpisodes();

  void mastersRecvEpisodes();

  void addComplete(Sequence* const EP, const bool bUpdateParams);

  bool bRunParameterServer = false;
  bool bLearnersEpsSharing = false;

private:

  void recvEP(const int ID2);

  void sendEp(const int ID2, Sequence* const EP);

  void IrecvSize(unsigned long& size, const int rank, const MPI_Comm C, MPI_Request&R) const
  {
    MPI(Irecv, &size, 1, MPI_UNSIGNED_LONG, rank, 37536+MDPID, C, &R);
  }
  void IsendSize(const unsigned long& size, const int rank, const MPI_Comm C, MPI_Request&R) const
  {
    MPI(Isend, &size, 1, MPI_UNSIGNED_LONG, rank, 37536+MDPID, C, &R);
  }

  void RecvSeq(Fvec&V, const int rank, const MPI_Comm C) const
  {
    MPI( Recv, V.data(), V.size(), MPI_Fval, rank, 737283+MDPID, C,
         MPI_STATUS_IGNORE);
  }
  void IsendSeq(const Fvec&V, const int rank, const MPI_Comm C, MPI_Request&R) const
  {
    MPI(Isend, V.data(), V.size(), MPI_Fval, rank, 737283+MDPID, C, &R);
  }

  bool isComplete(MPI_Request& req)
  {
    if(req == MPI_REQUEST_NULL) return false;
    int bRecvd = 0;
    MPI(Test, &req, &bRecvd, MPI_STATUS_IGNORE);
    return bRecvd > 0;
  }
};

}
#endif
