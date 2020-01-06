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
  //const Settings & settings = replay->settings;
  const DistributionInfo & distrib = replay->distrib;

  const Uint MDPID = replay->MDP.localID;
  const StateInfo& sI = replay->sI;
  const ActionInfo& aI = replay->aI;
  std::vector<Sequence> episodes;

  // allows masters to share episodes between each others
  // each master sends the size (in floats) of the episode
  // then sends the episode itself. same goes for receiving
  MPI_Comm sharingComm = MPI_COMM_NULL;
  Uint sharingSize=0, sharingRank=0, sharingDest=0;
  std::vector<MPI_Request> sharingReq;
  std::vector<Fvec> sharingSeq;

  MPI_Comm workerComm = MPI_COMM_NULL;
  Uint workerSize=0, workerRank=0;

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

  void addComplete(Sequence& EP, const bool bUpdateParams);

  bool bRunParameterServer = false;
  bool bLearnersEpsSharing = false;
};

}
#endif
