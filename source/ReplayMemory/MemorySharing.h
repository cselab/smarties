//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once
#include <thread>

#include "MemoryBuffer.h"
class Learner;

struct MemorySharing
{
  const Settings& settings;
  Learner* const learner;
  MemoryBuffer* const replay;

  const StateInfo& sI;
  const ActionInfo& aI;
  const Uint dimS = sI.dimUsed, dimA = aI.dim, dimP = aI.policyVecDim;

  const MPI_Comm comm = MPIComDup(settings.mastersComm);
  const int ID = settings.learner_rank, SZ = settings.learner_size;

  int EpOwnerID = ID; // first episode stays on rank

  std::vector<Sequence*> completed;
  std::vector<Uint> sendSz = std::vector<Uint>(SZ);
  std::vector<Uint> recvSz = std::vector<Uint>(SZ);

  std::vector<Fvec> sendSq = std::vector<Fvec>(SZ);
  std::vector<Fvec> recvSq = std::vector<Fvec>(SZ);

  std::vector<MPI_Request> RRq = std::vector<MPI_Request>(SZ, MPI_REQUEST_NULL);
  std::vector<MPI_Request> SRq = std::vector<MPI_Request>(SZ, MPI_REQUEST_NULL);
  std::vector<MPI_Request> CRq = std::vector<MPI_Request>(SZ, MPI_REQUEST_NULL);
  MPI_Request nObsRequest = MPI_REQUEST_NULL;

  std::mutex complete_mutex;
  const bool bAsync = settings.bAsync;
  std::mutex& mpi_mutex = settings.mpi_mutex;

  std::thread fetcher;
  std::atomic<Uint> bExit {0};

  std::atomic<long>& nSeenTransitions_loc = replay->nSeenTransitions_loc;
  std::atomic<long>& nSeenSequences_loc = replay->nSeenSequences_loc;
  long int globSeen[2] = {0, 0};

  MemorySharing(const Settings&S, Learner*const L, MemoryBuffer*const RM);
  ~MemorySharing();

  inline int testBuffer(MPI_Request& req);

  void recvEP(const int ID2);

  void sendEp(const int ID2, Sequence* const EP);

  void run();

  void addComplete(Sequence* const EP);
};
