//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemorySharing.h"
#include "../Learners/Learner_offPolicy.h"
#include "../Learners/CMALearner.h"
#include "../Learners/PPO.h"

MemorySharing::MemorySharing(const Settings&S, Learner*const L,
  MemoryBuffer*const RM) : settings(S), learner(L), replay(RM), sI(L->sInfo),
  aI(L->aInfo) {
  completed.reserve(S.nAgents);
  #pragma omp parallel
  {
    const int thrID = omp_get_thread_num();
    const int tgtCPU =  std::min(omp_get_num_threads()-1, 2);
    if( thrID==tgtCPU ) fetcher = std::thread( [ &, this ] () { run(); } );
  }
}

MemorySharing::~MemorySharing()
{
  bExit = 1;
  fetcher.join();
  for(const auto& S : completed) _dispose_object(S);
}

int MemorySharing::testBuffer(MPI_Request& req)
{
  if(req == MPI_REQUEST_NULL) return 0;
  int bRecvd = 0;
  MPI(Test, &req, &bRecvd, MPI_STATUS_IGNORE);
  return bRecvd;
}

void MemorySharing::recvEP(const int ID2)
{
  if( testBuffer(RRq[ID2]) ) {
   recvSq[ID2].resize(recvSz[ID2]);
   const auto NOSTS = MPI_STATUS_IGNORE;

   MPI(Recv, recvSq[ID2].data(), recvSz[ID2], MPI_Fval, ID2, 98, comm, NOSTS);
   MPI(Irecv, &recvSz[ID2], 1, MPI_UNSIGNED, ID2, 99, comm, &RRq[ID2]);

   Sequence * const tmp = new Sequence();
   tmp->unpackSequence(recvSq[ID2], dimS, dimA, dimP);
   replay->pushBackSequence(tmp);
  }
}

void MemorySharing::sendEp(const int ID2, Sequence* const EP)
{
  if(CRq[ID2] not_eq MPI_REQUEST_NULL) MPI(Wait, &CRq[ID2], MPI_STATUS_IGNORE);
  if(SRq[ID2] not_eq MPI_REQUEST_NULL) MPI(Wait, &SRq[ID2], MPI_STATUS_IGNORE);

  sendSq[ID2] = EP->packSequence(dimS, dimA, dimP);
  sendSz[ID2] = sendSq[ID2].size();

  MPI(Isend, &sendSz[ID2], 1, MPI_UNSIGNED, ID2, 99, comm, &CRq[ID2]);
  MPI(Isend, sendSq[ID2].data(),sendSz[ID2], MPI_Fval,ID2,98,comm,&SRq[ID2]);
}

void MemorySharing::run()
{
  for(int i=0; i<SZ; i++) if(i not_eq ID)
    MPI(Irecv, &recvSz[i], 1, MPI_UNSIGNED, i, 99, comm, &RRq[i]);

  while(true)
  {
    {
      std::lock_guard<std::mutex> lock(complete_mutex);
      while ( completed.size() ) {
        if (EpOwnerID == ID) replay->pushBackSequence(completed.back());
        else {
          sendEp(EpOwnerID, completed.back());
          _dispose_object(completed.back());
        }
        completed.pop_back();
        EpOwnerID = (EpOwnerID+1) % SZ;
      }
    }

    if(nObsRequest == MPI_REQUEST_NULL)
    {
     int ready = 0;
     if(learner->blockGradStep() and learner->blockDataAcquisition()) ready = 1;
     else if( learner->checkReady4Init() ) ready = 2;
     if(ready)
     {
       std::lock_guard<std::mutex> lock1(complete_mutex);
       if( completed.size() == 0) {
         globSeen[0] = nSeenTransitions_loc.load();
         globSeen[1] = nSeenSequences_loc.load();
         //_warn("Sent advance req %d %ld %ld",ready,globSeen[0],globSeen[1]);
         MPI(Iallreduce, MPI_IN_PLACE, globSeen, 2,
          MPI_LONG, MPI_SUM, comm, &nObsRequest);
       }
     }
    }

    for(int i=0; i<SZ; i++) if(i not_eq ID) recvEP(i);

    if( testBuffer(nObsRequest) ) {
      learner->globalDataCounterUpdate(globSeen[0], globSeen[1]);
      nObsRequest = MPI_REQUEST_NULL;
    }

    usleep(1);
    if( bExit.load() > 0 ) break;
  }
}

void MemorySharing::addComplete(Sequence* const EP)
{
  std::lock_guard<std::mutex> lock(complete_mutex);
  completed.push_back(EP);
}
