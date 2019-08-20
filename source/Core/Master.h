//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Master_h
#define smarties_Master_h

#include "Worker.h"
#include "../Utils/SocketsLib.h"
#include <thread>

namespace smarties
{

template <typename CommType, typename Request_t>
class Master : public Worker
{
  std::atomic<Uint> bExit {0};

protected:
  std::vector<std::thread> worker_replies;
  CommType * interface() { return static_cast<CommType*> (this); }

  void waitForStateActionCallers(const std::vector<Uint> givenWorkers);

public:
  Master(DistributionInfo& );
  virtual ~Master() {};
  void run();
  void spawnCallsHandlers();
};

class MasterSockets : public Master<MasterSockets, SOCKET_REQ>
{
public:

  void Irecv(void*const buffer, const Uint size, const int rank,
    const int tag, SOCKET_REQ& request) const {
    SOCKET_Irecv(buffer, size, getSocketID(rank), request);
  }

  void  Send(void*const buffer, const Uint size, const int rank,
    const int tag) const {
    SOCKET_Bsend(buffer, size, getSocketID(rank));
  }

  int TestComm(SOCKET_REQ& request) const {
    SOCKET_Test(request.completed, request);
    return request.completed;
  }
  void WaitComm(SOCKET_REQ& request) const {
    SOCKET_Wait(request);
  }

  MasterSockets( DistributionInfo& );
  ~MasterSockets(){}

  void run(const environment_callback_t& callback);
};

class MasterMPI : public Master<MasterMPI, MPI_Request>
{
public:

  void Irecv(void*const buffer, const Uint size, const int rank,
    const int tag, MPI_Request& req) const {
    MPI(Irecv, buffer, size, MPI_BYTE, rank, tag, master_workers_comm, & req);
  }

  void  Send(void*const buffer, const Uint size, const int rank,
    const int tag) const {
    MPI(Send, buffer, size, MPI_BYTE, rank, tag, master_workers_comm);
  }

  int TestComm(MPI_Request& request) const {
    int completed = 0; //MPI_Status mpistatus;
    MPI(Test, &request, &completed, MPI_STATUS_IGNORE);
    return completed;
  }
  void WaitComm(MPI_Request& request) const {
    //MPI_Status mpistatus;
    MPI(Wait, &request, MPI_STATUS_IGNORE);
  }

  MasterMPI( DistributionInfo& );
  ~MasterMPI(){}

  void run();
};

}
#endif
