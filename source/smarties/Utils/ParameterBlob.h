//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ParameterBlob_h
#define smarties_ParameterBlob_h

#include "Warnings.h"
#include "../Settings/ExecutionInfo.h"
#include "../ReplayMemory/ReplayStatsCounters.h"

#include <utility>
#include <vector>
#include <atomic>
#include <cstring>

namespace smarties
{

// Each part of the code where learner parameters need be sent
// from learning algorithm to agents/workers needs to pass
// pointer to this class.
// As of now, these parameters are only network parameters and
// scaling coefficient for state/rewards, therefore assumed
// ptr to nnReal. Might make it general into char in the future.
// Receiving should be blocking, because requested by user/app
// Sending should be issued when a learner cannot proceed with
// more training? Non-blocking?
// Cannot send parameters at the end of each episode because
// many agents may share weights... but that can be seen as
// a secondary optimization...

class ParameterBlob
{
  using dataInfo = std::pair<Uint, nnReal*>;
  const ExecutionInfo & distrib;
  const MPI_Comm comm = distrib.master_workers_comm;
  ReplayStats & stats;
  ReplayCounters & counters;
  std::vector<dataInfo> dataList;

  struct {
    long nDataB4startup;
    long nGradSteps;
    Real avgCumulativeRew;
  } counterMsg;

public:
  ParameterBlob(const ExecutionInfo& D, ReplayStats& S, ReplayCounters& C)
    : distrib(D), stats(S), counters(C) {}

  void add(const Uint size, nnReal * const data) {
    dataList.emplace_back(std::make_pair(size, data));
  }

  void recv(const Uint MDP_ID) const
  {
    MPI(Recv, (void *) & counterMsg, sizeof(counterMsg), MPI_BYTE,
        0, 72726 + MDP_ID, comm, MPI_STATUS_IGNORE);
    counters.nGatheredB4Startup = counterMsg.nDataB4startup;
    counters.nGradSteps         = counterMsg.nGradSteps;
    stats.avgReturn             = counterMsg.avgCumulativeRew;

    // workers always recv params from learner (rank 0)
    for(const auto& data : dataList ) {
      MPI(Recv, data.second, data.first, SMARTIES_MPI_NNVALUE_TYPE, 0,
        72726 + MDP_ID, comm, MPI_STATUS_IGNORE);
    }
  }

  void send(const Uint toRank, const Uint MDP_ID)
  {
    counterMsg.nDataB4startup = counters.nGatheredB4Startup;
    counterMsg.nGradSteps = counters.nGradSteps.load();
    counterMsg.avgCumulativeRew = stats.avgReturn;
    MPI(Send, (void *) & counterMsg, sizeof(counterMsg), MPI_BYTE,
        toRank, 72726 + MDP_ID, comm);

    for(const auto& data : dataList )
      MPI(Send, data.second, data.first, SMARTIES_MPI_NNVALUE_TYPE, toRank,
        72726 + MDP_ID, comm);
  }
};

} // end namespace smarties
#endif // smarties_ParameterBlob_h
