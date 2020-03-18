//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Collector_h
#define smarties_Collector_h

#include "MemoryBuffer.h"
#include "../Utils/StatsTracker.h"

namespace smarties
{

class DataCoordinator;

class Collector
{
private:
  MemoryBuffer * const replay;
  DataCoordinator * const sharing;
  //const MDPdescriptor & MDP = replay->MDP;
  //const Settings & settings = replay->settings;
  const DistributionInfo & distrib = replay->distrib;
  //const StateInfo& sI = replay->sI;
  const ActionInfo& aI = replay->aI;

  std::vector<Episode> inProgress;

  std::atomic<long>& nSeenEpisodes_loc = replay->nSeenEpisodes_loc;
  std::atomic<long>& nSeenTransitions_loc = replay->nSeenTransitions_loc;

  std::mutex envTerminationCheck;

public:
  void add_state(Agent&a);
  void add_action(const Agent& a, const Rvec pol);
  void terminate_seq(Agent&a);
  void push_back(const size_t agentId);

  inline Episode& get(const Uint ID) {
    return inProgress[ID];
  }
  inline Uint nInProgress() const {
    return inProgress.size();
  }

  Collector(MemoryBuffer*const RM, DataCoordinator*const C);

  ~Collector();
};

}
#endif
