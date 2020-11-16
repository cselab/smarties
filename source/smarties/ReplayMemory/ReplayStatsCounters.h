//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ReplayStatsCounters_h
#define smarties_ReplayStatsCounters_h

#include "../Settings/Definitions.h"
#include <atomic>

namespace smarties
{

struct ReplayStats
{
  Uint nFarPolicySteps = 0;
  Real avgKLdivergence = 0;
  Sint countReturnsEstimateUpdates = 0;
  Real sumReturnsEstimateErrors = 0;
  Real avgSquaredErr = 0;
  Real maxAbsError = 0;
  Real avgReturn = 0;
  Real stdevQ = 0;
  Real avgQ = 0;
  Real maxQ = 0;
  Real minQ = 0;
  Sint nPrunedEps = 0;
};

struct ReplayCounters
{
  // num of grad steps performed by owning learner:
  std::atomic<long> nGradSteps{0};
  // number of time steps collected before training begins:
  long nGatheredB4Startup = std::numeric_limits<long>::max();
  // num of samples contained in dataset:
  std::atomic<long> nEpisodes{0}; // num of episodes
  std::atomic<long> nTransitions{0}; // num of individual time steps
  // num of samples seen from the beginning
  std::atomic<long> nSeenEpisodes{0};
  std::atomic<long> nSeenTransitions{0};
  // num of samples seen from beginning on this learning process:
  std::atomic<long> nSeenEpisodes_loc{0};
  std::atomic<long> nSeenTransitions_loc{0};
};

}
#endif
