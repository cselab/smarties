//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MemoryProcessing_h
#define smarties_MemoryProcessing_h

#include "MemoryBuffer.h"
#include "../Utils/StatsTracker.h"

namespace smarties
{

class MemoryProcessing
{
private:
  MemoryBuffer* const RM;
  MDPdescriptor & MDP = RM->MDP;
  const Settings & settings = RM->settings;
  const DistributionInfo & distrib = RM->distrib;

  std::vector<nnReal> & invstd_state = RM->invstd_state;
  std::vector<nnReal> &   mean_state = RM->mean_state;
  std::vector<nnReal> &    std_state = RM->std_state;
  nnReal & invstd_reward = RM->invstd_reward;
  nnReal &   mean_reward = RM->mean_reward;
  nnReal &    std_reward = RM->std_reward;
  Real & beta = RM->beta;
  Real & alpha = RM->alpha;
  Real & CmaxRet = RM->CmaxRet;
  Real & CinvRet = RM->CinvRet;

  Uint nPrunedEps = 0;
  Uint oldestStoresTimeStamp = 0;
  Uint nFarPolicySteps = 0;
  Real avgKLdivergence =  0;
  Sint indexOfEpisodeToDelete = -1;

  DelayedReductor<long double> StateRewRdx;
  DelayedReductor<long> globalStep_reduce;
  DelayedReductor<long double> ReFER_reduce;

  std::vector<Episode>& episodes = RM->episodes;
  std::atomic<long>& nGradSteps = RM->nGradSteps;
  std::atomic<long>& nEpisodes = RM->nEpisodes;
  std::atomic<long>& nTransitions = RM->nTransitions;
  std::atomic<long>& nSeenEpisodes = RM->nSeenEpisodes;
  std::atomic<long>& nSeenTransitions = RM->nSeenTransitions;
  std::atomic<long>& nSeenEpisodes_loc = RM->nSeenEpisodes_loc;
  std::atomic<long>& nSeenTransitions_loc = RM->nSeenTransitions_loc;

public:

  MemoryProcessing(MemoryBuffer*const _RM);

  void updateRewardsStats(const Real WR, const Real WS, const bool bInit=false);

  static FORGET readERfilterAlgo(const std::string setting, const bool bReFER);

  // Algorithm for maintaining and filtering dataset, and optional imp weight range parameter
  void selectEpisodeToDelete(const FORGET ALGO);
  void prepareNextBatchAndDeleteStaleEp();

  void histogramImportanceWeights();
  void updateReFERpenalization();

  void getMetrics(std::ostringstream& buff);
  void getHeaders(std::ostringstream& buff);

  Uint nFarPol() {
    return nFarPolicySteps;
  }
};

}
#endif
