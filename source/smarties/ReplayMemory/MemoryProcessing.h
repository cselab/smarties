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

namespace MemoryProcessing
{
  void updateCounters(MemoryBuffer& RM, const bool bInit = false);
  void updateRewardsStats(MemoryBuffer& RM,
                          const bool bInit = false,
                          const Real learnRateFac = 1);

  FORGET readERfilterAlgo(const HyperParameters & S);

  // Algorithm for maintaining and filtering dataset, and optional imp weight range parameter
  void selectEpisodeToDelete(MemoryBuffer & RM, const FORGET ALGO);

  void prepareNextBatchAndDeleteStaleEp(MemoryBuffer & RM);

  void histogramImportanceWeights(const MemoryBuffer & RM);

  void computeReturnEstimator(const MemoryBuffer & RM, Episode & EP);

  void updateReturnEstimators(MemoryBuffer & RM,
                              const bool doAllEpisodes = false);

  void rescaleAllReturnEstimator(MemoryBuffer & RM);
}

}
#endif
