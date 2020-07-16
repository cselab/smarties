//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_StatsTracker_h
#define smarties_StatsTracker_h

#include "ThreadSafeVec.h"

#include <mutex>

namespace smarties
{

struct ExecutionInfo;

struct StatsTracker
{
  const Uint n_stats;
  const Uint nThreads, learn_rank;

  long double cnt = 0;
  LDvec avg = LDvec(n_stats, 0);
  LDvec std = LDvec(n_stats, 10);
  THRvec<long double> cntVec;
  THRvec<LDvec> avgVec, stdVec;

  LDvec instMean, instStdv;
  unsigned long nStep = 0;

  StatsTracker(const Uint N, const ExecutionInfo&);

  void track_vector(const Rvec& grad, const Uint thrID) const;

  void advance();

  void update();

  void printToFile(const std::string& base);

  void finalize(const LDvec&oldM, const LDvec&oldS);

  void reduce_stats(const std::string& base, const Uint iter = 0);

};

}
#endif
