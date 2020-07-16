//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "StatsTracker.h"
#include "Warnings.h"
#include "SstreamUtilities.h"
#include "../Settings/Bund.h"
#include "../Settings/ExecutionInfo.h"

#include <cassert>

namespace smarties
{

StatsTracker::StatsTracker(const Uint N, const ExecutionInfo& distrib) :
  n_stats(N), nThreads(distrib.nThreads),
  learn_rank(MPICommRank(distrib.learners_train_comm)), cntVec(nThreads, 0),
  avgVec(nThreads, LDvec(n_stats, 0)), stdVec(nThreads, LDvec(n_stats, 0))
{
  instMean.resize(n_stats, 0); instStdv.resize(n_stats, 0);
}

void StatsTracker::track_vector(const Rvec& grad, const Uint thrID) const
{
  assert(n_stats==grad.size());
  cntVec[thrID] += 1;
  for (Uint i=0; i<n_stats; ++i) {
    avgVec[thrID][i] += grad[i];
    stdVec[thrID][i] += grad[i]*grad[i];
  }
}

void StatsTracker::advance()
{
  std::fill(avg.begin(),  avg.end(), 0);
  std::fill(std.begin(),  std.end(), 0);
  cnt = 0;

  for (Uint i=0; i<nThreads; ++i) {
    cnt += cntVec[i];
    for (Uint j=0; j<n_stats; ++j) {
      avg[j] += avgVec[i][j];
      std[j] += stdVec[i][j];
    }
    cntVec[i] = 0;
    std::fill(avgVec[i].begin(),  avgVec[i].end(), 0);
    std::fill(stdVec[i].begin(),  stdVec[i].end(), 0);
  }
}

void StatsTracker::update()
{
  cnt = std::max((long double)2.2e-16, cnt);
  for (Uint j=0; j<n_stats; ++j) {
    const Real   mean = avg[j] / cnt;
    const Real sqmean = std[j] / cnt;
    std[j] = std::sqrt(sqmean); // - mean*mean
    avg[j] = mean;
  }
}

void StatsTracker::printToFile(const std::string& base)
{
  if(!learn_rank) {
    FILE * pFile;
    if(!nStep) {
      // write to log the number of variables, so that it can be then unwrangled
      pFile = fopen((base + "_outGrad_stats.raw").c_str(), "wb");
      float printvals = n_stats +.1; // to be floored to an integer in post
      fwrite(&printvals, sizeof(float), 1, pFile);
    }
    else pFile = fopen((base + "_outGrad_stats.raw").c_str(), "ab");
    std::vector<float> printvals(n_stats*2);
    for (Uint i=0; i<n_stats; ++i) {
      printvals[i]         = avg[i];
      printvals[i+n_stats] = std[i];
    }
    fwrite(printvals.data(), sizeof(float), n_stats*2, pFile);
    fflush(pFile); fclose(pFile);
  }
}

void StatsTracker::finalize(const LDvec&oldM, const LDvec&oldS)
{
  instMean = avg;
  instStdv = std;
  nStep++;
  for (Uint i=0; i<n_stats; ++i) {
    avg[i] = (1-CLIP_LEARNR)*oldM[i] +CLIP_LEARNR*avg[i];
    std[i] = (1-CLIP_LEARNR)*oldS[i] +CLIP_LEARNR*std[i];
    //stdVec[0][i]=std::max((1-CLIP_LEARNR)*oldstd[i], stdVec[0][i]);
  }
}

void StatsTracker::reduce_stats(const std::string& base, const Uint iter)
{
  const LDvec oldsum = avg, oldstd = std;
  advance();
  update();
  if(iter % 1000 == 0) printToFile(base);
  finalize(oldsum, oldstd);
}

}
