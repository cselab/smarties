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
#include "../Settings.h"

#include <mutex>

namespace smarties
{

template<typename T>
struct DelayedReductor
{
  const MPI_Comm mpicomm;
  const Uint arysize, mpisize;
  const DistributionInfo& distrib;
  MPI_Request buffRequest = MPI_REQUEST_NULL;
  std::vector<T> return_ret = std::vector<T>(arysize, 0);
  std::vector<T> reduce_ret = std::vector<T>(arysize, 0);
  std::vector<T> partialret = std::vector<T>(arysize, 0);

  static int getSize(const MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }

  DelayedReductor(const DistributionInfo&, const std::vector<T> init);
  ~DelayedReductor();

  std::vector<T> get(const bool accurate = false);

  template<Uint I>
  T get(const bool accurate = false)
  {
    const std::vector<T> ret = get(accurate);
    return ret[I];
  }

  void beginRDX();
  void update(const std::vector<T> ret);
};

struct TrainData
{
  const Uint n_extra, nThreads, bPolStats;
  const std::string name, extra_header;

  long double cnt = 0;
  LDvec q = LDvec(5, 0);
  LDvec p = LDvec(3, 0);
  LDvec e = LDvec(n_extra, 0);

  THRvec<long double> cntVec;
  THRvec<LDvec> qVec, pVec, eVec;

  TrainData(const std::string _name, const DistributionInfo&, bool bPPol=0,
    const std::string extrah = std::string(), const Uint nextra=0);

  ~TrainData();

  void log(const Real Q, const Real Qerr,
    const std::vector<Real>& polG, const std::vector<Real>& penal,
    const std::initializer_list<Real>& extra, const int thrID);
  void log(const Real Q, const Real Qerr,
    const std::initializer_list<Real>& extra, const int thrID);
  void log(const Real Q, const Real Qerr, const int thrID);

  void getMetrics(std::ostringstream& buff);
  void getHeaders(std::ostringstream& buff) const;

  void resetSoft();
  void resetHead();

  void reduce();

  void trackQ(const Real Q, const Real err, const int thrID);

  void trackPolicy(const std::vector<Real>& polG,
    const std::vector<Real>& penal, const int thrID);
};

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

  StatsTracker(const Uint N, const DistributionInfo&);

  void track_vector(const Rvec& grad, const Uint thrID) const;

  void advance();

  void update();

  void printToFile(const std::string& base);

  void finalize(const LDvec&oldM, const LDvec&oldS);

  void reduce_stats(const std::string& base, const Uint iter = 0);

};

}
#endif
