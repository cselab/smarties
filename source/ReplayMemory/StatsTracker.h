//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Settings.h"

struct DelayedReductor
{
  const MPI_Comm mpicomm;
  const bool bAsync;
  const Uint arysize, mpisize;
  std::mutex& mpi_mutex;
  MPI_Request buffRequest = MPI_REQUEST_NULL;
  LDvec return_ret = LDvec(arysize, 0);
  LDvec reduce_ret = LDvec(arysize, 0);
  LDvec partialret = LDvec(arysize, 0);

  static int getSize(const MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }

  DelayedReductor(const Settings& S, const LDvec init);

  LDvec get(const bool accurate = false);

  template<Uint I>
  long double get(const bool accurate = false)
  {
    const LDvec ret = get(accurate);
    return ret[I];
  }

  void update(const LDvec ret);
};

struct TrainData
{
  const Uint n_extra, nThreads, bPolStats;
  const string name, extra_header;

  long double cnt = 0;
  LDvec q = LDvec(5, 0);
  LDvec p = LDvec(3, 0);
  LDvec e = LDvec(n_extra, 0);

  THRvec<long double> cntVec = THRvec<long double>(nThreads, 0);
  THRvec<LDvec> qVec = THRvec<LDvec>(nThreads, LDvec(5, 0));
  THRvec<LDvec> pVec = THRvec<LDvec>(nThreads, LDvec(3, 0));
  THRvec<LDvec> eVec = THRvec<LDvec>(nThreads, LDvec(n_extra, 0));

  TrainData(const string _name, const Settings&set, bool bPPol=0,
    const string extrah = string(), const Uint nextra=0);

  ~TrainData();

  void log(const Real Q, const Real Qerr,
    const std::vector<Real> polG, const std::vector<Real> penal,
    std::initializer_list<Real> extra, const int thrID);
  void log(const Real Q, const Real Qerr,
    std::initializer_list<Real> extra, const int thrID);
  void log(const Real Q, const Real Qerr, const int thrID);

  void getMetrics(ostringstream& buff);
  void getHeaders(ostringstream& buff) const;

  void resetSoft();
  void resetHead();

  void reduce();

  void trackQ(const Real Q, const Real err, const int thrID);

  void trackPolicy(const std::vector<Real> polG,
    const std::vector<Real> penal, const int thrID);
};

struct StatsTracker
{
  const Uint n_stats;
  const MPI_Comm comm;
  const Uint nThreads, learn_size, learn_rank;

  long double cnt = 0;
  LDvec avg = LDvec(n_stats, 0);
  LDvec std = LDvec(n_stats, 10);
  THRvec<long double> cntVec = THRvec<long double>(nThreads, 0);
  THRvec<LDvec> avgVec = THRvec<LDvec>(nThreads, LDvec(n_stats, 0));
  THRvec<LDvec> stdVec = THRvec<LDvec>(nThreads, LDvec(n_stats, 0));

  LDvec instMean, instStdv;
  unsigned long nStep = 0;

  StatsTracker(const Uint N, const Settings& set);

  void track_vector(const Rvec grad, const Uint thrID) const;

  void advance();

  void update();

  void printToFile(const string base);

  void finalize(const LDvec&oldM, const LDvec&oldS);

  void reduce_stats(const string base, const Uint iter = 0);

};
