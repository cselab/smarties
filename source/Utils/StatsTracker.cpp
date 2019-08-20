//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "StatsTracker.h"
#include "../Settings.h"
#include "Warnings.h"
#include "SstreamUtilities.h"
#include "Bund.h"

#include <cassert>

namespace smarties
{

template<typename T>
DelayedReductor<T>::DelayedReductor(const DistributionInfo& D,
                                    const std::vector<T> I) :
mpicomm(MPICommDup(D.learners_train_comm)), arysize(I.size()),
mpisize(MPICommSize(D.learners_train_comm)), distrib(D), return_ret(I) {}


template<typename T>
DelayedReductor<T>::~DelayedReductor()
{
  MPI_Comm* commptr = const_cast<MPI_Comm *>(&mpicomm);
  MPI_Comm_free(commptr);
}

template<typename T>
std::vector<T> DelayedReductor<T>::get(const bool accurate)
{
  if(buffRequest not_eq MPI_REQUEST_NULL) {
    int completed = 0;
    if(accurate) {
      completed = 1;
      MPI(Wait, &buffRequest, MPI_STATUS_IGNORE);
    } else {
      MPI(Test, &buffRequest, &completed, MPI_STATUS_IGNORE);
    }
    if( completed ) {
      return_ret = reduce_ret;
      buffRequest = MPI_REQUEST_NULL;
    }
  }
  return return_ret;
}

template<typename T>
void DelayedReductor<T>::update(const std::vector<T> ret)
{
  assert(ret.size() == arysize);
  if (mpisize <= 1) {
    buffRequest = MPI_REQUEST_NULL;
    return_ret = ret;
    return;
  }

  if(buffRequest not_eq MPI_REQUEST_NULL) {
    MPI(Wait, &buffRequest, MPI_STATUS_IGNORE);
    buffRequest = MPI_REQUEST_NULL;
    return_ret = reduce_ret;
  }
  reduce_ret = ret;
  assert(mpicomm not_eq MPI_COMM_NULL);
  assert(buffRequest == MPI_REQUEST_NULL);
  beginRDX();
}

template<> void DelayedReductor<long double>::beginRDX()
{
  MPI(Iallreduce, MPI_IN_PLACE, reduce_ret.data(), arysize,
                 MPI_LONG_DOUBLE, MPI_SUM, mpicomm, &buffRequest);
}
template<> void DelayedReductor<long>::beginRDX()
{
  MPI(Iallreduce, MPI_IN_PLACE, reduce_ret.data(), arysize,
                 MPI_LONG, MPI_SUM, mpicomm, &buffRequest);
}

template struct DelayedReductor<long>;
template struct DelayedReductor<long double>;

TrainData::TrainData(const std::string _name, const DistributionInfo& distrib,
  bool bPPol, const std::string extrah, const Uint nextra) :
  n_extra(nextra), nThreads(distrib.nThreads), bPolStats(bPPol), name(_name), extra_header(extrah), cntVec(nThreads, 0), qVec(nThreads, LDvec(5, 0)),
  pVec(nThreads, LDvec(3, 0)), eVec(nThreads, LDvec(n_extra, 0))
{
  resetSoft();
  resetHead();
}

TrainData::~TrainData() { }

void TrainData::log(const Real Q, const Real Qerr,
  const std::vector<Real>& polG, const std::vector<Real>& penal,
  const std::initializer_list<Real>& extra, const int thrID)
{
  cntVec[thrID] ++;
  trackQ(Q, Qerr, thrID);
  trackPolicy(polG, penal, thrID);
  const std::vector<Real> tmp = extra;
  assert(tmp.size() == n_extra && bPolStats);
  for(Uint i=0; i<n_extra; ++i) eVec[thrID][i] += tmp[i];
}

void TrainData::log(const Real Q, const Real Qerr,
  const std::initializer_list<Real>& extra, const int thrID)
{
  cntVec[thrID] ++;
  trackQ(Q, Qerr, thrID);
  const std::vector<Real> tmp = extra;
  assert(tmp.size() == n_extra);
  for(Uint i=0; i<n_extra; ++i) eVec[thrID][i] += tmp[i];
}

void TrainData::log(const Real Q, const Real Qerr, const int thrID)
{
  cntVec[thrID] ++;
  trackQ(Q, Qerr, thrID);
  assert(not bPolStats);
}

void TrainData::getMetrics(std::ostringstream& buff)
{
  reduce();
  Utilities::real2SS(buff, q[0], 6, 1);
  Utilities::real2SS(buff, q[1], 6, 0);
  Utilities::real2SS(buff, q[2], 6, 1);
  Utilities::real2SS(buff, q[3], 6, 0);
  Utilities::real2SS(buff, q[4], 6, 0);
  if(bPolStats) {
    Utilities::real2SS(buff, p[0], 6, 1);
    Utilities::real2SS(buff, p[1], 6, 1);
    Utilities::real2SS(buff, p[2], 6, 0);
  }
  for(Uint i=0; i<n_extra; ++i) Utilities::real2SS(buff, e[i], 6, 1);
}

void TrainData::getHeaders(std::ostringstream& buff) const
{
  buff <<"| RMSE | avgQ | stdQ | minQ | maxQ ";

  // polG, penG : average norm of policy/penalization gradients
  // proj : average norm of projection of polG along penG
  //        it is usually negative because curr policy should be as far as
  //        possible from behav. pol. in the direction of update
  if(bPolStats) buff <<"| polG | penG | proj ";

  // beta: coefficient of update gradient to penalization gradient:
  //       g = g_loss * beta + (1-beta) * g_penal
  // dAdv : average magnitude of Qret update
  // avgW : average importance weight
  if(n_extra) buff << extra_header;
}

void TrainData::resetSoft()
{
  for(Uint i=0; i<nThreads; ++i) {
    cntVec[i] = 0;
    qVec[i][0] = 0;
    qVec[i][1] = 0;
    qVec[i][2] = 0;
    pVec[i][0] = 0;
    pVec[i][1] = 0;
    pVec[i][2] = 0;
    qVec[i][3] =  1e9;
    qVec[i][4] = -1e9;
    for(Uint j=0; j<n_extra; ++j) eVec[i][j] = 0;
  }
}

void TrainData::resetHead()
{
  cnt  = 0;

  q[0] = 0;
  q[1] = 0;
  q[2] = 0;
  q[3] =  1e9;
  q[4] = -1e9;

  p[0] = 0;
  p[1] = 0;
  p[2] = 0;

  for(Uint j=0; j<n_extra; ++j) e[j] = 0;
}

void TrainData::reduce()
{
  resetHead();
  for (Uint i=0; i<nThreads; ++i) {
    cnt += cntVec[i];
    q[0] += qVec[i][0];
    q[1] += qVec[i][1];
    q[2] += qVec[i][2];
    q[3]  = std::min(qVec[i][3], q[3]);
    q[4]  = std::max(qVec[i][4], q[4]);
    p[0] += pVec[i][0];
    p[1] += pVec[i][1];
    p[2] += pVec[i][2];
    for(Uint j=0; j<n_extra; ++j) e[j] += eVec[i][j];
  }
  resetSoft();

  q[0] = std::sqrt(q[0]/cnt);
  q[1] /= cnt; // average Q
  q[2] /= cnt; // second moment of Q
  q[2] = std::sqrt(q[2] - q[1]*q[1]); // sdev of Q

  p[0] /= cnt;
  p[1] /= cnt;
  p[2] /= cnt;
  for(Uint j=0; j<n_extra; ++j) e[j] /= cnt;
}

void TrainData::trackQ(const Real Q, const Real err, const int thrID)
{
  qVec[thrID][0] += err*err;
  qVec[thrID][1] += Q;
  qVec[thrID][2] += Q*Q;
  qVec[thrID][3] = std::min(qVec[thrID][3], static_cast<long double>(Q) );
  qVec[thrID][4] = std::max(qVec[thrID][4], static_cast<long double>(Q) );
}

void TrainData::trackPolicy(const std::vector<Real>& polG,
  const std::vector<Real>& penal, const int thrID)
{
  Real tmpPol = 0, tmpPen = 0, tmpPrj = 0;
  for(Uint i=0; i<polG.size(); ++i) {
    tmpPol +=  polG[i]* polG[i];
    tmpPen += penal[i]*penal[i];
    tmpPrj +=  polG[i]*penal[i];
  }
  pVec[thrID][0] += std::sqrt(tmpPol);
  pVec[thrID][1] += std::sqrt(tmpPen);
  static constexpr Real eps = std::numeric_limits<Real>::epsilon();
  pVec[thrID][2] += tmpPrj/(std::sqrt(tmpPen)+eps);
}

StatsTracker::StatsTracker(const Uint N, const DistributionInfo& distrib) :
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
