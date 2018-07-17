//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Settings.h"

struct ApproximateReductor
{
  const MPI_Comm mpicomm;
  const Uint mpisize, arysize;
  MPI_Request buffRequest = MPI_REQUEST_NULL;
  vector<long double> reduce_ret = vector<long double>(arysize, 0);
  vector<long double> local_vals = vector<long double>(arysize, 0);

  static int getSize(const MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }
  ApproximateReductor(const MPI_Comm c, const Uint N) :
  mpicomm(c), mpisize(getSize(c)), arysize(N)
  { }

  template<typename T> // , MPI_Datatype MPI_RDX_TYPE
  int sync(vector<T>& ret, bool accurate = false)
  {
    if (mpisize <= 1) return 1;
    const bool firstUpdate = buffRequest == MPI_REQUEST_NULL;
    if(not firstUpdate) MPI_Wait(&buffRequest, MPI_STATUS_IGNORE);
    assert(ret.size() == arysize);
    for(size_t i=0; i<ret.size(); i++) local_vals[i] = ret[i];

    if(accurate){
      if(not firstUpdate) die("undefined behavior");
      MPI_Allreduce( local_vals.data(), reduce_ret.data(), arysize,
                     MPI_LONG_DOUBLE, MPI_SUM, mpicomm);
      //accurate result after reduction:
      for(size_t i=0; i<ret.size(); i++) ret[i] = reduce_ret[i];
    } else {
      //inaccurate result coming from return of previous call:
      for(size_t i=0; i<ret.size(); i++) ret[i] = reduce_ret[i];
      MPI_Iallreduce(local_vals.data(), reduce_ret.data(), arysize,
                     MPI_LONG_DOUBLE, MPI_SUM, mpicomm, &buffRequest);
    }
    // if no reduction done, partial sums are meaningless
    return firstUpdate and not accurate;
  }
};

struct TrainData
{
  const Uint n_extra, nThreads, bPolStats;
  const string name, extra_header;

  mutable LDvec cntVec = LDvec(nThreads+1,0);
  mutable vector<LDvec> qVec = vector<LDvec>(nThreads+1, LDvec(5,0));
  mutable vector<LDvec> pVec = vector<LDvec>(nThreads+1, LDvec(3,0));
  mutable vector<LDvec> eVec = vector<LDvec>(nThreads+1, LDvec(n_extra,0));

  // used for debugging purposes to dump stats about gradient. will be removed
  //FILE * wFile = fopen("grads_dist.raw", "ab");
  //FILE * qFile = fopen("onpolQdist.raw", "ab");

  TrainData(const string _name, Settings&set, bool bPPol=0,
    const string extrah = string(), const Uint nextra=0) : n_extra(nextra),
    nThreads(set.nThreads), bPolStats(bPPol), name(_name), extra_header(extrah)
  {
    resetSoft();
    resetHead();
  }
  ~TrainData() {
    //fclose(wFile);
    //fclose(qFile);
  }

  void log(const Real Q, const Real Qerr,
    const std::vector<Real> polG, const std::vector<Real> penal,
    std::initializer_list<Real> extra, const int thrID) {
    cntVec[thrID+1] ++;
    trackQ(Q, Qerr, thrID);
    trackPolicy(polG, penal, thrID);
    const vector<Real> tmp = extra;
    assert(tmp.size() == n_extra && bPolStats);
    for(Uint i=0; i<n_extra; i++) eVec[thrID+1][i] += tmp[i];
  }
  void log(const Real Q, const Real Qerr,
    std::initializer_list<Real> extra, const int thrID) {
    cntVec[thrID+1] ++;
    trackQ(Q, Qerr, thrID);
    const vector<Real> tmp = extra;
    assert(tmp.size() == n_extra && not bPolStats);
    for(Uint i=0; i<n_extra; i++) eVec[thrID+1][i] += tmp[i];
  }
  void log(const Real Q, const Real Qerr, const int thrID) {
    cntVec[thrID+1] ++;
    trackQ(Q, Qerr, thrID);
    assert(not bPolStats);
  }

  void getMetrics(ostringstream& buff)
  {
    reduce();
    real2SS(buff, qVec[0][0], 6, 1);
    real2SS(buff, qVec[0][1], 6, 0);
    real2SS(buff, qVec[0][2], 6, 1);
    real2SS(buff, qVec[0][3], 6, 0);
    real2SS(buff, qVec[0][4], 6, 0);
    if(bPolStats) {
      real2SS(buff, pVec[0][0], 6, 1);
      real2SS(buff, pVec[0][1], 6, 1);
      real2SS(buff, pVec[0][2], 6, 0);
    }
    for(Uint i=0; i<n_extra; i++) real2SS(buff, eVec[0][i], 6, 1);
  }
  void getHeaders(ostringstream& buff) const
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

 private:
  void resetSoft() {
    for(Uint i=1; i<=nThreads; i++) {
      cntVec[i] = 0;
      qVec[i][0] = 0;
      qVec[i][1] = 0;
      qVec[i][2] = 0;
      pVec[i][0] = 0;
      pVec[i][1] = 0;
      pVec[i][2] = 0;
      qVec[i][3] =  1e9;
      qVec[i][4] = -1e9;
      for(Uint j=0; j<n_extra; j++) eVec[i][j] = 0;
    }
  }
  void resetHead() {
    cntVec[0]  = 0;
    qVec[0][0] = 0;
    qVec[0][1] = 0;
    qVec[0][2] = 0;
    pVec[0][0] = 0;
    pVec[0][1] = 0;
    pVec[0][2] = 0;
    qVec[0][3] =  1e9;
    qVec[0][4] = -1e9;
    for(Uint j=0; j<n_extra; j++) eVec[0][j] = 0;
  }

  void reduce()
  {
    resetHead();
    for (Uint i=0; i<nThreads; i++) {
      cntVec[0] += cntVec[i+1];
      qVec[0][0] += qVec[i+1][0];
      qVec[0][1] += qVec[i+1][1];
      qVec[0][2] += qVec[i+1][2];
      qVec[0][3]  = std::min(qVec[i+1][3], qVec[0][3]);
      qVec[0][4]  = std::max(qVec[i+1][4], qVec[0][4]);
      pVec[0][0] += pVec[i+1][0];
      pVec[0][1] += pVec[i+1][1];
      pVec[0][2] += pVec[i+1][2];
      for(Uint j=0; j<n_extra; j++)
        eVec[0][j] += eVec[i+1][j];
    }
    resetSoft();

    qVec[0][0] = std::sqrt(qVec[0][0]/cntVec[0]);
    qVec[0][1] /= cntVec[0]; // average Q
    qVec[0][2] /= cntVec[0]; // second moment of Q
    qVec[0][2] = std::sqrt(qVec[0][2] - qVec[0][1]*qVec[0][1]); // sdev of Q

    pVec[0][0] /= cntVec[0];
    pVec[0][1] /= cntVec[0];
    pVec[0][2] /= cntVec[0];
    for(Uint j=0; j<n_extra; j++) eVec[0][j] /= cntVec[0];

    #if 0
      if(outBuf.size()) {
        fwrite(outBuf.data(), sizeof(float), outBuf.size(), qFile);
        fflush(qFile);
        outBuf.resize(0);
      }
    #endif
  }

  inline void trackQ(const Real Q, const Real err, const int thrID) {
    qVec[thrID+1][0] += err*err;
    qVec[thrID+1][1] += Q;
    qVec[thrID+1][2] += Q*Q;
    qVec[thrID+1][3] = std::min(qVec[thrID+1][3], static_cast<long double>(Q));
    qVec[thrID+1][4] = std::max(qVec[thrID+1][4], static_cast<long double>(Q));
  }

  inline void trackPolicy(const std::vector<Real> polG,
    const std::vector<Real> penal, const int thrID) {
    #if 0
      if(thrID == 1) {
        float normT = 0, dot = 0;
        for(Uint i = 0; i < polG.size(); i++) {
          dot += polG[i] * penalG[i]; normT += penalG[i] * penalG[i];
        }
        float ret[]={dot/std::sqrt(normT)};
        fwrite(ret, sizeof(float), 1, wFile);
      }
    #endif

    #if 0
      if(thrID == 1) {
        Rvec Gcpy = gradient;
        F[0]->gradStats->clip_vector(Gcpy);
        Gcpy = Rvec(&Gcpy[pol_start[0]], &Gcpy[pol_start[0]+polG.size()]);
        float normT = 0, dot = 0;
        for(Uint i = 0; i < polG.size(); i++) {
          dot += Gcpy[i] * penalG[i]; normT += penalG[i] * penalG[i];
        }
        float ret[]={dot/std::sqrt(normT)};
        fwrite(ret, sizeof(float), 1, wFile);
      }
    #endif

    Real tmpPol = 0, tmpPen = 0, tmpPrj = 0;
    for(Uint i=0; i<polG.size(); i++) {
      tmpPol +=  polG[i]* polG[i];
      tmpPen += penal[i]*penal[i];
      tmpPrj +=  polG[i]*penal[i];
    }
    pVec[thrID+1][0] += std::sqrt(tmpPol);
    pVec[thrID+1][1] += std::sqrt(tmpPen);
    static constexpr Real eps = numeric_limits<Real>::epsilon();
    pVec[thrID+1][2] += tmpPrj/(std::sqrt(tmpPen)+eps);
  }
};

struct StatsTracker
{
  const Uint n_stats;
  const MPI_Comm comm;
  const Uint nThreads, learn_size, learn_rank;
  const Real grad_cut_fac, learnR;
  mutable LDvec cntVec = LDvec(nThreads+1,0);
  mutable vector<LDvec> avgVec = vector<LDvec>(nThreads+1, LDvec());
  mutable vector<LDvec> stdVec = vector<LDvec>(nThreads+1, LDvec());
  LDvec instMean, instStdv;
  mutable Real numCut = 0, numTot = 0;
  unsigned long nStep = 0;
  Real cutRatio = 0;

  ApproximateReductor reductor = ApproximateReductor(comm, 2*n_stats +1);

  StatsTracker(const Uint N, Settings& set, Real fac) :
  n_stats(N), comm(set.mastersComm), nThreads(set.nThreads),
  learn_size(set.learner_size), learn_rank(set.learner_rank), grad_cut_fac(fac),
  learnR(set.learnrate)
  {
    avgVec[0].resize(n_stats, 0); stdVec[0].resize(n_stats, 10);
    instMean.resize(n_stats, 0); instStdv.resize(n_stats, 0);
    #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
    for (Uint i=0; i<nThreads; i++) // numa aware allocation
     #pragma omp critical
     {
       avgVec[i+1].resize(n_stats, 0);
       stdVec[i+1].resize(n_stats, 0);
     }
  }

  inline void track_vector(const Rvec grad, const Uint thrID) const
  {
    assert(n_stats==grad.size());
    cntVec[thrID+1] += 1;
    for (Uint i=0; i<n_stats; i++) {
      avgVec[thrID+1][i] += grad[i];
      stdVec[thrID+1][i] += grad[i]*grad[i];
    }
  }
  inline void clip_vector(Rvec& grad) const
  {
    assert(grad.size() == n_stats);
    Uint ret = 0;
    Real change = 0;
    for (Uint i=0; i<n_stats && grad_cut_fac>=1; i++) {
      //#ifdef IMPORTSAMPLE
      //  assert(data->Set[seq]->tuples[samp]->weight>0);
      //  grad[i] *= data->Set[seq]->tuples[samp]->weight;
      //#endif
      if(grad[i]> grad_cut_fac*stdVec[0][i] && stdVec[0][i]>2.2e-16) {
        //printf("Cut %u was:%f is:%LG\n",i,grad[i], grad_cut_fac*stdVec[0][i]);
        change+=(grad[i]-grad_cut_fac*stdVec[0][i])/(grad_cut_fac*stdVec[0][i]);
        grad[i] = grad_cut_fac*stdVec[0][i];
        ret += 1;
      } else
      if(grad[i]< -grad_cut_fac*stdVec[0][i] && stdVec[0][i]>2.2e-16) {
        //printf("Cut %u was:%f is:%LG\n",i,grad[i],-grad_cut_fac*stdVec[0][i]);
        change-=(grad[i]+grad_cut_fac*stdVec[0][i])/(grad_cut_fac*stdVec[0][i]);
        grad[i] = -grad_cut_fac*stdVec[0][i];
        ret += 1;
      }
      //else printf("Not cut\n");
    }
    #pragma omp atomic
    //numCut += ret;
    numCut += change;
    #pragma omp atomic
    numTot += n_stats;
  }

  inline void advance()
  {
    std::fill(avgVec[0].begin(),  avgVec[0].end(), 0);
    std::fill(stdVec[0].begin(),  stdVec[0].end(), 0);
    cntVec[0] = 0;

    for (Uint i=1; i<=nThreads; i++) {
      cntVec[0] += cntVec[i];
      for (Uint j=0; j<n_stats; j++) {
        avgVec[0][j] += avgVec[i][j];
        stdVec[0][j] += stdVec[i][j];
      }
      cntVec[i] = 0;
      std::fill(avgVec[i].begin(),  avgVec[i].end(), 0);
      std::fill(stdVec[i].begin(),  stdVec[i].end(), 0);
    }
  }
  inline void update()
  {
    cntVec[0] = std::max((long double)2.2e-16, cntVec[0]);
    for (Uint j=0; j<n_stats; j++) {
      const Real   mean = avgVec[0][j] / cntVec[0];
      const Real sqmean = stdVec[0][j] / cntVec[0];
      stdVec[0][j] = std::sqrt(sqmean); // - mean*mean
      avgVec[0][j] = mean;
    }
  }
  inline void printToFile(const string base)
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
      vector<float> printvals(n_stats*2);
      for (Uint i=0; i<n_stats; i++) {
        printvals[i]         = avgVec[0][i];
        printvals[i+n_stats] = stdVec[0][i];
      }
      fwrite(printvals.data(), sizeof(float), n_stats*2, pFile);
      fflush(pFile); fclose(pFile);
    }
  }
  void finalize(const LDvec&oldM, const LDvec&oldS)
  {
    instMean = avgVec[0];
    instStdv = stdVec[0];
    nStep++;
    //const Real learnRate = learnR / (1 + nStep * ANNEAL_RATE);
    for (Uint i=0; i<n_stats; i++) {
      avgVec[0][i] = (1-CLIP_LEARNR)*oldM[i] +CLIP_LEARNR*avgVec[0][i];
      stdVec[0][i] = (1-CLIP_LEARNR)*oldS[i] +CLIP_LEARNR*stdVec[0][i];
      //stdVec[0][i]=std::max((1-CLIP_LEARNR)*oldstd[i], stdVec[0][i]);
    }
  }
  double clip_ratio()
  {
    cutRatio = numCut / (Real) numTot;
    numCut = 0; numTot = 0;
    return cutRatio;
  }
  inline void reduce_stats(const string base, const Uint iter = 0)
  {
    const LDvec oldsum = avgVec[0], oldstd = stdVec[0];
    assert(cntVec.size()>1);

    advance();

    if (learn_size > 1) {
      LDvec res = avgVec[0];
      res.insert(res.end(), stdVec[0].begin(), stdVec[0].end());
      res.push_back(cntVec[0]);
      assert(res.size() == 2*n_stats+1);
      bool skipped = reductor.sync(res);
      if(skipped) {
        avgVec[0] = oldsum; stdVec[0] = oldstd;
        return;
      } else {
        for (Uint i=0; i<n_stats; i++) {
          avgVec[0][i] = res[i]; stdVec[0][i] = res[i+n_stats];
        }
        cntVec[0] = res[2*n_stats];
      }
    }
    update();

    if(iter % 1000 == 0) printToFile(base);
    finalize(oldsum, oldstd);
  }
  inline void reduce_approx(const string base, const Uint iter = 0)
  {
    const LDvec oldsum = avgVec[0], oldstd = stdVec[0];
    assert(cntVec.size()>1);
    advance();
    update();
    if(iter % 1000 == 0) printToFile(base);
    finalize(oldsum, oldstd);
  }

  //void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  //{
  //  screenOut<<" "<<name<<" avg:["<<print(instMean)
  //                <<"] std:["<<print(instStdv)<<"]";
  //  fileOut<<" "<<print(instMean)<<" "<<print(stdVec[0]);
  //}
};
