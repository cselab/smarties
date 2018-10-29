//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Parameters.h"
#include <iomanip>

class Optimizer
{
 protected:
  const MPI_Comm mastersComm;
  const Uint learn_size, pop_size, nThreads;
  const Parameters * const weights;
  const Parameters * const tgt_weights;
  const vector<Parameters*> sampled_weights;
  const Uint pDim = weights->nParams;
  const Real eta_init;
  const Uint batchSize;
  const bool bAsync;
  Uint cntUpdateDelay = 0;
  std::mutex& mpi_mutex;
  std::mutex samples_mutex;

  std::vector<MPI_Request> wVecReq = std::vector<MPI_Request>(pop_size, MPI_REQUEST_NULL);

 public:
  bool bAnnealLearnRate = true;
  const Real lambda, epsAnneal, tgtUpdateAlpha;
  long unsigned nStep = 0;
  nnReal eta = eta_init;

  Optimizer(const Settings&S, const Parameters*const W,
    const Parameters*const WT, const vector<Parameters*>& samples);

  virtual ~Optimizer() {}
  virtual void save(const string fname, const bool bBackup) = 0;
  virtual int restart(const string fname) = 0;

  virtual void prepare_update(const Rvec&L) = 0;
  virtual void apply_update() = 0;

  virtual void getMetrics(ostringstream& buff) = 0;
  virtual void getHeaders(ostringstream& buff) = 0;

  inline const Parameters * getWeights(const int USEW) {
    if(USEW == 0) return weights;
    if(USEW <  0) return tgt_weights;
    if(wVecReq[USEW] == MPI_REQUEST_NULL) return sampled_weights[USEW];
    std::lock_guard<std::mutex> lockW(samples_mutex);
    if(wVecReq[USEW] not_eq MPI_REQUEST_NULL) {
      MPI(Wait, &wVecReq[USEW], MPI_STATUS_IGNORE);
      wVecReq[USEW] = MPI_REQUEST_NULL;
    }
    return sampled_weights[USEW];
  }
};

class AdamOptimizer : public Optimizer
{
 protected:
  const Real beta_1, beta_2;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  const Parameters * const gradSum = weights->allocateGrad(learn_size);
  const Parameters * const _1stMom = weights->allocateGrad(learn_size);
  const Parameters * const _2ndMom = weights->allocateGrad(learn_size);
  const Parameters * const _2ndMax = weights->allocateGrad(learn_size);
  vector<std::mt19937>& generators;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  //const Real alpha_eSGD, gamma_eSGD, eta_eSGD, eps_eSGD, delay;
  //const Uint L_eSGD;
  //nnReal *const _muW_eSGD, *const _muB_eSGD;
  const vector<Parameters*> grads;

 public:

  AdamOptimizer(const Settings&S, const Parameters*const W,
    const Parameters*const WT, const vector<Parameters*>& samples,
    const vector<Parameters*>&G, const Real B1=.9, const Real B2=.999);
  //alpha_eSGD(0.75), gamma_eSGD(10.), eta_eSGD(.1/_s.targetDelay),
  //eps_eSGD(1e-3), delay(_s.targetDelay), L_eSGD(_s.targetDelay),
  //_muW_eSGD(initClean(nWeights)), _muB_eSGD(initClean(nBiases))

  ~AdamOptimizer() {
   _dispose_object(gradSum); _dispose_object(_1stMom);
   _dispose_object(_2ndMom); _dispose_object(_2ndMax);
  }

  void prepare_update(const Rvec& L) override;
  void apply_update() override;

  void save(const string fname, const bool bBackup) override;
  int restart(const string fname) override;
  void getMetrics(ostringstream& buff) override {}
  void getHeaders(ostringstream& buff) override {}
};
