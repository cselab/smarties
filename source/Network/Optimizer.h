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
  const Uint learn_size;
  const Real eta, beta_1, beta_2;
  long unsigned nStep = 0;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  const Real lambda, epsAnneal, tgtUpdateAlpha;
  const Parameters * const weights;
  const Parameters * const tgt_weights;
  const Parameters * const gradSum;
  const Parameters * const _1stMom;
  const Parameters * const _2ndMom;
  const Parameters * const _2ndMax;
  vector<std::mt19937>& generators;
  Uint cntUpdateDelay = 0, totGrads = 0;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  MPI_Request batchRequest = MPI_REQUEST_NULL;
  //const Real alpha_eSGD, gamma_eSGD, eta_eSGD, eps_eSGD, delay;
  //const Uint L_eSGD;
  //nnReal *const _muW_eSGD, *const _muB_eSGD;

 public:
  bool bAnnealLearnRate = true;

  Optimizer(Settings&S, const Parameters*const W, const Parameters*const W_TGT,
    const Real B1=.9, const Real B2=.999) : mastersComm(S.mastersComm),
    learn_size(S.learner_size), eta(S.learnrate), beta_1(B1), beta_2(B2),
    lambda(S.nnLambda), epsAnneal(S.epsAnneal), tgtUpdateAlpha(S.targetDelay),
    weights(W), tgt_weights(W_TGT), gradSum(W->allocateGrad()),
    _1stMom(W->allocateGrad()), _2ndMom(W->allocateGrad()),
    _2ndMax(W->allocateGrad()), generators(S.generators) {
      //_2ndMom->set(std::sqrt(nnEPS));
      //_2ndMom->set(1);
    }
  //alpha_eSGD(0.75), gamma_eSGD(10.), eta_eSGD(.1/_s.targetDelay),
  //eps_eSGD(1e-3), delay(_s.targetDelay), L_eSGD(_s.targetDelay),
  //_muW_eSGD(initClean(nWeights)), _muB_eSGD(initClean(nBiases))

  virtual ~Optimizer() {
   _dispose_object(gradSum); _dispose_object(_1stMom);
   _dispose_object(_2ndMom); _dispose_object(_2ndMax);
  }

  inline void prepare_update(const int batchsize, const vector<Parameters*>& grads) {
    prepare_update(batchsize, &grads);
  }

  void prepare_update(const int batchsize, const vector<Parameters*>* grads = nullptr);

  void apply_update();

  void save(const string fname);
  int restart(const string fname);
};
