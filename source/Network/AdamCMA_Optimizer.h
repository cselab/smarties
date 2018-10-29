//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Optimizer.h"
#include <iomanip>
class Saru;
#define ACCEL_CMA
//#define FDIFF_CMA

class AdamCMA_Optimizer : public Optimizer
{
 protected:
  const vector<nnReal> popWeights = initializePopWeights(pop_size);
  const nnReal mu_eff = initializeMuEff(popWeights, pop_size);
  const nnReal sumW = initializeSumW(popWeights, pop_size);

  const vector<Parameters*> popNoiseVectors = initWpop(weights, pop_size, learn_size);

  const Parameters * const momNois = weights->allocateGrad(learn_size);
  const Parameters * const avgNois = weights->allocateGrad(learn_size);
  const Parameters * const pathCov = weights->allocateGrad(learn_size);
  const Parameters * const diagCov = weights->allocateGrad(learn_size);

  const vector<Parameters*> popGrads;
  const vector<Parameters*> popGradSums = initWpop(weights, pop_size, learn_size);
  const Parameters * const gradSum = weights->allocateGrad(learn_size);

  const Parameters * const _1stMom = weights->allocateGrad(learn_size);
  const Parameters * const _2ndMom = weights->allocateGrad(learn_size);
  const Parameters * const _2ndMax = weights->allocateGrad(learn_size);

  const int mpi_stride = roundUpSimd( std::ceil( pDim / (Real) learn_size ) );

  const std::vector<int> pStarts, pCounts;
  const Uint pStart, pCount;
  const Real beta_1 = 0.9, beta_2 = 0.999;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  std::vector<Saru *> generators;
  std::vector<std::mt19937 *> stdgens;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  vector<Real> losses = vector<Real>(pop_size, 0);
  //Uint Nswap = 0;

 public:

  AdamCMA_Optimizer(const Settings&S, const Parameters*const W,
    const Parameters*const WT, const vector<Parameters*>&G);

  ~AdamCMA_Optimizer();

  void prepare_update(const Rvec& L) override;
  void apply_update() override;

  void save(const string fname, const bool bBackup) override;
  int restart(const string fname) override;

 protected:
  static inline vector<nnReal> initializePopWeights(const Uint popsz)
  {
    vector<nnReal> ret(popsz); nnReal sum = 0;
    for(Uint i=0; i<popsz; i++) {
      ret[i] = std::log(0.5*(popsz+1)) - std::log(i+1.);
      sum += std::max( ret[i], (nnReal) 0 );
    }
    for(Uint i=0; i<popsz; i++) ret[i] /= sum;
    return ret;
  }

  static inline Real initializeMuEff(const vector<nnReal>popW, const Uint popsz)
  {
    Real sum = 0, sumsq = 0;
    for(Uint i=0; i<popsz; i++) {
      const nnReal W = std::max( popW[i], (nnReal) 0 );
      sumsq += W * W; sum += W;
    }
    return sum * sum / sumsq;
  }

  static inline Real initializeSumW(const vector<nnReal>popW, const Uint popsz)
  {
    Real sum = 0;
    for(Uint i=0; i<popsz; i++) sum += popW[i];
    return sum;
  }
  void getMetrics(ostringstream& buff) override;
  void getHeaders(ostringstream& buff) override;

  std::vector<int> computePstarts() const {
    std::vector<int> ret (learn_size, 0);
    for (int i=0; i < (int) learn_size; i++) ret[i] = mpi_stride * i;
    return ret;
  }
  std::vector<int> computePcounts() const {
    std::vector<int> ret (learn_size, 0);
    for (int i=0; i < (int) learn_size; i++)
      ret[i] = std::min(mpi_stride * (i+1), (int) pDim) - mpi_stride * i;
    return ret;
  }

  void startAllGather(const Uint ID);
};
