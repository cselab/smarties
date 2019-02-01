//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Optimizer.h"

class Saru;
#define ACCEL_CMA
//#define FDIFF_CMA

class CMA_Optimizer : public Optimizer
{
 protected:
  const vector<nnReal> popWeights = initializePopWeights(pop_size);
  const nnReal mu_eff = initializeMuEff(popWeights, pop_size);
  const nnReal sumW = initializeSumW(popWeights, pop_size);
  const vector<Parameters*> popNoiseVectors = initWpop(weights, pop_size, learn_size);
  const Parameters * const momNois = weights->allocateGrad(learn_size);
  const Parameters * const avgNois = weights->allocateGrad(learn_size);
  const Parameters * const negNois = weights->allocateGrad(learn_size);
  const Parameters * const pathCov = weights->allocateGrad(learn_size);
  const Parameters * const pathDif = weights->allocateGrad(learn_size);
  const Parameters * const diagCov = weights->allocateGrad(learn_size);
  const int mpi_stride = roundUpSimd( std::ceil( pDim / (Real) learn_size ) );

  const std::vector<int> pStarts, pCounts;
  const Uint pStart, pCount;
  std::vector<Saru *> generators;
  std::vector<std::mt19937 *> stdgens;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  vector<Real> losses = vector<Real>(pop_size, 0);
  //Uint Nswap = 0;

 public:

  CMA_Optimizer(const Settings&S, const Parameters*const W,
    const Parameters*const WT, const vector<Parameters*>&G);

  ~CMA_Optimizer();

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
      #ifdef FDIFF_CMA
        sum += std::fabs( ret[i] );
      #else
        sum += std::max( ret[i], (nnReal) 0 );
      #endif
    }
    for(Uint i=0; i<popsz; i++) ret[i] /= sum;
    return ret;
  }

  static inline Real initializeMuEff(const vector<nnReal>popW, const Uint popsz)
  {
    Real sum = 0, sumsq = 0;
    for(Uint i=0; i<popsz; i++) {
      #ifdef FDIFF_CMA
        const nnReal W = std::fabs( popW[i] );
      #else
        const nnReal W = std::max( popW[i], (nnReal) 0 );
      #endif
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
