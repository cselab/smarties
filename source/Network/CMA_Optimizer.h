//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_CMA_Optimizer_h
#define smarties_CMA_Optimizer_h

#include "Optimizer.h"

class Saru;
// #define ACCEL_CMA
// #define FDIFF_CMA

namespace smarties
{

class CMA_Optimizer : public Optimizer
{
protected:
  const std::vector<nnReal> popWeights = initializePopWeights(populationSize);
  const nnReal mu_eff = initializeMuEff(popWeights, populationSize);
  const nnReal sumW = initializeSumW(popWeights, populationSize);
  const Uint mpiDistribOpsStride = Utilities::roundUpSimd( std::ceil( weights->nParams/(Real)learn_size ) );
  const std::vector<std::shared_ptr<Parameters>> popNoiseVectors =
                                       allocManyParams(weights, populationSize);
  const std::shared_ptr<Parameters> momNois = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> avgNois = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> negNois = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> pathCov = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> pathDif = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> diagCov = weights->allocateEmptyAlike();

  const std::vector<Uint> pStarts, pCounts;
  const Uint pStart, pCount;
  std::vector<std::shared_ptr<Saru>> generators;
  std::vector<std::shared_ptr<std::mt19937>> stdgens;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  std::vector<Real> losses = std::vector<Real>(populationSize, 0);
  //Uint Nswap = 0;

  nnReal computeStdDevScale() const
  {
    return bAnnealLearnRate? Utilities::annealRate(eta, nStep, epsAnneal) : eta;
  }
public:

  CMA_Optimizer(const Settings& S, const DistributionInfo& D,
                const std::shared_ptr<Parameters>& W);

  void prepare_update(const Rvec& L) override;
  void apply_update() override;

  void save(const std::string fname, const bool bBackup) override;
  int restart(const std::string fname) override;

 protected:
  static inline std::vector<nnReal> initializePopWeights(const Uint popSize)
  {
    std::vector<nnReal> ret(popSize);
    nnReal sum = 0;
    for(Uint i=0; i<popSize; ++i)
    {
      ret[i] = std::log(0.5*(popSize+1)) - std::log(i+1.);
      #ifdef FDIFF_CMA
        sum += std::fabs( ret[i] );
      #else
        sum += std::max( ret[i], (nnReal) 0 );
      #endif
    }
    for(Uint i=0; i<popSize; ++i) ret[i] /= sum;
    return ret;
  }

  static inline Real initializeMuEff(const std::vector<nnReal>popW,
                                     const Uint popSize)
  {
    Real sum = 0, sumsq = 0;
    for(Uint i=0; i<popSize; ++i) {
      #ifdef FDIFF_CMA
        const nnReal W = std::fabs( popW[i] );
      #else
        const nnReal W = std::max( popW[i], (nnReal) 0 );
      #endif
      sumsq += W * W; sum += W;
    }
    return sum * sum / sumsq;
  }

  static inline Real initializeSumW(const std::vector<nnReal>popW, const Uint popsz)
  {
    Real sum = 0;
    for(Uint i=0; i<popsz; ++i) sum += popW[i];
    return sum;
  }
  void getMetrics(std::ostringstream& buff) override;
  void getHeaders(std::ostringstream& buff, const std::string nnName) override;

  std::vector<Uint> computePstarts() const
  {
    std::vector<Uint> ret (learn_size, 0);
    for (Uint i=0; i < learn_size; ++i)
      ret[i] = mpiDistribOpsStride * i;
    return ret;
  }
  std::vector<Uint> computePcounts() const
  {
    std::vector<Uint> ret (learn_size, 0);
    for (Uint i=0; i < learn_size; ++i) {
      const Uint end = std::min(mpiDistribOpsStride*(i+1), weights->nParams);
      const Uint beg = mpiDistribOpsStride * i;
      ret[i] = end - beg;
    }
    return ret;
  }

  bool ready2UpdateWeights() override
  {
    if(paramRequest == MPI_REQUEST_NULL) return true;
    int completed = 0;
    MPI(Test, &paramRequest, &completed, MPI_STATUS_IGNORE);
    return completed;
  }

  void startAllGather(const Uint ID);
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
