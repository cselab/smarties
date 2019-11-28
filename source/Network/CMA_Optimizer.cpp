//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMA_Optimizer.h"
#include "../Utils/SstreamUtilities.h"
#include "../extern/saruprng.h"
#include <algorithm>
#include <unistd.h>

namespace smarties
{

CMA_Optimizer::CMA_Optimizer(const Settings& S, const DistributionInfo& D,
                             const std::shared_ptr<Parameters>& W):
  Optimizer(S,D,W), pStarts(computePstarts()), pCounts(computePcounts()),
  pStart(pStarts[learn_rank]), pCount(pCounts[learn_rank])
{
  if(D.world_rank == 0)
    printf("Optimizer: Parameter updates using diagonal-only version of CMA-ES.\n");
  diagCov->set(1);
  pathCov->set(0);
  pathDif->set(0);

  std::vector<unsigned long> seed(3*populationSize);
  std::generate(seed.begin(), seed.end(), [&](){ return D.generators[0](); } );
  generators.resize(populationSize, nullptr);
  stdgens.resize(populationSize, nullptr);
  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for(Uint i=0; i<populationSize; ++i) {
    generators[i] = std::make_unique<Saru>(seed[3*i], seed[3*i+1], seed[3*i+2]);
    stdgens[i] = std::make_unique<std::mt19937>(seed[3*i +0]);
  }

  const nnReal* const _S = diagCov->params;
  const nnReal* const M = weights->params;
  //const nnReal* const D = pathDif->params;
  const nnReal _eta = computeStdDevScale();
  #pragma omp parallel num_threads(nThreads)
  for(Uint i=1; i<populationSize; ++i)
  {
    Saru & gen = * generators[omp_get_thread_num()].get();
    nnReal* const Y = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    #pragma omp for schedule(static)
    for(Uint w=pStart; w<pStart+pCount; ++w) {
      Y[w] = gen.f_mean0_var1() * _S[w];
      X[w] = M[w] + _eta * Y[w]; //+ _eta*1e-2*D[w];
    }
    #pragma omp single nowait
    startAllGather(i);
  }

  for(Uint i=1; i<populationSize; ++i)
    MPI(Wait, &weightsMPIrequests[i], MPI_STATUS_IGNORE);
}

void CMA_Optimizer::prepare_update(const Rvec&L)
{
  assert(L.size() == populationSize);
  losses = L;
  if (learn_size > 1) { //add up losses across master ranks
    MPI(Iallreduce, MPI_IN_PLACE, losses.data(), populationSize, SMARTIES_MPI_VALUE_TYPE,
                   MPI_SUM, learnersComm, &paramRequest);
  }
  nStep++;
}

void CMA_Optimizer::apply_update()
{
  if(nStep == 0) die("nStep == 0");
  if(learn_size > 1) {
    if(paramRequest == MPI_REQUEST_NULL) die("Did not start reduction");
    MPI(Wait, &paramRequest, MPI_STATUS_IGNORE);
  }

  // steps:
  // 0. Backup mean weights and prepare arrays for reductions
  //    (note that parameter vector [0] is always the mean, therefore
  //     associated sampled noise vector is 0)
  // 1. Compute new weighted mean. (MPI: start allgather for population mean)
  // 2. Compute weighted mean noise vector (for update path) and estimate
  //    of noise vector's second moment (for rank-mu update).
  // 3. Update path integral and diagonal covariance^(1/2) matrix. For stability
  //    we apply bounds to the entries of the covariance matrix.
  // 4. Sample noise vector and population from new mean and cov^(1/2) matrix.

  // Parallelization strategy:
  //    MPI) Avoid allreduce, prefer allgather. Each MPI rank works on a chunk
  //         of the various parameter vectors (from pStart to pStart + pCount).
  // OpenMP) Mixed. Some loops are parallelized over vector length pCount (1.)
  //         Some loops are parallelized over population size (3.)
  std::vector<Uint> inds(populationSize, 0);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );

  // 0.
  sampled_weights[0]->copy(weights);
  popNoiseVectors[0]->clear();
  momNois->clear();
  avgNois->clear();
  weights->clear();

  static constexpr nnReal c1cov = 1e-5;
  static constexpr nnReal c_sig = 1e-3;
  const nnReal alpha = 1 - c1cov - sumW * mu_eff * c1cov;
  const nnReal alphaP = 1 - c_sig;

  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  const nnReal _eta = computeStdDevScale();

  #pragma omp parallel num_threads(nThreads)
  {
    const Uint thrID = omp_get_thread_num();

    // 1.
    for(Uint i=0; i<populationSize; ++i)
    {
      nnReal * const M = weights->params;
      const nnReal wC = popWeights[i];
      if(wC <= 0) continue; // bad samples are not used to update mean

      const nnReal* const X = sampled_weights[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(M,X : VEC_WIDTH) nowait
      for(Uint w=pStart; w<pStart+pCount; ++w) M[w] += wC * X[w];
    }

    #pragma omp barrier // wait for all parellel-nowait loops above

    if(thrID == 0) startAllGather(0);

    // 2.
    for(Uint i=0; i<populationSize; ++i)
    {
      const nnReal wC = popWeights[i];
      const nnReal wZ = std::max(wC, (nnReal) 0 );

      nnReal * const B = momNois->params;
      nnReal * const A = avgNois->params;
      const nnReal* const Y = popNoiseVectors[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(A,B,Y : VEC_WIDTH) nowait
      for(Uint w=pStart; w<pStart+pCount; ++w) {
        B[w] += wC * Y[w]*Y[w];
        A[w] += wZ * Y[w];
      }
    }

    const nnReal * const B = momNois->params;
    const nnReal * const A = avgNois->params;

    nnReal * const P = pathCov->params;
    nnReal * const S = diagCov->params;

    // 3.
    #pragma omp for simd schedule(static) aligned(P,A,S,B : VEC_WIDTH)
    for(Uint w=pStart; w<pStart+pCount; ++w)
    {
      P[w] = alphaP * P[w] + updSigP * A[w];
      S[w] = std::sqrt( alpha*S[w]*S[w] + c1cov*P[w]*P[w] + mu_eff*c1cov*B[w] );
      S[w] = std::max(std::min(S[w], (nnReal) 10.0), (nnReal) 0.01); //safety
    }

    const nnReal* const M = weights->params;
    Saru & gen = * generators[thrID].get();

    // 4.
    #if 0 // Parellized over pop size. Benefit: gathers start asap.
      #pragma omp for schedule(static, 1) nowait
      for(Uint i=1; i<populationSize; ++i)
      {
        nnReal* const Y = popNoiseVectors[i]->params;
        nnReal* const X = sampled_weights[i]->params;
        for(Uint w=pStart; w<pStart+pCount; ++w) {
          Y[w] = gen.f_mean0_var1() * S[w];
          X[w] = M[w] + _eta * Y[w]; //+ _eta*1e-2*D[w];
        }
        startAllGather(i);
      }
    #else // Parellized over pCount. Benefit: less compulsory cache misses.
      for(Uint i=1; i<populationSize; ++i)
      {
        nnReal* const Y = popNoiseVectors[i]->params;
        nnReal* const X = sampled_weights[i]->params;
        #pragma omp for schedule(static) nowait
        for(Uint w=pStart; w<pStart+pCount; ++w) {
          Y[w] = gen.f_mean0_var1() * S[w];
          X[w] = M[w] + _eta * Y[w];
        }
      }
    #endif
  }
  #if 1 // Parellized over pCount. Benefit: less compulsory cache misses.
    for(Uint i=1; i<populationSize; ++i) startAllGather(i);
  #endif

  // Mean parameter vector needs to be available when we exit:
  MPI(Wait, &weightsMPIrequests[0], MPI_STATUS_IGNORE);
}

void CMA_Optimizer::save(const NetSaveF_t& saveFunc,
                         const std::string fname,
                         const bool backup)
{
  saveFunc(weights.get(), fname+"_weights", backup);
  saveFunc(pathCov.get(), fname+"_pathCov", backup);
  saveFunc(diagCov.get(), fname+"_diagCov", backup);

  if(backup) {
    std::ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    saveFunc(weights.get(), fname+"_"+ss.str()+"_weights", false);
    saveFunc(pathCov.get(), fname+"_"+ss.str()+"_pathCov", false);
    saveFunc(diagCov.get(), fname+"_"+ss.str()+"_diagCov", false);
  }
}

int CMA_Optimizer::restart(const NetLoadF_t& loadFunc, const std::string fname)
{
  char currDirectory[512];
  getcwd(currDirectory, 512);
  chdir(distrib.initial_runDir);

  int ret = loadFunc(weights.get(), fname+"_weights");
  loadFunc(pathCov.get(), fname+"_pathCov");
  loadFunc(diagCov.get(), fname+"_diagCov");

  chdir(currDirectory);
  return ret;
}

void CMA_Optimizer::getMetrics(std::ostringstream& buff)
{
  Utilities::real2SS(buff, weights->compute_weight_norm(), 7, 1);
  if(tgtUpdateAlpha > 0)
    Utilities::real2SS(buff, weights->compute_weight_dist(target_weights),6,1);
  //buff<<" "<<std::setw(5)<<Nswap/2; Nswap = 0; //each swap counted twice
  const Real norm = std::pow(diagCov->compute_weight_norm(),2)/weights->nParams;
  Utilities::real2SS(buff, norm, 6, 1);
}
void CMA_Optimizer::getHeaders(std::ostringstream&buff,const std::string nnName)
{
  buff << std::left << std::setfill(' ') <<"| " << std::setw(6) << nnName;
  if(tgtUpdateAlpha > 0) buff << "| dTgt ";
  buff << "| avgC "; //Nswp |
}

void CMA_Optimizer::startAllGather(const Uint ID)
{
  if( learn_size < 2 ) return;

  if(weightsMPIrequests[ID] not_eq MPI_REQUEST_NULL) {
    MPI(Wait, &weightsMPIrequests[ID], MPI_STATUS_IGNORE);
  }

  nnReal * const P = ID>0 ? sampled_weights[ID]->params : weights->params;
  //MPI(Iallgatherv, MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, P, pCounts.data(),
  //  pStarts.data(), SMARTIES_MPI_NNVALUE_TYPE, mastersComm, &wVecReq[ID]);
  MPI(Iallgather, MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, P, mpiDistribOpsStride,
    SMARTIES_MPI_NNVALUE_TYPE, learnersComm, &weightsMPIrequests[ID]);
}

} // end namespace smarties
