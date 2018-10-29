//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMA_Optimizer_MPI.h"
#include "saruprng.h"
#include <algorithm>

CMA_Optimizer::CMA_Optimizer(const Settings&S, const Parameters*const W,
  const Parameters*const WT, const vector<Parameters*>&G): Optimizer(S,W,WT,G),
  pStarts(computePstarts()), pCounts(computePcounts()),
  pStart(pStarts[S.learner_rank]), pCount(pCounts[S.learner_rank]) {
  diagCov->set(1);
  pathCov->set(0);
  pathDif->set(0);

  std::vector<unsigned long> seed(3*pop_size) ;
  std::generate(seed.begin(), seed.end(), [&](){return S.generators[0]();});
  generators.resize(pop_size, nullptr);
  stdgens.resize(pop_size, nullptr);
  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for(Uint i=0; i<pop_size; i++) {
    generators[i] = new Saru(seed[3*i +0], seed[3*i +1], seed[3*i +2]);
    stdgens[i] = new std::mt19937(seed[3*i +0]);
  }

  const nnReal* const _S = diagCov->params;
  const nnReal* const M = weights->params;
  //const nnReal* const D = pathDif->params;
  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;
  #pragma omp parallel num_threads(nThreads)
  for(Uint i=1; i<pop_size; i++)
  {
    Saru & gen = * generators[omp_get_thread_num()];
    nnReal* const Y = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    #pragma omp for schedule(static)
    for(Uint w=pStart; w<pStart+pCount; w++) {
      Y[w] = gen.f_mean0_var1() * _S[w];
      X[w] = M[w] + _eta * Y[w]; //+ _eta*1e-2*D[w];
    }
    #pragma omp single nowait
    startAllGather(i);
  }

  for(Uint i=1; i<pop_size; i++) MPI(Wait, &wVecReq[i], MPI_STATUS_IGNORE);
}

CMA_Optimizer::~CMA_Optimizer() {
  _dispose_object(momNois);
  _dispose_object(avgNois);
  _dispose_object(negNois);

  _dispose_object(pathCov);
  _dispose_object(pathDif);
  _dispose_object(diagCov);

 for(auto& ptr: popNoiseVectors) _dispose_object(ptr);
 for(auto& ptr: generators) _dispose_object(ptr);
}

void CMA_Optimizer::prepare_update(const Rvec&L) {
  assert(L.size() == pop_size);
  losses = L;
  if (learn_size > 1) { //add up losses across master ranks
    MPI(Iallreduce, MPI_IN_PLACE, losses.data(), pop_size, MPI_VALUE_TYPE,
                   MPI_SUM, mastersComm, &paramRequest);
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

  std::vector<Uint> inds(pop_size,0);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );

  sampled_weights[0]->copy(weights); // first backup mean weights
  popNoiseVectors[0]->clear();       // sample 0 is always mean W, no noise
  momNois->clear(); avgNois->clear(); // prepare for
  weights->clear(); //negNois->clear(); // reductions

  static constexpr nnReal c1cov = 1e-5;
  static constexpr nnReal c_sig = 1e-3;
  const nnReal alpha = 1 - c1cov - sumW*mu_eff*c1cov;
  const nnReal alphaP = 1 - c_sig;

  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;

  #pragma omp parallel num_threads(nThreads)
  {
    const Uint thrID = omp_get_thread_num();
    for(Uint i=0; i<pop_size; i++)
    {
      const nnReal wC = popWeights[i];
      #ifndef FDIFF_CMA
        if(wC <=0 ) continue;
      #endif
      nnReal * const M = weights->params;
      const nnReal* const X = sampled_weights[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(M,X : VEC_WIDTH) nowait
      for(Uint w=pStart; w<pStart+pCount; w++) M[w] += wC * X[w];
    }

    #pragma omp barrier
    if(thrID == 0) startAllGather(0);

    for(Uint i=0; i<pop_size; i++) {
      const nnReal wC = popWeights[i];
      #ifdef FDIFF_CMA
        const nnReal wZ = wC;
      #else
        const nnReal wZ = std::max(wC, (nnReal) 0 );
      #endif
      nnReal * const B = momNois->params;
      nnReal * const A = avgNois->params;
      const nnReal* const Y = popNoiseVectors[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(A,B,Y : VEC_WIDTH) nowait
      for(Uint w=pStart; w<pStart+pCount; w++) {
        B[w] += wC * Y[w]*Y[w]; A[w] += wZ * Y[w];
      }
    }

    //const nnReal * const C = negNois->params;
    const nnReal * const B = momNois->params;
    const nnReal * const A = avgNois->params;
    //nnReal * const D = pathDif->params;
    nnReal * const P = pathCov->params;
    nnReal * const S = diagCov->params;

    #pragma omp for simd schedule(static) aligned(P,A,S,B : VEC_WIDTH)
    for(Uint w=pStart; w<pStart+pCount; w++) {
      P[w] = alphaP * P[w] + updSigP * A[w];
      S[w] = std::sqrt( alpha*S[w]*S[w] + c1cov*P[w]*P[w] + mu_eff*c1cov*B[w] );
      S[w] = std::min(S[w], (nnReal) 10); //safety
      S[w] = std::max(S[w], (nnReal) .01); //safety
    }

    const nnReal* const M = weights->params;
    Saru & gen = * generators[thrID];
    #pragma omp for schedule(static) nowait
    for(Uint i=1; i<pop_size; i++)
    {
      nnReal* const Y = popNoiseVectors[i]->params;
      nnReal* const X = sampled_weights[i]->params;
      #ifdef ACCEL_CMA
      if(i==1) {
        popNoiseVectors[1]->copy(avgNois);
        for(Uint w=pStart; w<pStart+pCount; w++) X[w] = M[w] + _eta * A[w];
      } else
      #endif
      for(Uint w=pStart; w<pStart+pCount; w++) {
        Y[w] = gen.f_mean0_var1() * S[w];
        X[w] = M[w] + _eta * Y[w]; //+ _eta*1e-2*D[w];
      }
      if( i % nThreads == thrID ) startAllGather(i);
    }
  }
  //for(Uint i=1; i<pop_size; i++) startAllGather(i);
  MPI(Wait, &wVecReq[0], MPI_STATUS_IGNORE);
}

void CMA_Optimizer::save(const string fname, const bool backup) {
  weights->save(fname+"_weights");
  pathCov->save(fname+"_pathCov");
  diagCov->save(fname+"_diagCov");

  if(backup) {
    ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    weights->save(fname+"_"+ss.str()+"_weights");
    pathCov->save(fname+"_"+ss.str()+"_pathCov");
    diagCov->save(fname+"_"+ss.str()+"_diagCov");
  }
}
int CMA_Optimizer::restart(const string fname) {
  pathCov->restart(fname+"_pathCov");
  diagCov->restart(fname+"_diagCov");
  return weights->restart(fname+"_weights");
}

void CMA_Optimizer::getMetrics(ostringstream& buff) {
  //buff<<" "<<std::setw(5)<<Nswap/2; Nswap = 0; //each swap counted twice
  real2SS(buff, std::pow(diagCov->compute_weight_norm(), 2) / pDim, 6, 1);
}
void CMA_Optimizer::getHeaders(ostringstream& buff) {
  buff << "| avgC "; //Nswp |
}

void CMA_Optimizer::startAllGather(const Uint ID)
{
  nnReal * const P = ID? sampled_weights[ID]->params : weights->params;
  if( learn_size < 2 ) return;

  if(wVecReq[ID] not_eq MPI_REQUEST_NULL) {
    MPI(Wait, &wVecReq[ID], MPI_STATUS_IGNORE);
  }

  //MPI(Iallgatherv, MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, P, pCounts.data(),
  //  pStarts.data(), MPI_NNVALUE_TYPE, mastersComm, &wVecReq[ID]);
  MPI(Iallgather, MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, P, mpi_stride,
    MPI_NNVALUE_TYPE, mastersComm, &wVecReq[ID]);
}
