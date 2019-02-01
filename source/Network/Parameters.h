//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "Functions.h"

struct Parameters
{
 private:
  std::vector<Uint> indBiases, indWeights;
  std::vector<Uint> nBiases, nWeights;
 public:
  const Uint nParams, nLayers;
  mutable bool written = false;
  // array containing all parameters of network contiguously
  //(used by optimizer and for MPI reductions)
  nnReal*const params;

  //each layer requests a certain number of parameters, here compute contiguous
  //memory required such that each layer gets an aligned pointer to both
  //its first bias and and first weight, allowing SIMD ops on all layers
  Uint computeNParams(std::vector<Uint> _nWeights, std::vector<Uint> _nBiases)
  {
    assert(_nWeights.size() == _nBiases.size());
    const Uint nL = _nWeights.size();
    Uint nTotPara = 0;
    indBiases = std::vector<Uint>(nL, 0);
    indWeights = std::vector<Uint>(nL, 0);
    for(Uint i=0; i<nL; i++) {
      indWeights[i] = nTotPara;
      nTotPara += roundUpSimd(_nWeights[i]);
      indBiases[i] = nTotPara;
      nTotPara += roundUpSimd( _nBiases[i]);
    }
    //printf("Weight sizes:[%s] inds:[%s] Bias sizes:[%s] inds[%s] Total:%u\n",
    //  print(_nWeights).c_str(), print(indWeights).c_str(),
    //  print(_nBiases).c_str(), print(indBiases).c_str(), nTotPara);
    return nTotPara;
  }

  Parameters* allocateGrad(const Uint mpisize) const
  {
    return new Parameters(nWeights, nBiases, mpisize);
  }

  inline void broadcast(const MPI_Comm comm) const
  {
    MPI_Bcast(params, nParams, MPI_NNVALUE_TYPE, 0, comm);
  }

  inline void copy(const Parameters* const tgt) const
  {
    assert(nParams == tgt->nParams);
    memcpy(params, tgt->params, nParams*sizeof(nnReal));
  }

  Parameters(std::vector<Uint>_nWeights, std::vector<Uint>_nBiases, const Uint _mpisize) :
   nBiases(_nBiases), nWeights(_nWeights),
   nParams(computeNParams(_nWeights, _nBiases)), nLayers(_nWeights.size()),
   params(allocate_param(nParams, _mpisize))  { }

  ~Parameters() {
    if(params not_eq nullptr) free(params);
  }

  void reduceThreadsGrad(const std::vector<Parameters*>& g) const
  {
    #ifndef NDEBUG
      //vector<nnReal> gradMagn = vector<nnReal>(g.size(), 0);
    #endif
    #pragma omp parallel num_threads(g.size())
    {
      const Uint thrI = omp_get_thread_num(), thrN = omp_get_num_threads();
      const Uint shift = roundUpSimd(nParams/(Real)thrN);
      assert(thrN*shift>=nParams&& thrN==g.size()&& nParams==g[thrI]->nParams);
      const nnReal *const src = g[thrI]->params; nnReal *const dst = params;
      for(Uint i=0; i<thrN; i++)
      {
        const Uint turn = (thrI + i) % thrN;
        const Uint start = turn*shift, end = (turn+1)*shift;
        //#pragma omp critical
        //{ cout<<turn<<" "<<start<<" "<<end<<" "<<thrI<<" "
        //      <<thrN<<" "<<shift<<" "<<nParams<<endl; fflush(0); }
        if(g[thrI]->written) {
          #pragma omp simd aligned(dst, src : VEC_WIDTH)
          for(Uint j=start; j<std::min(nParams, end); j++) {
            assert(not std::isnan(src[j]) && not std::isinf(src[j]));
            dst[j] += src[j];
            #ifndef NDEBUG
              //gradMagn[thrI] += src[j]*src[j];
            #endif
          }
        }
        #pragma omp barrier
      }
      g[thrI]->clear();
    }
    //cout<<endl;
    #ifndef NDEBUG
    //cout<<"Grad magnitudes:"<<print(gradMagn)<<endl;
    #endif
  }

  void set(const Real val) const
  {
    for(Uint j=0; j<nParams; j++) params[j] = val;
  }

  long double compute_weight_norm() const
  {
    long double sumWeights = 0;
    #pragma omp parallel for schedule(static) reduction(+:sumWeights)
    for (Uint w=0; w<nParams; w++) sumWeights += std::pow(params[w],2);
    return std::sqrt(sumWeights);
  }

  long double compute_weight_dist(const Parameters*const TGT) const
  {
    long double dist = 0;
    #pragma omp parallel for schedule(static) reduction(+ : dist)
    for(Uint w=0; w<nParams; w++) dist += std::pow(params[w]-TGT->params[w], 2);
    return std::sqrt(dist);
  }

  inline void clear() const {
    std::memset(params, 0, nParams*sizeof(nnReal));
    written = false;
  }

  void save(const std::string fname) const {
    FILE * wFile = fopen((fname+".raw").c_str(), "wb");
    fwrite(params, sizeof(nnReal), nParams, wFile); fflush(wFile);
    fclose(wFile);
  }
  int restart(const std::string fname) const {
    FILE * wFile = fopen((fname+".raw").c_str(), "rb");
    if(wFile == NULL) {
      printf("Parameters restart file %s not found.\n", (fname+".raw").c_str());
      return 1;
    } else {
      printf("Restarting from file %s.\n", (fname+".raw").c_str());
      fflush(0);
    }
    size_t wsize = fread(params, sizeof(nnReal), nParams, wFile);
    fclose(wFile);
    if(wsize not_eq nParams)
      _die("Mismatch in restarted file %s; contains:%lu read:%lu.",
        fname.c_str(), wsize, nParams);
    return 0;
  }

  inline nnReal* W(const Uint layerID) const {
    assert(layerID < nLayers);
    return params + indWeights[layerID];
  }
  inline nnReal* B(const Uint layerID) const {
    assert(layerID < nLayers);
    return params + indBiases[layerID];
  }
  inline Uint NW(const Uint layerID) const {
    assert(layerID < nLayers);
    return nWeights[layerID];
  }
  inline Uint NB(const Uint layerID) const {
    assert(layerID < nLayers);
    return nBiases[layerID];
  }
};

inline std::vector<Parameters*> initWpop(const Parameters*const W,Uint popsz,Uint mpisize) {
  std::vector<Parameters*> ret(popsz, nullptr);
  for(Uint i=0; i<popsz; i++) ret[i] = W->allocateGrad(mpisize);
  return ret;
}
