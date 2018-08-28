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
  vector<Uint> indBiases, indWeights;
  vector<Uint> nBiases, nWeights;
 public:
  const Uint nParams, nLayers;
  mutable bool written = false;
  // array containing all parameters of network contiguously
  //(used by optimizer and for MPI reductions)
  nnReal*const params;
  nnReal* params_T = nullptr;

  //each layer requests a certain number of parameters, here compute contiguous
  //memory required such that each layer gets an aligned pointer to both
  //its first bias and and first weight, allowing SIMD ops on all layers
  Uint computeNParams(vector<Uint> _nWeights, vector<Uint> _nBiases)
  {
    assert(_nWeights.size() == _nBiases.size());
    const Uint nL = _nWeights.size();
    Uint nTotPara = 0;
    indBiases = vector<Uint>(nL, 0);
    indWeights = vector<Uint>(nL, 0);
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

  Parameters* allocateGrad() const
  {
    return new Parameters(nWeights, nBiases);
  }

  void allocateTransposed() {
    params_T = allocate_ptr(nParams);
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

  Parameters(vector<Uint> _nWeights, vector<Uint> _nBiases) :
   nBiases(_nBiases), nWeights(_nWeights),
   nParams(computeNParams(_nWeights, _nBiases)), nLayers(_nWeights.size()),
   params(allocate_ptr(nParams))  { }

  ~Parameters() {
    if(params not_eq nullptr) free(params);
    if(params_T not_eq nullptr) free(params_T);
  }

  void reduceThreadsGrad(const vector<Parameters*>& g) const
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
  inline nnReal* W_T(const Uint layerID) const {
    assert(layerID < nLayers && params_T not_eq nullptr);
    return params_T + indWeights[layerID];
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

#if 0
  inline void circle_region(Grads*const trust, Grads*const grad, const Real delta, const int ngrads)
  {
    #if 1
      assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
      long double norm = 0, fac = 1./(trust->nWeights+trust->nBiases)/ngrads/ngrads;
      for(Uint j=0; j<trust->nWeights; j++)
        norm += fac*std::pow(grad->_W[j]+trust->_W[j], 2);
        //norm += std::fabs(grad->_W[j]+trust->_W[j]);
      for(Uint j=0; j<trust->nBiases; j++)
        norm += fac*std::pow(grad->_B[j]+trust->_B[j], 2);
        //norm += std::fabs(grad->_B[j]+trust->_B[j]);

      const Real nG = std::sqrt(norm), softclip = delta/(nG+delta);
      //printf("%Lg %Lg %g %f\n",fac, norm, nG, softclip);
      //printf("grad norm %f\n",nG);
      for(Uint j=0; j<trust->nWeights; j++)
        grad->_W[j] = (grad->_W[j]+trust->_W[j])*softclip -trust->_W[j];
      for(Uint j=0; j<trust->nBiases; j++)
        grad->_B[j] = (grad->_B[j]+trust->_B[j])*softclip -trust->_B[j];
    #else
      Real dot=0, norm = numeric_limits<Real>::epsilon();
      for(Uint j=0; j<trust->nWeights; j++) {
        norm += std::pow(trust->_W[j]/ngrads, 2);
        dot += grad->_W[j]*trust->_W[j]/(ngrads*ngrads);
      }
      for(Uint j=0; j<trust->nBiases; j++)  {
        norm += std::pow(trust->_B[j]/ngrads, 2);
        dot += grad->_B[j]*trust->_B[j]/(ngrads*ngrads);
      }
      const Real proj = std::max( (Real)0, (dot - delta)/norm );
      //printf("grad norm %f %f %f\n", proj, dot, norm);
      for(Uint j=0; j<trust->nWeights; j++)
        grad->_W[j] = grad->_W[j] -proj*trust->_W[j];
      for(Uint j=0; j<trust->nBiases; j++)
        grad->_B[j] = grad->_B[j] -proj*trust->_B[j];
    #endif
    trust->clear();
  }

  inline void circle_region(Grads*const grad, Grads*const trust, Grads*const dest, const Real delta)
  {
    assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
    long double norm = 0, fac = 1./(trust->nWeights+trust->nBiases);
    {
      for(Uint j=0;j<trust->nWeights;j++)
      norm += fac*std::pow(grad->_W[j]+trust->_W[j],2);
      //norm += fac*std::fabs(grad->_W[j]+trust->_W[j]);
      for(Uint j=0;j<trust->nBiases; j++)
      norm += fac*std::pow(grad->_B[j]+trust->_B[j],2);
      //norm += fac*std::fabs(grad->_B[j]+trust->_B[j]);
    }
    const auto nG = std::sqrt(norm);
    //const Real nG = norm;
    const Real softclip = delta/(nG+delta);
    //printf("%Lg %Lg %Lg %f\n",fac, norm, nG, softclip);

    for(Uint j=0;j<trust->nWeights;j++)
      dest->_W[j] += (grad->_W[j]+trust->_W[j])*softclip -trust->_W[j];
    for(Uint j=0;j<trust->nBiases; j++)
      dest->_B[j] += (grad->_B[j]+trust->_B[j])*softclip -trust->_B[j];
    trust->clear();
    grad->clear();
  }

  inline void fullstats(Grads*const grad, Grads*const trust, Grads*const dest, const Real delta)
  {
    assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
    Real EO1 = 0, EO2 = 0, EO3 = 0, EO4 = 0;
    Real EL1 = 0, EL2 = 0, EL3 = 0, EL4 = 0;
    Real EC1 = 0, EC2 = 0, EC3 = 0, EC4 = 0;

    Real dot=0, norm=numeric_limits<Real>::epsilon(), sum=0,  dotL=0, dotC=0;
    for(Uint j=0; j<trust->nWeights; j++) {
      sum += std::pow(grad->_W[j]+trust->_W[j],2);
      norm += trust->_W[j]*trust->_W[j];
      dot +=   grad->_W[j]*trust->_W[j];
    }
    for(Uint j=0; j<trust->nBiases; j++)  {
      sum += std::pow(grad->_B[j]+trust->_B[j],2);
      norm += trust->_B[j]*trust->_B[j];
      dot +=   grad->_B[j]*trust->_B[j];
    }
    const Real proj = std::max( (Real)0, (dot - delta)/norm );
    const Real nG = std::sqrt(sum), clip = delta/(nG+delta);

    for(Uint j=0; j<trust->nWeights; j++) {
      const long double linear = grad->_W[j] -proj*trust->_W[j];
      const long double circle = (grad->_W[j]+trust->_W[j])*clip -trust->_W[j];
      dotL += linear*trust->_W[j];
      dotC += circle*trust->_W[j];
      dest->_W[j] += circle;
    }
    for(Uint j=0; j<trust->nBiases; j++)  {
      const long double linear = grad->_B[j] -proj*trust->_B[j];
      const long double circle = (grad->_B[j]+trust->_B[j])*clip -trust->_B[j];
      dotL += linear*trust->_B[j];
      dotC += circle*trust->_B[j];
      dest->_B[j] += circle;
    }

    if(omp_get_thread_num() == 1) {
      EO1 =          dot/std::sqrt(norm);      // to compute E[grad_proj_dkl]
      EO2 = std::pow(dot/std::sqrt(norm), 2);  //higher order statistics
      EO3 = std::pow(dot/std::sqrt(norm), 3);
      EO4 = std::pow(dot/std::sqrt(norm), 4);
      EL1 =          dotL/std::sqrt(norm);      // to compute E[grad_proj_dkl]
      EL2 = std::pow(dotL/std::sqrt(norm), 2);  //higher order statistics
      EL3 = std::pow(dotL/std::sqrt(norm), 3);
      EL4 = std::pow(dotL/std::sqrt(norm), 4);
      EC1 =          dotC/std::sqrt(norm);      // to compute E[grad_proj_dkl]
      EC2 = std::pow(dotC/std::sqrt(norm), 2);  //higher order statistics
      EC3 = std::pow(dotC/std::sqrt(norm), 3);
      EC4 = std::pow(dotC/std::sqrt(norm), 4);
      ofstream fs;
      fs.open("gradproj_dist.txt", ios::app);
      fs<<EO1<<"\t"<<EO2<<"\t"<<EO3<<"\t"<<EO4<<"\t"
        <<EL1<<"\t"<<EL2<<"\t"<<EL3<<"\t"<<EL4<<"\t"
        <<EC1<<"\t"<<EC2<<"\t"<<EC3<<"\t"<<EC4<<endl;
      fs.close(); fs.flush();
    }

    trust->clear();
    grad->clear();
  }
#endif
