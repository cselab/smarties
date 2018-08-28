//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Functions.h"

struct Memory //Memory light recipient for recurrent connections
{
  Memory(vector<Uint>_sizes, vector<Uint>_bOut): nLayers(_sizes.size()),
  outvals(allocate_vec(_sizes)), sizes(_sizes) {}

  inline void clearOutput() const {
    for(Uint i=0; i<nLayers; i++) {
      assert(outvals[i] not_eq nullptr);
      std::memset( outvals[i], 0, roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  ~Memory() { for(auto& p : outvals) if(p not_eq nullptr) free(p); }
  const Uint nLayers;
  const vector<nnReal*> outvals;
  const vector<Uint> sizes;
};

struct Activation
{
  Uint _nOuts(vector<Uint> _sizes, vector<Uint> _bOut) {
    assert(_sizes.size() == _bOut.size() && nLayers == _bOut.size());
    Uint ret = 0;
    for(Uint i=0; i<_bOut.size(); i++) if(_bOut[i]) ret += _sizes[i];
    if(!ret) die("err nOutputs");
    return ret;
  }
  Uint _nInps(vector<Uint> _sizes, vector<Uint> _bInp) {
    assert(_sizes.size() == _bInp.size() && nLayers == _bInp.size());
    Uint ret = 0;
    for(Uint i=0; i<_bInp.size(); i++) if(_bInp[i]) ret += _sizes[i];
    if(!ret) die("err nInputs");
    return ret;
  }

  Activation(vector<Uint>_sizes, vector<Uint>_bOut, vector<Uint>_bInp):
    nLayers(_sizes.size()), nOutputs(_nOuts(_sizes,_bOut)), nInputs(_nInps(_sizes,_bInp)), sizes(_sizes), output(_bOut), input(_bInp),
    suminps(allocate_vec(_sizes)), outvals(allocate_vec(_sizes)), errvals(allocate_vec(_sizes)) {
    assert(suminps.size()==nLayers);
    assert(outvals.size()==nLayers);
    assert(errvals.size()==nLayers);
  }

  ~Activation() {
    for(auto& p : suminps) if(p not_eq nullptr) free(p);
    for(auto& p : outvals) if(p not_eq nullptr) free(p);
    for(auto& p : errvals) if(p not_eq nullptr) free(p);
  }

  template<typename T>
  inline void setInput(const vector<T> inp) const {
    assert(nInputs == inp.size());
    for(Uint j=0; j<nInputs; j++)
      assert(!std::isnan(inp[j]) && !std::isinf(inp[j]));
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(input[i]) {
      std::copy(&inp[k], &inp[k]+sizes[i], outvals[i]);
      //memcpy(outvals[i], &inp[k], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nInputs);
  }
  inline vector<Real> getInput() const {
    vector<Real> ret(nInputs);
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(input[i]) {
      std::copy(outvals[i], outvals[i]+sizes[i], &ret[k]);
      //memcpy(&ret[k], outvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nInputs);
    return ret;
  }

  inline void clipDelta(const Uint ID, const Uint sizeLink, const nnReal clip=5) const
  {
    if(clip<=0 || output[ID]) return;
    nnReal norm = 0;
    nnReal* const delta = errvals[ID];
    #pragma omp simd aligned(delta : VEC_WIDTH) reduction(+ : norm)
    for(Uint i=0; i<sizes[ID]; i++) norm += delta[i]*delta[i];

    norm = clip/(std::sqrt(norm)/(sizes[ID]+sizeLink) + clip);
    /*
    if(omp_get_thread_num() == 1) {
      ofstream fout("clip"+to_string(ID)+".log", ios::app);
      fout << norm << endl;
      fout.close();
    }
    */
    #pragma omp simd aligned(delta : VEC_WIDTH)
    for(Uint i=0; i<sizes[ID]; i++) delta[i] *= norm;
  }

  inline vector<Real> getInputGradient(const Uint ID) const {
    assert(written == true);
    vector<Real> ret(sizes[ID]);
    std::copy(errvals[ID], errvals[ID]+sizes[ID], &ret[0]);
    //memcpy(&ret[0], errvals[ID], sizes[ID]*sizeof(nnReal));
    return ret;
  }

  template<typename T>
  inline void setOutputDelta(const vector<T> delta) const {
    assert(nOutputs == delta.size()); //alternative not supported
    for(Uint j=0; j<nOutputs; j++)
      assert(!std::isnan(delta[j]) && !std::isinf(delta[j]));
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i]) {
      std::copy(&delta[k], &delta[k]+sizes[i], errvals[i]);
      //memcpy(errvals[i], &delta[k], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nOutputs);
    written = true;
  }

  template<typename T>
  inline void addOutputDelta(const vector<T> delta) const {
    assert(nOutputs == delta.size()); //alternative not supported
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i])
      for (Uint j=0; j<sizes[i]; j++, k++) errvals[i][j] += delta[k];
    assert(k == nOutputs);
    written = true;
  }

  inline vector<nnReal> getOutputDelta() const {
    assert(written == true);
    vector<nnReal> ret(nOutputs);
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i]) {
      std::copy(errvals[i], errvals[i]+sizes[i], &ret[k]);
      //memcpy(&ret[k], errvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nOutputs);
    return ret;
  }

  inline vector<Real> getOutput() const {
    assert(written == true);
    vector<Real> ret(nOutputs);
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i]) {
      std::copy(outvals[i], outvals[i]+sizes[i], &ret[k]);
      //memcpy(&ret[k], outvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    for(Uint j=0; j<nOutputs; j++)
      assert(!std::isnan(ret[j]) && !std::isinf(ret[j]));
    assert(k == nOutputs);
    return ret;
  }

  inline void clearOutput() const {
    for(Uint i=0; i<nLayers; i++) {
      assert(outvals[i] not_eq nullptr);
      std::memset( outvals[i], 0, roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  inline void clearErrors() const {
    for(Uint i=0; i<nLayers; i++) {
      assert(errvals[i] not_eq nullptr);
      std::memset( errvals[i], 0, roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  inline void clearInputs() const {
    for(Uint i=0; i<nLayers; i++) {
      assert(suminps[i] not_eq nullptr);
      std::memset( suminps[i], 0, roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  inline void loadMemory(const Memory*const _M) const {
    for(Uint i=0; i<nLayers; i++) {
      assert(outvals[i] not_eq nullptr);
      assert(_M->outvals[i] not_eq nullptr);
      assert(sizes[i] == _M->sizes[i]);
      std::copy(_M->outvals[i], _M->outvals[i]+sizes[i], outvals[i]);
      //memcpy(outvals[i], _M->outvals[i], sizes[i]*sizeof(nnReal));
    }
  }

  inline void storeMemory(Memory*const _M) const {
    for(Uint i=0; i<nLayers; i++) {
      assert(outvals[i] not_eq nullptr);
      assert(_M->outvals[i] not_eq nullptr);
      assert(sizes[i] == _M->sizes[i]);
      std::copy(outvals[i], outvals[i]+sizes[i], _M->outvals[i]);
      //memcpy(_M->outvals[i], outvals[i], sizes[i]*sizeof(nnReal));
    }
  }

  inline nnReal* X(const Uint layerID) const {
    assert(layerID < nLayers);
    return suminps[layerID];
  }
  inline nnReal* Y(const Uint layerID) const {
    assert(layerID < nLayers);
    return outvals[layerID];
  }
  inline nnReal* E(const Uint layerID) const {
    assert(layerID < nLayers);
    return errvals[layerID];
  }

  const Uint nLayers, nOutputs, nInputs;
  const vector<Uint> sizes, output, input;
  //contains all inputs to each neuron (inputs to network input layer is empty)
  const vector<nnReal*> suminps;
  //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
  const vector<nnReal*> outvals;
  //deltas for each neuron
  const vector<nnReal*> errvals;
  mutable bool written = false;
};

inline void deallocateUnrolledActivations(vector<Activation*>& r)
{
  for (auto & ptr : r) _dispose_object(ptr);
  r.clear();
}
inline void deallocateUnrolledActivations(vector<Activation*>* r)
{
  for (auto & ptr : *r) _dispose_object(ptr);
  r->clear();
}

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
