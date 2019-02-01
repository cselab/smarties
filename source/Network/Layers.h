//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Parameters.h"
#include "Activation.h"
#include "Functions.h"
#include "../Utils/Profiler.h"

#ifndef __STDC_VERSION__ //it should never be defined with g++
#define __STDC_VERSION__ 0
#endif
#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif
#include <immintrin.h>

// Base class of all layer types. To insert a new layer type, overwrite all
// virtual functions.
class Layer
{
 public:
  const Uint size, ID, link, bInput;
  Uint bOutput;
  Uint spanCompInpGrads = 0, startCompInpGrads = 0;

  inline Uint number() const { return ID; }
  inline Uint nOutputs() const { return size; }

  // Should return the number of weights and biases required by layer
  virtual void requiredParameters(std::vector<Uint>& nWeight,
                                  std::vector<Uint>& nBiases ) const = 0;

  // Should return work memory that allows the network to compute forward step
  // and then, without re-calling forward, compute backward step.
  // See the LSTM class for an example on working out of the box.
  virtual void requiredActivation(std::vector<Uint>& sizes,
                                  std::vector<Uint>& bOutputs,
                                  std::vector<Uint>& bInputs) const = 0;
  // Some classes might allow user to specify an initial value for the bias
  // vector (eg. parametric layer or linear output layer)
  virtual void biasInitialValues(const std::vector<Real> init) = 0;

  Layer(
    Uint _ID,
    Uint _size,
    bool bOut,
    bool bInp = false,
    Uint _link = 0):
    size(_size), ID(_ID), link(_link), bInput(bInp), bOutput(bOut)  {}


  virtual std::string printSpecs() const = 0;

  virtual ~Layer() {}

  virtual void forward( const Activation*const prev,
                        const Activation*const curr,
                        const Parameters*const para) const = 0;
  // forward step without recurrent connection:
  inline void forward( const Activation*const curr,
                       const Parameters*const para) const {
    return forward(nullptr, curr, para);
  }

  virtual void backward( const Activation*const prev,
                         const Activation*const curr,
                         const Activation*const next,
                         const Parameters*const grad,
                         const Parameters*const para) const = 0;
  // forward step without recurrent connection:
  inline void backward( const Activation*const curr,
                        const Parameters*const grad,
                        const Parameters*const para) const {
    return backward(nullptr, curr, nullptr, grad, para);
  }

  void backward(const Uint NI, const Uint NO, const Uint NOsimd, const Uint NR,
                const Activation*const prev,
                const Activation*const curr,
                const Activation*const next,
                const Parameters*const grad,
                const Parameters*const para) const
  {
    const nnReal* const deltas = curr->E(ID);
    if( spanCompInpGrads )
    {
            nnReal* const errors = curr->E(ID-link);
      const nnReal* const weight = para->W(ID);
      #if 0 //def SINGLE_PREC
      for (Uint i = startCompInpGrads; i < spanCompInpGrads+startCompInpGrads; i++)
      {
        const nnReal* const W = weight + NOsimd*i;
        __m256 ret = _mm256_setzero_ps();
        for (Uint o = 0; o < NO; o+=8) {
         ret = _mm256_fmadd_ps(_mm256_load_ps(W+o), _mm256_load_ps(deltas+o), ret);
        }
        errors[i] += ((ret[0]+ret[4])+(ret[1]+ret[5])) + ((ret[2]+ret[6])+(ret[3]+ret[7]));
      }
      #else
      gemv(CblasRowMajor, CblasNoTrans,
        spanCompInpGrads,
        NO,
        1,
        weight + startCompInpGrads*NOsimd,
        NOsimd,
        deltas,
        1,
        1,
        errors + startCompInpGrads,
        1);
       #endif
    }

    if(NR && prev not_eq nullptr)
    {
            nnReal* const errors = prev->E(ID);
      const nnReal* const weight = para->W(ID) +NOsimd*NI;
      gemv(CblasRowMajor, CblasNoTrans, NR, NO, 1,
        weight, NOsimd, deltas, 1, 1, errors, 1);
    }

    if(grad == nullptr) return;

    {
      nnReal* const grad_b = grad->B(ID);
      #pragma omp simd aligned(deltas, grad_b : VEC_WIDTH)
      for(Uint o=0; o<NO; o++) grad_b[o] += deltas[o];
    }

    {
      const nnReal* const inputs = curr->Y(ID-link);
            nnReal* const grad_w = grad->W(ID);
      for(Uint i=0; i<NI;  i++) {
              nnReal* const G = grad_w + NOsimd*i;
        #pragma omp simd aligned(deltas,inputs,G : VEC_WIDTH)
        for(Uint o=0; o<NO; o++) G[o] += inputs[i] * deltas[o];
      }
    }

    if(NR && prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
            nnReal* const grad_w = grad->W(ID) +NOsimd*NI;
      for(Uint i=0; i<NR;  i++) {
        nnReal* const G = grad_w + NOsimd*i;
        #pragma omp simd aligned(deltas, inputs, G : VEC_WIDTH)
        for(Uint o=0; o<NO; o++) G[o] += inputs[i] * deltas[o];
      }
    }
  }

  // Initialize the weights and biases. Probably by sampling.
  virtual void transpose(const Parameters*const para) const {}
  virtual void initialize(std::mt19937* const gen, const Parameters*const para,
    Real initializationFac) const = 0;
};

class InputLayer: public Layer
{
 public:
  InputLayer(Uint _size, Uint _ID) : Layer(_ID, _size, false, true) { }
  std::string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") Input Layer of size:"<<size<<"\n";
    return o.str();
  }

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    assert(nWeight.size() == 0 && nBiases.size() == 0);
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    assert(sizes.size() == 0 && bOutputs.size() == 0);
    sizes.push_back(size);
    bOutputs.push_back(false);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override { }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override { }

  void initialize(std::mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class JoinLayer: public Layer
{
  const Uint nJoin;
 public:
  JoinLayer(Uint _ID, Uint _N, Uint _nJ): Layer(_ID,_N,false), nJoin(_nJ) {
    assert(nJoin>1);
  }
  std::string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") Join Layer of size:"<<size
     <<" joining the previous "<<nJoin<<" layers"<<"\n";
    return o.str();
  }

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    assert(nWeight.size() == 0 && nBiases.size() == 0);
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    assert(sizes.size() == 0 && bOutputs.size() == 0);
    sizes.push_back(size);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override {
    nnReal* const ret = curr->Y(ID);
    Uint k = 0;
    for (Uint i=1; i<=nJoin; i++) {
      const nnReal* const inputs = curr->Y(ID-i);
      for (Uint j=0; j<curr->sizes[ID-i]; j++) ret[k++] = inputs[j];
    }
    assert(k==size);
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override {
    const nnReal* const errors = curr->E(ID);
    Uint k = 0;
    for (Uint i=1; i<=nJoin; i++) {
      nnReal* const ret = curr->E(ID-i);
      for (Uint j=0; j<curr->sizes[ID-i]; j++) ret[j] = errors[k++];
    }
    assert(k==size);
  }

  void initialize(std::mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};


class ResidualLayer: public Layer
{
 public:
  ResidualLayer(Uint _ID, Uint _N): Layer(_ID,_N,false) { }

  std::string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") Residual Connection of size:"<<size<<"\n";
    return o.str();
  }

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(size);
    bOutputs.push_back(false);
    bInputs.push_back(false);
  }
  void biasInitialValues(const std::vector<Real> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override {
    nnReal* const ret = curr->Y(ID);
    std::memset( ret, 0, size * sizeof(nnReal) );
    for (Uint i=1; i<=2; i++) {
      const Uint sizeInp = std::min(curr->sizes[ID-i], size);
      const nnReal* const inputs = curr->Y(ID-i);
      #pragma omp simd aligned(ret, inputs : VEC_WIDTH)
      for (Uint j=0; j<sizeInp; j++) ret[j] += inputs[j];
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override {
    const nnReal* const errors = curr->E(ID);
    for (Uint i=1; i<=2; i++) {
      const Uint sizeInp = std::min(curr->sizes[ID-i], size);
      memcpy( curr->E(ID-i), errors, sizeInp * sizeof(nnReal) );
    }
  }

  void initialize(std::mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class ParamLayer: public Layer
{
  const Function * const func;
  std::vector<nnReal> initVals;
 public:
  ~ParamLayer() { delete func; }
  ParamLayer(Uint _ID, Uint _size, std::string funcType, std::vector<Real>init)
    : Layer(_ID, _size, true), func(makeFunction(funcType)) {
    biasInitialValues(init);
  }
  std::string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") "<<func->name()
     <<"Parameter Layer of size:"<<size<<". Initialized:"
     <<print(initVals, 3).c_str()<<"\n";
    return o.str();
  }

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nWeight.push_back(0); nBiases.push_back(size);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(size); bOutputs.push_back(true); bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals.resize(size, 0);
    std::copy(init.begin(), init.end(), initVals.begin());
  }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
          nnReal* const inputs = curr->X(ID);
          nnReal* const output = curr->Y(ID);
    const nnReal* const bias = para->B(ID);
    for (Uint n=0; n<size; n++) {
      inputs[n] = bias[n];
      output[n] = func->eval(bias[n]);
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
          nnReal* const deltas = curr->E(ID);
          nnReal* const grad_b = grad->B(ID);
    const nnReal* const inputs = curr->X(ID);
    const nnReal* const outval = curr->Y(ID);
    for(Uint o=0; o<size; o++) {
      deltas[o] *= func->evalDiff(inputs[o], outval[o]);
      grad_b[o] += deltas[o];
    }
  }

  void initialize(std::mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    nnReal* const biases = para->B(ID);
    for(Uint o=0; o<size; o++) biases[o] = func->inverse(initVals[o]);
  }
};


inline Activation* allocate_activation(const std::vector<Layer*>& layers) {
  std::vector<Uint> sizes, output, input;
  for(const auto & l : layers) l->requiredActivation(sizes, output, input);
  return new Activation(sizes, output, input);
}

inline Parameters* allocate_parameters(const std::vector<Layer*>&L, const Uint mpiSz)
{
  std::vector<Uint> nWeight, nBiases;
  for(const auto & l : L) l->requiredParameters(nWeight, nBiases);
  return new Parameters(nWeight, nBiases, mpiSz);
}
