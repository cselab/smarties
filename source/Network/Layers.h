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
#include "cblas.h"

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
  virtual void requiredParameters(vector<Uint>& nWeight,
                                  vector<Uint>& nBiases ) const = 0;

  // Should return work memory that allows the network to compute forward step
  // and then, without re-calling forward, compute backward step.
  // See the LSTM class for an example on working out of the box.
  virtual void requiredActivation(vector<Uint>& sizes,
                                  vector<Uint>& bOutputs,
                                  vector<Uint>& bInputs) const = 0;
  // Some classes might allow user to specify an initial value for the bias
  // vector (eg. parametric layer or linear output layer)
  virtual void biasInitialValues(const vector<Real> init) = 0;

  Layer(
    Uint _ID,
    Uint _size,
    bool bOut,
    bool bInp = false,
    Uint _link = 0):
    size(_size), ID(_ID), link(_link), bInput(bInp), bOutput(bOut)  {}


  virtual string printSpecs() const = 0;

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
  virtual void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const = 0;
};

class InputLayer: public Layer
{
 public:
  InputLayer(Uint _size, Uint _ID) : Layer(_ID, _size, false, true) { }
  string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") Input Layer of size:"<<size<<"\n";
    return o.str();
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    assert(nWeight.size() == 0 && nBiases.size() == 0);
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    assert(sizes.size() == 0 && bOutputs.size() == 0);
    sizes.push_back(size);
    bOutputs.push_back(false);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const vector<Real> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override { }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override { }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class JoinLayer: public Layer
{
  const Uint nJoin;
 public:
  JoinLayer(Uint _ID, Uint _N, Uint _nJ): Layer(_ID,_N,false), nJoin(_nJ) {
    assert(nJoin>1);
  }
  string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") Join Layer of size:"<<size
     <<" joining the previous "<<nJoin<<" layers"<<"\n";
    return o.str();
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    assert(nWeight.size() == 0 && nBiases.size() == 0);
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    assert(sizes.size() == 0 && bOutputs.size() == 0);
    sizes.push_back(size);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const vector<Real> init) override { }
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

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class ParamLayer: public Layer
{
  const Function * const func;
  vector<nnReal> initVals;
 public:
  ~ParamLayer() { delete func; }
  ParamLayer(Uint _ID, Uint _size, string funcType, vector<Real> init) :
    Layer(_ID, _size, true), func(makeFunction(funcType)) {
    biasInitialValues(init);
  }
  string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") "<<func->name()
     <<"Parameter Layer of size:"<<size<<". Initialized:"
     <<print(initVals).c_str()<<"\n";
    return o.str();
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    nWeight.push_back(0); nBiases.push_back(size);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    sizes.push_back(size); bOutputs.push_back(true); bInputs.push_back(bInput);
  }
  void biasInitialValues(const vector<Real> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals.resize(size, 0);
    std::copy(initVals.begin(), initVals.end(), initVals.begin());
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

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    nnReal* const biases = para->B(ID);
    for(Uint o=0; o<size; o++) biases[o] = func->inverse(initVals[o]);
  }
};


inline Activation* allocate_activation(const vector<Layer*>& layers) {
  vector<Uint> sizes, output, input;
  for(const auto & l : layers) l->requiredActivation(sizes, output, input);
  return new Activation(sizes, output, input);
}

inline Parameters* allocate_parameters(const vector<Layer*>& layers) {
  vector<Uint> nWeight, nBiases;
  for(const auto & l : layers) l->requiredParameters(nWeight, nBiases);
  return new Parameters(nWeight, nBiases);
}

inline Memory* allocate_memory(const vector<Layer*>& layers) {
  vector<Uint> sizes, output, input;
  for(const auto & l : layers) l->requiredActivation(sizes, output, input);
  return new Memory(sizes, output);
}
