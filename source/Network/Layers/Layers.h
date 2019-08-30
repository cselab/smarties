//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Layers_h
#define smarties_Layers_h

#include "Parameters.h"
#include "Activation.h"
#include "Functions.h"

#ifndef __STDC_VERSION__ //it should never be defined with g++
#define __STDC_VERSION__ 0
#endif

#ifndef SINGLE_PREC // NETWORK PRECISION
  #define gemv cblas_dgemv
  #define gemm cblas_dgemm
#else
  #define gemv cblas_sgemv
  #define gemm cblas_sgemm
#endif

#if   defined(USE_MKL)
#include "mkl_cblas.h"
#elif defined(USE_OPENBLAS)
#include "cblas.h"
#else
  #define USE_OMPSIMD_BLAS
#endif
//#include <immintrin.h>

namespace smarties
{

#ifdef USE_OMPSIMD_BLAS
template<typename T>
inline static void GEMVomp(const Uint NX, const Uint NY, const Uint S,
                           const T * __restrict__ const _W,
                           const T * __restrict__ const _X,
                                 T * __restrict__ const _Y)
{
  typedef __attribute__((aligned(VEC_WIDTH))) T alignT;
  static constexpr Uint safelen_ = VEC_WIDTH / sizeof(T);
  const alignT * __restrict__ const W_ = static_cast<const alignT *>(_W);
  const alignT * __restrict__ const X_ = static_cast<const alignT *>(_X);
  for (Uint o=0; o<NY; ++o)
  {
    const alignT* __restrict__ const W = W_ + S * o;
    T Y = 0;
    #pragma omp simd aligned(X_,W : VEC_WIDTH) safelen(safelen_) reduction(+:Y)
    for (Uint i=0; i<NX; ++i) Y += W[i] * X_[i];
    _Y[o] += Y;
  }
}
#endif

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
    if(NO == 0) return;

    if( spanCompInpGrads )
    {
            nnReal* const errors = curr->E(ID-link);
      const nnReal* const weight = para->W(ID);
      #ifdef USE_OMPSIMD_BLAS
        GEMVomp(NO, NI, NOsimd, weight, deltas, errors);
        // TODO : THERE IS AN ALIGNMENT ISSUE HERE:
        //GEMVomp(NO, spanCompInpGrads, NOsimd,
        //        weight + startCompInpGrads * NOsimd,
        //        deltas, errors + startCompInpGrads);
      #else
      gemv(CblasRowMajor, CblasNoTrans, spanCompInpGrads, NO, 1,
        weight + startCompInpGrads * NOsimd, NOsimd,
        deltas, 1, 1, errors + startCompInpGrads, 1);
      #endif
    }

    if(NR && prev not_eq nullptr)
    {
            nnReal* const errors = prev->E(ID);
      const nnReal* const weight = para->W(ID) +NOsimd*NI;
      #ifdef USE_OMPSIMD_BLAS
        GEMVomp(NO, NR, NOsimd, weight, deltas, errors);
      #else
      gemv(CblasRowMajor, CblasNoTrans, NR, NO, 1,
        weight, NOsimd, deltas, 1, 1, errors, 1);
      #endif
    }

    if(grad == nullptr) return;

    {
      nnReal* const grad_b = grad->B(ID);
      #pragma omp simd aligned(deltas, grad_b : VEC_WIDTH)
      for(Uint o=0; o<NO; ++o) grad_b[o] += deltas[o];
    }

    {
      const nnReal* const inputs = curr->Y(ID-link);
            nnReal* const grad_w = grad->W(ID);
      for(Uint i=0; i<NI;  ++i) {
              nnReal* const G = grad_w + NOsimd*i;
        #pragma omp simd aligned(deltas,inputs,G : VEC_WIDTH)
        for(Uint o=0; o<NO; ++o) G[o] += inputs[i] * deltas[o];
      }
    }

    if(NR && prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
            nnReal* const grad_w = grad->W(ID) +NOsimd*NI;
      for(Uint i=0; i<NR;  ++i) {
        nnReal* const G = grad_w + NOsimd*i;
        #pragma omp simd aligned(deltas, inputs, G : VEC_WIDTH)
        for(Uint o=0; o<NO; ++o) G[o] += inputs[i] * deltas[o];
      }
    }
  }

  // Initialize the weights and biases. Probably by sampling.
  virtual void initialize(std::mt19937& G, const Parameters*const W,
                          Real initializationFac) const = 0;
  virtual size_t   save(const Parameters * const para,
                                   float * tmp) const = 0;
  virtual size_t restart(const Parameters * const para,
                             const float * tmp) const = 0;
};

class InputLayer: public Layer
{
 public:
  InputLayer(Uint _size, Uint _ID) : Layer(_ID, _size, false, true) { }
  std::string printSpecs() const override {
    return "(" + std::to_string(ID) + ") Input Layer of size:"
           + std::to_string(size) + "\n";
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

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override { }
  size_t   save(const Parameters * const para,
                           float * tmp) const override { return 0; }
  size_t restart(const Parameters * const para,
                      const float * tmp) const override { return 0; }
};

class JoinLayer: public Layer
{
  const Uint nJoin;
 public:
  JoinLayer(Uint _ID, Uint _N, Uint _nJ): Layer(_ID,_N,false), nJoin(_nJ) {
    assert(nJoin>1);
  }
  std::string printSpecs() const override {
    return "(" + std::to_string(ID) + ") Join Layer of size:"
           + std::to_string(size) + " joining the previous "
           + std::to_string(nJoin) + " layers\n";
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
    for (Uint i=1; i<=nJoin; ++i) {
      const nnReal* const inputs = curr->Y(ID-i);
      for (Uint j=0; j<curr->sizes[ID-i]; ++j) ret[k++] = inputs[j];
    }
    assert(k==size);
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
    const nnReal* const errors = curr->E(ID);
    Uint k = 0;
    for (Uint i=1; i<=nJoin; ++i)
    {
      nnReal* const ret = curr->E(ID-i);
      for (Uint j=0; j<curr->sizes[ID-i]; ++j) ret[j] = errors[k++];
    }
    assert(k==size);
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override { }
  size_t  save(const Parameters * const para,
                          float * tmp) const override { return 0; }
  size_t restart(const Parameters * const para,
                    const float * tmp) const override { return 0; }
};


class ResidualLayer: public Layer
{
 public:
  ResidualLayer(Uint _ID, Uint _N): Layer(_ID,_N,false) { }

  std::string printSpecs() const override {
    return "(" + std::to_string(ID) + ") Residual Connection of size:"
           + std::to_string(size) + "\n";
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
    bOutputs.push_back(bOutput);
    bInputs.push_back(false);
  }
  void biasInitialValues(const std::vector<Real> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    nnReal* const ret = curr->Y(ID);
    std::memset( ret, 0, size * sizeof(nnReal) );
    for (Uint i=1; i<=2; ++i)
    {
      const Uint sizeInp = std::min(curr->sizes[ID-i], size);
      const nnReal* const inputs = curr->Y(ID-i);
      #pragma omp simd aligned(ret, inputs : VEC_WIDTH)
      for (Uint j=0; j<sizeInp; ++j) ret[j] += inputs[j];
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override {
    const nnReal* const errors = curr->E(ID);
    for (Uint i=1; i<=2; ++i) {
      const Uint sizeInp = std::min(curr->sizes[ID-i], size);
      memcpy( curr->E(ID-i), errors, sizeInp * sizeof(nnReal) );
    }
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override { }
  size_t  save(const Parameters * const para,
                          float * tmp) const override { return 0; }
  size_t restart(const Parameters * const para,
                    const float * tmp) const override { return 0; }
};

class ParamLayer: public Layer
{
  const std::unique_ptr<Function> func;
  std::vector<nnReal> initVals;
 public:

  ParamLayer(Uint _ID, Uint _size, std::string funcType, std::vector<Real>init)
    : Layer(_ID, _size, true), func(makeFunction(funcType)) {
    biasInitialValues(init);
  }
  std::string printSpecs() const override {
    std::string ret = "(" + std::to_string(ID) + ") Parameter Layer of size:"
           + std::to_string(size) + ". Initialized:";
    for(Uint i=0; i<size; ++i) { ret += " " + std::to_string(initVals[i]); }
    return ret + "\n";
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
    for (Uint n=0; n<size; ++n) {
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
    for(Uint o=0; o<size; ++o) {
      deltas[o] *= func->evalDiff(inputs[o], outval[o]);
      grad_b[o] += deltas[o];
    }
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override
  {
    nnReal* const biases = W->B(ID);
    for(Uint o=0; o<size; ++o) biases[o] = func->inverse(initVals[o]);
  }
  size_t  save(const Parameters * const para,
                          float * tmp) const override
  {
    const nnReal* const bias = para->B(ID);
    for (Uint n=0; n<size; ++n) tmp[n] = (float) bias[n];
    return size;
  }
  size_t restart(const Parameters * const para,
                      const float * tmp) const override
  {
    nnReal* const bias = para->B(ID);
    for (Uint n=0; n<size; ++n) bias[n] = (nnReal) tmp[n];
    return size;
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
