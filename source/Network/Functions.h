//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include <cmath>
#include "../Settings.h"
#include "Utils.h"

#ifndef PRELU_FAC
#define PRELU_FAC 0.1
#endif
using namespace std;

//List of non-linearities for neural networks
//- eval return f(in), also present as array in / array out
//- evalDiff returns f'(x)
//- initFactor: some prefer fan in fan out, some only fan-in dependency
//If adding a new function, edit function readFunction at end of file

struct Function {
  //weights are initialized with uniform distrib [-weightsInitFactor, weightsInitFactor]
  virtual Real initFactor(const Uint inps, const Uint outs) const = 0;

  virtual void eval(const nnReal*const in, nnReal*const out, const Uint N) const = 0; // f(in)

  virtual nnReal eval(const nnReal in) const = 0;
  virtual nnReal inverse(const nnReal in) const = 0; // f(in)
  virtual nnReal evalDiff(const nnReal in, const nnReal out) const = 0; // f'(in)
  virtual std::string name() const = 0;
  virtual ~Function() {}
};

struct Linear : public Function {
  std::string name() const override { return "Linear";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(1./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(1./inps);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    memcpy(out, in, N*sizeof(nnReal));
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    memcpy(out, in, N*sizeof(nnReal));
  }
  static inline nnReal _eval(const nnReal in) {
    return in;
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return 1;
  }
  nnReal eval(const nnReal in) const override {
    return in;
  }
  nnReal inverse(const nnReal in) const override {
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return 1;
  }
};

struct Tanh : public Function {
  std::string name() const override { return "Tanh";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    if(in > 0) {
      const nnReal e2x = std::exp(-2*in);
      return (1-e2x)/(1+e2x);
    } else {
      const nnReal e2x = std::exp( 2*in);
      return (e2x-1)/(1+e2x);
    }
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return 1 - out*out;
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(std::fabs(in)<1);
    return 0.5 * std::log((1+in)/(1-in));
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct Sigm : public Function {
  std::string name() const override { return "Sigm";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    if(in > 0) return 1/(1+safeExp(-in));
    else {
      const nnReal ex = safeExp(in);
      return ex/(1+ex);
    }
  }
  static inline nnReal _inv(const nnReal in) {
   assert(in > 0 && in < 1);
   return - std::log(1/in - 1);
  }
  static inline nnReal _evalDiff(const nnReal in) {
    const Real expx = safeExp(in);
    return expx / std::pow(expx+1, 2);
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return out*(1-out);
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    return _inv(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct HardSign : public Function {
  std::string name() const override { return "HardSign";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    return in/std::sqrt(1+in*in);
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    const nnReal denom = std::sqrt(1+in*in);
    return 1/(denom*denom*denom);
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0; i<N; i++) out[i] = in[i]/std::sqrt(1+in[i]*in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0 && in < 1);
    return in/std::sqrt(1 -in*in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

#define SoftSign_FAC 1
struct SoftSign : public Function {
  std::string name() const override { return "SoftSign";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    return SoftSign_FAC*in/(1 + SoftSign_FAC*std::fabs(in));
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    const nnReal denom = 1 + SoftSign_FAC*std::fabs(in);
    return SoftSign_FAC/(denom*denom);
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0 && in < 1);
    return in/(1-std::fabs(in))/SoftSign_FAC;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct Relu : public Function {
  std::string name() const override { return "Relu";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return in>0 ? in : 0;
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return in>0 ? 1 : 0;
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : 0;
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in>=0);
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct LRelu : public Function {
  std::string name() const override { return "LRelu";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(1./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(1./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return in>0 ? in : PRELU_FAC*in;
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return in>0 ? 1 : PRELU_FAC;
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    if(in >= 0) return in;
    else return in / PRELU_FAC;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct ExpPlus : public Function {
  std::string name() const override { return "ExpPlus";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static inline nnReal _inv(const nnReal in) {
    return std::log(safeExp(in)-1);
  }
  // Used here, std::exp is trigger happy with nans, therefore we clip it
  // between exp(-32) and exp(16).
  static inline nnReal _eval(const nnReal in) {
    return std::log(1+safeExp(in));
  }
  static inline nnReal _evalDiff(const nnReal in) {
    return 1/(1+safeExp(-in));
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return 1/(1+safeExp(-in));
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    return _inv(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct SoftPlus : public Function {
  std::string name() const override { return "SoftPlus";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return .5*(in + std::sqrt(1+in*in));
  }
  static inline nnReal _evalDiff(const nnReal in) {
    return .5*(1 + in/std::sqrt(1+in*in));
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return .5*(1 + in/std::sqrt(1+in*in));
  }
  static inline nnReal _inv(const nnReal in) {
    assert(in > 0);
    return (in*in - 0.25)/in;
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = .5*(in[i]+std::sqrt(1+in[i]*in[i]));
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override { return _eval(in); }
  nnReal inverse(const nnReal in) const override { return _inv(in); }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct Exp : public Function {
  std::string name() const override { return "Exp";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return nnSafeExp(in);
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal out) {
    return out;
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    for(Uint i=0;i<N;i++) out[i] = nnSafeExp(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0);
    return std::log(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override {
    return _evalDiff(in, out);
  }
};

struct DualRelu {
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; i++){
      out[2*i +0] = in[i]<0 ? in[i] : 0;
      out[2*i +1] = in[i]>0 ? in[i] : 0;
    }
  }
  static inline void _evalDiff(const nnReal*const I, const nnReal*const O, nnReal*const E, const Uint N) {
    #pragma omp simd aligned(I,E : VEC_WIDTH)
    for (Uint i=0;i<N; i++)
    E[i] = (I[i]<0 ? E[2*i+0] : 0) + (I[i]>0 ? E[2*i+1] : 0);
  }
};

struct DualLRelu {
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static inline void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; i++){
      out[2*i +0] = in[i]<0 ? in[i] : PRELU_FAC*in[i];
      out[2*i +1] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
    }
  }
  static inline void _evalDiff(const nnReal*const I, const nnReal*const O, nnReal*const E, const Uint N) {
    #pragma omp simd aligned(I,E : VEC_WIDTH)
    for (Uint i=0;i<N; i++)
    E[i] = E[2*i]*(I[i]<0? 1:PRELU_FAC) +E[2*i+1]*(I[i]>0? 1:PRELU_FAC);
  }
};

inline Function* makeFunction(const std::string name, const bool bOutput=false) {
  if (bOutput || name == "Linear") return new Linear();
  else
  if (name == "Tanh")   return new Tanh();
  else
  if (name == "Sigm") return new Sigm();
  else
  if (name == "HardSign") return new HardSign();
  else
  if (name == "SoftSign") return new SoftSign();
  else
  if (name == "Relu") return new Relu();
  else
  if (name == "LRelu") return new LRelu();
  else
  if (name == "ExpPlus") return new ExpPlus();
  else
  if (name == "SoftPlus") return new SoftPlus();
  else
  if (name == "Exp") return new Exp();
  else
  die("Activation function not recognized");
  return (Function*)nullptr;
}
