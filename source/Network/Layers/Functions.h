//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Function_h
#define smarties_Function_h

#include "../../Utils/FunctionUtilities.h"
#include "../../Utils/Warnings.h"
#include <memory>

#ifndef PRELU_FAC
#define PRELU_FAC 0.1
#endif

//List of non-linearities for neural networks
//- eval return f(in), also present as array in / array out
//- evalDiff returns f'(x)
//- initFactor: some prefer fan in fan out, some only fan-in dependency
//If adding a new function, edit function readFunction at end of file

namespace smarties
{

struct Function
{
  //weights are initialized with uniform distrib [-weightsInitFactor, weightsInitFactor]
  virtual Real initFactor(const Uint inps, const Uint outs) const = 0;

  virtual void eval(const nnReal*const in, nnReal*const out, const Uint N) const = 0; // f(in)

  virtual nnReal eval(const nnReal in) const = 0;
  virtual nnReal inverse(const nnReal in) const = 0; // f(in)
  virtual nnReal evalDiff(const nnReal in, const nnReal out) const = 0; // f'(in)
  virtual std::string name() const = 0;
  virtual ~Function() {}
};

struct Linear : public Function
{
  std::string name() const override { return "Linear";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(1./inps);
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(1./inps);
  }

  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    memcpy(out, in, N*sizeof(nnReal));
  }

  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    memcpy(out, in, N*sizeof(nnReal));
  }

  template <typename T> static T _eval(const T in)
  {
    return in;
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return 1;
  }

  nnReal eval(const nnReal in) const override
  {
    return in;
  }
  nnReal inverse(const nnReal in) const override
  {
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return 1;
  }
};

struct Tanh : public Function
{
  std::string name() const override { return "Tanh"; }

  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(6./(inps + outs));
  }

  template <typename T> static T _eval(const T in)
  {
    if(in > 0) {
      const T e2x = std::exp(-2*in);
      return (1-e2x)/(1+e2x);
    } else {
      const T e2x = std::exp( 2*in);
      return (e2x-1)/(1+e2x);
    }
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return 1 - out*out;
  }
  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    for(Uint i=0; i<N; ++i) out[i] = _eval(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    for(Uint i=0; i<N; ++i) out[i] = _eval(in[i]);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    assert(std::fabs(in)<1);
    return std::log((1+in)/(1-in)) / 2;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct Sigm : public Function
{
  std::string name() const override { return "Sigm";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(6./(inps + outs));
  }

  template <typename T> static T _eval(const T in)
  {
    if(in > 0) return 1/(1+Utilities::safeExp(-in));
    else {
      const T ex = Utilities::safeExp(in);
      return ex/(1+ex);
    }
  }

  template <typename T> static T _inv(const T in)
  {
   assert(in > 0 && in < 1);
   return - std::log(1/in - 1);
  }

  template <typename T> static T _evalDiff(const T in)
  {
    const T expx = Utilities::safeExp(in);
    return expx / std::pow(expx+1, 2);
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return out*(1-out);
  }

  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    for(Uint i=0; i<N; ++i) out[i] = _eval(in[i]);
  }

  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    for(Uint i=0; i<N; ++i) out[i] = _eval(in[i]);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    return _inv(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct HardSign : public Function
{
  std::string name() const override { return "HardSign";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(6./(inps + outs));
  }

  template <typename T> static T _eval(const T in)
  {
    return in/std::sqrt(1+in*in);
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    const T denom = std::sqrt(1+in*in);
    return 1/(denom*denom*denom);
  }

  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0; i<N; ++i) out[i] = in[i]/std::sqrt(1+in[i]*in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    assert(in > 0 && in < 1);
    return in/std::sqrt(1 -in*in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

#define SoftSign_FAC 1
struct SoftSign : public Function
{
  std::string name() const override { return "SoftSign";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(6./(inps + outs));
  }

  template <typename T> static T _eval(const T in)
  {
    return SoftSign_FAC * in/(1 + SoftSign_FAC*std::fabs(in));
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    const T denom = 1 + SoftSign_FAC*std::fabs(in);
    return SoftSign_FAC/(denom*denom);
  }

  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; ++i) out[i] = _eval(in[i]);
  }

  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    assert(in > 0 && in < 1);
    return in/(1-std::fabs(in))/SoftSign_FAC;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct Relu : public Function
{
  std::string name() const override { return "Relu";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(2./inps);
  }

  template <typename T> static T _eval(const T in)
  {
    return in>0 ? in : 0;
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return in>0 ? 1 : 0;
  }

  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; ++i) out[i] = in[i]>0 ? in[i] : 0;
  }

  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    assert(in>=0);
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct LRelu : public Function
{
  std::string name() const override { return "LRelu";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(1.0/inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(1.0/inps);
  }
  template <typename T> static T _eval(const T in)
  {
    return in>0 ? in : PRELU_FAC*in;
  }
  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return in>0 ? 1 : PRELU_FAC;
  }
  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; ++i) out[i] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    if(in >= 0) return in;
    else return in / PRELU_FAC;
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct ExpPlus : public Function
{
  std::string name() const override { return "ExpPlus";}
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  template <typename T> static T _inv(const T in) {
    return std::log(Utilities::safeExp(in) - 1);
  }
  // Used here, std::exp is trigger happy with nans, therefore we clip it
  // between exp(-32) and exp(16).
  template <typename T> static T _eval(const T in)
  {
    return std::log(1 + Utilities::safeExp(in));
  }
  template <typename T> static T _evalDiff(const T in)
  {
    return 1/(1 + Utilities::safeExp(-in));
  }
  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return 1/(1 + Utilities::safeExp(-in));
  }
  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    for(Uint i=0; i<N; ++i) out[i] = _eval(in[i]);
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    return _inv(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct SoftPlus : public Function
{
  std::string name() const override { return "SoftPlus";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }
  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(2./inps);
  }
  template <typename T> static T _eval(const T in)
  {
    return (in + std::sqrt(1+in*in)) / 2;
  }
  template <typename T> static T _evalDiff(const T in)
  {
    return (1 + in/std::sqrt(1+in*in)) / 2;
  }
  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return (1 + in/std::sqrt(1+in*in)) / 2;
  }
  template <typename T> static T _inv(const T in)
  {
    assert(in > 0);
    return (in*in - (T)0.25)/in;
  }
  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; ++i) out[i] = (in[i] + std::sqrt(1+in[i]*in[i])) / 2;
  }
  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override { return _eval(in); }
  nnReal inverse(const nnReal in) const override { return _inv(in); }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct Exp : public Function
{
  std::string name() const override { return "Exp";}
  Real initFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }

  static Real _initFactor(const Uint inps, const Uint outs)
  {
    return std::sqrt(2./inps);
  }

  template <typename T> static T _inv(const T in)
  {
    return std::log(in);
  }

  template <typename T> static T _eval(const T in)
  {
    return Utilities::nnSafeExp(in);
  }

  template <typename T> static T _evalDiff(const T in)
  {
    return Utilities::nnSafeExp(in);
  }

  template <typename T> static T _evalDiff(const T in, const T out)
  {
    return out;
  }

  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    for(Uint i=0; i<N; ++i) out[i] = Utilities::nnSafeExp(in[i]);
  }

  void eval(const nnReal*const in, nnReal*const out, const Uint N) const override
  {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override
  {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override
  {
    assert(in > 0);
    return std::log(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal out) const override
  {
    return _evalDiff(in, out);
  }
};

struct DualRelu
{
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static void _eval(const nnReal*const in, nnReal*const out, const Uint N)
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0; i<N; ++i){
      out[2*i +0] = in[i]<0 ? in[i] : 0;
      out[2*i +1] = in[i]>0 ? in[i] : 0;
    }
  }
  static void _evalDiff(const nnReal*const I, const nnReal*const O, nnReal*const E, const Uint N)
  {
    #pragma omp simd aligned(I,E : VEC_WIDTH)
    for (Uint i=0; i<N; ++i)
    E[i] = (I[i]<0 ? E[2*i+0] : 0) + (I[i]>0 ? E[2*i+1] : 0);
  }
};

struct DualLRelu
{
  static Real _initFactor(const Uint inps, const Uint outs) {
    return std::sqrt(2./inps);
  }
  static void _eval(const nnReal*const in, nnReal*const out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH)
    for (Uint i=0;i<N; ++i){
      out[2*i +0] = in[i]<0 ? in[i] : PRELU_FAC*in[i];
      out[2*i +1] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
    }
  }
  static void _evalDiff(const nnReal*const I, const nnReal*const O, nnReal*const E, const Uint N) {
    #pragma omp simd aligned(I,E : VEC_WIDTH)
    for (Uint i=0;i<N; ++i)
    E[i] = E[2*i]*(I[i]<0? 1:PRELU_FAC) +E[2*i+1]*(I[i]>0? 1:PRELU_FAC);
  }
};

inline std::unique_ptr<Function> makeFunction(const std::string name,
                                              const bool bOutput=false)
{
  if (bOutput || name == "Linear") return std::make_unique<Linear>();
  else
  if (name == "Tanh")   return std::make_unique<Tanh>();
  else
  if (name == "Sigm") return std::make_unique<Sigm>();
  else
  if (name == "HardSign") return std::make_unique<HardSign>();
  else
  if (name == "SoftSign") return std::make_unique<SoftSign>();
  else
  if (name == "Relu") return std::make_unique<Relu>();
  else
  if (name == "LRelu") return std::make_unique<LRelu>();
  else
  if (name == "ExpPlus") return std::make_unique<ExpPlus>();
  else
  if (name == "SoftPlus") return std::make_unique<SoftPlus>();
  else
  if (name == "Exp") return std::make_unique<Exp>();
  else
  die("Activation function not recognized");
  return std::make_unique<Linear>();
}

} // end namespace smarties
#endif // smarties_Quadratic_term_h
