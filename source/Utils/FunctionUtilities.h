//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_FunctionUtilities_h
#define smarties_FunctionUtilities_h

#include "Bund.h"
#include "Warnings.h"

#include <cassert>
#include <cmath> // log, exp, ...
#include <cstring> // memset, memcpy, ...
#include <string>
#include <numeric> // accumulate

namespace smarties
{

namespace Utilities
{

template <typename T>
inline bool isZero(const T vals)
{
  return std::fabs(vals) < std::numeric_limits<T>::epsilon();
}

template <typename T>
inline bool nonZero(const T vals)
{
  return std::fabs(vals) > std::numeric_limits<T>::epsilon();
}

template <typename T>
inline bool isPositive(const T vals)
{
  return vals > std::numeric_limits<T>::epsilon();
}

template <typename T>
inline bool isValidValue(const T vals) {
  return ( not std::isnan(vals) ) and ( not std::isinf(vals) );
}

template <typename T = Real>
inline T safeExp(const T val)
{
  return std::exp( std::min( (T)SMARTIES_EXP_CUT, std::max( - (T)SMARTIES_EXP_CUT, val) ) );
}
template <typename T = nnReal>
inline T nnSafeExp(const T val)
{
  return safeExp<T>(val);
}

inline std::vector<Uint> count_indices(const std::vector<Uint>& outs)
{
  std::vector<Uint> ret(outs.size(), 0); //index 0 is 0
  for(Uint i=1; i<outs.size(); ++i) ret[i] = ret[i-1] + outs[i-1];
  return ret;
}

template <typename T>
inline T annealRate(const T eta, const Real t, const Real time)
{
  return eta / (1 + (T) t * time);
}

inline Uint roundUp(const Real N, const Uint size)
{
  return std::ceil(N / size) * size;
}
template<typename T = nnReal>
inline Uint roundUpSimd(const Real N)
{
  static_assert(VEC_WIDTH % sizeof(T) == 0, "Invalid vectorization");
  return roundUp(N, VEC_WIDTH / sizeof(T) );
}

template<typename T = nnReal>
inline T* allocate_dirty(const Uint _size)
{
  T* ret = nullptr;
  assert(_size > 0);
  posix_memalign((void **) &ret, 64, roundUpSimd(_size) * sizeof(T));
  assert(((uintptr_t)ret % 64) == 0);
  return ret;
}

template<typename T = nnReal>
inline T* allocate_ptr(const Uint _size)
{
  T* const ret = allocate_dirty<T>(_size);
  memset(ret, 0, roundUpSimd(_size) * sizeof(T) );
  return ret;
}

template<typename T = nnReal>
inline std::vector<T*> allocate_vec(const std::vector<Uint>& _sizes)
{
  std::vector<T*> ret(_sizes.size(), nullptr);
  for(Uint i=0; i<_sizes.size(); ++i) ret[i] = allocate_ptr<T>(_sizes[i]);
  return ret;
}

#ifdef SMARTIES_CHEAP_SOFTPLUS

  template<typename T>
  inline T unbPosMap_func(const T in)
  {
    return ( in + std::sqrt(1+in*in) )/2;
  }

  template<typename T>
  inline T unbPosMap_diff(const T in)
  {
    return ( 1 + in/std::sqrt(1+in*in) )/2;
  }

  template<typename T>
  inline T unbPosMap_inverse(T in)
  {
    if(in<=0) {
      printf("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.\n");
      in = std::numeric_limits<float>::epsilon();
    }
    return (in*in - (T)0.25)/in;
  }
#else

  template<typename T>
  inline T unbPosMap_func(const T in)
  {
    return std::log(1+safeExp(in));
  }

  template<typename T>
  inline T unbPosMap_diff(const T in)
  {
    return 1/(1+safeExp(-in));
  }

  template<typename T>
  inline T unbPosMap_inverse(T in)
  {
    if(in<=0) {
      warn("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.");
      in = std::numeric_limits<float>::epsilon();
    }
    return std::log(safeExp(in)-1);
  }
#endif

template<typename T>
inline T noiseMap_func(const T val)
{
  #ifdef SMARTIES_UNBND_VAR
    return unbPosMap_func(val);
  #else
    return 1/(1 + safeExp(-val));
  #endif
}

template<typename T>
inline T noiseMap_diff(const T val)
{
  #ifdef SMARTIES_UNBND_VAR
    return unbPosMap_diff(val);
  #else
    const T expx = safeExp(val);
    return expx / std::pow(expx+1, 2);
  #endif
}

template<typename T>
inline T noiseMap_inverse(T val)
{
  #ifdef SMARTIES_UNBND_VAR
    return unbPosMap_inverse(val);
  #else
    if(val<=0 || val>=1) {
      warn("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.");
      if(val<=0) val =   std::numeric_limits<float>::epsilon();
      if(val>=1) val = 1-std::numeric_limits<float>::epsilon();
    }
    return - std::log(1/val - 1);
  #endif
}

template<typename T>
inline T clip(const T val, const T ub, const T lb)
{
  assert(!std::isnan(val) && !std::isnan(ub) && !std::isnan(lb));
  assert(!std::isinf(val) && !std::isinf(ub) && !std::isinf(lb));
  assert(ub>lb);
  return std::max(std::min(val, ub), lb);
}

inline Rvec sum3Grads(const Rvec& f, const Rvec& g, const Rvec& h)
{
  assert(g.size() == f.size());
  assert(h.size() == f.size());
  Rvec ret(f.size());
  for(Uint i=0; i<f.size(); ++i) ret[i] = f[i]+g[i]+h[i];
  return ret;
}

inline Rvec sum2Grads(const Rvec& f, const Rvec& g)
{
  assert(g.size() == f.size());
  Rvec ret(f.size());
  for(Uint i=0; i<f.size(); ++i) ret[i] = f[i]+g[i];
  return ret;
}

inline Rvec penalizeReFER(const Rvec& grad, const Rvec& penal, const Real beta)
{
  assert(grad.size() == penal.size());
  Rvec ret(grad.size());
  for(Uint i=0; i<grad.size(); ++i)
    ret[i] = beta * grad[i]+ (1-beta)/beta * penal[i];
  return ret;
}

inline Rvec weightSum2Grads(const Rvec& f, const Rvec& g, const Real W)
{
  assert(g.size() == f.size());
  Rvec ret(f.size());
  for(Uint i=0; i<f.size(); ++i) ret[i] = W*f[i]+ (1-W)*g[i];
  return ret;
}

inline Rvec trust_region_update(const Rvec& grad,
  const Rvec& trust, const Uint nA, const Real delta)
{
  assert(grad.size() == trust.size());
  Rvec ret(nA);
  Real dot=0, norm = std::numeric_limits<Real>::epsilon();
  for (Uint j=0; j<nA; ++j) {
    norm += trust[j] * trust[j];
    dot +=  trust[j] *  grad[j];
  }
  const Real proj = std::max((Real)0, (dot-delta)/norm);
  //#ifndef NDEBUG
  //if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
  //else {printf("Not Hit DKL constraint\n");fflush(0);}
  //#endif
  for (Uint j=0; j<nA; ++j) ret[j] = grad[j]-proj*trust[j];
  return ret;
}

template<typename T>
inline T sum(const std::vector<T> & vec)
{
  return std::accumulate(vec.begin(), vec.end(), (T) 0);
}

template<typename T>
inline Uint maxInd(const T& vec)
{
  auto maxVal = vec[0];
  Uint indBest = 0;
  for (Uint i=1; i<vec.size(); ++i)
    if (vec[i]>maxVal) {
      maxVal = vec[i];
      indBest = i;
    }
  return indBest;
}

inline Uint maxInd(const Rvec& pol, const Uint start, const Uint N)
{
  Real Val = -1e9;
  Uint Nbest = 0;
  for (Uint i=start; i<start+N; ++i)
      if (pol[i]>Val) { Val = pol[i]; Nbest = i-start; }
  return Nbest;
}

inline Real minAbsValue(const Real v, const Real w)
{
  return std::fabs(v)<std::fabs(w) ? v : w;
}

inline void copyFile(const std::string& fileFrom, const std::string& fileTo)
{
  FILE* sorc = fopen(fileFrom.c_str(), "rb");
  FILE* dest = fopen(fileTo.c_str(), "wb");
  static constexpr size_t BUFSIZE = 4096;
  char buf[BUFSIZE];
  while (true) {
    const size_t size = fread(buf, 1, BUFSIZE, sorc);
    fwrite(buf, 1, size, dest);
    if(size < BUFSIZE) break;
  }
  fflush(dest); fclose(dest); fclose(sorc);
}

template <typename T>
void dispose_object(T *& ptr)
{
  if(ptr == nullptr) return;
  delete ptr;
  ptr=nullptr;
}

template <typename T>
void dispose_object(T *const& ptr)
{
  if(ptr == nullptr) return;
  delete ptr;
}

template<typename It> class Range
{
    const It m_beg, m_end;
public:
    Range(It _beg, It _end) : m_beg(_beg), m_end(_end) {}
    It begin() const { return m_beg; }
    It end()   const { return m_end; }
};

template<typename ORange,
         typename OIt = decltype(std::begin(std::declval<ORange>())),
         typename It = std::reverse_iterator<OIt> >
Range<It> reverse(ORange && originalRange) {
  return Range<It>(It(std::end(originalRange)), It(std::begin(originalRange)));
}

#if 0 // unused:
struct BufferedPRNG {
    std::vector<size_t> seeds;
    void seed(std::mt19937 & gen, const size_t N) {
        while(seeds.size() < N) seeds.push_back(gen());
    }
    size_t min() { return std::mt19937::min(); }
    size_t max() { return std::mt19937::max(); }
    size_t operator() () {
        const size_t rng = seeds.back();
        seeds.pop_back();
        return s;
    }
    void discard() {
        seeds.pop_back();
    }
};
#endif

} // end namespace smarties
} // end namespace Utilities
#endif // smarties_FunctionUtilties_h


