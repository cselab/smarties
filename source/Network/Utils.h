//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include <cstring>

#define VEC_WIDTH 32
#define ARY_WIDTH (VEC_WIDTH/sizeof(nnReal))
static const int simdWidth = VEC_WIDTH/sizeof(nnReal);
static const nnReal nnEPS = std::numeric_limits<float>::epsilon();

inline Uint roundUpSimd(const Real N)
{
  return std::ceil(N/ARY_WIDTH)*ARY_WIDTH;
}

inline nnReal* allocate_dirty(const Uint _size)
{
  nnReal* ret = nullptr;
  assert(_size > 0);
  posix_memalign((void **) &ret, VEC_WIDTH, roundUpSimd(_size)*sizeof(nnReal));
  return ret;
}

inline nnReal* allocate_ptr(const Uint _size)
{
  nnReal* ret = allocate_dirty(_size);
  memset(ret, 0, roundUpSimd(_size)*sizeof(nnReal) );
  return ret;
}

inline nnReal* allocate_param(const Uint _size, const Real mpiSz)
{
  const Uint extraSize = roundUpSimd(std::ceil(_size / mpiSz)) * mpiSz;
  //printf("requested %u floats of size %lu, allocating %lu bytes\n",
  //  _size, sizeof(nnReal), extraSize*sizeof(nnReal));
  nnReal* ret = allocate_dirty(extraSize);
  memset(ret, 0, extraSize * sizeof(nnReal) );
  return ret;
}


inline std::vector<nnReal*> allocate_vec(std::vector<Uint> _sizes)
{
  std::vector<nnReal*> ret(_sizes.size(), nullptr);
  for(Uint i=0; i<_sizes.size(); i++) ret[i] = allocate_ptr(_sizes[i]);
  return ret;
}

#ifndef __CHECK_DIFF
  #define LSTM_PRIME_FAC 1 //input/output gates start closed, forget starts open
#else //else we are testing finite diffs
  #define LSTM_PRIME_FAC 0 //otherwise finite differences are small
  #define PRELU_FAC 1
#endif

inline nnReal nnSafeExp(const nnReal val) {
    return std::exp( std::min((nnReal)EXP_CUT, std::max(-(nnReal)EXP_CUT, val) ) );
}

inline Real annealRate(const Real eta, const Real t, const Real T) {
  return eta / (1 + t * T);
}
/*
inline Uint roundUpSimd(const Uint size)
{
  return std::ceil(size/(Real)simdWidth)*simdWidth;
}
static inline nnReal readCutStart(vector<nnReal>& buf)
{
  const Real ret = buf.front();
  buf.erase(buf.begin(),buf.begin()+1);
  assert(!std::isnan(ret) && !std::isinf(ret));
  return ret;
}
static inline nnReal readBuf(vector<nnReal>& buf)
{
  //const Real ret = buf.front();
  //buf.erase(buf.begin(),buf.begin()+1);
  const Real ret = buf.back();
  buf.pop_back();
  assert(!std::isnan(ret) && !std::isinf(ret));
  return ret;
}
static inline void writeBuf(const nnReal weight, vector<nnReal>& buf)
{
  buf.insert(buf.begin(), weight);
}

template <typename T>
inline void _myfree(T *const& ptr)
{
  if(ptr == nullptr) return;
  free(ptr);
}

//template <typename T>
inline void _allocate_quick(nnReal*& ptr, const Uint size)
{
  const Uint sizeSIMD = roundUpSimd(size)*sizeof(nnReal);
  posix_memalign((void **) &ptr, VEC_WIDTH, sizeSIMD);
}

//template <typename T>
inline void _allocate_clean(nnReal*& ptr, const Uint size)
{
  const Uint sizeSIMD = roundUpSimd(size)*sizeof(nnReal);
  posix_memalign((void **) &ptr, VEC_WIDTH, sizeSIMD);
  memset(ptr, 0, sizeSIMD);
}

inline nnReal* init(const Uint N, const nnReal ini)
{
  nnReal* ret;
  _allocate_quick(ret, N);
  for (Uint j=0; j<N; j++) ret[j] = ini;
  return ret;
}

inline nnReal* initClean(const Uint N)
{
  nnReal* ret;
  _allocate_clean(ret, N);
  return ret;
}

inline nnReal* init(const Uint N)
{
  nnReal* ret;
  _allocate_quick(ret, N);
  return ret;
}
*/
