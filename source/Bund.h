//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

//using namespace std;

#include <random>
#include <vector>
#include <cassert>
#include <limits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <omp.h>
#include <mpi.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// ALGORITHM TWEAKS ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Learn rate of moving stdev and mean of states. If <=0 averaging switched off
// and state scaling quantities are only computed from initial data.
// It can lead to small improvement of results with some computational cost
#define OFFPOL_ADAPT_STSCALE 1

// Switch between log(1+exp(x)) and (x+sqrt(x*x+1)/2 as mapping to R^+ for
// policies, advantages, and all math objects that require pos def net outputs
#define CHEAP_SOFTPLUS

// Switch between network computing \sigma (stdev) or \Sigma (covar).
// Does have an effect only if sigma is linked to network output rather than
// being a separate set of lerned parameters shared by all states.
//#define EXTRACT_COVAR

// Switch between \sigma in (0 1) or (0 inf).
#define UNBND_VAR

// Truncate gaussian dist from -3 to 3, resamples once every ~370 times.
// Without this truncation, high dim act spaces might fail the test rho==1
// with mixture of experts pols, because \pi is immediately equal to 0.
#define NORMDIST_MAX 3

// Bound of pol mean for bounded act. spaces (ie tanh(+/- 8)) Helps avoid nans
#define BOUNDACT_MAX 8

// Sample white Gaussian noise and add it to state vector before input to net
// This has been found to help in case of dramatic dearth of data
// The noise stdev for state s_t is = ($NOISY_INPUT) * || s_{t-1} - s_{t+1} ||
//#define NOISY_INPUT 0.01

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// OPTIMIZER TWEAKS ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Extra numerical stability for Adam optimizer: ensures M2 <= M1*M1/10
// (or in other words deltaW <= 3 \eta , which is what happens if M1 and M2 are
// initialized to 0 and hot started to something ). Can improve results.
//#define SAFE_ADAM

// Turn on Nesterov-style Adam:
//#define NESTEROV_ADAM

// Switch for amsgrad (grep for it, it's not vanilla but spiced up a bit):
//#define AMSGRAD

// Switch between L1 and L2 penalization, both with coef Settings::nnLambda
//#define NET_L1_PENAL

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// GRADIENT CLIPPING ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Learn rate for the exponential average of the gradient's second moment
// Used to learn the scale for the pre-backprop gradient clipping.
// (currently set to be the same as Adam's second moment learn rate)
#define CLIP_LEARNR 1e-3

// Default number of second moments to clip the pre-backprop gradient:
// Can be changed inside each learning algo by overwriting default arg of
// Approximator::initializeNetwork function. If 0 no gradient clipping.
#define STD_GRADCUT 0


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// BEHAVIOR TWEAKS ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//#define PRINT_ALL_RANKS

#define PRFL_DMPFRQ 50 // regulates how frequently print profiler info

// hint to reserve memory for the network workspaces, can be breached
#define MAX_SEQ_LEN 1200

//#define _dumpNet_ // deprecated

typedef unsigned Uint;
////////////////////////////////////////////////////////////////////////////////
#if 1 // MAIN CODE PRECISION
typedef double Real;
#define MPI_VALUE_TYPE MPI_DOUBLE
#else
typedef float Real;
#define MPI_VALUE_TYPE MPI_FLOAT
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef SINGLE_PREC // NETWORK PRECISION
  #define gemv cblas_dgemv
  #define gemm cblas_dgemm
  typedef double nnReal;
  #define MPI_NNVALUE_TYPE MPI_DOUBLE
  #define EXP_CUT 16 //prevent under/over flow with exponentials
#else
  #define gemv cblas_sgemv
  #define gemm cblas_sgemm
  #define MPI_NNVALUE_TYPE MPI_FLOAT
  typedef float nnReal;
  #define EXP_CUT 8 //prevent under/over flow with exponentials
#endif
////////////////////////////////////////////////////////////////////////////////
// Data format for storage in memory buffer. Switch to float for example for
// Atari where the memory buffer is in the order of GBs.
#ifndef SINGLE_PREC
typedef double memReal;
typedef double Fval;
#define MPI_Fval MPI_DOUBLE
#else
typedef float memReal;
typedef float Fval;
#define MPI_Fval MPI_FLOAT
#endif

typedef std::vector<Fval> Fvec;
typedef std::vector<Real> Rvec;
typedef std::vector<long double> LDvec;

////////////////////////////////////////////////////////////////////////////////

template <typename T>
void _dispose_object(T *& ptr)
{
    if(ptr == nullptr) return;
    delete ptr;
    ptr=nullptr;
}

template <typename T>
void _dispose_object(T *const& ptr)
{
    if(ptr == nullptr) return;
    delete ptr;
}

inline MPI_Comm MPIComDup(const MPI_Comm C) {
  MPI_Comm ret;
  MPI_Comm_dup(C, &ret);
  return ret;
}

inline bool isZero(const Real vals)
{
  return std::fabs(vals) < std::numeric_limits<Real>::epsilon();
}

inline bool nonZero(const Real vals)
{
  return std::fabs(vals) > std::numeric_limits<Real>::epsilon();
}

inline bool positive(const Real vals)
{
  return vals > std::numeric_limits<Real>::epsilon();
}

template <typename T>
inline bool bValid(const T vals) {
  return ( not std::isnan(vals) ) and ( not std::isinf(vals) );
}

inline Real safeExp(const Real val)
{
  return std::exp( std::min((Real)EXP_CUT, std::max(-(Real)EXP_CUT, val) ) );
}

inline std::vector<Uint> count_indices(const std::vector<Uint> outs)
{
  std::vector<Uint> ret(outs.size(), 0); //index 0 is 0
  for(Uint i=1; i<outs.size(); i++) ret[i] = ret[i-1] + outs[i-1];
  return ret;
}

#ifdef __APPLE__
#include <cpuid.h>
#define CPUID(INFO, LEAF, SUBLEAF)                     \
  __cpuid_count(LEAF, SUBLEAF, INFO[0], INFO[1], INFO[2], INFO[3])
#define GETCPU(CPU) do {                               \
        uint32_t CPUInfo[4];                           \
        CPUID(CPUInfo, 1, 0);                          \
        /* CPUInfo[1] is EBX, bits 24-31 are APIC ID */\
        if ( (CPUInfo[3] & (1 << 9)) == 0) {           \
          CPU = -1;  /* no APIC on chip */             \
        }                                              \
        else {                                         \
          CPU = (unsigned)CPUInfo[1] >> 24;            \
        }                                              \
        if (CPU < 0) CPU = 0;                          \
      } while(0)
#else
#define GETCPU(CPU) do { CPU=sched_getcpu(); } while(0)
#endif

#define MPI(NAME, ...)                                 \
do {                                                   \
  int mpiW = 0;                                        \
  if(bAsync) {                                         \
    mpiW = MPI_ ## NAME ( __VA_ARGS__ );               \
  } else {                                             \
    std::lock_guard<std::mutex> lock(mpi_mutex);       \
    mpiW = MPI_ ## NAME ( __VA_ARGS__ );               \
  }                                                    \
  if(mpiW not_eq MPI_SUCCESS) {                        \
    _warn("%s %d", #NAME, mpiW);                       \
    throw std::runtime_error("MPI ERROR");             \
  }                                                    \
} while(0)

inline float approxRsqrt( const float number )
{
	union { float f; uint32_t i; } conv;
	static constexpr float threehalfs = 1.5F;
	const float x2 = number * 0.5F;
	conv.f  = number;
	conv.i  = 0x5f3759df - ( conv.i >> 1 );
  // Uncomment to do 2 iterations:
  //conv.f  = conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
	return conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
}


template<typename T>
struct THRvec
{
  Uint nThreads;
  const T initial;
  mutable std::vector<T*> m_v = std::vector<T*> (nThreads, nullptr);
  THRvec(const Uint size, const T init=T()) : nThreads(size), initial(init) {}
  THRvec(const THRvec&c): nThreads(c.nThreads),initial(c.initial),m_v(c.m_v) {}

  ~THRvec() { for (Uint i=0; i<nThreads; i++) delete m_v[i]; }
  inline void resize(const Uint N)
  {
    nThreads = N;
    m_v.resize(N, nullptr);
  }
  inline Uint size() const { return nThreads; };
  inline T& operator[] (const Uint i) const {
    if(m_v[i] == nullptr) {
      m_v[i] = new T(initial);
      //printf("allocting %d\n", i);
    }
    return * m_v[i];
  }
};
