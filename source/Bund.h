//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

using namespace std;

#include <random>
#include <vector>
#include <cassert>
#include <sstream>
#include <cstring>
#include <utility>
#include <limits>
#include <cmath>
#include <immintrin.h>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

#include <omp.h>
#include <mpi.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// ALGORITHM TWEAKS ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Learn rate of moving stdev and mean of states. If <=0 averaging switched off
// and state scaling quantities are only computed from initial data.
// It can lead to small improvement of results with some computational cost
#define OFFPOL_ADAPT_STSCALE 0

// Switch between log(1+exp(x)) and (x+sqrt(x*x+1)/2 as mapping to R^+ for
// policies, advantages, and all math objects that require pos def net outputs
//#define CHEAP_SOFTPLUS

// Switch between network computing \sigma (stdev) or \Sigma (covar).
// Does have an effect only if sigma is linked to network output rather than
// being a separate set of lerned parameters shared by all states.
#define EXTRACT_COVAR

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

// #define PRIORITIZED_ER

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

// Data format for storage in memory buffer. Switch to float for example for
// Atari where the memory buffer is in the order of GBs.
typedef double memReal;
//typedef float memReal;

#define PRFL_DMPFRQ 50 // regulates how frequently print profiler info

// hint to reserve memory for the network workspaces, can be breached
#define MAX_SEQ_LEN 1200

//#define _dumpNet_ // deprecated

typedef unsigned Uint;

#if 0
typedef long double Real;
#define MPI_VALUE_TYPE MPI_LONG_DOUBLE
#else
#define MPI_VALUE_TYPE MPI_DOUBLE
#endif
typedef float  Fval;
typedef vector<Fval> Fvec;
typedef double Real;
typedef vector<Real> Rvec;
typedef vector<long double> LDvec;

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

template <typename T>
inline string print(const vector<T> vals)
{
  std::ostringstream o;
  if(!vals.size()) return o.str();
  for (Uint i=0; i<vals.size()-1; i++) o << vals[i] << " ";
  o << vals[vals.size()-1];
  return o.str();
}

inline void real2SS(ostringstream&B,const Real V,const int W, const bool bPos)
{
  B<<" "<<std::setw(W);
  if(std::fabs(V)>= 1e4) B << std::setprecision(std::max(W-7+bPos,0));
  else
  if(std::fabs(V)>= 1e3) B << std::setprecision(std::max(W-6+bPos,0));
  else
  if(std::fabs(V)>= 1e2) B << std::setprecision(std::max(W-5+bPos,0));
  else
  if(std::fabs(V)>= 1e1) B << std::setprecision(std::max(W-4+bPos,0));
  else
                         B << std::setprecision(std::max(W-3+bPos,0));
  B<<std::fixed<<V;
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

inline Real safeExp(const Real val)
{
  return std::exp( std::min((Real)16, std::max((Real)-32,val) ) );
}

inline vector<Uint> count_indices(const vector<Uint> outs)
{
  vector<Uint> ret(outs.size(), 0); //index 0 is 0
  for(Uint i=1; i<outs.size(); i++) ret[i] = ret[i-1] + outs[i-1];
  return ret;
}
