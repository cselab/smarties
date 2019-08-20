//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Bund_h
#define smarties_Bund_h

#include "Definitions.h"

namespace smarties
{

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

#ifndef __CHECK_DIFF
  #define LSTM_PRIME_FAC 1 //input/output gates start closed, forget starts open
#else //else we are testing finite diffs
  #define LSTM_PRIME_FAC 0 //otherwise finite differences are small
  #define PRELU_FAC 1
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// OPTIMIZER TWEAKS ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Extra numerical stability for Adam optimizer: ensures M2 <= M1*M1/10
// (or in other words deltaW <= 3 \eta , which is what happens if M1 and M2 are
// initialized to 0 and hot started to something ). Can improve results.
#define SAFE_ADAM

// Turn on Nesterov-style Adam:
#define NESTEROV_ADAM

// Switch for amsgrad (grep for it, it's not vanilla but spiced up a bit):
//#define AMSGRAD

// Switch between L1 and L2 penalization, both with coef Settings::nnLambda
//#define NET_L1_PENAL

// Switch between Adam (L = Lobj+Lpenal) and AdamW (penal is applied after Adam)
#define ADAMW

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

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// NETWORK ALLOC TWEAKS /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define VEC_WIDTH 32
#define ARY_WIDTH ( VEC_WIDTH / sizeof(nnReal) )
static constexpr int simdWidth = VEC_WIDTH / sizeof(nnReal);
static constexpr nnReal nnEPS = std::numeric_limits<float>::epsilon();

} // end namespace smarties
#endif // smarties_Bund_h
