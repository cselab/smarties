//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "MemoryBuffer.h"

enum FORGET {OLDEST, FARPOLFRAC, MAXKLDIV};
class MemoryProcessing
{
private:
  const Settings & settings;
  const int ID = settings.learner_rank;
  MemoryBuffer * const RM;

  std::vector<memReal>& invstd = RM->invstd;
  std::vector<memReal>& mean = RM->mean;
  std::vector<memReal>& std = RM->std;
  Real& invstd_reward = RM->invstd_reward;

  const Uint dimS = RM->dimS;
  Uint nPruned = 0, minInd = 0, nOffPol = 0;
  Real avgDKL = 0;
  int delPtr = -1;

  DelayedReductor Ssum1Rdx = DelayedReductor(settings, LDvec(dimS, 0) );
  DelayedReductor Ssum2Rdx = DelayedReductor(settings, LDvec(dimS, 1) );
  DelayedReductor Rsum2Rdx = DelayedReductor(settings, LDvec(   1, 1) );
  DelayedReductor Csum1Rdx = DelayedReductor(settings, LDvec(   1, 1) );

  const std::vector<Sequence*>& Set = RM->Set;
  std::atomic<long>& nSequences = RM->nSequences;
  std::atomic<long>& nTransitions = RM->nTransitions;
  std::atomic<long>& nSeenSequences = RM->nSeenSequences;
  std::atomic<long>& nSeenTransitions = RM->nSeenTransitions;
  std::atomic<long>& nSeenSequences_loc = RM->nSeenSequences_loc;
  std::atomic<long>& nSeenTransitions_loc = RM->nSeenTransitions_loc;

public:

  MemoryProcessing(const Settings&S, MemoryBuffer*const _RM);

  ~MemoryProcessing() { }

  void updateRewardsStats(const Real WR, const Real WS, const bool bInit=false);

  static FORGET readERfilterAlgo(const std::string setting, const bool bReFER);

  // Algorithm for maintaining and filtering dataset, and optional imp weight range parameter
  void prune(const FORGET ALGO, const Fval CmaxRho = 1);
  void finalize();

  void getMetrics(std::ostringstream& buff);
  void getHeaders(std::ostringstream& buff);

  Uint nFarPol() {
    return nOffPol;
  }
};
