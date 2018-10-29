//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Approximator.h"

enum RELAY {VEC, ACT, NET};
struct Aggregator
{
  const Settings& settings;
  const bool bRecurrent = settings.bRecurrent;
  const MemoryBuffer* const data;
  const ActionInfo& aI = data->aI;
  const Approximator* const approx;
  const Uint nOuts, nMaxBPTT = settings.nnBPTTseq;
  const Uint nThreads = settings.nThreads + settings.nAgents;

  THRvec<int> first_sample = THRvec<int>(nThreads,-1);
  THRvec<Sequence*> seq = THRvec<Sequence*>(nThreads, nullptr);
  // [thread][time][component]:
  THRvec<vector<Rvec>> inputs = THRvec<vector<Rvec>>(nThreads);
  // [thread]:
  THRvec<RELAY> usage = THRvec<RELAY>(nThreads, ACT);

  // Settings file, the memory buffer class from which all trajectory pointers
  // will be drawn from, the number of outputs from the aggregator. If 0 then
  // 1) output the actions of the sequence (default)
  // 2) output the result of NN approximator (pointer a)
  Aggregator(Settings& S, const MemoryBuffer*const d, const Uint nOut=0,
   const Approximator*const a = nullptr) : settings(S), data(d), approx(a),
   nOuts(nOut? nOut: d->aI.dim)  {}

  void prepare(Sequence*const traj,const Uint tID, const RELAY SET) const;

  void prepare_seq(Sequence*const traj, const Uint thrID,
    const RELAY SET = VEC) const;

  void prepare_one(Sequence*const traj, const Uint samp,
      const Uint thrID, const RELAY SET = VEC) const;

  void set(const Rvec vec, const Uint samp, const Uint thrID) const;

  Rvec get(const Uint samp, const Uint thrID, const int USEW) const;

  inline Uint nOutputs() const { return nOuts; }
};
