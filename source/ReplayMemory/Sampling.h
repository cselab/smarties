//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "Sequences.h"
#include <atomic>

class MemoryBuffer;

class Sampling
{
 protected:
  std::vector<std::mt19937>& gens;
  MemoryBuffer* const RM;
  const std::vector<Sequence*>& Set;
  const bool bSampleSequences;

  long nSequences() const;
  long nTransitions() const;
  void setMinMaxProb(const Real maxP, const Real minP);

 public:
  Sampling(const Settings& S, MemoryBuffer*const R);
  virtual void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) = 0;
  virtual void prepare(std::atomic<bool>& needs_pass) = 0;
  void IDtoSeqStep(std::vector<Uint>& seq, std::vector<Uint>& obs,
                  const std::vector<Uint>& ret, const Uint nSeqs);
  virtual bool requireImportanceWeights() = 0;
};

class Sample_uniform : public Sampling
{
 public:
  Sample_uniform(const Settings& S, MemoryBuffer*const R);
  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
  bool requireImportanceWeights() override;
};

class Sample_impLen : public Sampling
{
  std::discrete_distribution<Uint> dist;
 public:
  Sample_impLen(const Settings& S, MemoryBuffer*const R);
  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
  bool requireImportanceWeights() override;
};

class TSample_shuffle : public Sampling
{
  std::vector<std::pair<unsigned, unsigned>> samples;
 public:
  TSample_shuffle(const Settings& S, MemoryBuffer*const R);
  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
  bool requireImportanceWeights() override;
};

class TSample_impRank : public Sampling
{
  int stepSinceISWeep = 0;
  std::discrete_distribution<Uint> distObs;
 public:
  TSample_impRank(const Settings& S, MemoryBuffer*const R);
  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
  bool requireImportanceWeights() override;
};

class TSample_impErr : public Sampling
{
  int stepSinceISWeep = 0;
  std::discrete_distribution<Uint> distObs;
 public:
  TSample_impErr(const Settings& S, MemoryBuffer*const R);
  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
  bool requireImportanceWeights() override;
};

class Sample_impSeq : public Sampling
{
  int stepSinceISWeep = 0;
  std::discrete_distribution<Uint> distObs;
 public:
  Sample_impSeq(const Settings& S, MemoryBuffer*const R);
  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
  bool requireImportanceWeights() override;
};
