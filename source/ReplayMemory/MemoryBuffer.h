//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "Sampling.h"
#include "../Environments/Environment.h"
#include "StatsTracker.h"

class MemoryBuffer
{
 public:
  const Settings & settings;
  const Environment * const env;
  const StateInfo& sI = env->sI;
  const ActionInfo& aI = env->aI;
  const std::vector<Agent*>& agents = env->agents;
  Uint learnID = 0;
 private:

  friend class Sampling;
  friend class Collector;
  friend class MemorySharing;
  friend class MemoryProcessing;

  const Uint nAppended = settings.appendedObs;
  std::vector<std::mt19937>& generators = settings.generators;

  std::vector<memReal> invstd = sI.inUseInvStd();
  std::vector<memReal> mean = sI.inUseMean();
  std::vector<memReal> std = sI.inUseStd();
  Real invstd_reward = 1;

  const Uint dimS = mean.size();
  const Real gamma = settings.gamma;
  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;

  std::atomic<bool> needs_pass {false};

  std::vector<Sequence*> Set;
  std::vector<Uint> sampled;
  std::mutex dataset_mutex;

  std::atomic<long> nSequences{0};
  std::atomic<long> nTransitions{0};
  std::atomic<long> nSeenSequences{0};
  std::atomic<long> nSeenTransitions{0};
  std::atomic<long> nSeenSequences_loc{0};
  std::atomic<long> nSeenTransitions_loc{0};

  Sampling * const sampler;

  Real minPriorityImpW = 1;
  Real maxPriorityImpW = 1;

  void checkNData();

 public:

  MemoryBuffer(const Settings& settings, const Environment*const env);
  ~MemoryBuffer();

  void initialize();

  void clearAll();

  template<typename T>
  inline Rvec standardizeAppended(const std::vector<T>& state) const {
    Rvec ret(sI.dimUsed*(1+nAppended));
    assert(state.size() == sI.dimUsed*(1+nAppended));
    for (Uint j=0; j<1+nAppended; j++)
      for (Uint i=0; i<sI.dimUsed; i++)
        ret[j +i*(nAppended+1)] =(state[j +i*(nAppended+1)]-mean[i])*invstd[i];
    return ret;
  }
  template<typename T>
  inline Rvec standardize(const std::vector<T>& state) const {
    Rvec ret(sI.dimUsed);
    assert(state.size() == sI.dimUsed && mean.size() == sI.dimUsed);
    for (Uint i=0; i<sI.dimUsed; i++) ret[i] =(state[i]-mean[i])*invstd[i];
    return ret;
  }

  inline Real scaledReward(const Uint seq, const Uint samp) const {
    return scaledReward(Set[seq], samp);
  }
  inline Real scaledReward(const Sequence*const seq,const Uint samp) const {
    assert(samp < seq->tuples.size());
    return scaledReward(seq->tuples[samp]->r);
  }
  inline Real scaledReward(const Real r) const { return r * invstd_reward; }

  void restart(const std::string base);
  void save(const std::string base, const Uint nStep, const bool bBackup);

  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs);

  inline long readNSeen_loc() const {
    return nSeenTransitions_loc.load();
  }
  inline long readNSeenSeq_loc() const {
    return nSeenSequences_loc.load();
  }
  inline long readNSeenSeq() const {
    return nSeenSequences.load();
  }
  inline long readNSeen() const {
    return nSeenTransitions.load();
  }
  inline long readNData() const {
    return nTransitions.load();
  }
  inline long readNSeq() const {
    return nSequences.load();
  }

  inline void setNSeen_loc(const long val) {
    nSeenTransitions_loc = val;
  }
  inline void setNSeenSeq_loc(const long val) {
    nSeenSequences_loc = val;
  }
  inline void setNSeenSeq(const long val) {
    nSeenSequences = val;
  }
  inline void setNSeen(const long val) {
    nSeenTransitions = val;
  }
  inline void setNData(const long val) {
    nTransitions = val;
  }
  inline void setNSeq(const long val) {
    nSequences = val;
    Set.resize(val, nullptr);
  }

  void popBackSequence();
  void removeSequence(const Uint ind);
  void pushBackSequence(Sequence*const seq);

  inline Sequence* get(const Uint ID) {
    return Set[ID];
  }
  inline void set(Sequence*const S, const Uint ID) {
    assert(Set[ID] == nullptr);
    Set[ID] = S;
  }

  inline float getMinPriorityImpW() { return minPriorityImpW; }
  inline float getMaxPriorityImpW() { return maxPriorityImpW; }
  const bool requireImpWeights = sampler->requireImportanceWeights();

  const std::vector<Uint>& listSampled() { return sampled; }
  static Sampling* prepareSampler(const Settings&S, MemoryBuffer* const R);
};
