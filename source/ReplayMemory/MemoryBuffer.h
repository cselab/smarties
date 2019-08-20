//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MemoryBuffer_h
#define smarties_MemoryBuffer_h

#include "Sequences.h"
#include "../Core/Agent.h"
#include "../Settings.h"
#include <memory>
#include <atomic>
#include <mutex>

namespace smarties
{

class Sampling;
// algorithm to filter past episodes:
enum FORGET {OLDEST, FARPOLFRAC, MAXKLDIV};

class MemoryBuffer
{
 public:
  MDPdescriptor & MDP;
  const Settings & settings;
  const DistributionInfo & distrib;
  const StateInfo sI = StateInfo(MDP);
  const ActionInfo aI = ActionInfo(MDP);
  Uint learnID = 0;

  std::mutex dataset_mutex; // accessed by some samplers
 private:

  friend class Sampling;
  friend class Collector;
  friend class DataCoordinator;
  friend class MemoryProcessing;

  std::vector<std::mt19937>& generators = distrib.generators;
  std::vector<nnReal>& invstd = MDP.stateScale;
  std::vector<nnReal>& mean = MDP.stateMean;
  std::vector<nnReal>& std = MDP.stateStdDev;
  nnReal& stddev_reward = MDP.rewardsStdDev;
  nnReal& invstd_reward = MDP.rewardsScale;
  const bool bSampleSequences = settings.bSampleSequences;
  const Uint nAppended = MDP.nAppendedObs;
  const Real gamma = settings.gamma;

  std::atomic<bool> needs_pass {false};

  std::vector<Sequence*> Set;
  std::vector<Uint> lastSampledEps;

  // num of samples contained in dataset:
  std::atomic<long> nSequences{0}; // num of episodes
  std::atomic<long> nTransitions{0}; // num of individual time steps
  // num of samples seen from the beginning
  std::atomic<long> nSeenSequences{0};
  std::atomic<long> nSeenTransitions{0};
  // num of samples seen from beginning locally:
  std::atomic<long> nSeenSequences_loc{0};
  std::atomic<long> nSeenTransitions_loc{0};

  const std::unique_ptr<Sampling> sampler;
  nnReal minPriorityImpW = 1;
  nnReal maxPriorityImpW = 1;

  void checkNData();

 public:

  MemoryBuffer(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_);
  ~MemoryBuffer();

  void initialize();

  void clearAll();
  Uint clearOffPol(const Real C, const Real tol);

  template<typename V = nnReal, typename T>
  std::vector<V> standardizedState(const T seq, const T samp) const {
    return standardizedState<V>(Set[seq], samp);
  }
  template<typename V = nnReal, typename T>
  std::vector<V> standardizedState(const Sequence*const seq, const T samp) const
  {
    const Uint dimS = sI.dimObs();
    std::vector<V> ret( dimS * (1+nAppended) );
    for (Uint j=0, k=0; j <= nAppended; ++j)
    {
      const Sint t = std::max((Sint)samp - (Sint)j, (Sint)0);
      const auto& state = seq->states[t];
      assert(state.size() == dimS);
      for (Uint i=0; i<dimS; ++i, ++k) ret[k] = (state[i]-mean[i]) * invstd[i];
    }
    return ret;
  }

  template<typename T>
  Real scaledReward(const T seq, const T samp) const {
    return scaledReward(Set[seq], samp);
  }
  template<typename T>
  Real scaledReward(const Sequence*const seq, const T samp) const {
    assert(samp < (T) seq->rewards.size());
    return scaledReward(seq->rewards[samp]);
  }
  template<typename T>
  Real scaledReward(const Sequence& seq, const T samp) const {
    assert(samp < (T) seq.rewards.size());
    return scaledReward(seq.rewards[samp]);
  }
  Real scaledReward(const Real r) const { return r * invstd_reward; }

  void restart(const std::string base);
  void save(const std::string base, const Uint nStep, const bool bBackup);

  MiniBatch sampleMinibatch(const Uint batchSize, const Uint stepID);
  const std::vector<Uint>& lastSampledEpisodes() { return lastSampledEps; }

  MiniBatch agentToMinibatch(Sequence* const inProgress) const;

  bool bRequireImportanceSampling() const;

  long readNSeen_loc()    const { return nSeenTransitions_loc.load();  }
  long readNSeenSeq_loc() const { return nSeenSequences_loc.load();  }
  long readNData()        const { return nTransitions.load();  }
  long readNSeq()         const { return nSequences.load();  }
  void setNSeen_loc(const long val)    { nSeenTransitions_loc = val;  }
  void setNSeenSeq_loc(const long val) { nSeenSequences_loc = val;  }
  void setNData(const long val)        { nTransitions = val;  }
  void setNSeq(const long val) { nSequences = val; Set.resize(val, nullptr); }

  void removeSequence(const Uint ind);
  void pushBackSequence(Sequence*const seq);

  Sequence* get(const Uint ID) {
    return Set[ID];
  }
  void set(Sequence*const S, const Uint ID) {
    assert(Set[ID] == nullptr);
    Set[ID] = S;
  }

  static std::unique_ptr<Sampling> prepareSampler(MemoryBuffer* const R,
                                                  Settings&S_,
                                                  DistributionInfo&D_);
};

}
#endif
