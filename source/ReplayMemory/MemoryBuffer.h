//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MemoryBuffer_h
#define smarties_MemoryBuffer_h

#include "MiniBatch.h"
#include "../Core/Agent.h"
#include "../Settings.h"
#include <memory>
#include <atomic>
#include <mutex>

namespace smarties
{

class Sampling;
// algorithm to filter past episodes:
enum FORGET {OLDEST, FARPOLFRAC, MAXKLDIV, BATCHRL};

class MemoryBuffer
{
 public:
  MDPdescriptor & MDP;
  const Settings & settings;
  const DistributionInfo & distrib;
  const StateInfo sI = StateInfo(MDP);
  const ActionInfo aI = ActionInfo(MDP);
  Uint learnID = 0;

  // if clipImpWeight==0 do naive Exp Replay==0 do naive Exp Replay:
  Real beta = settings.clipImpWeight <= 0 ? 1 : 1e-4;
  Real alpha = 0.5; // UNUSED: weight between critic and policy used for CMA
  Real CmaxRet = 1 + settings.clipImpWeight;
  Real CinvRet = 1 / settings.clipImpWeight;
  Real avgCumulativeReward = 0;

  std::mutex dataset_mutex; // accessed by some samplers
 private:

  friend class Learner;
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

  std::vector<Sequence> episodes;
  std::vector<Uint> lastSampledEps;

  // num of grad steps performed by owning learner:
  std::atomic<long> nGradSteps{0};
  // number of time steps collected before training begins:
  long nGatheredB4Startup = std::numeric_limits<long>::max();

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
  MemoryBuffer(const MemoryBuffer& c) = delete;
  MemoryBuffer(MemoryBuffer && c) = delete;
  MemoryBuffer(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_);
  ~MemoryBuffer();

  void initialize();

  void clearAll();
  Uint clearOffPol(const Real C, const Real tol);

  template<typename V = nnReal, typename T>
  std::vector<V> standardizedState(const T seq, const T samp) const {
    return standardizedState<V>(episodes[seq], samp);
  }
  template<typename V = nnReal, typename T>
  std::vector<V> standardizedState(const Sequence& seq, const T samp) const
  {
    const Uint dimS = sI.dimObs();
    std::vector<V> ret( dimS * (1+nAppended) );
    for (Uint j=0, k=0; j <= nAppended; ++j)
    {
      const Sint t = std::max((Sint)samp - (Sint)j, (Sint)0);
      const auto& state = seq.states[t];
      assert(state.size() == dimS);
      for (Uint i=0; i<dimS; ++i, ++k) ret[k] = (state[i]-mean[i]) * invstd[i];
    }
    return ret;
  }

  template<typename T>
  Real scaledReward(const T seq, const T samp) const {
    return scaledReward(episodes[seq], samp);
  }
  template<typename T>
  Real scaledReward(const Sequence& seq, const T samp) const {
    assert(samp < (T) seq.rewards.size());
    return scaledReward(seq.rewards[samp]);
  }
  Real scaledReward(const Real r) const { return r * invstd_reward; }

  void restart(const std::string base);
  void save(const std::string base);

  MiniBatch sampleMinibatch(const Uint batchSize, const Uint stepID);
  const std::vector<Uint>& lastSampledEpisodes() { return lastSampledEps; }

  MiniBatch agentToMinibatch(Sequence & inProgress) const;

  bool bRequireImportanceSampling() const;

  long readNSeen_loc()    const { return nSeenTransitions_loc.load();  }
  long readNSeenSeq_loc() const { return nSeenSequences_loc.load();  }
  long readNData()        const { return nTransitions.load();  }
  long readNSeq()         const { return nSequences.load();  }
  long nLocTimeStepsTrain() const {
    return readNSeen_loc() - nGatheredB4Startup;
  }
  long nLocTimeSteps() const {
    return readNSeen_loc();
  }
  Real getAvgCumulativeReward() {
    return avgCumulativeReward;
  }

  void removeSequence(const Uint ind);
  void pushBackSequence(Sequence & seq);

  Sequence& get(const Uint ID) {
    return episodes[ID];
  }
};

}
#endif
