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
#include "../Utils/TaskQueue.h"
#include "../Utils/ParameterBlob.h"
#include "../Utils/DelayedReductor.h"
#include "../Settings/ExecutionInfo.h"
#include "../Settings/HyperParameters.h"
#include "ReplayStatsCounters.h"
#include "Sampling.h"

#include <memory>
#include <mutex>

namespace smarties
{

class DataCoordinator;

struct MemoryBuffer
{
 public:
  MDPdescriptor & MDP;
  const HyperParameters & settings;
  const ExecutionInfo & distrib;
  const StateInfo sI = StateInfo(MDP);
  const ActionInfo aI = ActionInfo(MDP);
  Uint learnID = 0;

  // if clipImpWeight==0 do naive Exp Replay==0 do naive Exp Replay:
  Real beta = settings.clipImpWeight <= 0 ? 1 : 1e-4;
  Real alpha = 0.5; // UNUSED: weight between critic and policy used for CMA
  Real CmaxRet = 1 + settings.clipImpWeight;
  Real CinvRet = 1 / settings.clipImpWeight;

  ReplayStats stats;
  ReplayCounters counters;
  ParameterBlob params = ParameterBlob(distrib, stats, counters);

  DataCoordinator * const sharing;

  DelayedReductor<long double> StateRewRdx;
  DelayedReductor<long> globalCounterRdx;

  std::mutex dataset_mutex; // accessed by some samplers
  std::mutex envTerminationCheck; // accessed when terminating episodes

  friend class Learner;
  friend class Sampling;
  friend class Collector;
  friend class DataCoordinator;

  std::vector<std::unique_ptr<Episode>> episodes;
  std::vector<std::unique_ptr<Episode>> inProgress;
  std::vector<Uint> lastSampledEps;

  const std::unique_ptr<Sampling> sampler;
  nnReal minPriorityImpW = 1;
  nnReal maxPriorityImpW = 1;

  void updateSampler();
  void setupDataCollectionTasks(TaskQueue& tasks);

  void checkNData();

  MemoryBuffer(const MemoryBuffer& c) = delete;
  MemoryBuffer(MemoryBuffer && c) = delete;
  MemoryBuffer(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_);
  ~MemoryBuffer();

  void initialize();

  void clearAll();
  Uint clearOffPol(const Real C, const Real tol);

  void restart(const std::string base);
  void save(const std::string base);

  void getMetrics(std::ostringstream& buff);
  void getHeaders(std::ostringstream& buff);

  MiniBatch sampleMinibatch(const Uint batchSize, const Uint stepID);
  const std::vector<Uint>& lastSampledEpisodes() { return lastSampledEps; }

  MiniBatch agentToMinibatch(const Uint ID);

  bool bRequireImportanceSampling() const;

  long nFarPolicySteps() const { return stats.nFarPolicySteps; }
  long nLocalSeenSteps() const { return counters.nSeenTransitions_loc.load(); }
  long nLocalSeenEps()   const { return counters.nSeenEpisodes_loc.load(); }
  long nSeenSteps() const { return counters.nSeenTransitions.load(); }
  long nSeenEps()   const { return counters.nSeenEpisodes.load(); }
  void increaseLocalSeenEps() { counters.nSeenEpisodes_loc ++; }
  void increaseLocalSeenSteps(const long N = 1) {
    counters.nSeenTransitions_loc += N;
  }
  long nStoredSteps()    const { return counters.nTransitions.load(); }
  long nStoredEps()      const { return counters.nEpisodes.load(); }
  long nGradSteps()      const { return counters.nGradSteps.load(); }
  void increaseGradStep() { counters.nGradSteps ++; }
  long nInProgress()     const { return inProgress.size(); }
  Real getAvgReturn()    const { return stats.avgReturn; }
  long nLocTimeStepsTrain() const {
    return nLocalSeenSteps() - counters.nGatheredB4Startup;
  }

  void storeState(Agent& a);
  void storeAction(const Agent& a);
  void terminateCurrentEpisode(Agent& a);
  void addEpisodeToTrainingSet(const Agent& a);

  void removeBackEpisode();
  void pushBackEpisode(std::unique_ptr<Episode> e);

  Episode& get(const Uint ID) {
    return * episodes[ID].get();
  }
  const Episode& get(const Uint ID) const {
    return * episodes[ID].get();
  }
  Episode& getInProgress(const Uint ID) {
    return * inProgress[ID].get();
  }
};

}
#endif
