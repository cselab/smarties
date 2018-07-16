//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "MemoryBuffer.h"
#include "Approximator.h"

#include <list>
using namespace std;

class Learner
{
protected:
  const MPI_Comm mastersComm;
  Environment * const env;
  const bool bSampleSequences, bTrain;
  const Uint totNumSteps, policyVecDim, batchSize, nAgents, nThreads, nWorkers;
  const Real gamma, learnR, ReFtol, explNoise, epsAnneal, CmaxPol;
  const int learn_rank, learn_size;
  Settings & settings;
  unsigned long nStep = 0;
  Uint nAddedGradients = 0;
  mutable Uint nSkipped = 0;

  mutable bool updateComplete = false;
  mutable bool updateToApply = false;

  const ActionInfo& aInfo;
  const StateInfo&  sInfo;
  std::vector<std::mt19937>& generators;
  MemoryBuffer* data;
  Encapsulator* input;
  TrainData* trainInfo = nullptr;
  vector<Approximator*> F;
  mutable std::mutex buffer_mutex;

  virtual void processStats();

  inline bool canSkip() const
  {
    Uint _nSkipped;
    #pragma omp atomic read
      _nSkipped = nSkipped;
    // If skipping too many samples return w/o sample to avoid code hanging.
    // If true smth is wrong. Approximator will print to screen a warning.
    return _nSkipped < 2*batchSize;
  }

  inline void resample(const Uint thrID) const // TODO resample sequence
  {
    #pragma omp atomic
    nSkipped++;

    Uint sequence, transition;
    data->sampleTransition(sequence, transition, thrID);
    data->Set[sequence]->setSampled(transition);
    return Train(sequence, transition, thrID);
  }

public:
  Profiler* profiler = nullptr;
  string learner_name;
  Uint learnID;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(profiler);
    _dispose_object(data);
  }

  inline void setLearnerName(const string lName, const Uint id) {
    learner_name = lName;
    learnID = id;
  }

  inline unsigned time() const
  {
    return data->readNSeen();
  }
  inline unsigned iter() const
  {
    return nStep;
  }
  inline unsigned nData() const
  {
    return data->readNData();
  }
  inline bool reachedMaxGradStep() const
  {
    return nStep > totNumSteps;
  }
  inline Real annealingFactor() const
  {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.);
    if(nStep*epsAnneal >= 1 || !bTrain) return 0;
    else return 1 - nStep*epsAnneal;
  }

  virtual void select(Agent& agent) = 0;
  virtual void TrainBySequences(const Uint seq, const Uint thrID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint thrID) const=0;

  virtual void getMetrics(ostringstream& buff) const;
  virtual void getHeaders(ostringstream& buff) const;
  //mass-handing of unfinished sequences from master
  void clearFailedSim(const int agentOne, const int agentEnd);
  void pushBackEndedSim(const int agentOne, const int agentEnd);
  bool workerHasUnfinishedSeqs(const int worker) const;

  //main training loop functions:
  virtual void spawnTrainTasks_par() = 0;
  virtual void spawnTrainTasks_seq() = 0;
  virtual bool bNeedSequentialTrain() = 0;

  virtual bool lockQueue() const = 0;

  virtual void prepareGradient();
  virtual void applyGradient();
  virtual void initializeLearner();
  bool predefinedNetwork(Builder& input_net);
  void restart();
};
