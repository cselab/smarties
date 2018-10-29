//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Learner.h"

template<typename Action_t>
class CMALearner: public Learner
{
protected:
  const Uint nWorkers_own = settings.nWorkers_own;
  const Uint ESpopSize_loc = ESpopSize / learn_size;
  const Uint nAgentsPerWorker = env->nAgentsPerRank;
  const Uint nSeqPerStep = batchSize * ESpopSize_loc * nAgentsPerWorker;
  const Uint nSeqPerWorker = batchSize * ESpopSize_loc / nWorkers_own;
  const Uint ESpopStart = ESpopSize_loc * learn_rank;

  THRvec<long> WiEnded = THRvec<long>(nWorkers_own, 0);
  THRvec<long> WnEnded = THRvec<long>(nWorkers_own, 0);
  THRvec<long> WwghtID = THRvec<long>(nWorkers_own, 0);

  std::vector<Rvec> R = std::vector<Rvec>(nWorkers_own, Rvec(ESpopSize, 0) );

  static vector<Uint> count_pol_outputs(const ActionInfo*const aI);
  static vector<Uint> count_pol_starts(const ActionInfo*const aI);

public:
  CMALearner(Environment*const _env, Settings&_s);

  //main training functions:
  void select(Agent& agent) override;

  void applyGradient() override;
  void prepareGradient() override;
  void initializeLearner() override;
  bool blockDataAcquisition() const override;
  void globalGradCounterUpdate() override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  bool bNeedSequentialTrain() override;

  static Uint getnDimPolicy(const ActionInfo*const aI);
};
