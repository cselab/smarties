//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Learner.h"

class Learner_onPolicy: public Learner
{
protected:
  const Uint nHorizon, nEpochs;
  mutable Uint cntBatch = 0, cntEpoch = 0, cntKept = 0;


public:
  Learner_onPolicy(Environment*const _env, Settings&_s);

  //main training functions:
  bool lockQueue() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  bool bNeedSequentialTrain() override;
  virtual void prepareGradient() override;
  virtual void initializeLearner() override;
};
