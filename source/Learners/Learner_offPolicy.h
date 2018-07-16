//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Learner.h"

class Learner_offPolicy: public Learner
{
protected:
  const Real obsPerStep_orig;
  const Uint nObsPerTraining;
  mutable int percData = -5;
  Uint nData_b4Startup = 0;
  Real nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig;

  //to be overwritten if algo needs specific filtering of old episodes
  FORGET MEMBUF_FILTER_ALGO = OLDEST;


  Real beta = CmaxPol<=0? 1 : 0.2; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;

  ApproximateReductor reductor = ApproximateReductor(mastersComm, 2);
public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;

  //main training functions:
  bool lockQueue() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  virtual void applyGradient() override;
  bool bNeedSequentialTrain() override;
  virtual void initializeLearner() override;
};
