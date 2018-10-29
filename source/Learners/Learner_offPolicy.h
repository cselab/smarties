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
  const Real obsPerStep_loc = settings.obsPerStep_loc;

  Real alpha = 0.5; // weight between critic and policy
  Real beta = CmaxPol<=0? 1 : 0.0; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;
  Real CinvRet = 1 / CmaxRet;
  bool computeQretrace = false;
  const FORGET ERFILTER =
    MemoryProcessing::readERfilterAlgo(settings.ERoldSeqFilter, CmaxPol>0);

  DelayedReductor ReFER_reduce = DelayedReductor(settings, LDvec{ 0.0, 1.0 } );
public:
  Learner_offPolicy(Environment*const env, Settings& _s);
  virtual void TrainBySequences(const Uint seq, const Uint wID,
    const Uint bID, const Uint tID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint tID) const = 0;

  //main training functions:
  bool blockDataAcquisition() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  virtual void applyGradient() override;
  virtual void prepareGradient() override;
  bool bNeedSequentialTrain() override;
  virtual void initializeLearner() override;
  void save() override;
  void restart() override;

  inline void backPropRetrace(Sequence*const S, const Uint t) {
    if(t == 0) return;
    const Fval W = S->offPolicImpW[t], R=data->scaledReward(S, t), G = gamma;
    const Fval C = W<1 ? W:1, V = S->state_vals[t], A = S->action_adv[t];
    S->setRetrace(t-1, R + G*V + G*C*(S->Q_RET[t] -A-V) );
  }
  inline Fval updateRetrace(Sequence*const S, const Uint t, const Fval A,
    const Fval V, const Fval W) const {
    assert(W >= 0);
    if(t == 0) return 0;
    S->setStateValue(t, V); S->setAdvantage(t, A);
    const Fval oldRet = S->Q_RET[t-1], C = W<1 ? W:1, G = gamma;
    S->setRetrace(t-1, data->scaledReward(S,t) +G*V + G*C*(S->Q_RET[t] -A-V) );
    return std::fabs(S->Q_RET[t-1] - oldRet);
  }

 protected:
  virtual void getMetrics(ostringstream& buff) const override;
  virtual void getHeaders(ostringstream& buff) const override;
};
