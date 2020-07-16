//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_DPG_h
#define smarties_DPG_h

#include "Learner_approximator.h"

namespace smarties
{

class DPG : public Learner_approximator
{
  const Uint nA = aInfo.dim();
  const Real explNoise = settings.explNoise;
  const Real OrUhDecay = settings.clipImpWeight <= 0 ? 0.85 : 0; // as in original
  //const Real OrUhDecay = 0; // no correlated noise
  std::vector<Rvec> OrUhState = std::vector<Rvec>(nAgents, Rvec(nA,0));
  Approximator* actor;
  Approximator* critc;

  void Train(const MiniBatch& MB, const Uint wID, const Uint bID) const override;

public:
  DPG(MDPdescriptor&, HyperParameters&, ExecutionInfo&);

  void setupTasks(TaskQueue& tasks) override;
  void selectAction(const MiniBatch& MB, Agent& agent) override;
  void processTerminal(const MiniBatch& MB, Agent& agent) override;
};

}

#endif // smarties_DPG_h
