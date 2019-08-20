//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_DQN_h
#define smarties_DQN_h

#include "Learner_approximator.h"

namespace smarties
{

class DQN : public Learner_approximator
{
  const Uint nA = MDP.maxActionLabel;

  Real annealingFactor() const
  {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.0);
    const auto mynstep = nGradSteps();
    if(mynstep*epsAnneal >= 1 || not bTrain) return 0;
    else return 1 - mynstep*epsAnneal;
  }

  void Train(const MiniBatch&MB, const Uint wID,const Uint bID) const override;

public:
  DQN(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_);

  void setupTasks(TaskQueue& tasks) override;
  void select(Agent& agent) override;
};

}
#endif
