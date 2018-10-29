//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"


class DQN : public Learner_offPolicy
{
  void TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
    const Uint thrID) const override;
  void Train(const Uint seq, const Uint t, const Uint wID,
    const Uint bID, const Uint thrID) const override;

public:
  DQN(Environment*const env, Settings & settings);
  void select(Agent& agent) override;
};
