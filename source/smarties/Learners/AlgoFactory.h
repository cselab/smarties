//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_AlgoFactory_h
#define smarties_AlgoFactory_h

#include "Learner.h"

namespace smarties
{

std::unique_ptr<Learner> createLearner(
  const Uint learnerID, MDPdescriptor& MDP, ExecutionInfo& distrib
);

}
#endif // smarties_AlgoFactory_h
