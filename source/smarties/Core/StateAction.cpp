//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#include "StateAction.h"
#include "../Utils/MPIUtilities.h"

namespace smarties
{

void MDPdescriptor::synchronize(const std::function<void(void*, size_t)>& sendRecvFunc )
{
  const int world_rank = MPIworldRank();

  // In this function we first recv all quantities of the descriptor
  // then we send them along. Idea is that simulation ranks will do nothing on
  // recv. Master ranks will first receive info from simulation, then pass it
  // along to worker-less masters.
  sendRecvFunc(&dimState, 1 * sizeof(Uint) );
  if(dimState == 0) warn("Stateless RL");

  sendRecvFunc(&dimAction,        1 * sizeof(Uint) );
  if(dimAction == 0)
    die("Application did not set up dimensionality of action vector.");

  sendRecvFunc(&bDiscreteActions, 1 * sizeof(bool) );
  sendRecvFunc(&bAgentsShareNoise, 1 * sizeof(bool) );

  sendRecvFunc(&nAppendedObs,            1 * sizeof(Uint) );
  sendRecvFunc(&isPartiallyObservable,   1 * sizeof(bool) );

  sendRecvVectorFunc(sendRecvFunc, conv2dDescriptors);

  // by default agent can observe all components of state vector
  if(bStateVarObserved.size() == 0)
    bStateVarObserved = std::vector<bool> (dimState, 1);
  sendRecvVectorFunc(sendRecvFunc, bStateVarObserved);
  if( bStateVarObserved.size() not_eq (size_t) dimState)
    die("Application error in setup of bStateVarObserved.");

  dimStateObserved = 0;
  for(Uint i=0; i<dimState; ++i) if(bStateVarObserved[i]) dimStateObserved++;
  if(world_rank == 0) {
   printf("SETUP: State vector has %u components, %u of which are observed. "
          "Action vector has %u %s-valued components.\n", (unsigned) dimState,
          (unsigned) dimStateObserved, (unsigned) dimAction,
          bDiscreteActions? "discrete" : "continuous");
  }

  // by default state vector scaling is assumed to be with mean 0 and std 1
  if(stateMean.size()==0) stateMean = std::vector<nnReal>(dimStateObserved, 0);
  if(stateMean.size()==dimState and dimState > dimStateObserved)
    stateMean = StateInfo::state2observed<nnReal>(stateMean, *this);
  sendRecvVectorFunc(sendRecvFunc, stateMean);
  if( stateMean.size() not_eq (size_t) dimStateObserved)
    die("Application error in setup of stateMean.");

  // by default agent can observer all components of action vector
  if(stateStdDev.size()==0) stateStdDev = std::vector<nnReal>(dimStateObserved, 1);
  if(stateStdDev.size()==dimState and dimState > dimStateObserved)
    stateStdDev = StateInfo::state2observed<nnReal>(stateStdDev, *this);
  sendRecvVectorFunc(sendRecvFunc, stateStdDev);
  if( stateStdDev.size() not_eq (size_t) dimStateObserved)
    die("Application error in setup of stateStdDev.");

  stateScale.resize(dimStateObserved);
  for(Uint i=0; i<dimStateObserved; ++i) {
    if( stateStdDev[i] < std::numeric_limits<Real>::epsilon() )
      _die("Invalid value in scaling of state component %u.", i);
    stateScale[i] = 1/stateStdDev[i];
  }

  // by default agent's action space is unbounded
  if(bActionSpaceBounded.size() == 0)
    bActionSpaceBounded = std::vector<bool> (dimAction, 0);
  sendRecvVectorFunc(sendRecvFunc, bActionSpaceBounded);
  if( bActionSpaceBounded.size() not_eq (size_t) dimAction)
    die("Application error in setup of bActionSpaceBounded.");

  // by default agent's action space not scaled (ie up/low vals are -1 and 1)
  if(upperActionValue.size() == 0)
    upperActionValue = std::vector<Real> (dimAction,  1);
  sendRecvVectorFunc(sendRecvFunc, upperActionValue);
  if( upperActionValue.size() not_eq (size_t) dimAction)
    die("Application error in setup of upperActionValue.");

  // by default agent's action space not scaled (ie up/low vals are -1 and 1)
  if(lowerActionValue.size() == 0)
    lowerActionValue = std::vector<Real> (dimAction, -1);
  sendRecvVectorFunc(sendRecvFunc, lowerActionValue);
  if( lowerActionValue.size() not_eq (size_t) dimAction)
    die("Application error in setup of lowerActionValue.");

  if(bDiscreteActions == false)
  {
    for (Uint i=0; i<dimAction; ++i) {
      const auto L = lowerActionValue[i], U = upperActionValue[i];
      lowerActionValue[i] = std::min(L, U);
      upperActionValue[i] = std::max(L, U);
    }
    if(world_rank==0) {
      printf("Action vector components :");
      for (Uint i=0; i<dimAction; ++i) {
        printf(" [ %u : %s to (%.1f:%.1f) ]", (unsigned) i,
          bActionSpaceBounded[i] ? "bound" : "scaled",
          lowerActionValue[i], upperActionValue[i]);
        // tidy-up formatting for very high-dim action spaces:
        if( ((i+1) % 3) == 0 && i+1 < dimAction )
          printf("\n                          ");
      }
      printf("\n");
    }
    return; // skip setup of discrete-action stuff
  }

  // Now some logic. The discreteActionValues vector should have size dimAction
  // If action space is continuous, these values are not used. If action space
  // is discrete at the end of synchronization we make sure that each component
  // has size greater than one. Otherwise agent has no options to choose from.
  if(discreteActionValues.size() == 0)
    discreteActionValues = std::vector<Uint> (dimAction, 0);
  sendRecvVectorFunc(sendRecvFunc, discreteActionValues);
  if( discreteActionValues.size() not_eq (size_t) dimAction)
    die("Application error in setup of discreteActionValues.");

  if(world_rank==0) printf("Discrete-action vector options :");
  for(size_t i=0; i<dimAction; ++i) {
    if( discreteActionValues[i] < 2 )
      die("Application error in setup of discreteActionValues: "
          "found less than 2 options to choose from.");
    if(world_rank==0)
      printf(" [ %u : %u options ]", (unsigned) i, (unsigned) discreteActionValues[i]);
  }
  if(world_rank==0) printf("\n");

  discreteActionShifts = std::vector<Uint>(dimAction);
  discreteActionShifts[0] = 1;
  for (Uint i=1; i < dimAction; ++i)
    discreteActionShifts[i] = discreteActionShifts[i-1] *
                              discreteActionValues[i-1];

  maxActionLabel = discreteActionShifts[dimAction-1] *
                   discreteActionValues[dimAction-1];
}

} // end namespace smarties
