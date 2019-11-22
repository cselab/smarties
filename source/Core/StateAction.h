//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#ifndef smarties_StateAction_h
#define smarties_StateAction_h

#include "../Utils/Definitions.h"
#include "../Utils/Warnings.h"
#include <array>
#include <cmath> // log, exp, ...
#include <cassert>

namespace smarties
{

static inline void sendRecvVectorFunc(
  const std::function<void(void*, size_t)>& sendRecvFunc, std::vector<bool>&vec)
{
  Uint vecSize = vec.size();
  sendRecvFunc(&vecSize, 1 * sizeof(Uint) );
  if(vec.size() not_eq vecSize) vec.resize(vecSize);
  if(vecSize == 0) return;
  //else assert( vecSize == (Uint) vec.size() );
  std::vector<int> intvec(vec.begin(), vec.end());
  sendRecvFunc( intvec.data(), vecSize * sizeof(int) );
  std::copy(intvec.begin(), intvec.end(), vec.begin());
}


template<typename T>
static inline void sendRecvVectorFunc(
  const std::function<void(void*, size_t)>& sendRecvFunc, std::vector<T>& vec )
{
  Uint vecSize = vec.size();
  sendRecvFunc(&vecSize, 1 * sizeof(Uint) );
  if(vec.size() not_eq vecSize) vec.resize(vecSize);
  if(vecSize == 0) return;
  //else assert( vecSize == (Uint) vec.size() );
  sendRecvFunc( vec.data(), vecSize * sizeof(T) );
}

struct MDPdescriptor
{
  Uint localID = 0; // ID of agent-int-environment which uses this MDP

  // This struct contains all information to fully define the state and action
  // space of an agent. Only source of complication is that this must hold for
  // both discrete and continuous action problems

  ///////////////////////////// STATE DESCRIPTION /////////////////////////////
  // Number of state dimensions and number of state dims observable to learner:
  Uint dimState = 0, dimStateObserved = 0;
  // vector specifying whether a state component is observable to the learner:
  std::vector<bool> bStateVarObserved;
  // mean and scale of state variables: will be computed from replay memory:
  std::vector<nnReal> stateMean, stateStdDev, stateScale;
  nnReal rewardsStdDev=1, rewardsScale=1;

  // TODO: vector describing shape of state. To enable environment having
  // separate preprocessing for sight as opposed to otehr sensors.
  // This is vector of vectors because each input type will have a vector
  // describing its shape. ( eg. [84 84 1] for atari )
  // std::vector<std::vector<int>> stateShape;

  ///////////////////////////// ACTION DESCRIPTION /////////////////////////////
  // dimensionality of action vector
  Uint dimAction = 0;
  // dimensionality of policy vector (typically 2*dimAction for continuous act,
  // which are mean and diag covariance, or dimAction for discrete policy)
  Uint policyVecDim = 0;

  // whether action have a lower && upper bounded (bool)
  // if true scaled action = tanh ( unscaled action )
  std::vector<bool> bActionSpaceBounded; // TODO 2 bools for semibounded
  // these values are used for scaling or, in case of bounded spaces, as bounds:
  Rvec upperActionValue, lowerActionValue;

  bool bDiscreteActions = false;
  // DISCRETE ACTION stuff:
  //each component of action vector has a vector of possible values:
  std::vector<Uint> discreteActionValues;
  Uint maxActionLabel; //number of actions options for discretized act spaces
  // to map between value and dicrete action option we need 'shifts':
  std::vector<Uint> discreteActionShifts;

  Uint nAppendedObs = 0;
  bool isPartiallyObservable = false;

  std::vector<Conv2D_Descriptor> conv2dDescriptors;

  void synchronize(const std::function<void(void*, size_t)>& sendRecvFunc )
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

    sendRecvFunc(&nAppendedObs,            1 * sizeof(Uint) );
    sendRecvFunc(&isPartiallyObservable,   1 * sizeof(bool) );

    sendRecvVectorFunc(sendRecvFunc, conv2dDescriptors);

    // by default agent can observe all components of state vector
    if(bStateVarObserved.size() == 0)
      bStateVarObserved = std::vector<bool> (dimState, 1);
    sendRecvVectorFunc(sendRecvFunc, bStateVarObserved);
    if( bStateVarObserved.size() not_eq (size_t) dimState)
      die("Application error in setup of bStateVarObserved.");

    // by default state vector scaling is assumed to be with mean 0 and std 1
    if(stateMean.size() == 0) stateMean = std::vector<nnReal> (dimState, 0);
    sendRecvVectorFunc(sendRecvFunc, stateMean);
    if( stateMean.size() not_eq (size_t) dimState)
      die("Application error in setup of stateMean.");

    // by default agent can observer all components of action vector
    if(stateStdDev.size() == 0) stateStdDev = std::vector<nnReal> (dimState, 1);
    sendRecvVectorFunc(sendRecvFunc, stateStdDev);
    if( stateStdDev.size() not_eq (size_t) dimState)
      die("Application error in setup of stateStdDev.");

    stateScale.resize(dimState);
    for(Uint i=0; i<dimState; ++i) {
      if( stateStdDev[i] < std::numeric_limits<Real>::epsilon() )
        _die("Invalid value in scaling of state component %u.", i);
      stateScale[i] = 1/stateStdDev[i];
    }

    dimStateObserved = 0;
    for(Uint i=0; i<dimState; ++i) if(bStateVarObserved[i]) dimStateObserved++;
    if(world_rank == 0) {
     printf("SETUP: State vector has %lu components, %lu of which are observed. "
     "Action vector has %lu %s-valued components.\n", dimState,
     dimStateObserved, dimAction, bDiscreteActions? "discrete" : "continuous");
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
      if(world_rank==0) {
        printf("Action vector components :");
        for (Uint i=0; i<dimAction; ++i) {
          printf(" [ %lu : %s to (%.1f:%.1f) ]", i,
          bActionSpaceBounded[i] ? "bound" : "scaled",
          upperActionValue[i], lowerActionValue[i]);
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
        printf(" [ %lu : %lu options ]", i, discreteActionValues[i]);
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
};

struct StateInfo
{
  const MDPdescriptor& MDP;
  StateInfo(const MDPdescriptor& MDP_) : MDP(MDP_) {}
  StateInfo(const StateInfo& SI) : MDP(SI.MDP) {}

  Uint dim() const { return MDP.dimState; }
  Uint dimObs() const { return MDP.dimStateObserved; }

  template<typename T = Real>
  std::vector<T> state2observed(const Rvec& state) const
  {
    assert(state.size() == MDP.dimState);
    std::vector<T> ret(MDP.dimStateObserved);
    for (Uint i=0, k=0; i<MDP.dimState; ++i)
      if (MDP.bStateVarObserved[i]) ret[k++] = state[i];
    return ret;
  }

  template<typename T = Real>
  void scale(std::vector<T>& observed) const
  {
    assert(observed.size() == MDP.dimStateObserved);
    for (Uint i=0; i<MDP.dimStateObserved; ++i)
      observed[i] = ( observed[i] - MDP.stateMean[i] ) * MDP.stateScale[i];
  }

  template<typename T = Real, typename S>
  std::vector<T> getScaled(const std::vector<S>& observed) const
  {
    assert(observed.size() == MDP.dimStateObserved);
    std::vector<T> ret(MDP.dimStateObserved);
    for (Uint i=0; i<MDP.dimStateObserved; ++i)
      ret = ( observed[i] - MDP.stateMean[i] ) * MDP.stateScale[i];
  }
};

struct ActionInfo
{
  const MDPdescriptor& MDP;
  ActionInfo(const MDPdescriptor & MDP_) : MDP(MDP_) {}
  ActionInfo(const ActionInfo& AI) : MDP(AI.MDP) {}

  ///////////////////////////// CONTINUOUS ACTIONS /////////////////////////////
  Real getActMaxVal(const Uint i) const { return MDP.upperActionValue[i]; }
  Real getActMinVal(const Uint i) const { return MDP.lowerActionValue[i]; }
  bool isBounded(const Uint i) const { return MDP.bActionSpaceBounded[i]; }
  Uint dim()         const { return MDP.dimAction;      }
  Uint dimPol()      const { return MDP.policyVecDim;   }
  Uint dimDiscrete() const { return MDP.maxActionLabel; }

  template<typename T>
  static inline T _tanh(const T x) {
    const Real e2x = std::exp( -2 * x );
    return (1-e2x)/(1+e2x);
  }
  template<typename T>
  static inline T _invTanh(const T y) {
    assert(std::fabs(y) < 1);
    return std::log( (y+1)/(1-y) ) / 2;
  }

  template<typename T>
  Rvec action2scaledAction(const std::vector<T>& unscaled) const
  {
    assert(not MDP.bDiscreteActions);
    Rvec ret(MDP.dimAction);
    assert( unscaled.size() == ret.size() );
    for (Uint i=0; i<MDP.dimAction; ++i)
    {
      const Real y = MDP.bActionSpaceBounded[i]? _tanh(unscaled[i]):unscaled[i];
      const Real min_a=MDP.lowerActionValue[i], max_a=MDP.upperActionValue[i];
      assert( max_a - min_a > std::numeric_limits<Real>::epsilon() );
      ret[i] = min_a + (max_a-min_a)/2 * (y + 1);
    }
    return ret;
  }

  template<typename T = Real>
  std::vector<T> scaledAction2action(const Rvec& scaled) const
  {
    assert(not MDP.bDiscreteActions);
    std::vector<T> ret(MDP.dimAction);
    assert( scaled.size() == ret.size() );
    for (Uint i=0; i<MDP.dimAction; ++i)
    {
      const T min_a=MDP.lowerActionValue[i], max_a=MDP.upperActionValue[i];
      assert( max_a - min_a > std::numeric_limits<Real>::epsilon() );
      const T y = 2 * (scaled[i] - min_a)/(max_a - min_a) - 1;
      ret[i] = MDP.bActionSpaceBounded[i] ? _invTanh(y) : y;
    }
    return ret;
  }
  /////////////////////////// CONTINUOUS ACTIONS END ///////////////////////////

  ////////////////////////////// DISCRETE ACTIONS //////////////////////////////
  template<typename T>
  Uint actionMessage2label(const std::vector<T>& action) const
  {
    //map from discretized action (entry per component of values vectors) to int
    assert(MDP.bDiscreteActions);
    assert(action.size() == MDP.dimAction);
    assert(MDP.discreteActionShifts.size() == MDP.dimAction);
    Uint label = 0;
    for (Uint i=0; i < MDP.dimAction; ++i) {
      // actions are passed around like doubles, but here map to int
      const Uint valI = std::floor(action[i]);
      assert(valI < MDP.discreteActionValues[i]);
      label += MDP.discreteActionShifts[i] * valI;
    }
    assert(label < MDP.maxActionLabel);
    return label;
  }

  template<typename T = Real>
  std::vector<T> label2actionMessage(Uint label) const
  {
    assert(MDP.bDiscreteActions);
    assert(label < MDP.maxActionLabel);
    //map an int to the corresponding entries in the values vec
    std::vector<T> action( MDP.dimAction );
    for (Uint i = MDP.dimAction; i>0; --i) {
      const Uint index_i = label / MDP.discreteActionShifts[i-1];
      assert(index_i < MDP.discreteActionValues[i]);
      action[i-1] = index_i + (T)0.1; // convert to real to move around
      label = label % MDP.discreteActionShifts[i-1];
    }
    return action;
  }

  void testDiscrete()
  {
    //for(Uint i=0; i<MDP.maxActionLabel; ++i)
    //  if(i != action2label(label2action(i)))
    //    _die("label %u does not match for action [%s]. returned %u",
    //      i, print(label2action(i)).c_str(), action2label(label2action(i)) );
  }
  //////////////////////////// DISCRETE ACTIONS END ////////////////////////////
};

} // end namespace smarties
#endif // smarties_StateAction_h
