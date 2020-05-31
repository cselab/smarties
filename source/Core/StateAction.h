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
  nnReal rewardsStdDev=1, rewardsScale=1, rewardsMean = 0;

  // TODO: vector describing shape of state. To enable environment having
  // separate preprocessing for sight as opposed to other sensors.
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

  // In the common case where act = policy (or mean) + stdev * N(0,1)
  // The following option allows sampling N(0,1) only once per time step
  // and share its value among all agents. Application in RL for LES model:
  // many agents (grid points) collaborate to dissipate right amount of energy
  // across the grid. Having different noise values makes the problem harder.
  bool bAgentsShareNoise = false;
  Rvec sharedNoiseVecTic, sharedNoiseVecToc;

  std::vector<Conv2D_Descriptor> conv2dDescriptors;
  void synchronize(const std::function<void(void*, size_t)>& sendRecvFunc);
};

struct StateInfo
{
  const MDPdescriptor& MDP;
  StateInfo(const MDPdescriptor& MDP_) : MDP(MDP_) {}
  StateInfo(const StateInfo& SI) : MDP(SI.MDP) {}

  Uint dim() const { return MDP.dimState; }
  Uint dimObs() const { return MDP.dimStateObserved; }
  Uint dimInfo() const { return  MDP.dimState - MDP.dimStateObserved; }

  template<typename T = Real>
  std::vector<T> state2observed(const Rvec& state) const {
    return state2observed<T>(state, MDP);
  }
  template<typename T = Real, typename TS>
  static std::vector<T> state2observed(const std::vector<TS>& S, const MDPdescriptor& MDP)
  {
    assert(S.size() == MDP.dimState);
    std::vector<T> ret(MDP.dimStateObserved);
    for (Uint i=0, k=0; i<MDP.dimState; ++i)
      if (MDP.bStateVarObserved[i]) ret[k++] = S[i];
    return ret;
  }
  template<typename T = Real, typename TS>
  std::vector<T> state2nonObserved(const std::vector<TS>& state) const
  {
    assert(state.size() == dim());
    std::vector<T> ret( dimInfo() );
    for (Uint i=0, k=0; i<dim(); ++i)
      if (not MDP.bStateVarObserved[i]) ret[k++] = state[i];
    return ret;
  }

  template<typename T = Real>
  std::vector<T> observedAndLatent2state(const std::vector<T>& observ,
                                         const std::vector<T>& latent) const
  {
    assert(observ.size() == dimObs() and latent.size() == dimInfo());
    std::vector<T> ret( dim() );
    for (Uint i=0, o=0, l=0; i<dim(); ++i) {
      if (    MDP.bStateVarObserved[i]) ret[i] = observ[o++];
      if (not MDP.bStateVarObserved[i]) ret[i] = latent[l++];
    }
    return ret;
  }

  template<typename T = Real>
  void scale(std::vector<T>& observed) const
  {
    assert(observed.size() == dimObs());
    for (Uint i=0; i<dimObs(); ++i)
      observed[i] = ( observed[i] - MDP.stateMean[i] ) * MDP.stateScale[i];
  }

  template<typename T = Real, typename S>
  std::vector<T> getScaled(const std::vector<S>& observed) const
  {
    assert(observed.size() == dimObs());
    std::vector<T> ret(dimObs());
    for (Uint i=0; i<dimObs(); ++i)
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

  Real getScale(const Uint i) const {
    assert(getActMaxVal(i)-getActMinVal(i) > std::numeric_limits<Real>::epsilon());
    return isBounded(i) ? getActMaxVal(i)-getActMinVal(i) : (getActMaxVal(i)-getActMinVal(i))/2;
  }
  Real getShift(const Uint i) const {
    return isBounded(i) ? getActMinVal(i) : (getActMaxVal(i)+getActMinVal(i))/2;
  }

  template<typename T>
  Rvec envAction2learnerAction(const std::vector<T>& envAct) const
  {
    assert(not MDP.bDiscreteActions);
    std::vector<T> learnerAct(dim());
    assert( envAct.size() == learnerAct.size() );
    for (Uint i=0; i<dim(); ++i) {
      learnerAct[i] = (envAct[i] - getShift(i)) / getScale(i);
      // if bounded action space learner samples a beta distribution:
      if(isBounded(i)) assert(learnerAct[i]>0 && learnerAct[i] < 1);
    }
    return learnerAct;
  }

  template<typename T = Real>
  std::vector<T> learnerPolicy2envPolicy(const Rvec& policy) const
  {
    if(MDP.bDiscreteActions)
      return std::vector<T>(policy.begin(), policy.end());
    assert(policy.size() == 2*dim() && "Supports only gaussian/beta distrib");
    std::vector<T> envPol(2*dim());
    for (Uint i=0; i<dim(); ++i) {
      envPol[i] = getScale(i) * policy[i] + getShift(i);
      envPol[i+dim()] = getScale(i) * policy[i+dim()];
      // if bounded action space learner samples a beta distribution:
      if(isBounded(i)) assert(policy[i]>=0 && policy[i] < 1);
    }
    return envPol;
  }

  template<typename T = Real>
  std::vector<T> learnerAction2envAction(const Rvec& learnerAct) const
  {
    if(MDP.bDiscreteActions)
        return std::vector<T>(learnerAct.begin(), learnerAct.end());
    std::vector<T> envAct(dim());
    assert( learnerAct.size() == envAct.size() );
    for (Uint i=0; i<dim(); ++i) {
      envAct[i] = getScale(i) * learnerAct[i] + getShift(i);
      // if bounded action space learner samples a beta distribution:
      if(isBounded(i)) assert(learnerAct[i]>=0 && learnerAct[i] < 1);
    }
    return envAct;
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
