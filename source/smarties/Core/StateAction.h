//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#ifndef smarties_StateAction_h
#define smarties_StateAction_h

#include "../Settings/Definitions.h"
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
  nnReal rewardsStdDev = 1, rewardsScale = 1, rewardsMean = 0;

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

  Uint dimS()    const { return dimState; }
  Uint dimObs()  const { return dimStateObserved; }
  Uint dimInfo() const { return  dimState - dimStateObserved; }

  Real getActMaxVal(const Uint i) const { return upperActionValue[i]; }
  Real getActMinVal(const Uint i) const { return lowerActionValue[i]; }
  bool isActBounded(const Uint i) const { return bActionSpaceBounded[i]; }
  Uint dimAct()         const { return dimAction;      }
  Uint dimPol()         const { return policyVecDim;   }
  Uint dimDiscreteAct() const { return maxActionLabel; }
  bool bDiscreteActions() const { return discreteActionValues.size() > 0; }

  Real getActScale(const Uint i) const {
    assert(getActMaxVal(i)-getActMinVal(i) > std::numeric_limits<Real>::epsilon());
    return (getActMaxVal(i)-getActMinVal(i))/2;
  }
  Real getActShift(const Uint i) const {
    return (getActMaxVal(i)+getActMinVal(i))/2;
  }
};

struct StateInfo
{
  const MDPdescriptor& MDP;
  StateInfo(const MDPdescriptor& MDP_) : MDP(MDP_) {}
  StateInfo(const StateInfo& SI) : MDP(SI.MDP) {}

  template<typename T = Real, typename TS> std::vector<T>
  state2observed(const std::vector<TS>& state) const {
    return state2observed<T>(state, MDP);
  }
  template<typename T = Real, typename TS> static std::vector<T>
  state2observed(const std::vector<TS>& S, const MDPdescriptor& MDP)
  {
    assert(S.size() == MDP.dimS());
    std::vector<T> ret(MDP.dimObs());
    for (Uint i=0, k=0; i<MDP.dimS(); ++i)
      if (MDP.bStateVarObserved[i]) ret[k++] = S[i];
    return ret;
  }
  template<typename T = Real, typename TS> std::vector<T>
  state2nonObserved(const std::vector<TS>& state) const
  {
    return state2nonObserved<T>(state, MDP);
  }
  template<typename T = Real, typename TS> static std::vector<T>
  state2nonObserved(const std::vector<TS>& S, const MDPdescriptor& MDP)
  {
    assert(S.size() == MDP.dimS());
    std::vector<T> ret(MDP.dimInfo());
    for (Uint i=0, k=0; i<MDP.dimS(); ++i)
      if (not MDP.bStateVarObserved[i]) ret[k++] = S[i];
    return ret;
  }

  template<typename T> T
  observedAndLatent2state(const T& observ, const T& latent) const
  {
    return observedAndLatent2state(observ, latent, MDP);
  }
  template<typename T = std::vector<Real>> static T
  observedAndLatent2state(const T& observ, const T& latent, const MDPdescriptor& MDP)
  {
    assert(observ.size() == MDP.dimObs() and latent.size() == MDP.dimInfo());
    T ret(MDP.dimS());
    for (Uint i=0, o=0, l=0; i<MDP.dimS(); ++i) {
      if (    MDP.bStateVarObserved[i]) ret[i] = observ[o++];
      else
      if (not MDP.bStateVarObserved[i]) ret[i] = latent[l++];
    }
    return ret;
  }

  template<typename T = Real> void
  scale(std::vector<T>& observed) const
  {
    return scale(observed, MDP);
  }
  template<typename T = Real> static void
  scale(std::vector<T>& observed, const MDPdescriptor& MDP)
  {
    assert(observed.size() == MDP.dimObs());
    for (Uint i=0; i<MDP.dimObs(); ++i)
      observed[i] = ( observed[i] - MDP.stateMean[i] ) * MDP.stateScale[i];
  }

  template<typename T = Real, typename S> std::vector<T>
  getScaled(const std::vector<S>& observed) const
  {
    return getScaled(observed, MDP);
  }
  template<typename T = Real, typename S> static std::vector<T>
  getScaled(const std::vector<S>& observed, const MDPdescriptor& MDP)
  {
    assert(observed.size() == MDP.dimObs());
    std::vector<T> ret(MDP.dimObs());
    for (Uint i=0; i<MDP.dimObs(); ++i)
      ret = ( observed[i] - MDP.stateMean[i] ) * MDP.stateScale[i];
  }
};

struct ActionInfo
{
  const MDPdescriptor& MDP;
  ActionInfo(const MDPdescriptor & MDP_) : MDP(MDP_) {}
  ActionInfo(const ActionInfo& AI) : MDP(AI.MDP) {}

  Real getActMaxVal(const Uint i) const { return MDP.getActMaxVal(i); }
  Real getActMinVal(const Uint i) const { return MDP.getActMinVal(i); }
  bool isBounded(const Uint i)    const { return MDP.isActBounded(i); }
  bool isDiscrete()    const { return MDP.bDiscreteActions(); }
  Uint dim()         const { return MDP.dimAct();         }
  Uint dimPol()      const { return MDP.dimPol();         }
  Uint dimDiscrete() const { return MDP.dimDiscreteAct(); }

  Real getActScale(const Uint i) const {
    return MDP.getActScale(i);
  }
  Real getActShift(const Uint i) const {
    return MDP.getActShift(i);
  }

  ///////////////////////////// CONTINUOUS ACTIONS /////////////////////////////

  template<typename T> std::vector<T>
  envAction2learnerAction(const std::vector<T>& envAct) const
  {
    return envAction2learnerAction(envAct, MDP);
  }
  template<typename T> static std::vector<T>
  envAction2learnerAction(const std::vector<T>& envAct, const MDPdescriptor& MDP)
  {
    assert(not MDP.bDiscreteActions());
    std::vector<T> learnerAct(MDP.dimAct());
    assert( envAct.size() == learnerAct.size() );
    for (Uint i=0; i<MDP.dimAct(); ++i) {
      const Real descaled = (envAct[i] - MDP.getActShift(i))/MDP.getActScale(i);
      if(MDP.isActBounded(i)) assert(descaled > -1 && descaled < 1);
      learnerAct[i] = std::log((1+descaled)/(1-descaled)) / 2;
    }
    return learnerAct;
  }

  template<typename T = Real> std::vector<T>
  learnerPolicy2envPolicy(const Rvec& policy) const
  {
    return learnerPolicy2envPolicy<T>(policy, MDP);
  }
  template<typename T = Real> static std::vector<T>
  learnerPolicy2envPolicy(const Rvec& policy, const MDPdescriptor& MDP)
  {
    if(MDP.bDiscreteActions())
      return std::vector<T>(policy.begin(), policy.end());

    const auto stdFactor = [&] (const Uint i, const Real mean) {
      if (not MDP.isActBounded(i)) return static_cast<Real>(1);
      // var(f(x)) ~= (f'(mean(x)))^2 var(x), here f(x) is tanh(x)
      // f'(x)^2 = (2*exp(x)/(1+exp(x)*exp(x)))^4, but we write the stdev
      const Real expmean = std::exp(mean);
      return std::pow((2*expmean/(1 + expmean*expmean)), 2);
    };

    const Uint dimA = MDP.dimAct();
    assert(policy.size() == 2*dimA && "Supports only gaussian/beta distrib");
    std::vector<T> envPol(2*dimA);
    for (Uint i=0; i<dimA; ++i) {
      const auto scale = MDP.getActScale(i), shift = MDP.getActShift(i);
      const auto bound = MDP.isActBounded(i)? std::tanh(policy[i]) : policy[i];
      envPol[i]      = scale * bound + shift;
      envPol[i+dimA] = scale * stdFactor(i, policy[i]) * policy[i+dimA];
    }
    return envPol;
  }

  template<typename T = Real> std::vector<T>
  learnerAction2envAction(const Rvec& learnerAct) const
  {
    return learnerAction2envAction(learnerAct, MDP);
  }
  template<typename T = Real> static std::vector<T>
  learnerAction2envAction(const Rvec& learnerAct, const MDPdescriptor& MDP)
  {
    if(MDP.bDiscreteActions())
        return std::vector<T>(learnerAct.begin(), learnerAct.end());
    std::vector<T> envAct(MDP.dimAct());
    assert( learnerAct.size() == envAct.size() );
    for (Uint i=0; i<MDP.dimAct(); ++i) {
      const auto bound = MDP.isActBounded(i)? std::tanh(learnerAct[i]) : learnerAct[i];
      envAct[i] = MDP.getActScale(i) * bound + MDP.getActShift(i);
    }
    return envAct;
  }
  /////////////////////////// CONTINUOUS ACTIONS END ///////////////////////////

  ////////////////////////////// DISCRETE ACTIONS //////////////////////////////
  template<typename T> Uint
  actionMessage2label(const std::vector<T>& action) const
  {
    return actionMessage2label(action, MDP);
  }
  template<typename T> static Uint
  actionMessage2label(const std::vector<T>& action, const MDPdescriptor& MDP)
  {
    //map from discretized action (entry per component of values vectors) to int
    assert(MDP.bDiscreteActions());
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

  template<typename T = Real> std::vector<T>
  label2actionMessage(Uint label) const
  {
    return label2actionMessage(label, MDP);
  }
  template<typename T = Real> static std::vector<T>
  label2actionMessage(Uint label, const MDPdescriptor& MDP)
  {
    assert(MDP.bDiscreteActions());
    assert(label < MDP.maxActionLabel);
    //map an int to the corresponding entries in the values vec
    std::vector<T> action( MDP.dimAction );
    for (Uint i = MDP.dimAction; i>0; --i) {
      const Uint index_i = label / MDP.discreteActionShifts[i-1];
      assert(index_i < MDP.discreteActionValues[i-1]);
      action[i-1] = index_i + (T)0.1; // convert to real to move around
      label = label % MDP.discreteActionShifts[i-1];
    }
    return action;
  }
  //////////////////////////// DISCRETE ACTIONS END ////////////////////////////
};

} // end namespace smarties
#endif // smarties_StateAction_h
