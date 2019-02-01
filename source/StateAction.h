//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#pragma once

#include "Settings.h"
#include <cassert>

struct StateInfo
{
  // number of state dimensions and number of state dims observable to learner:
  Uint dim, dimUsed;
  // vector specifying whether a state component is observable to the learner:
  std::vector<bool> inUse;
  Rvec mean, scale; // mean and scale of state variables. env. defined

  StateInfo& operator= (const StateInfo& stateInfo)
  {
    dim     = stateInfo.dim;
    dimUsed = stateInfo.dimUsed;
    assert(dimUsed<=dim);
    inUse.resize(dim);
    for (Uint i=0; i<dim; i++)
      inUse[i] = (stateInfo.inUse[i]);
    return *this;
  }

  //functions returning std, mean, 1/std of observale state components
  std::vector<memReal> inUseStd() const;
  std::vector<memReal> inUseMean() const;
  std::vector<memReal> inUseInvStd() const;
};

class State
{
 public:
  const StateInfo& sInfo;
  Rvec vals;

  State(const StateInfo& newSInfo) : sInfo(newSInfo) {
    vals.resize(sInfo.dim);
  };

  State& operator= (const State& s) {
    if (sInfo.dim != s.sInfo.dim) die("Dimension of states differ!!!\n");
    for (Uint i=0; i<sInfo.dim; i++) vals[i] = s.vals[i];
    return *this;
  }

  inline std::string _print() const {
    return print(vals);
  }

  inline void copy_observed(Rvec& res, const Uint append=0) const {
    //copy state into res, append is used to chain multiple states together
    Uint k = append*sInfo.dimUsed;
    assert(res.size() >= k+sInfo.dimUsed);
    for (Uint i=0; i<sInfo.dim; i++)
      if (sInfo.inUse[i]) res[k++] = vals[i];
  }

  inline Rvec copy_observed() const {
    Rvec ret(sInfo.dimUsed);
    for (Uint i=0, k=0; i<sInfo.dim; i++)
      if (sInfo.inUse[i]) ret[k++] = vals[i];
    return ret;
  }

  inline void copy(Rvec& res) const {
    assert(res.size() == sInfo.dim);
    for (Uint i=0; i<sInfo.dim; i++) res[i] = vals[i];
  }

  template<typename T>
  inline void set(const std::vector<T>& data) {
    assert(data.size() == sInfo.dim);
    for (Uint i=0; i<sInfo.dim; i++) vals[i] = data[i];
  }
};

struct ActionInfo
{
  Uint dim;      //number of actions per turn
  Uint maxLabel; //number of actions options for discretized act spaces
  Uint policyVecDim;
  // whether action have a lower && upper bounded (bool)
  // if true scaled action = tanh ( unscaled action )
  std::vector<bool> bounded;
  //vector<int> boundedTOP, boundedBOT; TODO

  bool discrete = false;
  //each component of action vector has a vector of possible values that action can take with DQN
  std::vector<Rvec> values; //max and min of this vector also used for rescaling
  std::vector<Uint> shifts; //used by DQN to map int to an (entry in each component of values)

  ActionInfo() {}

  ActionInfo& operator= (const ActionInfo& actionInfo)
  {
    dim = actionInfo.dim;
    maxLabel = actionInfo.maxLabel;
    assert(actionInfo.values.size() ==dim);
    assert(actionInfo.shifts.size() ==dim);
    assert(actionInfo.bounded.size()==dim);
    values = actionInfo.values;
    shifts = actionInfo.shifts;
    bounded = actionInfo.bounded;
    discrete = actionInfo.discrete;
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////
  //CONTINUOUS ACTION STUFF
  Real getActMaxVal(const Uint i) const;
  Real getActMinVal(const Uint i) const;

  Real getScaled(const Real unscaled, const Uint i) const;

  Real getDactDscale(const Real unscaled, const Uint i) const;

  Real getInvScaled(const Real scaled, const Uint i) const;

  Rvec getInvScaled(const Rvec scaled) const;

  Rvec getScaled(Rvec unscaled) const;

  Real getUniformProbability() const;

  //////////////////////////////////////////////////////////////////////////////
  //DISCRETE ACTION STUFF
  void updateShifts();

  Uint actionToLabel(const Rvec vals) const;

  Rvec labelToAction(Uint lab) const;

  Uint realActionToIndex(const Real val, const Uint i) const;
};

class Action
{
 public:
  const ActionInfo& actInfo;
  Rvec vals;

  Action(const ActionInfo& newActInfo) :  actInfo(newActInfo)
  {
    vals.resize(actInfo.dim);
  }

  Action& operator= (const Action& a)
  {
    if (actInfo.dim != a.actInfo.dim) die("Dimension of actions differ!!!");
    for (Uint i=0; i<actInfo.dim; i++) vals[i] = a.vals[i];
    return *this;
  }

  inline std::string _print() const
  {
    return print(vals);
  }

  inline void set(const Rvec& data)
  {
    assert(data.size() == actInfo.dim);
    vals = data;
  }

  inline void set(const Uint label)
  {
    vals = actInfo.labelToAction(label);
  }

  inline Uint getActionLabel() const
  {
    return actInfo.actionToLabel(vals);
  }
};
