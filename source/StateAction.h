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
#include <algorithm>
#include <math.h>

using namespace std;

struct StateInfo
{
  // number of state dimensions and number of state dims observable to learner:
  Uint dim, dimUsed;
  // vector specifying whether a state component is observable to the learner:
  vector<bool> inUse;
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
  vector<memReal> inUseStd() const {
    vector<memReal> ret(dimUsed, 0);
    for(Uint i=0, k=0; i<dim && scale.size(); i++) {
      if(inUse[i]) ret[k++] = scale[i];
      if(i+1 == dim) assert(k == dimUsed);
    }
    return ret;
  }
  vector<memReal> inUseMean() const {
    vector<memReal> ret(dimUsed, 1);
    for(Uint i=0, k=0; i<dim && mean.size(); i++) {
      if(inUse[i]) ret[k++] = mean[i];
      if(i+1 == dim) assert(k == dimUsed);
    }
    return ret;
  }
  vector<memReal> inUseInvStd() const {
    vector<memReal> ret(dimUsed, 1);
    for(Uint i=0, k=0; i<dim && scale.size(); i++) {
      if(inUse[i]) ret[k++] = 1/scale[i];
      if(i+1 == dim) assert(k == dimUsed);
    }
    return ret;
  }
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

  inline string _print() const {
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
  inline void set(const vector<T>& data) {
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
  vector<bool> bounded;
  //vector<int> boundedTOP, boundedBOT; TODO

  bool discrete = false;
  //each component of action vector has a vector of possible values that action can take with DQN
  vector<Rvec> values; //max and min of this vector also used for rescaling
  vector<Uint> shifts; //used by DQN to map int to an (entry in each component of values)

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
  inline Real getActMaxVal(const Uint i) const
  {
    assert(i<dim && dim==values.size());
    assert(values[i].size()>1); //otherwise scaling is impossible
    return *std::max_element(std::begin(values[i]), std::end(values[i]));
  }

  inline Real getActMinVal(const Uint i) const
  {
    assert(i<dim && dim==values.size());
    assert(values[i].size()>1); //otherwise scaling is impossible
    return *std::min_element(std::begin(values[i]), std::end(values[i]));
  }

  static Real _tanh(const Real inp)
  {
    if(inp>0) {
      const Real e2x = std::exp(-std::min((Real) 16, 2*inp));
      return (1-e2x)/(1+e2x);
    } else {
      const Real e2x = std::exp( std::max((Real)-16, 2*inp));
      return (e2x-1)/(1+e2x);
    }
  }

  static Real Dtanh(const Real inp)
  {
    const Real arg = inp < 0 ? -inp : inp; //symmetric
    const Real e2x = std::exp(-std::min((Real)16, 2*arg));
    return 4*e2x/((1+e2x)*(1+e2x));
  }

  inline Real getScaled(const Real unscaled, const Uint i) const
  {
    //unscaled value and i is to which component of action vector it corresponds
    //if action space is bounded, return the scaled component, else return unscaled
    //scaling is between max and min of values vector (user specified in environment)
    //scaling function is x/(1+abs(x)) (between -1 and 1 for x in -inf, inf)
    const Real min_a = getActMinVal(i), max_a = getActMaxVal(i);
    assert(max_a-min_a > std::numeric_limits<Real>::epsilon());
    if (bounded[i]) {
      const Real soft_sign = _tanh(unscaled);
      //const Real soft_sign = unscaled/(1. + std::fabs(unscaled));
      return       min_a + 0.5*(max_a - min_a)*(soft_sign + 1);
    } else  return min_a + 0.5*(max_a - min_a)*(unscaled  + 1);

  }

  inline Real getDactDscale(const Real unscaled, const Uint i) const
  {
    //derivative of scaled action wrt to unscaled action, see getScaled()
    const Real min_a = getActMinVal(i), max_a = getActMaxVal(i);
    if (bounded[i]) {
      return 0.5*(max_a-min_a)*Dtanh(unscaled);
      //const Real denom = 1. + std::fabs(unscaled);
      //return 0.5*(max_a-min_a)/denom/denom;
    } else return 0.5*(max_a-min_a);
  }

  inline Real getInvScaled(const Real scaled, const Uint i) const
  {
    //opposite operation
    const Real min_a = getActMinVal(i), max_a = getActMaxVal(i);
    assert(max_a-min_a > std::numeric_limits<Real>::epsilon());
    if (bounded[i]) {
      assert(scaled>min_a && scaled<max_a);
      const Real y = 2*(scaled - min_a)/(max_a - min_a) -1;
      assert(std::fabs(y) < 1);
      //return  y/(1.-std::fabs(y));
      return  0.5*std::log((y+1)/(1-y));
    } else {
      return 2*(scaled - min_a)/(max_a - min_a) -1;
    }
  }

  inline Rvec getInvScaled(const Rvec scaled) const
  {
    Rvec ret = scaled;
    assert(ret.size()==dim);
    for (Uint i=0; i<dim; i++)
      ret[i] = getInvScaled(scaled[i], i);
    return ret;
  }

  inline Rvec getScaled(Rvec unscaled) const
  {
    Rvec ret(dim);
    assert(unscaled.size()==dim);
    for (Uint i=0; i<dim; i++) ret[i] = getScaled(unscaled[i], i);
    return ret;
  }

  inline Real getUniformProbability() const
  {
    Real P = 1;
    for (Uint i=0; i<dim; i++) {
      const Real lB = getActMinVal(i), uB = getActMaxVal(i);
      assert(uB-lB > std::numeric_limits<Real>::epsilon());
      P /= (uB-lB);
    }
    return P;
  }

  ///////////////////////////////////////////////////////////////////////////////
  //DISCRETE ACTION STUFF
  void updateShifts()
  {
    shifts.resize(dim);
    shifts[0] = 1;
    for (Uint i=1; i < dim; i++)
      shifts[i] = shifts[i-1] * values[i-1].size();

    maxLabel = shifts[dim-1] * values[dim-1].size();

    #ifndef NDEBUG
    for (Uint i=0; i<maxLabel; i++)
      if(i!=actionToLabel(labelToAction(i)))
        _die("label %u, action [%s], ret %u",
            i, print(labelToAction(i)).c_str(),
            actionToLabel(labelToAction(i)));
    #endif
  }

  inline Uint actionToLabel(const Rvec vals) const
  {
    assert(vals.size() == dim && shifts.size() == dim);
    //map from discretized action (entry per component of values vectors) to int
    Uint lab=0;
    for (Uint i=0; i<dim; i++)
      lab += shifts[i]*realActionToIndex(vals[i],i);

    #ifndef NDEBUG
      vector<Uint> test(dim);
      Uint max = 1;
      for (Uint i=0; i < dim; i++) {
        test[i] = i==0 ? 1 : test[i-1] * values[i-1].size();
        assert(test[i] == shifts[i]);
        max *= values[i].size();
      }
      assert(max == maxLabel);
    #endif

    return lab;
  }

  inline Rvec labelToAction(Uint lab) const
  {
    //map an int to the corresponding entries in the values vec
    Rvec ret(dim);
    for (Uint i=dim; i>0; i--) {
      Uint tmp = lab/shifts[i-1]; //in opposite op: add shifts*index
      ret[i-1] = values[i-1][tmp];
      lab = lab % shifts[i-1];
    }
    return ret;
  }

  inline Uint realActionToIndex(const Real val, const Uint i) const
  {
    //From continous action for i-th component of action vector
    // convert to an entry in values vector
    Real dist = 1e9; Uint ret = 0;
    for (Uint j=0; j<values[i].size(); j++) {
      const Real _dist = std::fabs(values[i][j]-val);
      if (_dist<dist) { dist = _dist; ret = j; }
    }
    return ret;
  }
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

  inline string _print() const
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
