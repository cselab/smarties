//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#include "StateAction.h"
#include <algorithm>
#include <math.h>


static inline Real _tanh(const Real inp)
{
  if(inp>0) {
    //const Real e2x = std::exp(-2*std::min( (Real)EXP_CUT, inp));
    const Real e2x = std::exp(-2*inp);
    return (1-e2x)/(1+e2x);
  } else {
    //const Real e2x = std::exp( 2*std::max(-(Real)EXP_CUT, inp));
    const Real e2x = std::exp( 2*inp);
    return (e2x-1)/(1+e2x);
  }
}

static inline Real Dtanh(const Real inp)
{
  const Real arg = inp < 0 ? -inp : inp; //symmetric
  //const Real e2x = std::exp(-2*std::min((Real)EXP_CUT, arg));
  const Real e2x = std::exp(-2*arg);
  return 4*e2x/((1+e2x)*(1+e2x));
}

std::vector<memReal> StateInfo::inUseStd() const {
  std::vector<memReal> ret(dimUsed, 0);
  for(Uint i=0, k=0; i<dim && scale.size(); i++) {
    if(inUse[i]) ret[k++] = scale[i];
    if(i+1 == dim) assert(k == dimUsed);
  }
  return ret;
}
std::vector<memReal> StateInfo::inUseMean() const {
  std::vector<memReal> ret(dimUsed, 1);
  for(Uint i=0, k=0; i<dim && mean.size(); i++) {
    if(inUse[i]) ret[k++] = mean[i];
    if(i+1 == dim) assert(k == dimUsed);
  }
  return ret;
}
std::vector<memReal> StateInfo::inUseInvStd() const {
  std::vector<memReal> ret(dimUsed, 1);
  for(Uint i=0, k=0; i<dim && scale.size(); i++) {
    if(inUse[i]) ret[k++] = 1/scale[i];
    if(i+1 == dim) assert(k == dimUsed);
  }
  return ret;
}

///////////////////////////////////////////////////////////////////////////////
//CONTINUOUS ACTION STUFF
Real ActionInfo::getActMaxVal(const Uint i) const
{
  assert(i<dim && dim==values.size());
  assert(values[i].size()>1); //otherwise scaling is impossible
  return * std::max_element(std::begin(values[i]), std::end(values[i]));
}

Real ActionInfo::getActMinVal(const Uint i) const
{
  assert(i<dim && dim==values.size());
  assert(values[i].size()>1); //otherwise scaling is impossible
  return * std::min_element(std::begin(values[i]), std::end(values[i]));
}

Real ActionInfo::getScaled(const Real unscaled, const Uint i) const
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

Real ActionInfo::getDactDscale(const Real unscaled, const Uint i) const
{
  //derivative of scaled action wrt to unscaled action, see getScaled()
  const Real min_a = getActMinVal(i), max_a = getActMaxVal(i);
  if (bounded[i]) {
    return 0.5*(max_a-min_a)*Dtanh(unscaled);
    //const Real denom = 1. + std::fabs(unscaled);
    //return 0.5*(max_a-min_a)/denom/denom;
  } else return 0.5*(max_a-min_a);
}

Real ActionInfo::getInvScaled(const Real scaled, const Uint i) const
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

Rvec ActionInfo::getInvScaled(const Rvec scaled) const
{
  Rvec ret = scaled;
  assert(ret.size()==dim);
  for (Uint i=0; i<dim; i++)
    ret[i] = getInvScaled(scaled[i], i);
  return ret;
}

Rvec ActionInfo::getScaled(Rvec unscaled) const
{
  Rvec ret(dim);
  assert(unscaled.size()==dim);
  for (Uint i=0; i<dim; i++) ret[i] = getScaled(unscaled[i], i);
  return ret;
}

Real ActionInfo::getUniformProbability() const
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
void ActionInfo::updateShifts()
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

Uint ActionInfo::actionToLabel(const Rvec vals) const
{
  assert(vals.size() == dim && shifts.size() == dim);
  //map from discretized action (entry per component of values vectors) to int
  Uint lab=0;
  for (Uint i=0; i<dim; i++)
    lab += shifts[i]*realActionToIndex(vals[i],i);

  #ifndef NDEBUG
    std::vector<Uint> test(dim);
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

Rvec ActionInfo::labelToAction(Uint lab) const
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

Uint ActionInfo::realActionToIndex(const Real val, const Uint i) const
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
