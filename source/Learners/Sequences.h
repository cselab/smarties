//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "../StateAction.h"
#include "../Settings.h"
#include "../Environments/Environment.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>

struct Tuple
{
  const vector<memReal> s;
  Rvec a, mu;
  const Real r;
  Tuple(const Tuple*const c): s(c->s), a(c->a), mu(c->mu), r(c->r) {}
  Tuple(const Rvec _s, const Real _r) : s(convert(_s)), r(_r) {}
  static inline vector<memReal> convert(const Rvec _s)
  {
    vector<memReal> ret ( _s.size() );
    for(Uint i=0; i < _s.size(); i++) ret[i] = toFinitePrec(_s[i]);
    return ret;
  }
 private:
  static inline memReal toFinitePrec(const Real val) {
    if(val > numeric_limits<float>::max()) return  numeric_limits<float>::max();
    else
    if(val <-numeric_limits<float>::max()) return -numeric_limits<float>::max();
    else
    return val;
  }
};

struct Sequence
{
  vector<Tuple*> tuples;
  int ended = 0, ID = -1, just_sampled = -1;
  Uint prefix = 0;
  Fval nOffPol = 0, MSE = 0, sumKLDiv = 0, totR = 0;
  Fvec action_adv;
  Fvec state_vals;
  Fvec Q_RET;
  //Used for sampling, filtering, and sorting off policy data:
  Fvec SquaredError;
  Fvec offPolicImpW;
  #ifdef PRIORITIZED_ER
    vector<float> priorityImpW;
  #endif
  Fvec KullbLeibDiv;
  mutable std::mutex seq_mutex;

  inline Uint ndata() const {
    assert(tuples.size());
    if(tuples.size()==0) return 0;
    return tuples.size()-1;
  }
  inline bool isLast(const Uint t) const {
    return t+1 >= tuples.size();
  }
  inline bool isTerminal(const Uint t) const {
    return t+1 == tuples.size() && ended;
  }
  inline bool isTruncated(const Uint t) const {
    return t+1 == tuples.size() && not ended;
  }
  ~Sequence() { clear(); }
  void clear()
  {
    for(auto &t : tuples) _dispose_object(t);
    tuples.clear();
    ended=0; ID=-1; just_sampled=-1; nOffPol=0; MSE=0; sumKLDiv=0; totR=0;
    //priorityImpW.clear();
    SquaredError.clear();
    offPolicImpW.clear();
    #ifdef PRIORITIZED_ER
      priorityImpW.clear();
    #endif
    KullbLeibDiv.clear();
    action_adv.clear();
    state_vals.clear();
    Q_RET.clear();
  }
  inline void setSampled(const int t) //update index of latest sampled time step
  {
    lock_guard<mutex> lock(seq_mutex);
    if(just_sampled < t) just_sampled = t;
  }
  inline void setRetrace(const Uint t, const Fval Q)
  {
    assert( t < Q_RET.size() );
    lock_guard<mutex> lock(seq_mutex);
    Q_RET[t] = Q;
  }
  inline void setAdvantage(const Uint t, const Fval A)
  {
    assert( t < action_adv.size() );
    lock_guard<mutex> lock(seq_mutex);
    action_adv[t] = A;
  }
  inline void setStateValue(const Uint t, const Fval V)
  {
    assert( t < state_vals.size() );
    lock_guard<mutex> lock(seq_mutex);
    state_vals[t] = V;
  }
  inline void setMseDklImpw(const Uint t,const Fval E,const Fval D,const Fval W)
  {
    lock_guard<mutex> lock(seq_mutex);
    SquaredError[t] = E;
    KullbLeibDiv[t] = D;
    offPolicImpW[t] = W;
  }

  inline bool isFarPolicyPPO(const Uint t, const Fval W, const Fval C) const
  {
    assert(C<1) ;
    const bool isOff = W > (Fval)1 + C || W < (Fval)1 - C;
    return isOff;
  }
  inline bool isFarPolicy(const Uint t, const Fval W, const Fval C) const
  {
    const bool isOff = W > C || W < (Fval)1 / C;
    // If C<=1 assume we never filter far policy samples
    return C > (Fval)1 && isOff;
  }
  inline bool distFarPolicy(const Uint t, const Fval D, const Fval target) const
  {
    // If target<=0 assume we never filter far policy samples
    return target>0 && D > target;
  }
  inline void add_state(const Rvec state, const Real reward=0)
  {
    Tuple * t = new Tuple(state, reward);
    if(tuples.size()) totR += reward;
    else assert(std::fabs(reward)<2.2e-16);
    tuples.push_back(t);
  }
  inline void add_action(const Rvec act, const Rvec mu)
  {
    assert( tuples.back()->s.size() && 0==tuples.back()->a.size() && 0==tuples.back()->mu.size() );
    tuples.back()->a = act;
    tuples.back()->mu = mu;
  }
  void finalize(const Uint index)
  {
    ID = index;
    // whatever the meaning of SquaredError, initialize with all zeros
    // this must be taken into account when sorting/filtering
    SquaredError = Fvec(ndata(), 0);
    // off pol importance weights are initialized to 1s
    offPolicImpW = Fvec(ndata(), 1);
    #ifdef PRIORITIZED_ER
      priorityImpW = vector<float>(ndata(), 1);
    #endif
    KullbLeibDiv = Fvec(ndata(), 0);
  }
};
