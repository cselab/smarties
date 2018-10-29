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

struct Tuple // data obtained at time t
{
  // vector containing the vector at time t
  const std::vector<memReal> s;
  // actions and behavior follower at time t
  Rvec a, mu;
  // reward obtained with state s (at time 0 this is 0)
  const Real r;
  Tuple(const Tuple*const c): s(c->s), a(c->a), mu(c->mu), r(c->r) {}
  Tuple(const Rvec _s, const Real _r) : s(convert(_s)), r(_r) {}
  Tuple(const Fval*_s, const Uint dS, float _r) : s(convert(_s,dS)), r(_r) {}

  //convert to probably double (Rvec) to probably single precision for storage
  static inline vector<memReal> convert(const Rvec _s) {
    vector<memReal> ret ( _s.size() );
    for(Uint i=0; i < _s.size(); i++) ret[i] = _s[i];
    return ret;
  }
  static inline vector<memReal> convert(const Fval* _s, const Uint dS) {
    vector<memReal> ret ( dS );
    for(Uint i=0; i < dS; i++) ret[i] = _s[i];
    return ret;
  }
  inline void setAct(const Fval* _b, const Uint dA, const Uint dP) {
    a = Rvec( dA ); mu = Rvec( dP );
    for(Uint i=0; i < dA; i++)  a[i] = *_b++;
    for(Uint i=0; i < dP; i++) mu[i] = *_b++;
  }
};

struct Sequence
{
  std::vector<Tuple*> tuples;
  long ended = 0, ID = -1, just_sampled = -1;
  Uint prefix = 0;
  Fval nOffPol = 0, MSE = 0, sumKLDiv = 0, totR = 0;

  Fvec Q_RET, action_adv, state_vals;
  //Used for sampling, filtering, and sorting off policy data:
  Fvec SquaredError, offPolicImpW, KullbLeibDiv;
  std::vector<float> priorityImpW;

  std::mutex seq_mutex;

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
  void clear() {
    for(auto &t : tuples) _dispose_object(t);
    tuples.clear();
    ended=0; ID=-1; just_sampled=-1; nOffPol=0; MSE=0; sumKLDiv=0; totR=0;
    //priorityImpW.clear();
    SquaredError.clear();
    offPolicImpW.clear();
    priorityImpW.clear();
    KullbLeibDiv.clear();
    action_adv.clear();
    state_vals.clear();
    Q_RET.clear();
  }
  inline void setSampled(const int t) {//update ind of latest sampled time step
    if(just_sampled < t) just_sampled = t;
  }
  inline void setRetrace(const Uint t, const Fval Q) {
    assert( t < Q_RET.size() );
    Q_RET[t] = Q;
  }
  inline void setAdvantage(const Uint t, const Fval A) {
    assert( t < action_adv.size() );
    action_adv[t] = A;
  }
  inline void setStateValue(const Uint t, const Fval V) {
    assert( t < state_vals.size() );
    state_vals[t] = V;
  }
  inline void setMseDklImpw(const Uint t, const Fval E, const Fval D,
    const Fval W, const Fval C, const Fval invC)
  {
    const bool wasOff = offPolicImpW[t] > C || offPolicImpW[t] < invC;
    const bool isOff = W > C || W < invC;
    {
      std::lock_guard<std::mutex> lock(seq_mutex);
      sumKLDiv = sumKLDiv - KullbLeibDiv[t] + D;
      MSE = MSE - SquaredError[t] + E;
      nOffPol = nOffPol - wasOff + isOff;
    }
    SquaredError[t] = E;
    KullbLeibDiv[t] = D;
    offPolicImpW[t] = W;
  }

  inline bool isFarPolicyPPO(const Uint t, const Fval W, const Fval C) const {
    assert(C<1) ;
    const bool isOff = W > (Fval)1 + C || W < (Fval)1 - C;
    return isOff;
  }
  inline bool isFarPolicy(const Uint t, const Fval W,
    const Fval C, const Fval invC) const {
    const bool isOff = W > C || W < invC;
    // If C<=1 assume we never filter far policy samples
    return C > (Fval)1 && isOff;
  }
  inline bool distFarPolicy(const Uint t,const Fval D,const Fval target) const {
    // If target<=0 assume we never filter far policy samples
    return target>0 && D > target;
  }
  inline void add_state(const Rvec state, const Real reward=0) {
    Tuple * t = new Tuple(state, reward);
    if(tuples.size()) totR += reward;
    else assert(std::fabs(reward)<2.2e-16);
    tuples.push_back(t);
  }
  inline void add_action(const Rvec act, const Rvec mu) {
    assert( 0==tuples.back()->a.size() && 0==tuples.back()->mu.size() );
    tuples.back()->a = act;
    tuples.back()->mu = mu;
  }
  void finalize(const Uint index) {
    ID = index;
    const Uint seq_len = tuples.size();
    // whatever the meaning of SquaredError, initialize with all zeros
    // this must be taken into account when sorting/filtering
    SquaredError = Fvec(seq_len, 0);
    // off pol importance weights are initialized to 1s
    offPolicImpW.resize(seq_len, 1);

    KullbLeibDiv = Fvec(seq_len, 0);
  }

  int restart(FILE * f, const Uint dS, const Uint dA, const Uint dP);
  void save(FILE * f, const Uint dS, const Uint dA, const Uint dP);

  void unpackSequence(const vector<Fval>& data, const Uint dS,
    const Uint dA, const Uint dP);
  vector<Fval> packSequence(const Uint dS, const Uint dA, const Uint dP);

  static inline Uint computeTotalEpisodeSize(const Uint dS, const Uint dA,
    const Uint dP, const Uint Nstep)
  {
    const Uint tuplSize = dS+dA+dP+1;
    const Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    const Uint ret = (tuplSize+infoSize)*Nstep + 6;
    return ret;
  }
  static inline Uint computeTotalEpisodeNstep(const Uint dS, const Uint dA,
    const Uint dP, const Uint size)
  {
    const Uint tuplSize = dS+dA+dP+1;
    const Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    const Uint nStep = (size - 6)/(tuplSize+infoSize);
    assert(Sequence::computeTotalEpisodeSize(dS,dA,dP,nStep) == size);
    return nStep;
  }
};
