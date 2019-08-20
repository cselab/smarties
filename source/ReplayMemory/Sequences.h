//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Sequences_h
#define smarties_Sequences_h

#include "../Utils/Bund.h"
#include "../Utils/Warnings.h"
#include <cassert>
#include <mutex>

namespace smarties
{

struct Sequence
{
  Sequence()
  {
    states.reserve(MAX_SEQ_LEN);
    actions.reserve(MAX_SEQ_LEN);
    policies.reserve(MAX_SEQ_LEN);
    rewards.reserve(MAX_SEQ_LEN);
  }

  bool isEqual(const Sequence * const S) const;

  // Fval is just a storage format, probably float while Real is prob. double
  std::vector<Fvec> states;
  std::vector<Rvec> actions;
  std::vector<Rvec> policies;
  std::vector<Real> rewards;

  // additional quantities which may be needed by algorithms:
  NNvec Q_RET, action_adv, state_vals;
  //Used for sampling, filtering, and sorting off policy data:
  Fvec SquaredError, offPolicImpW, KullbLeibDiv;
  std::vector<float> priorityImpW;

  // some quantities needed for processing of experiences
  Fval nOffPol = 0, MSE = 0, sumKLDiv = 0, totR = 0;

  // did episode terminate (i.e. terminal state) or was a time out (i.e. V(s_end) != 0
  bool ended = false;
  // unique identifier of the episode, counter
  Sint ID = -1;
  // used for prost processing eps: idx of latest time step sampled during past gradient update
  Sint just_sampled = -1;
  // used for uniform sampling : prefix sum
  Uint prefix = 0;
  // local agent id (agent id within environment) that generated epiosode
  Uint agentID;

  std::mutex seq_mutex;

  Uint ndata() const // how much data to train from? ie. not terminal
  {
    assert(states.size());
    if(states.size()==0) return 0;
    else return states.size() - 1;
  }
  Uint nsteps() const // total number of time steps observed
  {
    return states.size();
  }
  bool isLast(const Uint t) const
  {
    return t+1 >= states.size();
  }
  bool isTerminal(const Uint t) const
  {
    return t+1 == states.size() && ended;
  }
  bool isTruncated(const Uint t) const
  {
    return t+1 == states.size() && not ended;
  }
  ~Sequence() { clear(); }
  void clear()
  {
    ended=0; ID=-1; just_sampled=-1; nOffPol=0; MSE=0; sumKLDiv=0; totR=0;
    states.clear();
    actions.clear();
    policies.clear();
    rewards.clear();
    //priorityImpW.clear();
    SquaredError.clear();
    offPolicImpW.clear();
    priorityImpW.clear();
    KullbLeibDiv.clear();
    action_adv.clear();
    state_vals.clear();
    Q_RET.clear();
  }
  void setSampled(const int t) //update ind of latest sampled time step
  {
    if(just_sampled < t) just_sampled = t;
  }
  void setRetrace(const Uint t, const Fval Q)
  {
    assert( t < Q_RET.size() );
    Q_RET[t] = Q;
  }
  void setAdvantage(const Uint t, const Fval A)
  {
    assert( t < action_adv.size() );
    action_adv[t] = A;
  }
  void setStateValue(const Uint t, const Fval V)
  {
    assert( t < state_vals.size() );
    state_vals[t] = V;
  }
  void setMseDklImpw(const Uint t, const Fval E, const Fval D,
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

  bool isFarPolicyPPO(const Uint t, const Fval W, const Fval C) const
  {
    assert(C<1) ;
    const bool isOff = W > (Fval)1 + C || W < (Fval)1 - C;
    return isOff;
  }
  bool isFarPolicy(const Uint t, const Fval W,
    const Fval C, const Fval invC) const {
    const bool isOff = W > C || W < invC;
    // If C<=1 assume we never filter far policy samples
    return C > (Fval)1 && isOff;
  }
  bool distFarPolicy(const Uint t,const Fval D,const Fval target) const
  {
    // If target<=0 assume we never filter far policy samples
    return target>0 && D > target;
  }

  void finalize(const Uint index)
  {
    ID = index;
    const Uint seq_len = states.size();
    // whatever the meaning of SquaredError, initialize with all zeros
    // this must be taken into account when sorting/filtering
    SquaredError = std::vector<Fval>(seq_len, 0);
    // off pol importance weights are initialized to 1s
    offPolicImpW = std::vector<Fval>(seq_len, 1);
    KullbLeibDiv = std::vector<Fval>(seq_len, 0);
  }

  int restart(FILE * f, const Uint dS, const Uint dA, const Uint dP);
  void save(FILE * f, const Uint dS, const Uint dA, const Uint dP);

  void unpackSequence(const std::vector<Fval>& data, const Uint dS,
    const Uint dA, const Uint dP);
  std::vector<Fval> packSequence(const Uint dS, const Uint dA, const Uint dP);

  static Uint computeTotalEpisodeSize(const Uint dS, const Uint dA,
    const Uint dP, const Uint Nstep)
  {
    const Uint tuplSize = dS+dA+dP+1;
    static constexpr Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    //extras : ended,ID,sampled,prefix,agentID x 2 for conversion safety
    static constexpr Uint extraSize = 14;
    const Uint ret = (tuplSize+infoSize)*Nstep + extraSize;
    return ret;
  }
  static Uint computeTotalEpisodeNstep(const Uint dS, const Uint dA,
    const Uint dP, const Uint size)
  {
    const Uint tuplSize = dS+dA+dP+1;
    static constexpr Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    static constexpr Uint extraSize = 14;
    const Uint nStep = (size - extraSize)/(tuplSize+infoSize);
    assert(Sequence::computeTotalEpisodeSize(dS,dA,dP,nStep) == size);
    return nStep;
  }
};

struct MiniBatch
{
  const Uint size;
  MiniBatch(const Uint _size) : size(_size)
  {
    episodes.resize(size);
    begTimeStep.resize(size);
    endTimeStep.resize(size);
    sampledTimeStep.resize(size);
    S.resize(size); A.resize(size); MU.resize(size); R.resize(size);
    W.resize(size);
  }

  std::vector<Sequence*> episodes;
  std::vector<Uint> begTimeStep;
  std::vector<Uint> endTimeStep;
  std::vector<Uint> sampledTimeStep;
  Uint getBegStep(const Uint b) const { return begTimeStep[b]; }
  Uint getEndStep(const Uint b) const { return endTimeStep[b]; }
  Uint getTstep(const Uint b) const { return sampledTimeStep[b]; }
  Uint getNumSteps(const Uint b) const {
    assert(begTimeStep.size() > b);
    assert(endTimeStep.size() > b);
    return endTimeStep[b] - begTimeStep[b];
  }
  Uint mapTime2Ind(const Uint b, const Uint t) const
  {
    assert(begTimeStep.size() >  b);
    assert(begTimeStep[b]     <= t);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return t - begTimeStep[b];
  }
  Uint mapInd2Time(const Uint b, const Uint k) const
  {
    assert(begTimeStep.size() > b);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return k + begTimeStep[b];
  }

  // episodes | time steps | dimensionality
  std::vector< std::vector< NNvec > > S;  // state
  std::vector< std::vector< Rvec* > > A;  // action pointer
  std::vector< std::vector< Rvec* > > MU; // behavior pointer
  std::vector< std::vector< Real  > > R;  // reward
  std::vector< std::vector< nnReal> > W;  // importance sampling

  Sequence& getEpisode(const Uint b) const
  {
    return * episodes[b];
  }
  NNvec& state(const Uint b, const Uint t)
  {
    return S[b][mapTime2Ind(b, t)];
  }
  Rvec& action(const Uint b, const Uint t)
  {
    return * A[b][mapTime2Ind(b, t)];
  }
  Rvec& mu(const Uint b, const Uint t)
  {
    return * MU[b][mapTime2Ind(b, t)];
  }
  const NNvec& state(const Uint b, const Uint t) const
  {
    return S[b][mapTime2Ind(b, t)];
  }
  const Rvec& action(const Uint b, const Uint t) const
  {
    return * A[b][mapTime2Ind(b, t)];
  }
  const Rvec& mu(const Uint b, const Uint t) const
  {
    return * MU[b][mapTime2Ind(b, t)];
  }
  void set_action(const Uint b, const Uint t, std::vector<Real>& act)
  {
    A[b][mapTime2Ind(b, t)] = & act;
  }
  void set_mu(const Uint b, const Uint t, std::vector<Real>& pol)
  {
    MU[b][mapTime2Ind(b, t)] = & pol;
  }
  Real& reward(const Uint b, const Uint t)
  {
    return R[b][mapTime2Ind(b, t)];
  }
  nnReal& importanceWeight(const Uint b, const Uint t)
  {
    return W[b][mapTime2Ind(b, t)];
  }
  const Real& reward(const Uint b, const Uint t) const
  {
    return R[b][mapTime2Ind(b, t)];
  }
  const nnReal& importanceWeight(const Uint b, const Uint t) const
  {
    return W[b][mapTime2Ind(b, t)];
  }

  void resizeStep(const Uint b, const Uint nSteps)
  {
    assert( S.size()>b); assert( A.size()>b);
    assert(MU.size()>b); assert( R.size()>b);
    S [b].resize(nSteps); A[b].resize(nSteps);
    MU[b].resize(nSteps); R[b].resize(nSteps); W[b].resize(nSteps);
  }
};

} // namespace smarties
#endif // smarties_Sequences_h
