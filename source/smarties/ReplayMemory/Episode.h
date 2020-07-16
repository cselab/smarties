//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Episode_h
#define smarties_Episode_h

#include "../Utils/FunctionUtilities.h"
#include "../Core/StateAction.h"

#include <cassert>
#include <mutex>
#include <cmath>

namespace smarties
{

inline bool isFarPolicyPPO(const Fval W, const Fval C)
{
  assert(C<1) ;
  const bool isOff = W > (Fval)1 + C || W < (Fval)1 - C;
  return isOff;
}
inline bool isFarPolicy(const Fval W, const Fval C, const Fval invC)
{
  const bool isOff = W > C || W < invC;
  // If C<=1 assume we never filter far policy samples
  return C > (Fval)1 && isOff;
}
inline bool distFarPolicy(const Fval D, const Fval target)
{
  // If target<=0 assume we never filter far policy samples
  return target>0 && D > target;
}

struct Episode
{
  static constexpr Fval FVAL_EPS = std::numeric_limits<Fval>::epsilon();
  const MDPdescriptor & MDP;

  Episode(const MDPdescriptor & _MDP) : MDP(_MDP)
  {
    states.reserve(MAX_SEQ_LEN);
    actions.reserve(MAX_SEQ_LEN);
    policies.reserve(MAX_SEQ_LEN);
    rewards.reserve(MAX_SEQ_LEN);
    latent_states.reserve(MAX_SEQ_LEN);
  }
  Episode(const std::vector<Fval>&, const MDPdescriptor& MDP);
  ~Episode() = default;
  Episode(const Episode &p) = delete;
  Episode& operator=(const Episode &p) = delete;

  #define MOVE_SEQUENCE() do {                                                \
    bReachedTermState= p.bReachedTermState;         p.bReachedTermState=false;\
    ID               = p.ID;                        p.ID = -1;                \
    just_sampled     = p.just_sampled;              p.just_sampled = -1;      \
    prefix           = p.prefix;                    p.prefix = 0;             \
    agentID          = p.agentID;                   p.agentID = -1;           \
    totR             = p.totR;                      p.totR = 0;               \
    states           = std::move(p.states);         p.states.clear();         \
    latent_states    = std::move(p.latent_states);  p.latent_states.clear();  \
    actions          = std::move(p.actions);        p.actions.clear();        \
    policies         = std::move(p.policies);       p.policies.clear();       \
    rewards          = std::move(p.rewards);        p.rewards.clear();        \
    stateValue       = std::move(p.stateValue);     p.stateValue.clear();     \
    actionAdvantage  = std::move(p.actionAdvantage);p.actionAdvantage.clear();\
    returnEstimator  = std::move(p.returnEstimator);p.returnEstimator.clear();\
    deltaValue       = std::move(p.deltaValue);     p.deltaValue.clear();     \
    offPolicImpW     = std::move(p.offPolicImpW);   p.offPolicImpW.clear();   \
    KullbLeibDiv     = std::move(p.KullbLeibDiv);   p.KullbLeibDiv.clear();   \
    priorityImpW     = std::move(p.priorityImpW);   p.priorityImpW.clear();   \
    nFarOverPolSteps = p.nFarOverPolSteps;          p.nFarOverPolSteps = 0;   \
    nFarUndrPolSteps = p.nFarUndrPolSteps;          p.nFarUndrPolSteps = 0;   \
    sumKLDivergence  = p.sumKLDivergence;           p.sumKLDivergence = 0;    \
    sumSquaredErr    = p.sumSquaredErr;             p.sumSquaredErr = 0;      \
    sumAbsError      = p.sumAbsError;               p.sumAbsError = 0;        \
    sumSquaredQ      = p.sumSquaredQ;               p.sumSquaredQ = 0;        \
    sumQ             = p.sumQ;                      p.sumQ = 0;               \
    maxQ             = p.maxQ;                      p.maxQ = -1e9;            \
    minQ             = p.minQ;                      p.minQ =  1e9;            \
  } while (0)

  Episode(Episode && p) : MDP(p.MDP)
  {
    MOVE_SEQUENCE();
  }
  Episode& operator=(Episode && p)
  {
    MOVE_SEQUENCE();
    return * this;
  }

  #undef MOVE_SEQUENCE

  //////////////////////////////////////////////////////////////////////////////
  // did ep terminate (i.e. terminal state) or was a time out (i.e. V(s_end)!=0
  bool bReachedTermState = false;
  Sint ID = -1; //identifier of the episode, counter
  Sint just_sampled = -1; //largest time step sampled during latest grad update
  Uint prefix = 0; //used for uniform sampling: prefix sum of episodes lengths
  Sint agentID = -1; //agent id within environment which generated the epiosode
  Fval totR = 0; // sum of rewards obtained during the episode

  // Fval is just a storage format, probably float while Real is prob. double
  // latent_states : auxilliary state variables, not passed to the network
  std::vector<Fvec> states, latent_states;
  std::vector<Rvec> actions, policies;
  std::vector<Real> rewards;

  // additional quantities which may be needed by algorithms:
  NNvec stateValue, actionAdvantage, returnEstimator;
  //Used for sampling, filtering, and sorting off policy data:
  Fvec deltaValue, offPolicImpW, KullbLeibDiv;
  std::vector<float> priorityImpW;

  // some quantities needed for processing of experiences
  Uint nFarOverPolSteps = 0; // pi/mu > c
  Uint nFarUndrPolSteps = 0; // pi/mu < 1/c
  Fval sumKLDivergence = 0, sumSquaredErr = 0, sumAbsError = 0;
  Fval sumSquaredQ = 0, sumQ = 0;
  Fval maxQ = -1e9, minQ = 1e9;
  //////////////////////////////////////////////////////////////////////////////

  void clear()
  {
    bReachedTermState = false;
    ID = -1; just_sampled = -1; prefix =0; agentID =-1; totR =0;
    states.clear(); latent_states.clear(); actions.clear(); policies.clear();
    rewards.clear(); stateValue.clear(); actionAdvantage.clear();
    returnEstimator.clear(); deltaValue.clear(); offPolicImpW.clear();
    KullbLeibDiv.clear(); priorityImpW.clear();
    nFarOverPolSteps = 0; nFarUndrPolSteps = 0; sumKLDivergence  = 0;
    sumSquaredErr = 0; sumAbsError = 0;
    sumSquaredQ = 0; sumQ = 0; maxQ = -1e9; minQ = 1e9;
  }

  void clearNonTrackedAgent()
  {
    latent_states.clear(); actions.clear();         policies.clear();
    stateValue.clear();    actionAdvantage.clear(); returnEstimator.clear();
    deltaValue.clear();    offPolicImpW.clear();    KullbLeibDiv.clear();
    assert(agentID == -1 && "Untracked sequences are not tagged to agent");
  }

  void updateCumulative(const Fval C, const Fval invC);

  mutable std::mutex dataset_mutex; // used to update stats

  void updateCumulative_atomic(const Uint t, const Fval E, const Fval D,
                               const Fval W, const Fval C, const Fval invC)
  {
    assert(nsteps() == deltaValue.size());
    assert(nsteps() == KullbLeibDiv.size());
    assert(nsteps() == offPolicImpW.size());
    const Uint wasFarOver = offPolicImpW[t] >    C, isFarOver = W >    C;
    const Uint wasFarUndr = offPolicImpW[t] < invC, isFarUndr = W < invC;
    {
      std::lock_guard<std::mutex> lock(dataset_mutex);
      nFarOverPolSteps += isFarOver - wasFarOver;
      nFarUndrPolSteps += isFarUndr - wasFarUndr;
      sumKLDivergence += D - KullbLeibDiv[t];
      sumSquaredErr += E*E - deltaValue[t]*deltaValue[t];
      sumAbsError += std::fabs(E) - std::fabs(deltaValue[t]);
    }
    deltaValue[t] = E; KullbLeibDiv[t] = D; offPolicImpW[t] = W;
  }

  void updateValues_atomic(const Uint t, const Fval V, const Fval Q)
  {
    assert(nsteps() == stateValue.size());
    assert(nsteps() == actionAdvantage.size());
    assert(nsteps() == returnEstimator.size());
    const Fval oldQ = actionAdvantage[t] + stateValue[t];
    {
      std::lock_guard<std::mutex> lock(dataset_mutex);
      sumSquaredQ += Q*Q - oldQ*oldQ; sumQ += Q - oldQ;
      maxQ = std::max(maxQ, Q); minQ = std::min(minQ, Q);
    }
    stateValue[t] = V; actionAdvantage[t] = Q-V;
  }

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
  Uint nFarPolicySteps() const
  {
    return nFarOverPolSteps + nFarUndrPolSteps;
  }

  bool isTerminal(const Uint t) const
  {
    return t+1 == states.size() && bReachedTermState;
  }
  bool isTruncated(const Uint t) const
  {
    return t+1 == states.size() && not bReachedTermState;
  }
  void setSampled(const int t) //update ind of latest sampled time step
  {
    if(just_sampled < t) just_sampled = t;
  }

  template<typename V = nnReal, typename T>
  std::vector<V> standardizedState(const T samp) const
  {
    const Uint dimS = MDP.dimStateObserved, nAppended = MDP.nAppendedObs;
    std::vector<V> ret( dimS * (1+nAppended) );
    for (Uint j=0, k=0; j <= nAppended; ++j) {
      const Uint t = std::max((Uint) samp - j, (Uint) 0);
      assert(states[t].size() == dimS);
      for (Uint i=0; i<dimS; ++i, ++k)
        ret[k] = (states[t][i] - MDP.stateMean[i]) * MDP.stateScale[i];
    }
    return ret;
  }
  template<typename V = Real, typename T>
  V scaledReward(const T samp) const
  {
    assert((Uint) samp < rewards.size());
    return (rewards[samp] - MDP.rewardsMean) * MDP.rewardsScale;
  }
  template<typename V = Real, typename T>
  V clippedOffPolW(const T samp) const
  {
    return offPolicImpW[samp]<1 ? offPolicImpW[samp] : 1;
  }
  template<typename V = Real, typename T>
  V SquaredError(const T samp) const
  {
    return deltaValue[samp] * deltaValue[samp];
  }

  void finalize(const Uint index);

  int restart(FILE * f);
  void save(FILE * f);

  void unpackEpisode(const std::vector<Fval>& data);
  std::vector<Fval> packEpisode();
  bool isEqual(const Episode & S) const;

  static Uint computeTotalEpisodeSize(const MDPdescriptor& MDP, const Uint Nstep)
  {
    const Uint tuplSize = MDP.dimState + MDP.dimAction + MDP.policyVecDim + 1;
    static constexpr Uint infoSize = 6; //adv,val,ret, mse,dkl,impW
    //extras: bReachedTermState,ID,sampled,prefix,agentID x2 for safety
    static constexpr Uint extraSize = 10;
    const Uint ret = (tuplSize+infoSize)*Nstep + extraSize;
    return ret;
  }
  static Uint computeTotalEpisodeNstep(const MDPdescriptor& MDP, const Uint size)
  {
    const Uint tuplSize = MDP.dimState + MDP.dimAction + MDP.policyVecDim + 1;
    static constexpr Uint infoSize = 6; //adv,val,ret, mse,dkl,impW
    static constexpr Uint extraSize = 10;
    const Uint nStep = (size - extraSize)/(tuplSize+infoSize);
    assert(Episode::computeTotalEpisodeSize(MDP, nStep) == size);
    return nStep;
  }

  std::vector<float> logToFile(const Uint iterStep) const;
};

} // namespace smarties
#endif // smarties_Episode_h
