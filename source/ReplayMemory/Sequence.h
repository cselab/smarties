//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Sequence_h
#define smarties_Sequence_h

#include "../Utils/FunctionUtilities.h"

#include <cassert>
#include <atomic>
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

struct Sequence
{
  static constexpr Fval FVAL_EPS = std::numeric_limits<Fval>::epsilon();
  Sequence()
  {
    states.reserve(MAX_SEQ_LEN);
    actions.reserve(MAX_SEQ_LEN);
    policies.reserve(MAX_SEQ_LEN);
    rewards.reserve(MAX_SEQ_LEN);
  }
  Sequence(const std::vector<Fval>&, const Uint dS,const Uint dA,const Uint dP);
  ~Sequence() = default;
  Sequence(const Sequence &p) = delete;
  Sequence& operator=(const Sequence &p) = delete;

  #define MOVE_SEQUENCE() do {                                                \
    ended            = p.ended;                     p.ended = false;          \
    ID               = p.ID;                        p.ID = -1;                \
    just_sampled     = p.just_sampled;              p.just_sampled = -1;      \
    prefix           = p.prefix;                    p.prefix = 0;             \
    agentID          = p.agentID;                   p.agentID = 0;            \
    totR             = p.totR;                      p.totR = 0;               \
    nFarOverPolSteps = p.nFarOverPolSteps.load();   p.nFarOverPolSteps = 0;   \
    nFarUndrPolSteps = p.nFarUndrPolSteps.load();   p.nFarUndrPolSteps = 0;   \
    sumKLDivergence  = p.sumKLDivergence.load();    p.sumKLDivergence = 0;    \
    sumSquaredErr    = p.sumSquaredErr.load();      p.sumSquaredErr = 0;      \
    minImpW          = p.minImpW.load();            p.minImpW = 1;            \
    avgImpW          = p.avgImpW.load();            p.avgImpW = 1;            \
    states           = std::move(p.states);         p.states.clear();         \
    actions          = std::move(p.actions);        p.actions.clear();        \
    policies         = std::move(p.policies);       p.policies.clear();       \
    rewards          = std::move(p.rewards);        p.rewards.clear();        \
    action_adv       = std::move(p.action_adv);     p.action_adv.clear();     \
    state_vals       = std::move(p.state_vals);     p.state_vals.clear();     \
    Q_RET            = std::move(p.Q_RET);          p.Q_RET.clear();          \
    SquaredError     = std::move(p.SquaredError);   p.SquaredError.clear();   \
    offPolicImpW     = std::move(p.offPolicImpW);   p.offPolicImpW.clear();   \
    KullbLeibDiv     = std::move(p.KullbLeibDiv);   p.KullbLeibDiv.clear();   \
    priorityImpW     = std::move(p.priorityImpW);   p.priorityImpW.clear();   \
  } while (0)

  Sequence(Sequence && p)
  {
    MOVE_SEQUENCE();
  }
  Sequence& operator=(Sequence && p)
  {
    MOVE_SEQUENCE();
    return * this;
  }

  #undef MOVE_SEQUENCE

  void clear()
  {
    ended = false; ID = -1; just_sampled = -1; prefix =0; agentID =0; totR =0;
    nFarOverPolSteps = 0;
    nFarUndrPolSteps = 0;
    sumKLDivergence  = 0;
    sumSquaredErr    = 0;
    minImpW          = 1;
    avgImpW          = 1;

    states.clear(); actions.clear(); policies.clear(); rewards.clear();
    SquaredError.clear(); offPolicImpW.clear(); KullbLeibDiv.clear();
    action_adv.clear(); state_vals.clear(); Q_RET.clear(); priorityImpW.clear();
  }

  //////////////////////////////////////////////////////////////////////////////
  // did ep terminate (i.e. terminal state) or was a time out (i.e. V(s_end)!=0
  bool ended = false;
  // unique identifier of the episode, counter:
  Sint ID = -1;
  // used for prost processing eps: idx of latest time step sampled during past grad update
  Sint just_sampled = -1;
  // used for uniform sampling, prefix sum:
  Uint prefix = 0;
  // local agent id (agent id within environment) that generated epiosode:
  Uint agentID = 0;
  // sum of rewards obtained during the episode:
  Fval totR = 0;

  // Fval is just a storage format, probably float while Real is prob. double
  std::vector<Fvec> states;
  std::vector<Rvec> actions;
  std::vector<Rvec> policies;
  std::vector<Real> rewards;

  // additional quantities which may be needed by algorithms:
  NNvec action_adv, state_vals, Q_RET;
  //Used for sampling, filtering, and sorting off policy data:
  Fvec SquaredError, offPolicImpW, KullbLeibDiv;
  std::vector<float> priorityImpW;

  // some quantities needed for processing of experiences
  std::atomic<Uint> nFarOverPolSteps{0}; // pi/mu > c
  std::atomic<Uint> nFarUndrPolSteps{0}; // pi/mu < 1/c
  std::atomic<Real> sumKLDivergence{0};
  std::atomic<Real> sumSquaredErr{0};
  std::atomic<Real> minImpW{1};
  std::atomic<Real> avgImpW{1};
  //////////////////////////////////////////////////////////////////////////////

  void updateCumulative(const Fval C, const Fval invC)
  {
    const Uint N = ndata();
    Uint nOverFarPol = 0, nUndrFarPol = 0;
    Real minRho = 9e9, avgRho = 1;
    const Real invN = 1.0 / N;
    for (Uint t = 0; t < N; ++t) {
      // float precision may cause DKL to be slightly negative:
      assert(KullbLeibDiv[t] >= - FVAL_EPS && offPolicImpW[t] >= 0);
      // sequence is off policy if offPol W is out of 1/C : C
      if (offPolicImpW[t] >    C) nOverFarPol++;
      if (offPolicImpW[t] < invC) nUndrFarPol++;
      if (offPolicImpW[t] < minRho) minRho = offPolicImpW[t];
      const Real clipRho = std::min((Fval) 1, offPolicImpW[t]);
      avgRho *= std::pow(clipRho, invN);
    }
    nFarOverPolSteps = nOverFarPol;
    nFarUndrPolSteps = nUndrFarPol;
    minImpW = minRho;
    avgImpW = avgRho;

    totR = Utilities::sum(rewards);
    sumSquaredErr = Utilities::sum(SquaredError);
    sumKLDivergence = Utilities::sum(KullbLeibDiv);
  }

  void updateCumulative_atomic(const Uint t, const Fval E, const Fval D,
                               const Fval W, const Fval C, const Fval invC)
  {
    const Fval oldW = offPolicImpW[t];
    const Uint wasFarOver = oldW > C, wasFarUndr = oldW < invC;
    const Uint  isFarOver =    W > C,  isFarUndr =    W < invC;
    const Real clipOldW = std::min((Fval) 1, oldW);
    const Real clipNewW = std::min((Fval) 1,    W);
    const Real invN = 1.0 / ndata();

    sumKLDivergence.store(sumKLDivergence.load() - KullbLeibDiv[t] + D);
    sumSquaredErr.store(sumSquaredErr.load() - SquaredError[t] + E);
    nFarOverPolSteps += isFarOver - wasFarOver;
    nFarUndrPolSteps += isFarUndr - wasFarUndr;
    avgImpW.store(avgImpW.load() * std::pow(clipNewW/clipOldW, invN));
    if(W < minImpW.load()) minImpW = W;

    SquaredError[t] = E;
    KullbLeibDiv[t] = D;
    offPolicImpW[t] = W;
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
    return t+1 == states.size() && ended;
  }
  bool isTruncated(const Uint t) const
  {
    return t+1 == states.size() && not ended;
  }
  bool isEqual(const Sequence & S) const;

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
    #ifndef NDEBUG
      Fval dbg_sumR = std::accumulate(rewards.begin(), rewards.end(), (Fval)0);
      //Fval dbg_norm = std::max(std::fabs(totR), std::fabs(dbg_sumR));
      Fval dbg_norm = std::max((Fval)1, std::fabs(totR));
      assert(std::fabs(totR-dbg_sumR)/dbg_norm < 100*FVAL_EPS);
    #endif
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
    static constexpr Uint infoSize = 6; //adv,val,ret, mse,dkl,impW
    //extras : ended,ID,sampled,prefix,agentID x 2 for conversion safety
    static constexpr Uint extraSize = 10;
    const Uint ret = (tuplSize+infoSize)*Nstep + extraSize;
    return ret;
  }
  static Uint computeTotalEpisodeNstep(const Uint dS, const Uint dA,
    const Uint dP, const Uint size)
  {
    const Uint tuplSize = dS+dA+dP+1;
    static constexpr Uint infoSize = 6; //adv,val,ret, mse,dkl,impW
    static constexpr Uint extraSize = 10;
    const Uint nStep = (size - extraSize)/(tuplSize+infoSize);
    assert(Sequence::computeTotalEpisodeSize(dS,dA,dP,nStep) == size);
    return nStep;
  }

  void propagateRetrace(const Uint t, const Fval gamma, const Fval R)
  {
    if(t == 0) return;
    const Fval V = state_vals[t], A = action_adv[t];
    const Fval clipW = offPolicImpW[t]<1 ? offPolicImpW[t] : 1;
    Q_RET[t-1] = R + gamma * V + gamma * clipW * (Q_RET[t] - A - V);
  }

  std::vector<float> logToFile(const Uint dimS, const Uint iterStep) const;
};

} // namespace smarties
#endif // smarties_Sequence_h
