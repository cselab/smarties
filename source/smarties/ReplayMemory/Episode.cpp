//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Episode.h"
#include "../Core/Agent.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace smarties
{

Episode::Episode(const std::vector<Fval>& data, const MDPdescriptor& _MDP)
: MDP(_MDP)
{
  unpackEpisode(data);
}

std::vector<Fval> Episode::packEpisode()
{
  const Uint dS = MDP.dimStateObserved, dI = MDP.dimState-MDP.dimStateObserved;
  const Uint dA = MDP.dimAction, dP = MDP.policyVecDim, N = states.size();
  assert(states.size() == actions.size() && states.size() == policies.size());
  const Uint totalSize = Episode::computeTotalEpisodeSize(MDP, N);
  assert( N == Episode::computeTotalEpisodeNstep(MDP, totalSize) );
  std::vector<Fval> ret(totalSize, 0);
  Fval* buf = ret.data();

  for (Uint i = 0; i<N; ++i)
  {
    assert(states[i].size() == dS and latent_states[i].size() == dI);
    assert(actions[i].size() == dA and policies[i].size() == dP);
    std::copy(states[i].begin(), states[i].end(), buf);
    buf[dS] = rewards[i]; buf += dS + 1;
    std::copy(actions[i].begin(),  actions[i].end(),  buf); buf += dA;
    std::copy(policies[i].begin(), policies[i].end(), buf); buf += dP;
    std::copy(latent_states[i].begin(), latent_states[i].end(), buf); buf += dI;
  }

  /////////////////////////////////////////////////////////////////////////////
  // following vectors may be of size less than N because
  // some algorithms do not allocate them. I.e. Q-learning-based
  // algorithms do not need to advance retrace-like value estimates

  assert(returnEstimator.size() <= N);           returnEstimator.resize(N);
  std::copy(returnEstimator.begin(), returnEstimator.end(), buf); buf += N;

  assert(actionAdvantage.size() <= N);           actionAdvantage.resize(N);
  std::copy(actionAdvantage.begin(), actionAdvantage.end(), buf); buf += N;

  assert(stateValue.size() <= N);           stateValue.resize(N);
  std::copy(stateValue.begin(), stateValue.end(), buf); buf += N;

  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // post processing quantities might not be already allocated

  assert(deltaValue.size() <= N); deltaValue.resize(N);
  std::copy(deltaValue.begin(), deltaValue.end(), buf); buf += N;

  assert(offPolicImpW.size() <= N); offPolicImpW.resize(N);
  std::copy(offPolicImpW.begin(), offPolicImpW.end(), buf); buf += N;

  assert(KullbLeibDiv.size() <= N); KullbLeibDiv.resize(N);
  std::copy(KullbLeibDiv.begin(), KullbLeibDiv.end(), buf); buf += N;

  /////////////////////////////////////////////////////////////////////////////

  assert((Uint) (buf-ret.data()) == (dS + dA + dP + dI + 7) * N);

  char * charPos = (char*) buf;
  memcpy(charPos, &bReachedTermState, sizeof(bool)); charPos += sizeof(bool);
  memcpy(charPos, &          ID, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &just_sampled, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &     agentID, sizeof(Sint)); charPos += sizeof(Sint);

  // assert(buf - ret.data() == (ptrdiff_t) totalSize);
  return ret;
}

void Episode::save(FILE * f)
{
  const Uint seq_len = states.size();
  fwrite(& seq_len, sizeof(Uint), 1, f);
  Fvec buffer = packEpisode();
  fwrite(buffer.data(), sizeof(Fval), buffer.size(), f);
}

void Episode::unpackEpisode(const std::vector<Fval>& data)
{
  const Uint dS = MDP.dimStateObserved, dI = MDP.dimState-MDP.dimStateObserved;
  const Uint dA = MDP.dimAction, dP = MDP.policyVecDim;
  const Uint N = Episode::computeTotalEpisodeNstep(MDP, data.size());
  assert( data.size() == Episode::computeTotalEpisodeSize(MDP, N) );
  const Fval* buf = data.data();
  assert(states.size() == 0);
  for (Uint i = 0; i<N; ++i) {
    states.push_back(  Fvec(buf, buf + dS));
    rewards.push_back(buf[dS]); buf += dS + 1;
    actions.push_back( Rvec(buf, buf + dA)); buf += dA;
    policies.push_back(Rvec(buf, buf + dP)); buf += dP;
    latent_states.push_back(Fvec(buf, buf + dI)); buf += dI;
  }

  /////////////////////////////////////////////////////////////////////////////
  returnEstimator = std::vector<nnReal>(buf, buf + N); buf += N;
  actionAdvantage = std::vector<nnReal>(buf, buf + N); buf += N;
  stateValue      = std::vector<nnReal>(buf, buf + N); buf += N;
  /////////////////////////////////////////////////////////////////////////////
  deltaValue   = std::vector<Fval>(buf, buf + N); buf += N;
  offPolicImpW = std::vector<Fval>(buf, buf + N); buf += N;
  KullbLeibDiv = std::vector<Fval>(buf, buf + N); buf += N;
  /////////////////////////////////////////////////////////////////////////////
  assert((Uint) (buf - data.data()) == (dS + dA + dP + dI + 7) * N);
  priorityImpW = std::vector<float>(N, 1);
  totR = Utilities::sum(rewards);
  /////////////////////////////////////////////////////////////////////////////

  const char * charPos = (const char *) buf;
  memcpy(&bReachedTermState, charPos, sizeof(bool)); charPos += sizeof(bool);
  memcpy(&          ID, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&just_sampled, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&     agentID, charPos, sizeof(Sint)); charPos += sizeof(Sint);
}

int Episode::restart(FILE * f)
{
  Uint N = 0;
  if(fread(& N, sizeof(Uint), 1, f) != 1) return 1;
  const Uint totalSize = Episode::computeTotalEpisodeSize(MDP, N);
  std::vector<Fval> buffer(totalSize);
  if(fread(buffer.data(), sizeof(Fval), totalSize, f) != totalSize)
    die("mismatch");
  unpackEpisode(buffer);
  return 0;
}

template<typename T>
inline bool isDifferent(const std::atomic<T>& a, const std::atomic<T>& b) {
  static constexpr T EPS = std::numeric_limits<float>::epsilon(), tol = 100*EPS;
  const auto norm = std::max({std::fabs(a.load()), std::fabs(b.load()), EPS});
  return std::fabs(a-b)/norm > tol;
}
template<typename T>
inline bool isDifferent(const T& a, const T& b) {
  static constexpr T EPS = std::numeric_limits<float>::epsilon(), tol = 100*EPS;
  const auto norm = std::max({std::fabs(a), std::fabs(b), EPS});
  return std::fabs(a-b)/norm > tol;
}
template<typename T>
inline bool isDifferent(const std::vector<T>& a, const std::vector<T>& b) {
  if(a.size() not_eq b.size()) return true;
  for(size_t i=0; i<b.size(); ++i) if( isDifferent(a[i], b[i]) ) return true;
  return false;
}

bool Episode::isEqual(const Episode & S) const
{
  if(isDifferent(S.states      , states      )) assert(false && "states");
  if(isDifferent(S.actions     , actions     )) assert(false && "actions");
  if(isDifferent(S.policies    , policies    )) assert(false && "policies");
  if(isDifferent(S.rewards     , rewards     )) assert(false && "rewards");

  if(isDifferent(S.returnEstimator, returnEstimator)) assert(false && "ret");
  if(isDifferent(S.actionAdvantage, actionAdvantage)) assert(false && "adv");
  if(isDifferent(S.stateValue     , stateValue     )) assert(false && "val");

  if(isDifferent(S.deltaValue, deltaValue)) assert(false && "deltaValue");
  if(isDifferent(S.offPolicImpW, offPolicImpW)) assert(false && "offPolicImpW");
  if(isDifferent(S.KullbLeibDiv, KullbLeibDiv)) assert(false && "KullbLeibDiv");

  if(S.bReachedTermState not_eq bReachedTermState) assert(false && "ended");
  if(S.ID           not_eq ID          ) assert(false && "ID");
  if(S.just_sampled not_eq just_sampled) assert(false && "just_sampled");
  if(S.agentID      not_eq agentID     ) assert(false && "agentID");
  return true;
}

std::vector<float> Episode::logToFile(const Uint iterStep) const
{
  const Uint dS = MDP.dimStateObserved, dI = MDP.dimState - dS;
  const Uint dA = MDP.dimAction, dP = MDP.policyVecDim, N = states.size();

  std::vector<float> buffer(N * (4 + dS + dI + dA + dP));
  float * pos = buffer.data();
  for (Uint t=0; t<N; ++t)
  {
    assert(states[t].size() == dS and dI == latent_states[t].size());
    assert(actions[t].size() == dA and dP == policies[t].size());
    *(pos++) = iterStep + 0.1;
    const auto steptype = t==0 ? INIT : ( isTerminal(t) ? TERM : (
                          isTruncated(t) ? LAST : CONT ) );
    *(pos++) = status2int(steptype) + 0.1;
    *(pos++) = t + 0.1;
    const auto S = StateInfo::observedAndLatent2state(states[t], latent_states[t], MDP);
    std::copy(S.begin(), S.end(), pos); pos += dS + dI;
    const auto A = ActionInfo::learnerAction2envAction<float>(actions[t], MDP);
    std::copy(A.begin(), A.end(), pos); pos += dA;
    *(pos++) = rewards[t];
    const auto P = ActionInfo::learnerPolicy2envPolicy<float>(policies[t], MDP);
    std::copy(P.begin(), P.end(), pos); pos += dP;
    assert(A.size() == dA and P.size() == dP and S.size() == dS + dI);
  }
  return buffer;
}

void Episode::updateCumulative(const Fval C, const Fval invC)
{
  const Uint N = ndata();
  const Fval invN = 1 / (Fval) N;
  Uint nFarPol = 0;
  Fval _sumE2=0, _maxAE = -1e9, _maxQ = -1e9, _sumQ2=0, _minQ = 1e9, _sumQ1=0;
  for (Uint t = 0; t < N; ++t) {
    // float precision may cause DKL to be slightly negative:
    assert(KullbLeibDiv[t] >= - FVAL_EPS && offPolicImpW[t] >= 0);
    // sequence is off policy if offPol W is out of 1/C : C
    if (offPolicImpW[t] > C or offPolicImpW[t] < invC) ++nFarPol;
    _sumE2 += deltaValue[t] * deltaValue[t];
    _maxAE  = std::max(_maxAE, std::fabs(deltaValue[t]));
    const Fval Q = actionAdvantage[t] + stateValue[t];
    _maxQ   = std::max(_maxQ,  Q);
    _minQ   = std::min(_minQ,  Q);
    _sumQ2 += Q*Q;
    _sumQ1 += Q;
  }
  fracFarPolSteps = invN * nFarPol;
  avgSquaredErr   = invN * _sumE2;
  maxAbsError     = _maxAE;
  sumSquaredQ     = _sumQ2;
  sumQ            = _sumQ1;
  maxQ            = _maxQ;
  minQ            = _minQ;
  assert(std::fabs(rewards[0])<1e-16);
  totR = Utilities::sum(rewards);
  avgKLDivergence = invN * Utilities::sum(KullbLeibDiv);
}

void Episode::finalize(const Uint index)
{
  ID = index;
  const Uint N = nsteps();
  stateValue.resize(N, 0);
  actionAdvantage.resize(N, 0);
  // whatever the meaning of deltaValue, initialize with all zeros
  // this must be taken into account when sorting/filtering
  deltaValue.resize(N, 0);
  KullbLeibDiv.resize(N, 0);
  // off pol and priority importance weights are initialized to 1
  offPolicImpW.resize(N, 1);
  offPolicImpW.back() = 0;
  priorityImpW.resize(N, 1);
  returnEstimator.resize(N, 0);

  #ifndef NDEBUG
    Fval dbg_sumR = std::accumulate(rewards.begin(), rewards.end(), (Fval)0);
    //Fval dbg_norm = std::max(std::fabs(totR), std::fabs(dbg_sumR));
    Fval dbg_norm = std::max((Fval)1, std::fabs(totR));
    assert(std::fabs(totR-dbg_sumR)/dbg_norm < 100*FVAL_EPS);
  #endif
}

void Episode::initPreTrainErrorPlaceholder(const Fval maxError)
{
  deltaValue = Fvec(nsteps(), maxError);
  avgSquaredErr = maxError * maxError;
  maxAbsError = maxError;
}

}
