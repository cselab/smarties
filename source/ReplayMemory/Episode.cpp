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

Episode::Episode(const std::vector<Fval>& data, const MDPdescriptor& MDP)
{
  unpackEpisode(data, MDP);
}

std::vector<Fval> Episode::packEpisode(const MDPdescriptor& MDP)
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

  assert(Q_RET.size() <= N);        Q_RET.resize(N);
  std::copy(Q_RET.begin(), Q_RET.end(), buf); buf += N;

  assert(action_adv.size() <= N);   action_adv.resize(N);
  std::copy(action_adv.begin(), action_adv.end(), buf); buf += N;

  assert(state_vals.size() <= N);   state_vals.resize(N);
  std::copy(state_vals.begin(), state_vals.end(), buf); buf += N;

  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // post processing quantities might not be already allocated

  assert(SquaredError.size() <= N); SquaredError.resize(N);
  std::copy(SquaredError.begin(), SquaredError.end(), buf); buf += N;

  assert(offPolicImpW.size() <= N); offPolicImpW.resize(N);
  std::copy(offPolicImpW.begin(), offPolicImpW.end(), buf); buf += N;

  assert(KullbLeibDiv.size() <= N); KullbLeibDiv.resize(N);
  std::copy(KullbLeibDiv.begin(), KullbLeibDiv.end(), buf); buf += N;

  /////////////////////////////////////////////////////////////////////////////

  assert((Uint) (buf-ret.data()) == (dS + dA + dP + dI + 7) * N);

  char * charPos = (char*) buf;
  memcpy(charPos, &       ended, sizeof(bool)); charPos += sizeof(bool);
  memcpy(charPos, &          ID, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &just_sampled, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &      prefix, sizeof(Uint)); charPos += sizeof(Uint);
  memcpy(charPos, &     agentID, sizeof(Sint)); charPos += sizeof(Sint);

  // assert(buf - ret.data() == (ptrdiff_t) totalSize);
  return ret;
}

void Episode::save(FILE * f, const MDPdescriptor& MDP) {
  const Uint seq_len = states.size();
  fwrite(& seq_len, sizeof(Uint), 1, f);
  Fvec buffer = packEpisode(MDP);
  fwrite(buffer.data(), sizeof(Fval), buffer.size(), f);
}

void Episode::unpackEpisode(const std::vector<Fval>& data, const MDPdescriptor& MDP)
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
  Q_RET      = std::vector<nnReal>(buf, buf + N); buf += N;
  action_adv = std::vector<nnReal>(buf, buf + N); buf += N;
  state_vals = std::vector<nnReal>(buf, buf + N); buf += N;
  /////////////////////////////////////////////////////////////////////////////
  SquaredError = std::vector<Fval>(buf, buf + N); buf += N;
  offPolicImpW = std::vector<Fval>(buf, buf + N); buf += N;
  KullbLeibDiv = std::vector<Fval>(buf, buf + N); buf += N;
  /////////////////////////////////////////////////////////////////////////////
  assert((Uint) (buf - data.data()) == (dS + dA + dP + dI + 7) * N);
  priorityImpW = std::vector<float>(N, 1);
  totR = Utilities::sum(rewards);
  /////////////////////////////////////////////////////////////////////////////

  const char * charPos = (const char *) buf;
  memcpy(&       ended, charPos, sizeof(bool)); charPos += sizeof(bool);
  memcpy(&          ID, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&just_sampled, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&      prefix, charPos, sizeof(Uint)); charPos += sizeof(Uint);
  memcpy(&     agentID, charPos, sizeof(Sint)); charPos += sizeof(Sint);
}

int Episode::restart(FILE * f, const MDPdescriptor& MDP)
{
  Uint N = 0;
  if(fread(& N, sizeof(Uint), 1, f) != 1) return 1;
  const Uint totalSize = Episode::computeTotalEpisodeSize(MDP, N);
  std::vector<Fval> buffer(totalSize);
  if(fread(buffer.data(), sizeof(Fval), totalSize, f) != totalSize)
    die("mismatch");
  unpackEpisode(buffer, MDP);
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

  if(isDifferent(S.Q_RET       , Q_RET       )) assert(false && "Q_RET");
  if(isDifferent(S.action_adv  , action_adv  )) assert(false && "action_adv");
  if(isDifferent(S.state_vals  , state_vals  )) assert(false && "state_vals");

  if(isDifferent(S.SquaredError, SquaredError)) assert(false && "SquaredError");
  if(isDifferent(S.offPolicImpW, offPolicImpW)) assert(false && "offPolicImpW");
  if(isDifferent(S.KullbLeibDiv, KullbLeibDiv)) assert(false && "KullbLeibDiv");

  if(S.ended        not_eq ended       ) assert(false && "ended");
  if(S.ID           not_eq ID          ) assert(false && "ID");
  if(S.just_sampled not_eq just_sampled) assert(false && "just_sampled");
  if(S.prefix       not_eq prefix      ) assert(false && "prefix");
  if(S.agentID      not_eq agentID     ) assert(false && "agentID");
  return true;
}

std::vector<float> Episode::logToFile(
    const StateInfo& sInfo, const ActionInfo& aInfo, const Uint iterStep) const
{
  const Uint dS = sInfo.dimObs(), dI = sInfo.dimInfo();
  const Uint dA = aInfo.dim(), dP = aInfo.dimPol(), N = states.size();

  std::vector<float> buffer(N * (4 + dS + dI + dA + dP));
  float * pos = buffer.data();
  for (Uint t=0; t<N; ++t)
  {
    assert(states[t].size() == dS and dI == latent_states[t].size());
    assert(actions[t].size() == dA and dP == policies[t].size());
    *(pos++) = iterStep + 0.1;
    const auto steptype = t==0 ? INIT : ( isTerminal(t) ? TERM : (
                          isTruncated(t) ? TRNC : CONT ) );
    *(pos++) = status2int(steptype) + 0.1;
    *(pos++) = t + 0.1;
    const auto S = sInfo.observedAndLatent2state(states[t], latent_states[t]);
    std::copy(S.begin(), S.end(), pos);           pos += dS + dI;
    const auto envAct = aInfo.learnerAction2envAction<float>(actions[t]);
    std::copy(envAct.begin(), envAct.end(), pos); pos += dA;
    *(pos++) = rewards[t];
    const auto envPol = aInfo.learnerPolicy2envPolicy<float>(policies[t]);
    std::copy(envPol.begin(), envPol.end(), pos); pos += dP;
    assert(envAct.size() == dA and envPol.size() == dP and S.size() == dS + dI);
  }
  return buffer;
}

}
