//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Sequence.h"
#include "../Core/Agent.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace smarties
{

Sequence::Sequence(const std::vector<Fval>& data,
                   const Uint dS, const Uint dA, const Uint dP)
{
  unpackSequence(data, dS, dA, dP);
}

std::vector<Fval> Sequence::packSequence(const Uint dS, const Uint dA, const Uint dP)
{
  const Uint seq_len = states.size();
  assert(states.size() == actions.size() && states.size() == policies.size());
  const Uint totalSize = Sequence::computeTotalEpisodeSize(dS, dA, dP, seq_len);
  assert( seq_len == Sequence::computeTotalEpisodeNstep(dS,dA,dP,totalSize) );
  std::vector<Fval> ret(totalSize, 0);
  Fval* buf = ret.data();

  for (Uint i = 0; i<seq_len; ++i)
  {
    assert(states[i].size() == dS);
    assert(actions[i].size() == dA);
    assert(policies[i].size() == dP);
    std::copy(states[i].begin(), states[i].end(), buf);
    buf[dS] = rewards[i]; buf += dS + 1;
    std::copy(actions[i].begin(),  actions[i].end(),  buf); buf += dA;
    std::copy(policies[i].begin(), policies[i].end(), buf); buf += dP;
  }

  /////////////////////////////////////////////////////////////////////////////
  // following vectors may be of size less than seq_len because
  // some algorithms do not allocate them. I.e. Q-learning-based
  // algorithms do not need to advance retrace-like value estimates

  assert(Q_RET.size() <= seq_len);        Q_RET.resize(seq_len);
  std::copy(Q_RET.begin(), Q_RET.end(), buf); buf += seq_len;

  assert(action_adv.size() <= seq_len);   action_adv.resize(seq_len);
  std::copy(action_adv.begin(), action_adv.end(), buf); buf += seq_len;

  assert(state_vals.size() <= seq_len);   state_vals.resize(seq_len);
  std::copy(state_vals.begin(), state_vals.end(), buf); buf += seq_len;

  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // post processing quantities might not be already allocated

  assert(SquaredError.size() <= seq_len); SquaredError.resize(seq_len);
  std::copy(SquaredError.begin(), SquaredError.end(), buf); buf += seq_len;

  assert(offPolicImpW.size() <= seq_len); offPolicImpW.resize(seq_len);
  std::copy(offPolicImpW.begin(), offPolicImpW.end(), buf); buf += seq_len;

  assert(KullbLeibDiv.size() <= seq_len); KullbLeibDiv.resize(seq_len);
  std::copy(KullbLeibDiv.begin(), KullbLeibDiv.end(), buf); buf += seq_len;

  /////////////////////////////////////////////////////////////////////////////

  assert((Uint) (buf-ret.data()) == (dS+dA+dP+7) * seq_len);

  char * charPos = (char*) buf;
  memcpy(charPos, &       ended, sizeof(bool)); charPos += sizeof(bool);
  memcpy(charPos, &          ID, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &just_sampled, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &      prefix, sizeof(Uint)); charPos += sizeof(Uint);
  memcpy(charPos, &     agentID, sizeof(Uint)); charPos += sizeof(Uint);

  // assert(buf - ret.data() == (ptrdiff_t) totalSize);
  return ret;
}

void Sequence::save(FILE * f, const Uint dS, const Uint dA, const Uint dP) {
  const Uint seq_len = states.size();
  fwrite(& seq_len, sizeof(Uint), 1, f);
  Fvec buffer = packSequence(dS, dA, dP);
  fwrite(buffer.data(), sizeof(Fval), buffer.size(), f);
}

void Sequence::unpackSequence(const std::vector<Fval>& data, const Uint dS,
  const Uint dA, const Uint dP)
{
  const Uint seq_len = Sequence::computeTotalEpisodeNstep(dS,dA,dP,data.size());
  assert( data.size() == Sequence::computeTotalEpisodeSize(dS,dA,dP,seq_len) );
  const Fval* buf = data.data();
  assert(states.size() == 0);
  for (Uint i = 0; i<seq_len; ++i) {
    states.push_back(  Fvec(buf, buf+dS));
    rewards.push_back(buf[dS]); buf += dS + 1;
    actions.push_back( Rvec(buf, buf+dA)); buf += dA;
    policies.push_back(Rvec(buf, buf+dP)); buf += dP;
  }

  /////////////////////////////////////////////////////////////////////////////
  Q_RET      = std::vector<nnReal>(buf, buf + seq_len); buf += seq_len;
  action_adv = std::vector<nnReal>(buf, buf + seq_len); buf += seq_len;
  state_vals = std::vector<nnReal>(buf, buf + seq_len); buf += seq_len;
  /////////////////////////////////////////////////////////////////////////////
  SquaredError = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  offPolicImpW = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  KullbLeibDiv = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  /////////////////////////////////////////////////////////////////////////////
  assert((Uint) (buf - data.data()) == (dS+dA+dP+7) * seq_len);
  priorityImpW = std::vector<float>(seq_len, 1);
  totR = Utilities::sum(rewards);
  /////////////////////////////////////////////////////////////////////////////

  const char * charPos = (const char *) buf;
  memcpy(&       ended, charPos, sizeof(bool)); charPos += sizeof(bool);
  memcpy(&          ID, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&just_sampled, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&      prefix, charPos, sizeof(Uint)); charPos += sizeof(Uint);
  memcpy(&     agentID, charPos, sizeof(Uint)); charPos += sizeof(Uint);
  //assert(buf-data.data()==(ptrdiff_t)computeTotalEpisodeSize(dS,dA,dP,seq_len));
}

int Sequence::restart(FILE * f, const Uint dS, const Uint dA, const Uint dP)
{
  Uint seq_len = 0;
  if(fread(& seq_len, sizeof(Uint), 1, f) != 1) return 1;
  const Uint totalSize = Sequence::computeTotalEpisodeSize(dS, dA, dP, seq_len);
  std::vector<Fval> buffer(totalSize);
  if(fread(buffer.data(), sizeof(Fval), totalSize, f) != totalSize)
    die("mismatch");
  unpackSequence(buffer, dS, dA, dP);
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

bool Sequence::isEqual(const Sequence & S) const
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

std::vector<float> Sequence::logToFile(const Uint dimS, const Uint iterStep) const
{
  const Uint seq_len = states.size();
  const Uint dimA = actions[0].size(), dimP = policies[0].size();
  std::vector<float> buffer(seq_len * (4 + dimS + dimA + dimP));
  float * pos = buffer.data();
  for (Uint t=0; t<seq_len; ++t) {
    *(pos++) = iterStep + 0.1;
    const auto steptype = t==0 ? INIT : ( isTerminal(t) ? TERM : (
                          isTruncated(t) ? TRNC : CONT ) );
    *(pos++) = status2int(steptype) + 0.1;
    *(pos++) = t + 0.1;
    std::copy(  states[t].begin(),   states[t].end(), pos);
    pos += dimS;
    std::copy( actions[t].begin(),  actions[t].end(), pos);
    pos += dimA;
    *(pos++) = rewards[t];
    std::copy(policies[t].begin(), policies[t].end(), pos);
    pos += dimP;
  }
  return buffer;
}

}
