//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Sequences.h"

vector<Fval> Sequence::packSequence(const Uint dS, const Uint dA, const Uint dP)
{
  const Uint seq_len = tuples.size();
  const Uint totalSize = Sequence::computeTotalEpisodeSize(dS, dA, dP, seq_len);
  std::vector<Fval> ret(totalSize, 0);
  Fval* buf = ret.data();
  for (Uint i = 0; i<seq_len; i++) {
    std::copy(tuples[i]->s.begin(), tuples[i]->s.end(), buf);
    buf[dS] = tuples[i]->r; buf += dS + 1;
    std::copy(tuples[i]->a.begin(),  tuples[i]->a.end(),  buf); buf += dA;
    std::copy(tuples[i]->mu.begin(), tuples[i]->mu.end(), buf); buf += dP;
  }

  assert(Q_RET.size() <= seq_len);        Q_RET.resize(seq_len);
  std::copy(Q_RET.begin(), Q_RET.end(), buf); buf += seq_len;

  assert(action_adv.size() <= seq_len);   action_adv.resize(seq_len);
  std::copy(action_adv.begin(), action_adv.end(), buf); buf += seq_len;

  assert(state_vals.size() <= seq_len);   state_vals.resize(seq_len);
  std::copy(state_vals.begin(), state_vals.end(), buf); buf += seq_len;

  assert(SquaredError.size() <= seq_len); SquaredError.resize(seq_len);
  std::copy(SquaredError.begin(), SquaredError.end(), buf); buf += seq_len;

  assert(offPolicImpW.size() <= seq_len); offPolicImpW.resize(seq_len);
  std::copy(offPolicImpW.begin(), offPolicImpW.end(), buf); buf += seq_len;

  assert(KullbLeibDiv.size() <= seq_len); KullbLeibDiv.resize(seq_len);
  std::copy(KullbLeibDiv.begin(), KullbLeibDiv.end(), buf); buf += seq_len;

  *(buf++) = ended+.5; *(buf++) = ID+.5; *(buf++) = nOffPol;
  *(buf++) = MSE; *(buf++) = sumKLDiv; *(buf++) = totR;
  assert(buf - ret.data() == totalSize);
  return ret;
}

void Sequence::save(FILE * f, const Uint dS, const Uint dA, const Uint dP) {
  const Uint seq_len = tuples.size();
  fwrite(& seq_len, sizeof(Uint), 1, f);
  Fvec buffer = packSequence(dS, dA, dP);
  fwrite(buffer.data(), sizeof(Fval), buffer.size(), f);
}

void Sequence::unpackSequence(const vector<Fval>& data, const Uint dS,
  const Uint dA, const Uint dP)
{
  const Uint seq_len = Sequence::computeTotalEpisodeNstep(dS,dA,dP,data.size());
  const Fval* buf = data.data();
  assert(tuples.size() == 0);
  tuples = vector<Tuple*>(seq_len, nullptr);
  for (Uint i = 0; i<seq_len; i++) {
    tuples[i] = new Tuple(buf, dS, buf[dS]); buf += dS + 1;
    tuples[i]->setAct(buf, dA, dP); buf += dA + dP;
  }
  Q_RET = Fvec(buf, buf + seq_len); buf += seq_len;
  action_adv = Fvec(buf, buf + seq_len); buf += seq_len;
  state_vals = Fvec(buf, buf + seq_len); buf += seq_len;
  SquaredError = Fvec(buf, buf + seq_len); buf += seq_len;
  offPolicImpW = Fvec(buf, buf + seq_len); buf += seq_len;
  KullbLeibDiv = Fvec(buf, buf + seq_len); buf += seq_len;
  priorityImpW = std::vector<float>(seq_len, 1);
  ended = *(buf++); ID = *(buf++); nOffPol = *(buf++);
  MSE = *(buf++); sumKLDiv = *(buf++); totR = *(buf++);
  assert(buf-data.data()==Sequence::computeTotalEpisodeSize(dS,dA,dP,seq_len));
}

int Sequence::restart(FILE * f, const Uint dS, const Uint dA, const Uint dP)
{
  Uint seq_len = 0;
  if(fread(& seq_len, sizeof(Uint), 1, f) != 1) return 1;
  const Uint totalSize = Sequence::computeTotalEpisodeSize(dS, dA, dP, seq_len);
  vector<Fval> buffer(totalSize);
  if(fread(buffer.data(), sizeof(Fval), totalSize, f) != totalSize)
    die("mismatch");
  unpackSequence(buffer, dS, dA, dP);
  return 0;
}
