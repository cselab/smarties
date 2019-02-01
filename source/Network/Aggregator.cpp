//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Aggregator.h"
#include "Builder.h"

void Aggregator::prepare(Sequence*const traj, const Uint thrID,
  const RELAY SET) const {
  seq[thrID] = traj;
  usage[thrID] = SET;
}

void Aggregator::prepare_seq(Sequence*const traj, const Uint thrID,
  const RELAY SET) const {
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(traj->tuples.size(), Rvec());
  first_sample[thrID] = 0;
  usage[thrID] = SET;
  seq[thrID] = traj;
}

void Aggregator::prepare_one(Sequence*const traj, const Uint samp,
  const Uint thrID, const RELAY SET) const {
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 2;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, Rvec());
  first_sample[thrID] = samp - nRecurr;
  usage[thrID] = SET;
  seq[thrID] = traj;
}

void Aggregator::prepare(Sequence*const traj, const Uint samp, const Uint N,
  const Uint thrID, const RELAY SET) const {
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 1 + N;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, Rvec());
  first_sample[thrID] = samp - nRecurr;
  usage[thrID] = SET;
  seq[thrID] = traj;
}

void Aggregator::set(const Rvec vec, const Uint samp, const Uint thrID) const {
  usage[thrID] = VEC;
  const int ind = (int)samp - first_sample[thrID];
  assert(first_sample[thrID] <= (int)samp);
  assert(ind >= 0 && (int) inputs[thrID].size() > ind);
  inputs[thrID][ind] = vec;
}

Rvec Aggregator::get(const Uint samp, const Uint thrID, const int USEW) const {
  if(usage[thrID] == VEC) {
    assert(first_sample[thrID] >= 0);
    const int ind = (int)samp - first_sample[thrID];
    assert(first_sample[thrID] <= (int)samp);
    assert(ind >= 0 && (int) inputs[thrID].size() > ind);
    assert(inputs[thrID][ind].size());
    return inputs[thrID][ind];
  } else if (usage[thrID] == ACT) {
    assert(aI.dim == nOuts);
    return aI.getInvScaled(seq[thrID]->tuples[samp]->a);
  } else {
    assert(USEW >= -1);
    // if target net, use target net's workspace, otherwise workspaces are
    // defined per thread and not per whichever weight sample we consider:
    const int USEA = USEW > 0 ? 0 : -1;
    Rvec ret = approx->forward(samp, thrID, USEW, USEA);
    assert(ret.size() >= nOuts); // in DPG we now also output parametric stdev
    ret.resize(nOuts);           // vector, therefore ... this workaround
    return ret;
  }
}
