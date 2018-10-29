//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"
#include "../Math/Utils.h"

class NAF : public Learner_offPolicy
{
  //Network produces a vector. The two following vectors specify:
  // - the sizes of the elements that compose the vector
  // - the starting indices along the output vector of each
  const vector<Uint> net_outputs = {1, compute_nL(aInfo.dim), aInfo.dim};
  const vector<Uint> net_indices = {0, 1, 1+compute_nL(aInfo.dim)};
  #ifdef EXTRACT_COVAR
    const Real stdParam = noiseMap_inverse(explNoise*explNoise);
  #else
    const Real stdParam = noiseMap_inverse(explNoise);
  #endif

  const Uint nA = env->aI.dim;
  const Real OrUhDecay = CmaxPol<=0? .85 : 0;
  //const Real OrUhDecay = 0; // as in original
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );

  void TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
    const Uint thrID) const override;
  void Train(const Uint seq, const Uint t, const Uint wID,
    const Uint bID, const Uint thrID) const override;

public:
  NAF(Environment*const env, Settings & settings);
  void select(Agent& agent) override;
  void test();
  static inline Uint compute_nL(const Uint NA)
  {
    return (NA*NA + NA)/2;
  }
};
