//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MiniBatch_h
#define smarties_MiniBatch_h

#include "Episode.h"

namespace smarties
{

struct MiniBatch
{
  const Uint size;
  const Fval gamma;
  MiniBatch(const Uint _size, const Fval G) : size(_size), gamma(G)
  {
    episodes.resize(size);
    begTimeStep.resize(size);
    endTimeStep.resize(size);
    sampledTimeStep.resize(size);
    S.resize(size); R.resize(size); PERW.resize(size);
  }
  MiniBatch(MiniBatch && p) = default;
  MiniBatch& operator=(MiniBatch && p) = delete;
  MiniBatch(const MiniBatch &p) = delete;
  MiniBatch& operator=(const MiniBatch &p) = delete;

  std::vector<Episode*> episodes;
  std::vector<Sint> begTimeStep;
  std::vector<Sint> endTimeStep;
  std::vector<Sint> sampledTimeStep;
  Sint sampledBegStep(const Uint b) const { return begTimeStep[b]; }
  Sint sampledEndStep(const Uint b) const { return endTimeStep[b]; }
  Sint sampledTstep(const Uint b) const { return sampledTimeStep[b]; }
  Sint sampledNumSteps(const Uint b) const {
    assert(begTimeStep.size() > b);
    assert(endTimeStep.size() > b);
    return endTimeStep[b] - begTimeStep[b];
  }
  Sint mapTime2Ind(const Uint b, const Sint t) const
  {
    assert(begTimeStep.size() >  b);
    assert(begTimeStep[b]     <= t);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return t - begTimeStep[b];
  }
  Sint mapInd2Time(const Uint b, const Sint k) const
  {
    assert(begTimeStep.size() > b);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return k + begTimeStep[b];
  }

  // episodes | time steps | dimensionality
  std::vector< std::vector< NNvec > > S;  // scaled state
  std::vector< std::vector< Real  > > R;  // scaled reward
  std::vector< std::vector< nnReal> > PERW;  // prioritized sampling

  Episode& getEpisode(const Uint b) const
  {
    return * episodes[b];
  }

  NNvec& state(const Uint b, const Sint t)
  {
    return S[b][mapTime2Ind(b, t)];
  }
  const NNvec& state(const Uint b, const Sint t) const
  {
    return S[b][mapTime2Ind(b, t)];
  }
  Real& reward(const Uint b, const Sint t)
  {
    return R[b][mapTime2Ind(b, t)];
  }
  const Real& reward(const Uint b, const Sint t) const
  {
    return R[b][mapTime2Ind(b, t)];
  }
  nnReal& PERweight(const Uint b, const Sint t)
  {
    return PERW[b][mapTime2Ind(b, t)];
  }
  const nnReal& PERweight(const Uint b, const Sint t) const
  {
    return PERW[b][mapTime2Ind(b, t)];
  }
  const Rvec& action(const Uint b, const Uint t) const
  {
    return episodes[b]->actions[t];
  }
  const Rvec& mu(const Uint b, const Uint t) const
  {
    return episodes[b]->policies[t];
  }
  nnReal& Q_RET(const Uint b, const Uint t) const
  {
    return episodes[b]->Q_RET[t];
  }
  std::vector<nnReal> Q_RET(const Uint dt = 0) const
  {
    std::vector<nnReal> ret(size, 0);
    for(Uint b=0; b<size; ++b) {
      const auto t = sampledTstep(b);
      assert(t >= (Sint) dt);
      ret[b] = episodes[b]->Q_RET[t-dt];
    }
    return ret;
  }
  nnReal& value(const Uint b, const Uint t) const
  {
    return episodes[b]->state_vals[t];
  }
  nnReal& advantage(const Uint b, const Uint t) const
  {
    return episodes[b]->action_adv[t];
  }


  bool isTerminal(const Uint b, const Uint t) const
  {
    return episodes[b]->isTerminal(t);
  }
  std::vector<int> isNextTerminal() const // pybind will not like vector of bool
  {
    std::vector<int> ret(size, 0);
    for(Uint b=0; b<size; ++b)
      ret[b] = episodes[b]->isTerminal(sampledTstep(b) + 1);
    return ret;
  }
  bool isTruncated(const Uint b, const Uint t) const
  {
    return episodes[b]->isTruncated(t);
  }
  std::vector<int> isNextTruncated() const //pybind will not like vector of bool
  {
    std::vector<int> ret(size, 0);
    for(Uint b=0; b<size; ++b)
      ret[b] = episodes[b]->isTruncated(sampledTstep(b) + 1);
    return ret;
  }
  Uint nTimeSteps(const Uint b) const
  {
    return episodes[b]->nsteps();
  }
  Uint nDataSteps(const Uint b) const //terminal/truncated state not actual data
  {
    return episodes[b]->ndata();
  }

  void setMseDklImpw(const Uint b, const Uint t, // batch id and time id
    const Fval E, const Fval D, const Fval W,    // error, dkl, offpol weight
    const Fval C, const Fval invC) const         // bounds of offpol weight
  {
    getEpisode(b).updateCumulative_atomic(t, E, D, W, C, invC);
  }

  Fval updateRetrace(const Uint b, const Uint t,
    const Fval A, const Fval V, const Fval W) const
  {
    assert(W >= 0);
    if(t == 0) return 0; // at time 0, no reward, QRET is undefined
    Episode& EP = getEpisode(b);
    EP.action_adv[t] = A; EP.state_vals[t] = V;
    const Fval reward = R[b][mapTime2Ind(b, t)];
    const Fval oldRet = EP.Q_RET[t-1], clipW = W<1 ? W:1;
    EP.Q_RET[t-1] = reward + gamma*V + gamma*clipW * (EP.Q_RET[t] - A - V);
    return std::fabs(EP.Q_RET[t-1] - oldRet);
  }

  template<typename T>
  void updateAllRetrace(const std::vector<T>& advantages,
                        const std::vector<T>& values,
                        const std::vector<T>& rhos) const
  {
    assert(advantages.size() == size);
    assert(values.size() == size);
    assert(rhos.size() == size);
    #pragma omp parallel for schedule(static)
    for(Uint b=0; b<size; ++b)
      updateRetrace(b, sampledTstep(b), advantages[b], values[b], rhos[b]);
  }

  template<typename T>
  void updateAllLastStepRetrace(const std::vector<T>& values) const
  {
    assert(values.size() == size);
    #pragma omp parallel for schedule(static)
    for(Uint b=0; b<size; ++b) {
      if( isTruncated(b, sampledTstep(b)+1) )
        updateRetrace(b, sampledTstep(b)+1, 0, values[b], 0);
      if( isTerminal (b, sampledTstep(b)+1) )
        updateRetrace(b, sampledTstep(b)+1, 0, 0, 0);
    }
  }

  template<typename T>
  void setAllMseDklImpw(const std::vector<T>& L2errs,
                        const std::vector<T>& DKLs,
                        const std::vector<T>& rhos,
                        const Fval C, const Fval invC) const
  {
    assert(L2errs.size() == size);
    assert(DKLs.size() == size);
    assert(rhos.size() == size);
    for(Uint b=0; b<size; ++b)
      setMseDklImpw(b, sampledTstep(b), L2errs[b], DKLs[b], rhos[b], C,invC);
  }

  void resizeStep(const Uint b, const Uint nSteps)
  {
    assert( S.size()>b); assert( R.size()>b);
    S[b].resize(nSteps); R[b].resize(nSteps); PERW[b].resize(nSteps);
  }
};
} // namespace smarties
#endif // smarties_MiniBatch_h
