//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ThreadContext_h
#define smarties_ThreadContext_h

#include "../ReplayMemory/MiniBatch.h"
#include "Network.h"
#include "../Core/Agent.h"

namespace smarties
{

enum ADDED_INPUT {NONE, NETWORK, ACTION, VECTOR};
struct ThreadContext
{
  const Uint threadID;
  const Uint nAddedSamples;
  const bool bHaveTargetW;
  const Sint targetWeightIndex;
  const Uint allSamplCnt = 1 + nAddedSamples + bHaveTargetW;

  //vector over evaluations (eg. target/current or many samples) and over time:
  std::vector<std::vector<std::unique_ptr<Activation>>> activations;
  std::vector<ADDED_INPUT> _addedInputType;
  std::vector<std::vector<NNvec>> _addedInputVec;

  std::shared_ptr<Parameters> partialGradient;

  std::vector<Sint> lastGradTstep = std::vector<Sint>(allSamplCnt, -1);
  std::vector<Sint> weightIndex = std::vector<Sint>(allSamplCnt, 0);
  const MiniBatch * batch;
  Uint batchIndex;

  ThreadContext(const Uint thrID,
                const std::shared_ptr<Parameters> grad,
                const Uint nAdded,
                const bool bHasTargetWeights,
                const Sint tgtWeightsID) :
    threadID(thrID), nAddedSamples(nAdded), bHaveTargetW(bHasTargetWeights),
    targetWeightIndex(tgtWeightsID), partialGradient(grad)
  {
    activations.resize(allSamplCnt);
    _addedInputVec.resize(allSamplCnt);
    for(Uint i=0; i < allSamplCnt; ++i) {
      activations[i].reserve(MAX_SEQ_LEN);
      _addedInputVec[i].reserve(MAX_SEQ_LEN);
    }
    _addedInputType.resize(allSamplCnt, NONE);
  }

  void setSampleAddedInputType(const Sint sample, ADDED_INPUT type)
  {
    if(sample < 0) _addedInputType.back() = type;
    else _addedInputType[sample] = type;
  }

  void load(const std::shared_ptr<Network> NET,
            const MiniBatch & B,
            const Uint batchID,
            const Uint weightID)
  {
    batch = & B;
    batchIndex = batchID;
    lastGradTstep = std::vector<Sint>(allSamplCnt, -1);
    weightIndex = std::vector<Sint>(allSamplCnt, weightID);
    if(bHaveTargetW) weightIndex.back() = targetWeightIndex;

    for(Uint i=0; i < allSamplCnt; ++i)
      NET->allocTimeSeries(activations[i], batch->sampledNumSteps(batchIndex));
  }

  void overwrite(const Sint t, const Sint sample) const
  {
    if(sample<0) target(t)->written = false;
    else activation(t, sample)->written = false; // what about backprop?
  }

  Sint& endBackPropStep(const Sint sample = 0) {
    assert(sample<0 || lastGradTstep.size() > (Uint) sample);
    if(sample<0) return lastGradTstep.back();
    else return lastGradTstep[sample];
  }
  Sint& usedWeightID(const Sint sample = 0) {
    assert(sample<0 || weightIndex.size() > (Uint) sample);
    if(sample<0) return weightIndex.back();
    else return weightIndex[sample];
  }
  ADDED_INPUT& addedInputType(const Sint sample = 0) {
    assert(sample<0 || _addedInputType.size() > (Uint) sample);
    if(sample<0) return _addedInputType.back();
    else return _addedInputType[sample];
  }
  NNvec& addedInputVec(const Sint t, const Sint sample = 0) {
    assert(sample<0 || _addedInputVec.size() > (Uint) sample);
    if(sample<0) return _addedInputVec.back()[ mapTime2Ind(t) ];
    else return _addedInputVec[sample][ mapTime2Ind(t) ];
  }

  const Sint& endBackPropStep(const Sint sample = 0) const {
    assert(sample<0 || lastGradTstep.size() > (Uint) sample);
    if(sample<0) return lastGradTstep.back();
    else return lastGradTstep[sample];
  }
  const Sint& usedWeightID(const Sint sample = 0) const {
    assert(sample<0 || weightIndex.size() > (Uint) sample);
    if(sample<0) return weightIndex.back();
    else return weightIndex[sample];
  }
  const ADDED_INPUT& addedInputType(const Sint sample = 0) const {
    assert(sample<0 || _addedInputType.size() > (Uint) sample);
    if(sample<0) return _addedInputType.back();
    else return _addedInputType[sample];
  }
  const NNvec& addedInputVec(const Sint t, const Sint sample = 0) const {
    assert(sample<0 || _addedInputVec.size() > (Uint) sample);
    if(sample<0) return _addedInputVec.back()[ mapTime2Ind(t) ];
    else return _addedInputVec[sample][ mapTime2Ind(t) ];
  }

  Activation* activation(const Sint t, const Sint sample) const
  {
    assert(sample<0 || activations.size() > (Uint) sample);
    const auto& timeSeries = sample<0? activations.back() : activations[sample];
    assert( timeSeries.size() > (Uint) mapTime2Ind(t) );
    return timeSeries[ mapTime2Ind(t) ].get();
  }
  Activation* target(const Sint t) const
  {
    assert(bHaveTargetW);
    return activation(t, -1);
  }
  Sint mapTime2Ind(const Sint t) const
  {
    assert(batch not_eq nullptr);
    return batch->mapTime2Ind(batchIndex, t);
  }
  Sint mapInd2Time(const Sint k) const
  {
    assert(batch not_eq nullptr);
    return batch->mapInd2Time(batchIndex, k);
  }
  const NNvec& getState(const Sint t) const
  {
    assert(batch not_eq nullptr);
    return batch->state(batchIndex, t);
  }
  const Rvec& getAction(const Sint t) const
  {
    assert(batch not_eq nullptr);
    return batch->action(batchIndex, t);
  }
};

struct AgentContext
{
  static constexpr Uint nAddedSamples = 0;
  const Uint agentID;
  const MiniBatch* batch;
  //vector over time:
  std::vector<std::unique_ptr<Activation>> activations;
  //std::shared_ptr<Parameters>> partialGradient;
  ADDED_INPUT _addedInputType = NONE;
  std::vector<NNvec> _addedInputVec;
  Sint lastGradTstep;
  Sint weightIndex;

  AgentContext(const Uint aID) : agentID(aID)
  {
    activations.reserve(MAX_SEQ_LEN);
    _addedInputVec.reserve(MAX_SEQ_LEN);
  }

  void setAddedInputType(const Sint sample, const ADDED_INPUT type)
  {
    _addedInputType = type;
  }

  void load(const std::shared_ptr<Network> NET,
            const MiniBatch& B,
            const Agent& agent,
            const Uint weightID)
  {
    batch = & B;
    assert(agent.ID == agentID);
    lastGradTstep = -1;
    weightIndex = weightID;
    NET->allocTimeSeries(activations, batch->sampledNumSteps(0));
    _addedInputVec.resize(batch->sampledNumSteps(0));
  }

  void overwrite(const Sint t, const Sint sample = -1) const
  {
    activation(t)->written = false; // what about backprop?
  }

  Sint& endBackPropStep(const Sint sample =-1) {
    return lastGradTstep;
  }
  Sint& usedWeightID(const Sint sample =-1) {
    return weightIndex;
  }
  ADDED_INPUT& addedInputType(const Sint sample =-1) {
    return _addedInputType;
  }
  NNvec& addedInputVec(const Sint t, const Sint sample = -1)
  {
    return _addedInputVec[ mapTime2Ind(t) ];
  }
  const Sint& endBackPropStep(const Sint sample =-1) const {
    return lastGradTstep;
  }
  const Sint& usedWeightID(const Sint sample =-1) const {
    return weightIndex;
  }
  const ADDED_INPUT& addedInputType(const Sint sample =-1) const {
    return _addedInputType;
  }
  const NNvec& addedInputVec(const Sint t, const Sint sample = -1) const {
    return _addedInputVec[ mapTime2Ind(t) ];
  }

  Activation* activation(const Sint t, const Sint sample = -1) const
  {
    return activations[ mapTime2Ind(t) ].get();
  }
  Sint mapTime2Ind(const Sint t) const
  {
    assert(batch not_eq nullptr);
    return batch->mapTime2Ind(0, t);
  }
  Sint mapInd2Time(const Sint k) const
  {
    assert(batch not_eq nullptr);
    return batch->mapInd2Time(0, k);
  }
  const NNvec& getState(const Sint t) const
  {
    assert(batch not_eq nullptr);
    return batch->state(0, t);
  }
  const Rvec& getAction(const Sint t) const
  {
    assert(batch not_eq nullptr);
    return batch->action(0, t);
  }
  const Episode* episode() const
  {
    assert(batch not_eq nullptr);
    assert(batch->episodes.size() == 1 && batch->episodes[0] not_eq nullptr);
    return batch->episodes[0];
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
