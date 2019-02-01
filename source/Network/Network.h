//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Layers.h"

class Builder;

class Network
{
public:
  const Uint nThreads, nInputs, nOutputs, nLayers;
  const bool bDump;
  const Real gradClip;
  const std::vector<Layer*> layers;
  const Parameters* const weights;
  const Parameters* const tgt_weights;
  const std::vector<Parameters*> Vgrad;
  const std::vector<Parameters*> sampled_weights;
  std::vector<std::mt19937>& generators;

  Uint getnOutputs() const { return nOutputs; }
  Uint getnInputs()  const { return nInputs;  }
  Uint getnLayers()  const { return nLayers;  }

  //inline vector<Activation*> allocateUnrolledActivations(const Uint num) const {
  //  vector<Activation*> ret(num, nullptr);
  //  for (Uint j=0; j<num; j++) ret[j] = allocate_activation(layers);
  //  return ret;
  //}
  inline Activation* allocateActivation() const {
    std::vector<Uint> sizes, output, input;
    for(const auto & l : layers) l->requiredActivation(sizes, output, input);
    return new Activation(sizes, output, input);
  }

  //inline Parameters* allocateParameters() const {
  //  vector<Uint> nWeight, nBiases;
  //  for(const auto & l : layers) l->requiredParameters(nWeight, nBiases);
  //  return new Parameters(nWeight, nBiases);
  //}

  void updateTransposed() const {
    for(const auto & l : layers) l->transpose(weights);
  }

  inline void prepForBackProp(std::vector<Activation*>& series, const Uint N) const
  {
    if (series.size() < N)
      for(Uint j=series.size(); j<N; j++)
        series.push_back(allocateActivation());
    assert(series.size()>=N);

    for(Uint j=0; j<series.size(); j++) {
      series[j]->clearErrors();
      series[j]->written = false;
    }

    #ifndef NDEBUG
    for(Uint j=0; j<series.size(); j++) assert(not series[j]->written);
    #endif
  }
  inline void prepForFwdProp (std::vector<Activation*>& series, const Uint N) const
  {
    prepForBackProp(series, N);
    #if 0
    if (series.size() < N)
      for(Uint j=series.size(); j<N; j++)
        series.push_back(allocateActivation());
    assert(series.size()>=N);

    for(Uint j=0; j<series.size() && series[j]->written; j++) //; j++)
      series[j]->written = false;

    #ifndef NDEBUG
    for(Uint j=0; j<series.size(); j++) assert(not series[j]->written);
    #endif
    #endif
  }

  Network(Builder* const B, const Settings & settings) ;

  ~Network() {
    for(auto& ptr: sampled_weights) _dispose_object(ptr);
    for(auto& ptr: layers) _dispose_object(ptr);
    for(auto& ptr: Vgrad) _dispose_object(ptr);
    _dispose_object(tgt_weights);
    _dispose_object(weights);
  }

  inline std::vector<Real> predict(const std::vector<Real>& _inp,
    const std::vector<Activation*>& timeSeries, const Uint step,
    const Parameters*const _weights = nullptr) const
  {
    assert(timeSeries.size() > step);
    const Activation*const currStep = timeSeries[step];
    const Activation*const prevStep = step==0 ? nullptr : timeSeries[step-1];
    return predict(_inp, prevStep, currStep, _weights);
  }

  inline std::vector<Real> predict(const std::vector<Real>& _inp,
    const Activation* const currStep,
    const Parameters*const _weights = nullptr) const
  {
    return predict(_inp,  nullptr, currStep, _weights);
  }

  std::vector<Real> predict(const std::vector<Real>& _inp,
    const Activation* const prevStep, const Activation* const currStep,
    const Parameters*const _weights = nullptr) const;

  void backProp(const Activation*const currStep, const Parameters*const _grad,
                const Parameters*const _weights = nullptr) const
  {
    return backProp(nullptr, currStep, nullptr, _grad, _weights);
  }
  void backProp(const std::vector<Real>& _errors, const Activation*const currStep,
    const Parameters*const _grad, const Parameters*const _weights=nullptr) const
  {
    currStep->clearErrors();
    currStep->setOutputDelta(_errors);
    assert(currStep->written);
    _grad->written = true;
    backProp(nullptr, currStep, nullptr, _grad, _weights);
  }

  std::vector<Real> inpBackProp(const std::vector<Real>&err, Activation*const act,
    const Parameters*const W, const Uint ID) const
  {
    act->clearErrors();
    act->setOutputDelta(err);
    assert(act->written && act->input[ID]);
    for(Uint i=layers.size()-1; i>ID; i--) //skip below layer we want grad for
      layers[i]->backward(nullptr, act, nullptr, nullptr, W);
    return act->getInputGradient(ID);
  }


  void backProp(const std::vector<Activation*>& timeSeries,
                const Uint stepLastError,
                const Parameters*const _gradient,
                const Parameters*const _weights = nullptr) const;

  void backProp(const Activation*const prevStep,
                const Activation*const currStep,
                const Activation*const nextStep,
                const Parameters*const _gradient,
                const Parameters*const _weights = nullptr) const;

  void checkGrads();
  //void dump(const int agentID);
};
