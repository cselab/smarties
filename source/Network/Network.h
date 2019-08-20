//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Network_h
#define smarties_Network_h

#include "Layers/Layers.h"

namespace smarties
{

class Builder;

class Network
{
public:
  const std::vector<std::unique_ptr<Layer>> layers;
  const Uint nInputs, nOutputs, nLayers = layers.size();
  const std::shared_ptr<Parameters> weights;
  Uint getnOutputs() const { return nOutputs; }
  Uint getnInputs()  const { return nInputs;  }
  Uint getnLayers()  const { return nLayers;  }

  static std::shared_ptr<Parameters> allocParameters(
        const std::vector<std::unique_ptr<Layer>>& layers, const Uint mpiSize)
  {
    std::vector<Uint> nWeight, nBiases;
    for(const auto & l : layers) l->requiredParameters(nWeight, nBiases);
    return std::make_shared<Parameters>(nWeight, nBiases, mpiSize);
  }
  static std::unique_ptr<Activation> allocActivation(
        const std::vector<std::unique_ptr<Layer>>& layers)
  {
    std::vector<Uint> sizes, output, input;
    for(const auto & l : layers) l->requiredActivation(sizes, output, input);
    return std::make_unique<Activation>(sizes, output, input);
  }
  std::unique_ptr<Activation> allocActivation() const
  {
    return allocActivation(layers);
  }
  std::shared_ptr<Parameters> allocParameters() const
  {
    return weights->allocateEmptyAlike();
  }

  void allocTimeSeries(std::vector<std::unique_ptr<Activation>>& series,
                       const Uint N) const
  {
    if (series.size() < N)
      for(Uint j=series.size(); j<N; ++j)
        series.emplace_back( allocActivation() );
    assert(series.size()>=N);

    for(Uint j=0; j<series.size(); ++j) {
      series[j]->clearErrors();
      series[j]->written = false;
    }

    #ifndef NDEBUG
    for(Uint j=0; j<series.size(); ++j) assert(not series[j]->written);
    #endif
  }

  Network(const Uint _nInp, const Uint _nOut,
          std::vector<std::unique_ptr<Layer>>& _layers,
          const std::shared_ptr<Parameters>& _weights);

  std::vector<Real> predict(const std::vector<Real>& _inp,
    const std::vector<Activation*>& timeSeries, const Uint step,
    const Parameters*const _weights = nullptr) const
  {
    assert(timeSeries.size() > step);
    const Activation*const currStep = timeSeries[step];
    const Activation*const prevStep = step==0 ? nullptr : timeSeries[step-1];
    return predict(_inp, prevStep, currStep, _weights);
  }

  std::vector<Real> predict(const std::vector<Real>& _inp,
    const Activation* const currStep,
    const Parameters*const _weights = nullptr) const
  {
    return predict(_inp,  nullptr, currStep, _weights);
  }

  /*
    predict output of network given input:
    - vector<Real> _inp: must be same size as input layer
    - Activation prevStep: for recurrent connections. Will use field `outvals`.
    - Activation currStep: work memory where prediction will be computed.
                           Will overwrite fields `suminps` (containing layer's
                           W-matrix input-vector + bias in case of MLP) and
                           `outvals` (containing func(suminps)). No need to clear.
    - Parameters _weights: network weights to use. If nullptr then we use default.
  */
  template<typename T>
  std::vector<Real> predict(const std::vector<T>& input,
                            const Activation* const prevStep,
                            const Activation* const currStep,
                            const Parameters* const _weights = nullptr) const
  {
    assert(input.size()==nInputs && layers.size()==nLayers);
    currStep->setInput(input);
    const Parameters*const W = _weights==nullptr? weights.get() : _weights;
    for(Uint j=0; j<nLayers; ++j) layers[j]->forward(prevStep, currStep, W);
    currStep->written = true;
    return currStep->getOutput();
  }

  void backProp(const Activation*const currStep, const Parameters*const _grad,
                const Parameters*const _weights = nullptr) const
  {
    return backProp(nullptr, currStep, nullptr, _grad, _weights);
  }

  template<typename T>
  void backProp(const std::vector<T>& _errors,
                const Activation* const currStep,
                const Parameters* const _grad,
                const Parameters* const _weights = nullptr) const
  {
    currStep->clearErrors();
    currStep->setOutputDelta(_errors);
    assert(currStep->written);
    _grad->written = true;
    backProp(nullptr, currStep, nullptr, _grad, _weights);
  }

  template<typename T>
  std::vector<Real> backPropToLayer(const std::vector<T>& gradient,
                                    const Uint toLayerID,
                                    const Activation*const activation,
                                    const Parameters*const _weights) const
  {
    const Parameters*const W = _weights==nullptr? weights.get() : _weights;
    activation->clearErrors();
    activation->setOutputDelta(gradient);
    assert(activation->written && activation->input[toLayerID]);
    //backprop from output layer down to layer we want grad for
    for(Uint i=layers.size()-1; i>toLayerID; --i)
      layers[i]->backward(nullptr, activation, nullptr, nullptr, W);
    return activation->getInputGradient(toLayerID);
  }

  /*
    cache friendly backprop for time series: backprops from top layers to bottom
    layers and from last time step to first, with layer being the 'slow index'
    maximises reuse of weights in cache by getting each layer done in turn
  */
  void backProp(const std::vector<std::unique_ptr<Activation>>& series,
                const Uint stepLastError, const Parameters*const _grad,
                const Parameters*const _weights = nullptr) const
  {
    assert(stepLastError <= series.size());
    const Parameters*const W = _weights == nullptr ? weights.get() : _weights;

    if (stepLastError == 0) return; //no errors placed
    else
    if (stepLastError == 1)
    { //errors placed at first time step
      assert(series[0]->written);
      for (Sint i = (Sint)layers.size()-1; i>=0; --i)
        layers[i]->backward(nullptr, series[0].get(), nullptr, _grad, W);
    }
    else
    {
      const Uint T = stepLastError - 1;
      for (Sint i = (Sint)layers.size()-1; i>=0; --i)
      {
        assert(series[T]->written);
        layers[i]->backward(series[T-1].get(), series[T].get(), nullptr,
                            _grad, W);

        for (Uint k = T-1; k>0; --k) {
        assert(series[k]->written);
        layers[i]->backward(series[k-1].get(), series[k].get(), series[k+1].get(),
                            _grad, W);
        }

        assert(series[0]->written);
        layers[i]->backward(          nullptr, series[0].get(), series[  1].get(),
                            _grad, W);
      }
    }
    _grad->written = true;
  }

  /*
    backProp to compute the gradients wrt the errors at step currStep:
    - Activation prevStep: to backProp gradients to previous step
    - Activation currStep: where dE/dOut placed (ie. Activation::setOutputDelta
                           was already used on currStep). This will update all
                           `errvals` fields with +=, therefore field should first
                           be cleared to 0 if it contains spurious gradients.
                           (ie. grads not due to same BPTT loop)
    - Activation nextStep: next step in the time series. Needed by LSTM and such.
    - Parameters gradient: will cointain d Err / d Params. Accessed with +=
                           as minibatch update is implied.
    - Parameters _weights: network weights to use. If nullptr then we use default.
  */
  void backProp(const Activation*const prevStep,
                const Activation*const currStep,
                const Activation*const nextStep,
                const Parameters*const _gradient,
                const Parameters*const _weights = nullptr) const
  {
    assert(currStep->written);
    _gradient->written = true;
    const Parameters*const W = _weights == nullptr ? weights.get() : _weights;
    for (Sint i = (Sint)layers.size()-1; i>=0; --i)
      layers[i]->backward(prevStep, currStep, nextStep, _gradient, W);
  }

  void checkGrads();
  //void dump(const int agentID);
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
