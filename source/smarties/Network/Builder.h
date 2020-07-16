//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Builder_h
#define smarties_Builder_h

#include "Optimizer.h"
#include "Network.h"

namespace smarties
{

class Builder
{
public:
  void addInput(const Uint size);

  /*
    addLayer adds fully conn. layer:
      - nNeurons: simply the size of the layer (for LSTM is number of cells)
      - funcType: non-linearity applied to the matrix-vector mul
                  (for LSTM is function applied cell input, gates have sigmoid)
      - bOutput: whether layer is output and therefore copied into return
                 vector when calling Network:forward
      - layerType: LSTM, RNN, else assumed MLP
      - iLink: how many layers back should layer take the input from.
               iLink=1 means that input is previous layer
               iLink=2 means input is *only* the output of 2 layers below
               This allows networks with multiple heads, but always each
               layer has only one input layer (+ eventual recurrent connection).
  */
  void addLayer(const Uint nNeurons,
                const std::string funcType,
                const bool bOutput=false,
                const std::string layerType="",
                const Uint iLink = 1);

  void addParamLayer(Uint size, std::string funcType="Linear", Real init=0);

  void addParamLayer(Uint size, std::string funcType, std::vector<Real> init);


  template<typename T>
  void setLastLayersBias(const std::vector<T> init_vals)
  {
    Rvec init = Rvec(init_vals.begin(), init_vals.end());
    layers.back()->biasInitialValues(init);
  }

  void addConv2d(const Conv2D_Descriptor&, bool bOutput=false, Uint iLink = 1);

  // Function that initializes and constructs net and optimizer.
  // Once this is called number of layers or weights CANNOT be modified.
  void build(const bool isInputNet = false);

private:
  bool bBuilt = false;
public:
  const ExecutionInfo & distrib;
  const HyperParameters & settings;
  Uint nInputs=0, nOutputs=0, nLayers=0;

  std::vector<std::shared_ptr<Parameters>> threadGrads;
  std::vector<std::unique_ptr<Layer>> layers;

  std::shared_ptr<Network> net;
  std::shared_ptr<Optimizer> opt;

  Builder(const HyperParameters& S, const ExecutionInfo& D);
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
