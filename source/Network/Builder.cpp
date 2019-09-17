//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Builder.h"

#include "CMA_Optimizer.h"
#include "Optimizer.h"
#include "Network.h"
#include "../Utils/SstreamUtilities.h"
#include "Layers/Layer_Base.h"
#include "Layers/Layer_Conv2D.h"
#include "Layers/Layer_LSTM.h"
#include "Layers/Layer_GRU.h"

namespace smarties
{

Builder::Builder(const Settings& S, const DistributionInfo& D)
  : distrib(D), settings(S) { }

void Builder::addInput(const Uint size)
{
  if(size==0) die("Requested an empty input layer");
  if(bBuilt) die("Cannot build the network multiple times");
  const Uint ID = layers.size();
  layers.emplace_back(
    std::make_unique<InputLayer>(size, ID)
  );
  assert(layers[ID]->nOutputs() == size);

  // if this is not first layer, glue this layer and previous together:
  if(nInputs > 0) {
    assert(ID>0);
    const Uint twoLayersSize = layers[ID-1]->nOutputs() + size;
    layers.emplace_back(
      std::make_unique<JoinLayer>(ID+1, twoLayersSize, 2)
    );
  } else assert(ID == 0);
  nInputs += size;
}

void Builder::addLayer(const Uint layerSize,
                       const std::string funcType,
                       const bool isOutputLayer,
                       const std::string layerType,
                       const Uint iLink)
{
  if(bBuilt) die("Cannot build the network multiple times");

  const Uint ID = layers.size();
  if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
    die("Missing input layer.");
  if(layerSize <= 0)  die("Requested empty layer.");

  const Uint layerInputSize = layers[ID-iLink]->nOutputs();

  if (layerType == "LSTM")
  {
    layers.emplace_back(
      std::make_unique<LSTMLayer>(ID, layerInputSize, layerSize, funcType, isOutputLayer, iLink)
    );
  }
  else if (layerType == "MGU" || layerType == "GRU")
  {
    layers.emplace_back(
      std::make_unique<MGULayer>(ID, layerInputSize, layerSize, funcType, isOutputLayer, iLink)
    );
  }
  else
  {
    const bool bRecurrent = (layerType=="RNN") || (layerType=="Recurrent");
    layers.emplace_back(
      std::make_unique<BaseLayer>(ID, layerInputSize, layerSize, funcType, bRecurrent, isOutputLayer, iLink)
    );
  }

  #if 0
    const bool bResidualLayer = layers[ID-1]->nOutputs() == layerSize
                                && not isOutputLayer;
    if(bResidualLayer)
      layers.emplace_back(std::make_unique<ResidualLayer>(ID+1, layerSize));
  #else
    const bool bResidualLayer = not isOutputLayer;
    if(bResidualLayer)
      layers.emplace_back(
        std::make_unique<ParametricResidualLayer>(ID+1, layerSize) );
  #endif

  if(isOutputLayer) nOutputs += layers.back()->nOutputs();
}

void Builder::addParamLayer(Uint size, std::string funcType, Real init_vals)
{
  addParamLayer(size, funcType, std::vector<Real>(size, init_vals) );
}

void Builder::addParamLayer(Uint size,
                            std::string funcType,
                            std::vector<Real> init_vals)
{
  const Uint ID = layers.size();
  if(bBuilt) die("Cannot build the network multiple times\n");
  if(size<=0) die("Requested an empty layer\n");
  layers.emplace_back(
    std::make_unique<ParamLayer>(ID, size, funcType, init_vals)
  );
  nOutputs += layers.back()->nOutputs();
}

void Builder::build(const bool isInputNet)
{
  if(bBuilt) die("Cannot build the network multiple times\n");
  bBuilt = true;

  nLayers = layers.size();
  unsigned long lsize = MPICommSize(distrib.learners_train_comm);
  const MPI_Comm & tmpComm = distrib.learnersOnWorkers ? distrib.world_comm :
                             distrib.learners_train_comm;
  MPI_Bcast( &lsize, 1, MPI_UNSIGNED_LONG, 0, tmpComm);

  std::shared_ptr<Parameters> weights = Network::allocParameters(layers, lsize);

  std::mt19937& gen = distrib.generators[0];
  // Initialize weights
  for(const auto & l : layers)
    l->initialize(gen, weights.get(),
      l->bOutput && not isInputNet ? settings.outWeightsPrefac : 1);

  if(MPICommRank(distrib.world_comm) == 0) {
    for(const auto & l : layers) printf( "%s", l->printSpecs().c_str() );
  }

  // Make sure that all ranks have the same weights (copy from rank 0)
  if(distrib.learnersOnWorkers) weights->broadcast(distrib.world_comm);
  else weights->broadcast(distrib.learners_train_comm);

  // Initialize network workspace to check that all is ok
  const std::unique_ptr<Activation> test = Network::allocActivation(layers);
  if(test->nInputs not_eq (int) nInputs)
    _die("Mismatch between Builder's computed inputs:%u and Activation's:%u",
         nInputs, test->nInputs);

  if(test->nOutputs not_eq (int) nOutputs) {
    _warn("Mismatch between Builder's computed outputs:%u and Activation's:%u. "
          "Overruled Builder: probable cause is that user net did not specify "
          "which layers are output. If multiple output layers expect trouble\n",
          nOutputs, test->nOutputs);
    nOutputs = test->nOutputs;
  }

  threadGrads = allocManyParams(weights, distrib.nThreads);

  net = std::make_shared<Network>(nInputs, nOutputs, layers, weights);
  // ownership of layers passed onto network, builder should have an empty vec:
  assert(layers.size() == 0);

  if(settings.ESpopSize>1)
    opt = std::make_shared<CMA_Optimizer>(settings,distrib,weights);
  else
    opt = std::make_shared<AdamOptimizer>(settings,distrib,weights,threadGrads);
}

inline bool matchConv2D(const Conv2D_Descriptor& DESCR,
                    Uint InX, Uint InY, Uint InC, Uint KnX, Uint KnY, Uint KnC,
                    Uint Sx,  Uint Sy,  Uint Px,  Uint Py,  Uint OpX, Uint OpY)
{
  bool sameInp = DESCR.inpFeatures==InC && DESCR.inpY==InX && DESCR.inpX==InY;
  bool sameOut = DESCR.outFeatures==KnC && DESCR.outY==OpY && DESCR.outX==OpX;
  bool sameFilter  = DESCR.filterx==KnX && DESCR.filtery==KnY;
  bool sameStride  = DESCR.stridex== Sx && DESCR.stridey== Sy;
  bool samePadding = DESCR.paddinx== Px && DESCR.paddiny== Py;
  if( KnC*OpX*OpY == 0 ) die("Requested empty layer.");
  return sameInp && sameOut && sameFilter && sameStride && samePadding;
}

void Builder::addConv2d(const Conv2D_Descriptor& descr, bool bOut, Uint iLink)
{
  if(bBuilt) die("Cannot build the network multiple times");
  const Uint ID = layers.size();
  if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
    die("Missing input layer.");

  const Uint inpSize = descr.inpFeatures * descr.inpY * descr.inpX;
  if( layers[ID-iLink]->nOutputs() not_eq inpSize )
    _die("Mismatch between input size (%d) and previous layer size (%d).",
      inpSize, layers.back()->nOutputs() );

  // I defined here the conv layers used in the Atari paper. To add new ones add
  // an if-pattern matching the other ones and refer to the `matchConv2D`
  // function above to interpret the arguments. Useful rule of thumb to remember
  // is: outSize should be : (InSize - FilterSize + 2*Padding)/Stride + 1
  if (      matchConv2D(descr, 84,84, 4, 8,8,32, 4,4, 0,0, 20,20) ) {
    layers.emplace_back(
      std::make_unique<Mat2ImLayer<         84,84, 4, 8,8,32, 4,4, 0,0, 20,20>>
        (ID, false, iLink) );
    layers.emplace_back(
      std::make_unique<Conv2DLayer<SoftSign,84,84, 4, 8,8,32, 4,4, 0,0, 20,20>>
        (ID+1, bOut, 1) );
  }
  else
  if (      matchConv2D(descr, 20,20,32, 4,4,64, 2,2, 0,0,  9, 9) ) {
    layers.emplace_back(
      std::make_unique<Mat2ImLayer<         20,20,32, 4,4,64, 2,2, 0,0,  9, 9>>
        (ID, false, iLink) );
    layers.emplace_back(
      std::make_unique<Conv2DLayer<SoftSign,20,20,32, 4,4,64, 2,2, 0,0,  9, 9>>
        (ID+1, bOut, 1) );
  }
  else
  if (      matchConv2D(descr,  9, 9,64, 3,3,64, 1,1, 0,0,  7, 7) ) {
    layers.emplace_back(
      std::make_unique<Mat2ImLayer<          9, 9,64, 3,3,64, 1,1, 0,0,  7, 7>>
        (ID, false, iLink) );
    layers.emplace_back(
      std::make_unique<Conv2DLayer<SoftSign, 9, 9,64, 3,3,64, 1,1, 0,0,  7, 7>>
        (ID+1, bOut, 1) );
  }
  else
    die("Detected undeclared conv2d description. This will be frustrating... "
        "In order to remove dependencies, keep the code low latency, and high "
        "performance, conv2d are templated. Whatever conv2d op you want must "
        "be specified in the Builder.cpp file. You'll see, it's easy.");

  assert(layers.size()>ID && layers.back());
  if(bOut) nOutputs += layers.back()->nOutputs();
}

} // end namespace smarties
