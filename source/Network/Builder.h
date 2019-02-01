//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Optimizer.h"
#include "CMA_Optimizer_MPI.h"
#include "Network.h"
#include "Layer_Base.h"
#include "Layer_Conv2D.h"
//#include "Layer_IntFire.h"
#include "Layer_LSTM.h"
#include "Layer_MGU.h"

class Builder
{
public:
  void addInput(const int size)
  {
    if(bBuilt) die("Cannot build the network multiple times");
    if(size<=0) die("Requested an empty input layer\n");
    const int ID = layers.size();
    layers.push_back(new InputLayer(size, ID));
    assert(layers[ID]->nOutputs() == (Uint) size);
    if(nInputs > 0) {
      const Uint twoLayersSize = layers[ID-1]->nOutputs() + size;
      layers.push_back(new JoinLayer(ID+1, twoLayersSize, 2));
    } else assert(ID == 0);
    nInputs += size;
  }

  /*
    addLayer adds fully conn. layer:
      - nNeurons: simply the size of the layer (for LSTM is number of cells)
      - funcType: non-linearity applied to the matrix-vector mul
                  (for LSTM is function applied cell input, gates have sigmoid)
      - bOutput: whether layer is output and therefore copied into return
                 vector when calling Network:predict
      - layerType: LSTM, RNN, else assumed MLP
      - iLink: how many layers back should layer take the input from.
               iLink=1 means that input is previous layer
               iLink=2 means input is *only* the output of 2 layers below
               This allows networks with multiple heads, but always each
               layer has only one input layer (+ eventual recurrent connection).
  */
  void addLayer(const int nNeurons, const string funcType,
    const bool bOutput=false, const string layerType="", const int iLink = 1)
  {
    if(bBuilt) die("Cannot build the network multiple times");
    const int ID = layers.size();
    if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
      die("Missing input layer.");
    if(nNeurons <= 0)  die("Requested empty layer.");
    const Uint layInp = layers[ID-iLink]->nOutputs();
    Layer* l = nullptr;
           if (layerType == "LSTM") {
      l = new LSTMLayer(ID, layInp, nNeurons, funcType, bOutput, iLink);
    } else if (layerType == "MGU" || layerType == "GRU") {
      l = new MGULayer(ID, layInp, nNeurons, funcType, bOutput, iLink);
    } else if (layerType == "IntegrateFire") {
      //l = new IntegrateFireLayer(nInputs, nNeurons, layers.size());
    } else {
      const bool bRecur = (layerType=="RNN") || (layerType=="Recurrent");
      l = new BaseLayer(ID, layInp, nNeurons, funcType, bRecur, bOutput, iLink);
    }
    assert(l not_eq nullptr);
    layers.push_back(l);

    const bool bResLayer = (int) layers[ID-1]->nOutputs()==nNeurons && !bOutput;
    //const bool bResLayer = not bOutput;
    if(bResLayer)
      layers.push_back(new ResidualLayer(ID+1, nNeurons));

    if(bOutput) nOutputs += l->nOutputs();
  }

  void setLastLayersBias(vector<Real> init_vals)
  {
    layers.back()->biasInitialValues(init_vals);
  }

  void addParamLayer(int size, string funcType = "Linear", Real init_vals = 0)
  {
    addParamLayer(size, funcType, vector<Real>(size, init_vals) );
  }
  void addParamLayer(int size, string funcType, vector<Real> init_vals)
  {
    const Uint ID = layers.size();
    if(bBuilt) die("Cannot build the network multiple times\n");
    if(size<=0) die("Requested an empty layer\n");
    Layer* l = new ParamLayer(ID, size, funcType, init_vals);
    layers.push_back(l);
    assert(l not_eq nullptr);
    nOutputs += l->nOutputs();
  }

  template<
  typename func,
  int In_X, int In_Y, int In_C, //input image: x:width, y:height, c:channels
  int Kn_X, int Kn_Y, int Kn_C,  //filter: x:width, y:height, c:channels
  int Sx=1, int Sy=1, //stride x/y
  int OutX=(In_X -Kn_X)/Sx+1,
  int OutY=(In_Y -Kn_Y)/Sy+1> //output image: same number of channels as KnC
  void addConv2d(const bool bOutput=false, const int iLink = 1)
  {
    if(bBuilt) die("Cannot build the network multiple times");
    const int ID = layers.size();
    if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
      die("Missing input layer.");
    if( Kn_C*OutX*OutY <= 0 ) die("Requested empty layer.");
    if( layers[ID-iLink]->nOutputs() not_eq In_X * In_Y * In_C )
      _die("Mismatch between input size (%d) and previous layer size (%d).",
        In_X * In_Y * In_C, layers.back()->nOutputs() );

    Layer* l = nullptr;
    l = new ConvLayer<func, In_X,In_Y,In_C, Kn_X,Kn_Y,Kn_C, Sx,Sy, OutX,OutY>(
      ID, bOutput, iLink);

    layers.push_back(l);
    assert(l not_eq nullptr);
    if(bOutput) nOutputs += l->nOutputs();

    #if 0 //TODO check
      assert((OutX-1)*Sx +Kn_X >= In_X);
      assert((OutX-1)*Sx +Kn_X <  Kn_X+In_X);
      assert((OutY-1)*Sy +Kn_Y >= In_Y);
      assert((OutY-1)*Sy +Kn_Y <  Kn_Y+In_Y);
      if(Kn_X<=0 || Kn_Y<=0 || Kn_C<=0) die("Bad request for conv2D: filter");
      if(OutX<=0 || OutY<=0) die("Bad request for conv2D: outSize");
      if(Sx<0 || Sy<0) die("Bad request for conv2D: padding or stride\n");
      //assert(Kn_X >= Sx && Kn_Y >= Sy && PadX < Kn_X && PadY < Kn_Y);
    #endif
  }

  // Function that initializes and constructs net and optimizer.
  // Once this is called number of layers or weights CANNOT be modified.
  Network* build(const bool isInputNet = false)
  {
    if(bBuilt) die("Cannot build the network multiple times\n");
    bBuilt = true;

    nLayers = layers.size();
    weights = allocate_parameters(layers, mpisize);
    tgt_weights = allocate_parameters(layers, mpisize);

    // Initialize weights
    for(const auto & l : layers)
      l->initialize(&generators[0], weights,
        l->bOutput && not isInputNet ? settings.outWeightsPrefac : 1);

    if(settings.learner_rank == 0) {
      for(const auto & l : layers) cout << l->printSpecs();
    }

    // Make sure that all ranks have the same weights (copy from rank 0)
    weights->broadcast(settings.mastersComm);
    //weights->allocateTransposed();
    tgt_weights->copy(weights); //copy weights onto tgt_weights

    // Allocate a gradient for each thread.
    #ifdef MIX_CMA_ADAM
      Vgrad.resize(nThreads*CMApopSize, nullptr);
    #else
      Vgrad.resize(nThreads, nullptr);
    #endif
    #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
    for (Uint i=0; i<Vgrad.size(); i++)
      #pragma omp critical // numa-aware allocation if OMP_PROC_BIND is TRUE
        Vgrad[i] = allocate_parameters(layers, mpisize);

    // Initialize network workspace to check that all is ok
    Activation*const test = allocate_activation(layers);

    if(test->nInputs not_eq (int) nInputs)
      _die("Mismatch between Builder's computed inputs:%u and Activation's:%u",
        nInputs, test->nInputs);

    if(test->nOutputs not_eq (int) nOutputs) {
      _warn("Mismatch between Builder's computed outputs:%u and Activation's:%u. Overruled Builder: probable cause is that user's net did not specify which layers are output. If multiple output layers expect trouble\n",
        nOutputs, test->nOutputs);
      nOutputs = test->nOutputs;
    }

    _dispose_object(test);

    popW = initWpop(weights, CMApopSize, mpisize);

    net = new Network(this, settings);
    if(CMApopSize>1)
    #ifdef MIX_CMA_ADAM
      opt = new AdamCMA_Optimizer(settings, weights,tgt_weights, popW, Vgrad);
    #else
      opt = new CMA_Optimizer(settings, weights,tgt_weights, popW);
    #endif
      else opt = new AdamOptimizer(settings, weights,tgt_weights, popW, Vgrad);

    return net;
  }

  // stackSimple reads from the settings file the amount of fully connected
  // layers (nnl1, nnl2, ...) and builds a network with given number of nInputs
  // and nOutputs. Supports LSTM, RNN and MLP (aka InnerProduct or Dense).
  //void stackSimple(Uint ninps,Uint nouts) { return stackSimple(ninps,{nouts}); }
  void stackSimple(const Uint ninps, const vector<Uint> nouts)
  {
    const int sumout=static_cast<int>(accumulate(nouts.begin(),nouts.end(),0));
    const string netType = settings.nnType, funcType = settings.nnFunc;
    const vector<int> lsize = settings.readNetSettingsSize();

    if(ninps == 0)
    {
      warn("network with no input space. will return a param layer");
      addParamLayer(sumout, settings.nnOutputFunc, vector<Real>(sumout,0));
      return;
    }

    addInput(ninps);

    //User can specify how many layers exist independendlty for each output
    // of the network. For example, if the settings file specifies 3 layer
    // sizes and splitLayers=1, the network will have 2 shared bottom Layers
    // (not counting input layer) and then for each of the outputs a separate
    // third layer each connected back to the second layer.
    const Uint nL = lsize.size();
    const Uint nsplit = std::min((Uint) settings.splitLayers, nL);
    const Uint firstSplit = nL - nsplit;

    for(Uint i=0; i<firstSplit; i++) addLayer(lsize[i],funcType,false,netType);
    if(sumout) {
      if(nsplit) {
        const Uint lastShared = layers.back()->number();
        for (Uint i=0; i<nouts.size(); i++) {
          //`link' specifies how many layers back should layer take input from
          // use layers.size()-lastShared >=1 to link back to last shared layer
          addLayer(lsize[lastShared], funcType, false, netType, nL-lastShared);

          for (Uint j=firstSplit+1; j<lsize.size(); j++)
            addLayer(lsize[j], funcType, false, netType);

          addLayer(nouts[i], settings.nnOutputFunc, true);
        }
      } else addLayer(sumout, settings.nnOutputFunc, true);
    }
  }

private:
  bool bBuilt = false;
public:
  const Settings & settings;
  const Uint nThreads = settings.nThreads;
  const Uint CMApopSize = settings.ESpopSize;
  const Uint mpisize = settings.learner_size;
  Uint nInputs=0, nOutputs=0, nLayers=0;
  Real gradClip = 1;
  std::vector<std::mt19937>& generators = settings.generators;
  Parameters *weights, *tgt_weights;
  vector<Parameters*> Vgrad;
  vector<Parameters*> popW;
  vector<Layer*> layers;

  Network* net = nullptr;
  Optimizer* opt = nullptr;

  Builder(const Settings& _sett) : settings(_sett) { }
};
