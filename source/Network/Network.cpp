//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Builder.h"
#include "Network.h"

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
vector<Real> Network::predict(const vector<nnReal>& _inp,
  const Activation*const prevStep, const Activation*const currStep,
  const Parameters*const _weights) const
{
  assert(_inp.size()==nInputs && layers.size()==nLayers);
  currStep->setInput(_inp);
  const Parameters*const W = _weights==nullptr? weights : _weights;
  for(Uint j=1; j<nLayers; j++) //skip 0: input layer
    layers[j]->forward(prevStep, currStep, W);
  currStep->written = true;

  return currStep->getOutput();
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
void Network::backProp( const Activation*const prevStep,
                        const Activation*const currStep,
                        const Activation*const nextStep,
                        const Parameters*const _gradient,
                        const Parameters*const _weights) const
{
  assert(currStep->written);
  _gradient->written = true;
  const Parameters*const W = _weights == nullptr ? weights : _weights;
  for (Uint i=layers.size()-1; i>0; i--) //skip 0: input layer
    layers[i]->backward(prevStep, currStep, nextStep, _gradient, W);
}

/*
  cache friendly backprop for time series: backprops from top layers to bottom
  layers and from last time step to first, with layer being the 'slow index'
  maximises reuse of weights in cache by getting each layer done in turn
*/
void Network::backProp(const vector<Activation*>& netSeries,
                       const Uint stepLastError,
                       const Parameters*const _grad,
                       const Parameters*const _weights) const
{
  assert(stepLastError <= netSeries.size());
  const Parameters*const W = _weights == nullptr ? weights : _weights;

  if (stepLastError == 0) return; //no errors placed
  else
  if (stepLastError == 1)
  { //errors placed at first time step
    assert(netSeries[0]->written);
    for(Uint i=layers.size()-1; i>0; i--)
      layers[i]->backward(nullptr, netSeries[0], nullptr, _grad, W);
  }
  else
  {
    const Uint T = stepLastError - 1;
    for(Uint i=layers.size()-1; i>0; i--) //skip 0: input layer
    {
      assert(netSeries[T]->written);
      layers[i]->backward(netSeries[T-1],netSeries[T],nullptr,        _grad,W);

      for (Uint k=T-1; k>0; k--) {
      assert(netSeries[k]->written);
      layers[i]->backward(netSeries[k-1],netSeries[k],netSeries[k+1], _grad,W);
      }

      assert(netSeries[0]->written);
      layers[i]->backward(       nullptr,netSeries[0],netSeries[1],   _grad,W);
    }
  }
  _grad->written = true;
}

Network::Network(Builder* const B, Settings & settings) :
  nAgents(B->nAgents), nThreads(B->nThreads), nInputs(B->nInputs),
  nOutputs(B->nOutputs), nLayers(B->nLayers), bDump(not settings.bTrain),
  gradClip(B->gradClip), layers(B->layers), weights(B->weights),
  tgt_weights(B->tgt_weights), Vgrad(B->Vgrad), mem(B->mem),
  generators(settings.generators) {
  updateTransposed();
  dump_ID.resize(nAgents, 0);
}

void Network::checkGrads()
{
  const Uint seq_len = 5;
  const nnReal incr = std::pow(2,-20), tol = incr;
  cout<<"Checking grads with increment "<<incr<<" and tolerance "<<tol<<endl;
  vector<Activation*> timeSeries;
  if(Vgrad.size() < 4) die("I'm the worst, just use 4 threads and forgive me");
  Vgrad[1]->clear(); Vgrad[2]->clear(); Vgrad[3]->clear();

  for(Uint t=0; t<seq_len; t++)
  for(Uint o=0; o<nOutputs; o++)
  {
    vector<vector<nnReal>> inputs(seq_len, vector<Real>(nInputs,0));
    prepForBackProp(timeSeries, seq_len);
    Vgrad[0]->clear();
    normal_distribution<nnReal> dis_inp(0, 1);
    for(Uint i=0; i<seq_len; i++)
      for(Uint j=0; j<nInputs; j++) inputs[i][j] = dis_inp(generators[0]);

    for (Uint k=0; k<seq_len; k++) {
      predict(inputs[k], timeSeries, k);
      vector<nnReal> errs(nOutputs, 0);
      if(k==t) {
        errs[o] = -1;
        timeSeries[k]->addOutputDelta(errs);
      }
    }
    backProp(timeSeries, t+1, Vgrad[0]);

    for (Uint w=0; w<weights->nParams; w++) {
      nnReal diff = 0;
      const auto copy = weights->params[w];
      //1
      weights->params[w] += incr;
      for (Uint k=0; k<seq_len; k++) {
        const vector<Real> ret = predict(inputs[k], timeSeries, k);
        if(k==t) diff = -ret[o]/(2*incr);
      }
      //2
      weights->params[w] = copy - incr;
      for (Uint k=0; k<seq_len; k++) {
        const vector<Real> ret = predict(inputs[k], timeSeries, k);
        if(k==t) diff += ret[o]/(2*incr);
      }
      //0
      weights->params[w] = copy;

      //const nnReal scale = std::max( fabs(Vgrad[0]->params[w]), fabs(diff) );
      //if (scale < nnEPS) continue;
      const nnReal err = fabs(Vgrad[0]->params[w]-diff);//relerr=err/scale;
      // if error now is bigger or if equal but grad magnitude is greater
      if( err>Vgrad[2]->params[w] || ( err>=Vgrad[2]->params[w] &&
         std::fabs(Vgrad[1]->params[w]) < std::fabs(Vgrad[0]->params[w]) ) ) {
        Vgrad[1]->params[w] = Vgrad[0]->params[w];
        Vgrad[2]->params[w] = err;
        Vgrad[3]->params[w] = diff;
      }
    }
  }

  long double sum1 = 0, sumsq1 = 0, sum2 = 0, sumsq2 = 0;
  for (Uint w=0; w<weights->nParams; w++) {
    if(Vgrad[2]->params[w]>tol)
    cout<<w<<" err:"<<Vgrad[2]->params[w]<<", grad:"<<Vgrad[1]->params[w]
        <<" diff:"<<Vgrad[3]->params[w]<<" param:"<<weights->params[w]<<endl;

    sum1+=std::fabs(Vgrad[1]->params[w]); sum2+=std::fabs(Vgrad[2]->params[w]);
    sumsq1 += Vgrad[1]->params[w]*Vgrad[1]->params[w];
    sumsq2 += Vgrad[2]->params[w]*Vgrad[2]->params[w];
  }

  long double NW = weights->nParams, avg1 = sum1/NW, avg2 = sum2/NW;
  auto std1=sqrt((sumsq1-sum1*avg1)/NW), std2=sqrt((sumsq2-sum2*avg2)/NW);
  cout<< "Abs gradient avg:" <<avg1<<" std:"<<std1
      <<" Abs error avg:"<<avg2<<" std:"<<std2<<endl;
  deallocateUnrolledActivations(&timeSeries);
  Vgrad[0]->clear(); Vgrad[1]->clear(); Vgrad[2]->clear(); Vgrad[3]->clear();
  die("done");
}

#if 0
void Network::dump(const int agentID)
{
  if (not bDump) return;
  char buf[500];
  sprintf(buf, "%07u", (Uint)dump_ID[agentID]);
  string nameNeurons  = "neuronOuts_"+to_string(agentID)+"_"+string(buf)+".dat";
  string nameMemories = "cellStates_"+to_string(agentID)+"_"+string(buf)+".dat";
  string nameOut_Mems = "out_states_"+to_string(agentID)+"_"+string(buf)+".dat";
  {
    ofstream out(nameOut_Mems.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameOut_Mems.c_str());
    for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
    for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
    out << "\n";
    out.close();
  }
  {
    ofstream out(nameNeurons.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameNeurons.c_str());
    for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
    out << "\n";
    out.close();
  }
  {
    ofstream out(nameMemories.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameMemories.c_str());
    for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
    out << "\n";
    out.close();
  }
  dump_ID[agentID]++;
}
#endif
