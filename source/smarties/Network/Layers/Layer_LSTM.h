//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_LSTMLayer_h
#define smarties_LSTMLayer_h

#include "Layers.h"

namespace smarties
{

class LSTMLayer: public Layer
{
  const Uint nInputs, nCells;
  const std::unique_ptr<Function> cell;

 public:
  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override
  {
    //cell, input, forget, output gates all linked to inp and prev LSTM output
    nWeight.push_back(4*nCells * (nInputs + nCells) );
    nBiases.push_back(4*nCells);
  }
  /*
  organization of Activation work memory:
  `suminps` field  spans 4 blocks each of size nCells. Each contains the result
    from a matvec multiplication: for the cell's input neuron and for each gate.
    Gates during forward overwrite their suminps with the output of the sigmoid.
        nCells               nCells             nCells             nCells
    |================| |================| |================| |================|
       cell' Input        input Gate        forget Gate         output Gate

  `outvals` field is more complex. First nCells fields will be read by upper
   layer and by recurrent connection at next time step therefore contain LSTM
   cell output. Then we store states, cell output b4 outpGate, and dErr/dState
    |================| |================| |================| |================|
     LSTM layer output    cell states      pre-Ogate output   state error signal

   `errvals`: simple again to do backprop with `gemv'
    |================| |================| |================| |================|
        dE/dInput        dE/dInput Gate     dE/dForget Gate    dE/dOutput Gate
  */
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override
  {
    sizes.push_back(4*nCells);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }

  LSTMLayer(Uint _ID, Uint _nInputs, Uint _nCells, std::string funcType,
    bool bOut, Uint iLink) :  Layer(_ID, _nCells, bOut, false, iLink),
    nInputs(_nInputs), nCells(_nCells), cell(makeFunction(funcType))
  {
    spanCompInpGrads = _nInputs;
  }

  std::string printSpecs() const override
  {
    std::ostringstream o;
    o<<"("<<ID<<") "<<cell->name()
     <<std::string(bOutput? " output ":" ")
     <<"LSTM Layer of size:"<<nCells
     <<" linked to Layer:"<<ID-link
     <<" of size:"<<nInputs<<"\n";
    return o.str();
  }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    // suminp contains input to all cell inputs and gates
    // only one matrix-vector multiplication
    nnReal* const suminp = curr->X(ID);
    memcpy(suminp, para->B(ID), 4*nCells*sizeof(nnReal));
    {
      const nnReal* const inputs = curr->Y(ID-link);
      const nnReal* const weight = para->W(ID);
      for (Uint i = 0; i < nInputs; ++i) {
        const nnReal* const W = weight + (4*nCells)*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < 4*nCells; ++o) suminp[o] += inputs[i] * W[o];
      }
    }

    if(prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
      const nnReal* const weight = para->W(ID) +(4*nCells)*nInputs;
      //first input loop, here input only prev step LSTM's output
      for (Uint i = 0; i < nCells; ++i) {
        const nnReal* const W = weight + (4*nCells)*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < 4*nCells; ++o) suminp[o] += inputs[i] * W[o];
      }
    }
    {
      // Input, forget, output gates output overwrite their input
      Sigm::_eval(suminp +nCells, suminp +nCells, 3*nCells);

      // state is placed onto output work mem, shifted by nCells
      const nnReal*const prevSt = prev==nullptr? nullptr : prev->Y(ID)+nCells;
            nnReal*const output = curr->Y(ID)+ 0*nCells;
            nnReal*const currSt = curr->Y(ID)+ 1*nCells;
            nnReal*const cellOp = curr->Y(ID)+ 2*nCells;
      const nnReal*const inputG = curr->X(ID)+ 1*nCells;
      const nnReal*const forgtG = curr->X(ID)+ 2*nCells;
      const nnReal*const outptG = curr->X(ID)+ 3*nCells;

      for (Uint o=0; o<nCells; ++o) {
       const nnReal oldStatePass = prev==nullptr? 0 : prevSt[o] * forgtG[o];
       currSt[o] = suminp[o] * inputG[o] + oldStatePass;
       cellOp[o] = Tanh::_eval(currSt[o]);
       output[o] = outptG[o] * cellOp[o];
      }
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
          nnReal*const deltas = curr->E(ID); //error signal from above/future
    // Also need pre-outGate cell output
    const nnReal*const cellOutput = curr->Y(ID) + 2*nCells;
    // Will also need to copy the state's error signal, use last available slot:
          nnReal*const stateDelta = curr->Y(ID) + 3*nCells;

    const nnReal*const cellInpt = curr->X(ID);
    const nnReal*const IGate = curr->X(ID)+ 1*nCells;
    const nnReal*const FGate = curr->X(ID)+ 2*nCells;
    const nnReal*const OGate = curr->X(ID)+ 3*nCells;
    // prevState, nextState's delta and next output of forget gate
    const nnReal*const prvState = prev==nullptr? nullptr :prev->Y(ID) +1*nCells;
    const nnReal*const nxtStErr = next==nullptr? nullptr :next->Y(ID) +3*nCells;
    const nnReal*const nxtFGate = next==nullptr? nullptr :next->X(ID) +2*nCells;

    for (Uint o=0; o<nCells; ++o) {
      const nnReal D = deltas[o]; //before overwriting it
      //                  |      derivative of tanh     |
      const nnReal diff = (1-cellOutput[o]*cellOutput[o]) * deltas[o];
      // Compute state's error signal
      stateDelta[o] = diff*OGate[o] +(next==nullptr?0: nxtStErr[o]*nxtFGate[o]);
      // Compute deltas for cell input and gates
      deltas[o+0*nCells] = IGate[o] * stateDelta[o];
      //                  | sigmoid derivative |
      deltas[o+1*nCells] = IGate[o]*(1-IGate[o]) * cellInpt[o] * stateDelta[o];
      if(prev not_eq nullptr)
      deltas[o+2*nCells] = FGate[o]*(1-FGate[o]) * prvState[o] * stateDelta[o];
      else deltas[o+2*nCells] = 0;
      deltas[o+3*nCells] = OGate[o]*(1-OGate[o]) * D * cellOutput[o];
    }

    Layer::backward(nInputs, 4*nCells, 4*nCells, nCells, prev,curr,next, grad,para);
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * cell->initFactor(nInputs, nCells);
    std::uniform_real_distribution<nnReal> dis(-init, init);
    { // forget gate starts open, inp/out gates are closed
     nnReal* const BB = W->B(ID);
     for(Uint o=0*nCells; o<1*nCells; ++o) BB[o]=0;
     //for(Uint o=1*nCells; o<2*nCells; ++o) BB[o]=dis(*gen)+LSTM_PRIME_FAC;
     //for(Uint o=2*nCells; o<3*nCells; ++o) BB[o]=dis(*gen)-LSTM_PRIME_FAC;
     //for(Uint o=3*nCells; o<4*nCells; ++o) BB[o]=dis(*gen)+LSTM_PRIME_FAC;
     for(Uint o=1*nCells; o<2*nCells; ++o) BB[o]=0-LSTM_PRIME_FAC;
     for(Uint o=2*nCells; o<3*nCells; ++o) BB[o]=0+LSTM_PRIME_FAC;
     for(Uint o=3*nCells; o<4*nCells; ++o) BB[o]=0-LSTM_PRIME_FAC;
    }
    {
     nnReal* const weight = W->W(ID);
     for(Uint w=0; w<4*nCells*(nInputs+nCells); ++w) weight[w] = dis(G);
    }
  }
  size_t  save(const Parameters * const para,
                          float * tmp) const override
  {
    const nnReal* const bias = para->B(ID);
    const nnReal* const weight = para->W(ID);
    for (Uint n=0; n<4*nCells * (nInputs+nCells); ++n)
        *(tmp++) = (float) weight[n];
    for (Uint n=0; n<4*nCells; ++n)
        *(tmp++) = (float) bias[n];
    return 4*nCells * (nInputs+nCells + 1);
  }
  size_t restart(const Parameters * const para,
                    const float * tmp) const override
  {
    nnReal* const bias = para->B(ID);
    nnReal* const weight = para->W(ID);
    for (Uint n=0; n<4*nCells * (nInputs+nCells); ++n)
        weight[n] = (nnReal) *(tmp++);
    for (Uint n=0; n<4*nCells; ++n)
        bias[n] = (nnReal) *(tmp++);
    return 4*nCells * (nInputs+nCells + 1);
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
