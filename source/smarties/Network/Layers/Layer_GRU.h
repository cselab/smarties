//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MGULayer_h
#define smarties_MGULayer_h

#include "Layers.h"

namespace smarties
{

class MGULayer: public Layer
{
  // MGU (Minimal Gated Unit)  input(t) -> [GRU] -> output(t)
  // forget(t) = sigmoid (Wfr output(t-1) + Wff input(t) + bf)
  // state(t)  =    tanh (Wsr [forget(t) * output(t-1)] + Wsf input(t) + bs
  // output(t) = (1 - forget(t)) * output(t-1) + forget(t) * state(t)
  // Where * and + are element-wise ops, weight-vector multiplication is implied
  const Uint nInputs, nCells;
  const std::unique_ptr<Function> cell;

 public:
  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override
  {
    //cell, input, forget, output gates all linked to inp and prev LSTM output
    nWeight.push_back(2*nCells * (nInputs + nCells) );
    nBiases.push_back(2*nCells);
  }

  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(2*nCells);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }

  MGULayer(Uint _ID, Uint _nInputs, Uint _nCells, std::string funcType,
    bool bOut, Uint iLink) :  Layer(_ID, _nCells, bOut, false, iLink),
    nInputs(_nInputs), nCells(_nCells), cell(makeFunction(funcType)) {
    spanCompInpGrads = _nInputs;
    if(_nCells % ARY_WIDTH)
      die("hardcoded simd: pick size multiple of 8 for float and 4 for double");
  }

  std::string printSpecs() const override
  {
    std::ostringstream o;
    o<<"("<<ID<<") "<<cell->name()
     <<std::string(bOutput? " output ":" ")
     <<"MGU Layer of size:"<<nCells
     <<" linked to Layer:"<<ID-link
     <<" of size:"<<nInputs<<"\n";
    return o.str();
  }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    // linearOutput contains input to all cell inputs and gates
    // it then gets split into first nCell components and last nCell components
    nnReal* const forget = curr->X(ID);          // first nCell is forget gate
    nnReal* const state  = curr->X(ID) + nCells; // last nCell is cell state
    nnReal* const output = curr->Y(ID);          // MGU output
    // para->W(ID) contains [Wff Wsf Wfr Wsr] := [ weights feedforward forget,
    // w ff cellstate, w recurrent forget, w recur cellstate ]
    {
      nnReal* const linearOutput = curr->X(ID); // both forget and cell state
      memcpy(linearOutput, para->B(ID), 2*nCells*sizeof(nnReal)); // add bias
      const nnReal* const inputs = curr->Y(ID-link); // output of prev layer
      const nnReal* const weight = para->W(ID); // weights for feedforward op
      for (Uint i = 0; i < nInputs; ++i) {
        const nnReal* const W = weight + (2*nCells)*i;
        #pragma omp simd aligned(linearOutput, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < 2*nCells; ++o) linearOutput[o] += inputs[i] * W[o];
      }
    }

    if(prev not_eq nullptr) // if not at first time step
    {
      // forget = = sigm [ Wfr prevOutput + Wff inputs + b ]
      const nnReal* const inputs = prev->Y(ID);
      // recurrent connection weights are shifted by (2*nCells)*nInputs:
      const nnReal* const weightRecur = para->W(ID) +(2*nCells)*nInputs;
      for (Uint i=0; i<nCells; ++i) {
        const nnReal* const Wfr = weightRecur + (2*nCells)*i;
        #pragma omp simd aligned(forget, inputs, Wfr : VEC_WIDTH)
        for(Uint o=0; o<nCells; ++o) forget[o] += Wfr[o] * inputs[i];
      }
      Sigm::_eval(forget, forget, nCells);
      // state = tanh [ Wsr (forget \elemProd prevOut) + Wsf inputs + b ]
      for (Uint i=0; i<nCells; ++i) {
        const nnReal* const Wsr = weightRecur + (2*nCells)*i +nCells;
        #pragma omp simd aligned(state, forget, inputs, Wsr : VEC_WIDTH)
        for(Uint o=0; o<nCells; ++o) state[o] += Wsr[o] * inputs[i] * forget[i];
      }
      Tanh::_eval(state, state, nCells);
      // output = = (1 - forget) \elemProd prevOut + forget \elemProd state
      #pragma omp simd aligned(output, forget, inputs, state : VEC_WIDTH)
      for (Uint o=0; o<nCells; ++o)
        output[o] = forget[o]*state[o] + (1-forget[o])*inputs[o];
    }
    else
    {
      Sigm::_eval(forget, forget, nCells);
      Tanh::_eval(state, state, nCells);
      #pragma omp simd aligned(output, forget, state : VEC_WIDTH)
      for (Uint o=0; o<nCells; ++o) output[o] = forget[o]*state[o];
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
    using Utilities::allocate_ptr;
    const nnReal* const forget = curr->X(ID);
    const nnReal* const state  = curr->X(ID) + nCells;
    const nnReal* const dLdO = curr->E(ID); // dLossdGRU, comes from backprop
    nnReal* const dLdF = curr->E(ID) + nCells; // dLoss dForgetGate
    // curr->Y(ID) + nCells is unused memory: used here to store dLoss dState
    nnReal* const dLdS = curr->Y(ID) + nCells;
    nnReal* const prevOut = prev==nullptr? allocate_ptr(nCells) : prev->Y(ID);
    nnReal* const dLdprevOut = prev==nullptr? nullptr : prev->E(ID);
    // temp buffer for dLoss d(forget * previousInput) through state update
    nnReal* const dLdFprevOut = allocate_ptr(nCells);

    // 1) dLdS = forget * dLdO * tanh' (so it is actually dLoss d InputToTanh)
    #pragma omp simd aligned(dLdS, dLdO, forget, state : VEC_WIDTH)
    for (Uint o=0; o<nCells; ++o)
      dLdS[o] = dLdO[o] * forget[o] * (1-state[o]*state[o]);

    // 2) dLdFprevOut = Wsr * dLdS
    if(prev not_eq nullptr)
    {
      const nnReal*const Wsr = para->W(ID) + (2*nCells)*nInputs + nCells;
      #ifdef USE_OMPSIMD_BLAS
        GEMVomp(nCells, nCells, 2*nCells, Wsr, dLdS, dLdFprevOut);
      #else
        SMARTIES_gemv(CblasRowMajor, CblasNoTrans, nCells, nCells, 1,
          Wsr, 2*nCells, dLdS, 1, 0, dLdFprevOut, 1);
      #endif
    }

    // 3) dLdF = ((state - prevOut) * dLdO + dLdFprevOut * prevOut) * sigm'
    #pragma omp simd aligned(dLdF,prevOut,state,dLdO,forget,dLdFprevOut : VEC_WIDTH)
    for (Uint o=0; o<nCells; ++o)
      dLdF[o] = ((state[o]-prevOut[o])*dLdO[o] + dLdFprevOut[o]*prevOut[o]) *
                forget[o] * (1-forget[o]);

    // 4) dLdprevOut = (1-forget)*dLdO + dLdFprevOut*forget + Wfr*dFdL
    if(prev not_eq nullptr)
    {
      #pragma omp simd aligned(dLdprevOut,forget,dLdO,dLdFprevOut : VEC_WIDTH)
      for(Uint o=0; o<nCells; ++o) // first two terms of 4) are elt-wise:
        dLdprevOut[o] += (1-forget[o])*dLdO[o] + forget[o]*dLdFprevOut[o];
      // last term of 4):
      const nnReal * const Wfr = para->W(ID) +(2*nCells)*nInputs;
      #ifdef USE_OMPSIMD_BLAS
        GEMVomp(nCells, nCells, 2*nCells, Wfr, dLdF, dLdprevOut);
      #else
        SMARTIES_gemv(CblasRowMajor, CblasNoTrans, nCells, nCells, 1,
          Wfr, 2*nCells, dLdF, 1, 1, dLdprevOut, 1);
      #endif
    }
    free(dLdFprevOut);

    // backprop dL to input dLdI = Wff * dLdF + Wsf * dLdS
    if( spanCompInpGrads )
    {
      nnReal* const dLdInput = curr->E(ID-link) + startCompInpGrads;
      const nnReal* const Wff = para->W(ID) +startCompInpGrads*2*nCells;
      const nnReal* const Wsf = para->W(ID) +startCompInpGrads*2*nCells +nCells;
      #ifdef USE_OMPSIMD_BLAS
        GEMVomp(nCells, spanCompInpGrads, 2*nCells, Wff, dLdF, dLdInput);
        GEMVomp(nCells, spanCompInpGrads, 2*nCells, Wsf, dLdS, dLdInput);
      #else
        SMARTIES_gemv(CblasRowMajor, CblasNoTrans, spanCompInpGrads, nCells, 1,
          Wff, 2*nCells, dLdF, 1, 1, dLdInput, 1);
        SMARTIES_gemv(CblasRowMajor, CblasNoTrans, spanCompInpGrads, nCells, 1,
          Wsf, 2*nCells, dLdS, 1, 1, dLdInput, 1);
      #endif
    }

    if(prev==nullptr) { free(prevOut); }

    if(grad == nullptr) return; // then no need to compute grad w.r.t. params

    {
      nnReal* const grad_b = grad->B(ID);
      #pragma omp simd aligned(grad_b, dLdF, dLdS : VEC_WIDTH)
      for(Uint o=0; o<nCells; ++o) {
        grad_b[o]        += dLdF[o];
        grad_b[o+nCells] += dLdS[o];
      }
    }

    {
      const nnReal* const inputs = curr->Y(ID-link);
      for(Uint i=0; i<nInputs;  ++i) {
        nnReal* const G = grad->W(ID) + (2*nCells)*i;
        #pragma omp simd aligned(G, inputs, dLdF, dLdS : VEC_WIDTH)
        for(Uint o=0; o<nCells; ++o) {
          G[o]        += inputs[i] * dLdF[o];
          G[o+nCells] += inputs[i] * dLdS[o];
        }
      }
    }

    if(prev not_eq nullptr)
    {
      for(Uint i=0; i<nCells; ++i) {
        nnReal* const G = grad->W(ID) + 2*nCells * (nInputs + i);
        #pragma omp simd aligned(G, prevOut, dLdF, dLdS, forget : VEC_WIDTH)
        for(Uint o=0; o<nCells; ++o) {
          G[o]        += prevOut[i] * dLdF[o];
          G[o+nCells] += prevOut[i] * dLdS[o] * forget[i];
        }
      }
    }
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * cell->initFactor(nInputs, nCells);
    std::uniform_real_distribution<nnReal> dis(-init, init);
    { // forget gate starts open, inp/out gates are closed
      nnReal* const BB = W->B(ID);
      for(Uint o=0*nCells; o<1*nCells; ++o) BB[o] = 0+LSTM_PRIME_FAC;
      for(Uint o=1*nCells; o<2*nCells; ++o) BB[o] = 0;
    }
    {
      nnReal* const weight = W->W(ID);
      for(Uint w=0; w<2*nCells*(nInputs+nCells); ++w) weight[w] = dis(G);
    }
  }

  size_t  save(const Parameters * const para,
                          float * tmp) const override
  {
    const nnReal* const bias = para->B(ID);
    const nnReal* const weight = para->W(ID);
    for (Uint n=0; n<2*nCells * (nInputs+nCells); ++n)
      *(tmp++) = (float) weight[n];
    for (Uint n=0; n<2*nCells; ++n)
      *(tmp++) = (float) bias[n];
    return 2*nCells * (nInputs+nCells + 1);
  }
  size_t restart(const Parameters * const para,
                    const float * tmp) const override
  {
    nnReal* const bias = para->B(ID);
    nnReal* const weight = para->W(ID);
    for (Uint n=0; n<2*nCells * (nInputs+nCells); ++n)
      weight[n] = (nnReal) *(tmp++);
    for (Uint n=0; n<2*nCells; ++n)
      bias[n] = (nnReal) *(tmp++);
    return 2*nCells * (nInputs+nCells + 1);
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
